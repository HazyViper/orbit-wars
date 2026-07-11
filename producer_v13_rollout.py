"""producer_v13_rollout — producer_v12_rollout の計算精度・汎化性を改良.

ベースは v12（producer_v11 + value-free 深い MC rollout 選択）。各ターン:
  1. producer で max_waves を振った候補プラン（no-op / 中 / フル）を生成
  2. 各候補を「自分の手→以降両者 producer」で **深さ DEPTH ターン**ロールアウト
  3. ロールアウト後の評価で最良プランを選択
学習 value を使わない → optimizer's curse 免疫。DEPTH は >=12 必須。

v12 → v13 の変更（計算精度・汎化性）:
  A. シミュレータの copy() を deepcopy から手書き再構築に置換（simulator/state.py）。
     step() の約80%を占めていた deepcopy を排し、エンジンを ~4x 高速化。
     意味は deepcopy と等価（simulator の golden test 44件で検証済）。
  B. 時間校正をハードコード係数(×1.1)から**実測エンジンコスト**に置換。
     v12 は per-step を t_plan×num_players×1.1 と仮定。A の高速化でエンジン実コスト比は
     ~3% に低下し 1.1 は不正確。v13 は engine step を毎ターン1回計測し
     unit=(t_plan×num_players + t_engine)×SAFETY とする（任意の実機速度に正しく適応）。
  C. リーフ評価に生産力項（_PROD_W）。最終総艦数（500ターン目的）の予測精度向上。
     leaf_ships は depth 分の生産しか織り込まないため残り horizon の farming を
     prod_w×production で線形外挿。_PROD_W=0 なら v12 と完全一致。

提出: orbit_lite/（torch）+ simulator/（standalone）+ 本ファイルを同梱。torch 提出可。

Attribution: ベース planner（producer_v11 / orbit_lite）は slawekbiel 氏の
Kaggle 公開ノートブック "The Producer V2" の設計に基づく移植です。
    https://www.kaggle.com/code/slawekbiel/the-producer-v2
"""

from __future__ import annotations

import dataclasses
import importlib.util as _ilu
import os
import sys
import time


def _setup_paths() -> str:
    cands = []
    try:
        cands.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    cands.append(os.getcwd())
    cands.extend([p for p in sys.path if p])
    roots = []
    for c in cands:
        for d in (c, os.path.dirname(c)):
            if d and d not in roots:
                roots.append(d)
    root = None
    for d in roots:
        try:
            entries = set(os.listdir(d))
        except OSError:
            continue
        if "orbit_lite" in entries and "simulator" in entries:
            root = d
            break
    if root is None:
        # フォールバック: orbit_lite を持つ root + simulator を別所から探す
        for d in roots:
            try:
                if "orbit_lite" in set(os.listdir(d)):
                    root = d
                    break
            except OSError:
                continue
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    # simulator が root 直下にない場合、兄弟 exp1 を探す
    if not os.path.isdir(os.path.join(root, "simulator")):
        exp1 = os.path.join(os.path.dirname(root), "exp1")
        if os.path.isdir(os.path.join(exp1, "simulator")) and exp1 not in sys.path:
            sys.path.insert(0, exp1)
    return root


_ROOT = _setup_paths()

import torch

_num_threads = os.environ.get("TORCH_NUM_THREADS")
if _num_threads is not None:
    torch.set_num_threads(int(_num_threads))

from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.movement_step import disambiguate_duplicate_launches, ensure_planet_movement
from orbit_lite.planner_core import (
    entries_to_sparse_payload, largest_initial_player_count,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

from simulator.state import GameState
from simulator.turn_logic import step as _engine_step

# producer_v11 本体（plan_lite_waves と config を流用）
def _load_base():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else _ROOT
    for cand in (os.path.join(here, "producer_v11.py"),
                 os.path.join(_ROOT, "agents", "producer_v11.py"),
                 os.path.join(_ROOT, "producer_v11.py")):
        if os.path.exists(cand):
            spec = _ilu.spec_from_file_location("producer_v11_base_v12", cand)
            mod = _ilu.module_from_spec(spec)
            sys.modules["producer_v11_base_v12"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError("producer_v11.py not found")


_base = _load_base()
plan_lite_waves = _base.plan_lite_waves
_movement_config = _base._movement_config
_config_for = _base._config_for

# --- 設定（env で上書き可）---------------------------------------------------
# DEPTH: ロールアウト深さ（>=12 必須。浅いと飛行中の艦を評価して壊滅）。
# 2P は 1 ステップが相手1体ぶんで安いので深く（16）回せる → 81%→97.5% に大幅向上。
# 4P は相手3体ぶんで高く timeout リスクがあるので 12。
_DEPTH_2P = int(os.environ.get("EXP4_LA_DEPTH_2P", "16"))
_DEPTH_4P = int(os.environ.get("EXP4_LA_DEPTH_4P", "12"))
# 後方互換: EXP4_LA_DEPTH を指定したら両方を上書き。
_DEPTH_OVERRIDE = os.environ.get("EXP4_LA_DEPTH")
if _DEPTH_OVERRIDE is not None:
    _DEPTH_2P = _DEPTH_4P = int(_DEPTH_OVERRIDE)


_MIN_DEPTH = int(os.environ.get("EXP4_LA_MIN_DEPTH", "12"))  # これ未満は壊滅するので使わない


def _depth_cap(num_players):
    return _DEPTH_2P if int(num_players) <= 2 else _DEPTH_4P
# 候補の max_waves（攻撃性スペクトラム）。v11 既定=6 を必ず含む。
_WAVES = [int(x) for x in os.environ.get("EXP4_LA_WAVES", "3,6").split(",")]
# 1 ターンの wall-clock 予算（秒）。actTimeout=1s に対し margin を残す。
# 自己校正: 1ステップ probe で実機速度を測り、full rollout が収まらない見込みなら
# rollout を行わず v11 プランにフォールバック（遅い Kaggle ハードでも timeout 回避）。
_TIME_BUDGET = float(os.environ.get("EXP4_LA_BUDGET", "0.80"))
_SAFETY = float(os.environ.get("EXP4_LA_SAFETY", "1.3"))  # per-step 推定の安全係数
# C: リーフ評価の生産力項の重み。leaf_ships に prod_w×production を加算し、
# 残り horizon の farming を線形外挿する。0.0 で v12（実艦アドバンテージのみ）と完全一致。
# 既定 16.0 = rollout depth に一致（farming をもう 1 horizon 外挿）。評価で決定:
#   vs v11(2P) prod_w 0→86.7% / 8→96.7% / 16→100% / 32→100%（同一シード, n=30）。
#   vs v117/v60b は全 prod_w で 96.7-100% に飽和、4P 1位率も中立 → どこでも悪化なし。
_PROD_W = float(os.environ.get("EXP4_LA_PROD_W", "16.0"))


def _plan_moves(obs, obs_tensors, player_id, player_count, config):
    obs_p = parse_obs(obs_tensors)
    if obs_p.P == 0:
        return []
    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=player_count),
        cached_movement=None,
    )
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]
    planet_ids = obs_tensors["planets"][..., 0].long()
    entries = plan_lite_waves(
        movement=movement, obs=obs_p, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=player_count,
    )
    entries = disambiguate_duplicate_launches(entries)
    row = entries_to_sparse_payload(entries, planet_ids=planet_ids)
    return sparse_action_row_to_moves(row, obs, player_id=player_id)


def _v11_moves_for(state, p, num_players):
    obs = state.to_observation(p)
    obst = single_obs_to_tensor(obs, player_id=p, device="cpu")
    cfg = _config_for(num_players, player_id=p)
    return _plan_moves(obs, obst, p, num_players, cfg)


def _move_key(moves):
    return tuple(sorted((int(m[0]), round(float(m[1]), 4), int(m[2])) for m in moves))


def _v11_full_plan(obs, obs_tensors, player_id, player_count):
    """v11 の既定プラン（max_waves 既定）。安全 fallback 兼 最攻撃候補。"""
    config0 = _config_for(player_count, player_id=player_id)
    return _plan_moves(obs, obs_tensors, player_id, player_count, config0)


def _extra_candidates(obs, obs_tensors, player_id, player_count, exclude_key):
    """v11 フル以外の候補（less-aggressive + no-op）。重複・exclude を除く。"""
    config0 = _config_for(player_count, player_id=player_id)
    default_waves = int(config0.max_waves_per_turn)
    ws = sorted(set(w for w in _WAVES if w < default_waves), reverse=True)
    out = []
    seen = {exclude_key, ()}
    for w in ws:
        cfg = dataclasses.replace(config0, max_waves_per_turn=int(w))
        moves = _plan_moves(obs, obs_tensors, player_id, player_count, cfg)
        k = _move_key(moves)
        if k not in seen:
            seen.add(k)
            out.append(moves)
    out.append([])  # no-op
    return out


def _terminal(state, num_players):
    owners = set()
    for pl in state.planets:
        o = int(pl.to_list()[1])
        if o >= 0:
            owners.add(o)
    return int(getattr(state, "step", 0)) >= 500 or len(owners) <= 1


def _ship_advantage(state, player_id, num_players, prod_w=0.0):
    """リーフ評価: (my - 最強敵) / (my + 最強敵 + 1)。

    各プレイヤーの総艦数（惑星駐留 + 飛行中）を集計する。prod_w>0 のとき、所有惑星の
    production を prod_w 倍して加算し、残り horizon の farming（将来の艦生成）を線形外挿
    する（最終総艦数の予測精度向上）。prod_w=0 で v12 と完全一致。
    """
    totals = [0.0] * num_players
    for pl in state.planets:
        row = pl.to_list()
        o = int(row[1])
        if 0 <= o < num_players:
            totals[o] += float(row[5]) + prod_w * float(row[6])
    for fl in state.fleets:
        row = fl.to_list()
        o = int(row[1])
        if 0 <= o < num_players:
            totals[o] += float(row[6])
    my = totals[player_id]
    enemies = [totals[q] for q in range(num_players) if q != player_id]
    me = max(enemies) if enemies else 0.0
    return (my - me) / (my + me + 1.0)


def _rollout_score(state, my_moves, player_id, num_players, depth, prod_w=0.0):
    actions = [[] for _ in range(num_players)]
    actions[player_id] = my_moves
    for opp in range(num_players):
        if opp != player_id:
            actions[opp] = _v11_moves_for(state, opp, num_players)
    s = _engine_step(state.copy(), actions)
    for _ in range(max(0, depth - 1)):
        if _terminal(s, num_players):
            break
        acts = [_v11_moves_for(s, p, num_players) for p in range(num_players)]
        s = _engine_step(s, acts)
    return _ship_advantage(s, player_id, num_players, prod_w=prod_w)


def agent(obs):
    t0 = time.time()
    player_id = int(obs.get("player", 0) if isinstance(obs, dict) else obs.player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id, device="cpu")
    num_players = largest_initial_player_count(obs_tensors)

    # 1) v11 フルプラン（安全 fallback）。これだけは必ず計算し、その所要で実機速度を自己校正する。
    tp = time.time()
    v11_moves = _v11_full_plan(obs, obs_tensors, player_id, num_players)
    t_plan = time.time() - tp                           # producer 1 回の所要（≒ v11 の per-turn コスト）
    if (time.time() - t0) > _TIME_BUDGET:
        return v11_moves

    # state 構築（ロールアウト + エンジンコスト計測に必要）。
    try:
        state = GameState.from_observation(obs, num_players=num_players)
    except Exception:
        return v11_moves

    # 2) 適応的 depth: 計測した実機速度で「2 候補が予算内に収まる最大 depth」を選ぶ。
    #    深いほど強い（depth12→16 で 81%→97.5%）が、深いほど 1 ロールアウトが高コスト。
    #    per rollout-step = num_players 回のプランナ(t_plan) + 1 回のエンジン step。
    #    v12 はエンジン step を t_plan×0.1 とハードコード(×1.1)していたが、copy() 高速化で
    #    実比は ~3% に低下。v13 はエンジン step を実測してハードコード比を排す（汎化性）。
    te = time.time()
    try:
        _engine_step(state.copy(), [v11_moves if p == player_id else []
                                    for p in range(num_players)])
    except Exception:
        return v11_moves
    t_engine = time.time() - te
    unit = (t_plan * num_players + t_engine) * _SAFETY  # 1 ロールアウトの depth あたりコスト
    budget_left = _TIME_BUDGET - (time.time() - t0)
    max_depth = int(budget_left / (2.0 * unit)) if unit > 0 else 0   # 2 候補が収まる最大 depth
    depth = min(_depth_cap(num_players), max_depth)
    if depth < _MIN_DEPTH:
        # depth>=12 で 2 候補すら afford できない → v11 フォールバック（timeout/後退なし）。
        return v11_moves
    est_rollout = unit * depth
    n_afford = int(budget_left / est_rollout)          # 構成上 >= 2

    # 3) 追加候補生成 → n_afford 個を depth でロールアウトし最良を選ぶ。
    cand = [v11_moves] + _extra_candidates(obs, obs_tensors, player_id, num_players, _move_key(v11_moves))
    best_moves, best_v = v11_moves, -1e9
    for i, moves in enumerate(cand):
        if i >= n_afford:
            break
        if best_v > -1e9 and (time.time() - t0) + est_rollout > _TIME_BUDGET:
            break
        try:
            v = _rollout_score(state, moves, player_id, num_players, depth, prod_w=_PROD_W)
        except Exception:
            continue
        if v > best_v:
            best_v, best_moves = v, moves
    return best_moves
