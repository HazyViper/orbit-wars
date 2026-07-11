"""Orbit Wars シミュレータの状態表現。

公式 orbit_wars.py の list ベース表現を、安全な dataclass で包む。
両者の相互変換は from_observation / to_observation で提供する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .utils import COMET_SPAWN_STEPS


def observation_get(obs: Any, key: str, default: Any = None) -> Any:
    """kaggle の observation が dict でも SimpleNamespace でも読めるようにする。"""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


@dataclass
class PlanetState:
    """惑星1個の状態（list [id, owner, x, y, radius, ships, production] に対応）。"""

    id: int  # 惑星 ID（一意な整数）
    owner: int  # 所有者プレイヤーインデックス（-1 は中立）
    x: float  # 盤面上の x 座標（タイル座標系）
    y: float
    radius: float  # 描画・衝突判定用の半径（ゲーム定数に依存）
    ships: int  # 現在停泊中の艦数
    production: int  # 毎ターンの生産量（production）

    @classmethod
    def from_list(cls, lst: List[Any]) -> PlanetState:
        """公式形式の list から PlanetState を構築する。"""
        return cls(
            id=lst[0],
            owner=lst[1],
            x=lst[2],
            y=lst[3],
            radius=lst[4],
            ships=lst[5],
            production=lst[6],
        )

    def to_list(self) -> List[Any]:
        """公式 observation 用の list に戻す（可逆変換のため型は list のまま）。"""
        return [
            self.id,
            self.owner,
            self.x,
            self.y,
            self.radius,
            self.ships,
            self.production,
        ]


@dataclass
class FleetState:
    """艦隊1つ（list [id, owner, x, y, angle, from_planet_id, ships]）。"""

    id: int
    owner: int
    x: float
    y: float
    angle: float  # 進行方向（ラジアン）
    from_planet_id: int  # 出発した惑星 ID
    ships: int

    @classmethod
    def from_list(cls, lst: List[Any]) -> FleetState:
        return cls(
            id=lst[0],
            owner=lst[1],
            x=lst[2],
            y=lst[3],
            angle=lst[4],
            from_planet_id=lst[5],
            ships=lst[6],
        )

    def to_list(self) -> List[Any]:
        return [
            self.id,
            self.owner,
            self.x,
            self.y,
            self.angle,
            self.from_planet_id,
            self.ships,
        ]


@dataclass
class CometGroup:
    """彗星グループ（observation 内は dict 形式）。"""

    planet_ids: List[int] = field(default_factory=list)  # 関連付けられた惑星 ID
    paths: List[List[Tuple[float, float]]] = field(
        default_factory=list
    )  # 各パスは (x,y) の折れ線
    path_index: int = 0  # 現在どのセグメント／進行位置か（エンジン仕様）

    @classmethod
    def from_dict(cls, d: Any) -> CometGroup:
        """dict または kaggle の SimpleNamespace から読む。"""
        if isinstance(d, dict):
            planet_ids = d.get("planet_ids", [])
            paths_raw = d.get("paths", [])
            path_index = d.get("path_index", 0)
        else:
            planet_ids = getattr(d, "planet_ids", [])
            paths_raw = getattr(d, "paths", [])
            path_index = getattr(d, "path_index", 0)
        return cls(
            planet_ids=list(planet_ids),
            paths=[list(p) for p in paths_raw],
            path_index=path_index,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planet_ids": list(self.planet_ids),
            "paths": [list(p) for p in self.paths],
            "path_index": self.path_index,
        }


def _default_comet_spawn_steps() -> Tuple[int, ...]:
    """公式 COMET_SPAWN_STEPS と同じタプル（ミュータブルなリストを避ける）。"""
    return tuple(COMET_SPAWN_STEPS)


@dataclass
class GameState:
    """シミュレータが保持する 1 スナップショット全体。"""

    planets: List[PlanetState] = field(default_factory=list)
    fleets: List[FleetState] = field(default_factory=list)
    comets: List[CometGroup] = field(default_factory=list)
    comet_planet_ids: List[int] = field(default_factory=list)
    initial_planets: List[PlanetState] = field(default_factory=list)
    angular_velocity: float = 0.0  # 盤の回転に関するパラメータ（公式 observation と同名）
    step: int = 0  # 経過ターン数（kaggle core が observation に書く値と同義）
    next_fleet_id: int = 0  # 次に割り当てる艦隊 ID
    num_players: int = 2  # メタ情報（observation の dict には含めない場合あり）
    # kaggle configuration に相当するメタ（observation には通常含まれない）
    comet_speed: float = 4.0  # コメットがパス上を進む速さ（cometSpeed）
    comet_spawn_steps: Tuple[int, ...] = field(default_factory=_default_comet_spawn_steps)
    # エピソード乱数の種。kaggle 新版では configuration.seed が無い場合 0 扱いでコメット RNG が決まる。
    episode_seed: int = 0

    @classmethod
    def from_observation(cls, obs: Dict[str, Any], num_players: int = 2) -> GameState:
        """エージェントの observation dict から GameState を構築する。"""
        spawn_meta = observation_get(obs, "comet_spawn_steps", None)
        if spawn_meta is not None:
            comet_spawn_steps = tuple(spawn_meta)
        else:
            comet_spawn_steps = _default_comet_spawn_steps()
        return cls(
            planets=[
                PlanetState.from_list(p) for p in observation_get(obs, "planets", []) or []
            ],
            fleets=[
                FleetState.from_list(f) for f in observation_get(obs, "fleets", []) or []
            ],
            comets=[
                CometGroup.from_dict(c)
                for c in observation_get(obs, "comets", []) or []
            ],
            comet_planet_ids=list(observation_get(obs, "comet_planet_ids", []) or []),
            initial_planets=[
                PlanetState.from_list(p)
                for p in observation_get(obs, "initial_planets", []) or []
            ],
            angular_velocity=observation_get(obs, "angular_velocity", 0.0),
            step=observation_get(obs, "step", 0),
            next_fleet_id=observation_get(obs, "next_fleet_id", 0),
            num_players=num_players,
            comet_speed=float(observation_get(obs, "comet_speed", 4.0)),
            comet_spawn_steps=comet_spawn_steps,
            episode_seed=int(observation_get(obs, "episode_seed", 0) or 0),
        )

    def to_observation(self, player: int = 0) -> Dict[str, Any]:
        """エージェント観測形式の dict を返す（player は観測側のコンテキスト用）。"""
        return {
            "planets": [p.to_list() for p in self.planets],
            "fleets": [f.to_list() for f in self.fleets],
            "comets": [c.to_dict() for c in self.comets],
            "comet_planet_ids": list(self.comet_planet_ids),
            "initial_planets": [p.to_list() for p in self.initial_planets],
            "angular_velocity": self.angular_velocity,
            "step": self.step,
            "next_fleet_id": self.next_fleet_id,
            "player": player,
        }

    def copy(self) -> GameState:
        """MCTS／ロールアウト用の独立コピー。

        copy.deepcopy は汎用ゆえ遅く（step() の約80%を占める）、ロールアウトの
        スループット＝実質探索深さを律速する。ここではターン処理で実際に変更され得る
        要素のみを手で再構築し、不変オブジェクト（点座標タプル・spawn_steps タプル・
        スカラー）は共有する。意味は deepcopy と等価（golden test で検証済）。

        変更され得るもの → 新規オブジェクト/リストを作る:
          planets / fleets / initial_planets の各 dataclass（フィールドが書き換わる・
          リストが append/filter される）、comets の planet_ids リストと path_index。
        不変で共有して安全なもの:
          paths 内の点座標（生成後に読み取りのみ）、comet_spawn_steps（タプル）、
          全スカラーフィールド。
        """
        return GameState(
            planets=[
                PlanetState(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
                for p in self.planets
            ],
            fleets=[
                FleetState(f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships)
                for f in self.fleets
            ],
            comets=[
                CometGroup(
                    planet_ids=list(c.planet_ids),
                    paths=[list(p) for p in c.paths],
                    path_index=c.path_index,
                )
                for c in self.comets
            ],
            comet_planet_ids=list(self.comet_planet_ids),
            initial_planets=[
                PlanetState(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
                for p in self.initial_planets
            ],
            angular_velocity=self.angular_velocity,
            step=self.step,
            next_fleet_id=self.next_fleet_id,
            num_players=self.num_players,
            comet_speed=self.comet_speed,
            comet_spawn_steps=self.comet_spawn_steps,
            episode_seed=self.episode_seed,
        )
