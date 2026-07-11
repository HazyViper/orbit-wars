"""kaggle_environments との完全一致を検証するゴールデンテスト。

シミュレータが公式実装と1ステップごとに一致することを保証する。
コメット出現は専用 RNG（orbit_wars-comet-*）で再現し、スケジュール境界でも検証する。
"""

import random
from typing import Any, List, Tuple

from kaggle_environments import make

from simulator.state import GameState, observation_get
from simulator.turn_logic import step

# env.run 後の env.steps[0] は planets/fleets が空のプレースホルダ。
# 実ボードは steps[1] 以降。比較は steps[k-1] の観測 + steps[k] に記録された action → steps[k] を k>=2 で行う。
_GOLDEN_LOOP_START = 2

# 公式 COMET_SPAWN_STEPS に対応する「遷移後の observation.step」
_COMET_GOLDEN_GAME_STEPS: Tuple[int, ...] = (50, 150, 250, 350, 450)


def _apply_kaggle_configuration_meta(state: GameState, env: Any) -> None:
    """kaggle の configuration からシミュレータ専用メタ（episode_seed, comet_speed）を埋める。

    observation には含まれないため、ゴールデン再現時に明示的に同期する。
    """
    cfg = env.configuration
    seed = getattr(cfg, "seed", None)
    if seed is None and hasattr(cfg, "get"):
        seed = cfg.get("seed")
    state.episode_seed = int(seed) if seed is not None else 0
    state.comet_speed = float(getattr(cfg, "cometSpeed", 4.0))


def get_actions_from_step(env_step, num_players):
    """env.steps[i] から actions を抽出する。"""
    actions = []
    for player_id in range(num_players):
        action = env_step[player_id].action
        if action is None:
            action = []
        actions.append(action)
    return actions


def assert_planets_equal(sim_planets, actual_planets, step_idx, tolerance=1e-6):
    """惑星リストが一致するか検証する（浮動小数点誤差を許容）。"""
    assert len(sim_planets) == len(actual_planets), (
        f"Step {step_idx}: planet count mismatch "
        f"sim={len(sim_planets)}, actual={len(actual_planets)}"
    )

    sim_dict = {p[0]: p for p in sim_planets}
    actual_dict = {p[0]: p for p in actual_planets}

    for pid in actual_dict:
        assert pid in sim_dict, f"Step {step_idx}: planet {pid} missing in sim"
        sim_p = sim_dict[pid]
        actual_p = actual_dict[pid]

        # owner, ships, production, radius の整数値は厳密一致
        assert sim_p[1] == actual_p[1], (
            f"Step {step_idx}, planet {pid}: owner mismatch "
            f"sim={sim_p[1]}, actual={actual_p[1]}"
        )
        assert sim_p[5] == actual_p[5], (
            f"Step {step_idx}, planet {pid}: ships mismatch "
            f"sim={sim_p[5]}, actual={actual_p[5]}"
        )
        assert sim_p[6] == actual_p[6], (
            f"Step {step_idx}, planet {pid}: production mismatch"
        )

        # x, y は浮動小数点なので tolerance 比較
        assert abs(sim_p[2] - actual_p[2]) < tolerance, (
            f"Step {step_idx}, planet {pid}: x mismatch "
            f"sim={sim_p[2]}, actual={actual_p[2]}"
        )
        assert abs(sim_p[3] - actual_p[3]) < tolerance, (
            f"Step {step_idx}, planet {pid}: y mismatch "
            f"sim={sim_p[3]}, actual={actual_p[3]}"
        )
        assert abs(sim_p[4] - actual_p[4]) < tolerance, (
            f"Step {step_idx}, planet {pid}: radius mismatch"
        )


def assert_fleets_equal(sim_fleets, actual_fleets, step_idx, tolerance=1e-6):
    """艦隊リストが一致するか検証する。"""
    assert len(sim_fleets) == len(actual_fleets), (
        f"Step {step_idx}: fleet count mismatch "
        f"sim={len(sim_fleets)}, actual={len(actual_fleets)}"
    )

    sim_dict = {f[0]: f for f in sim_fleets}
    actual_dict = {f[0]: f for f in actual_fleets}

    for fid in actual_dict:
        assert fid in sim_dict, f"Step {step_idx}: fleet {fid} missing in sim"
        sim_f = sim_dict[fid]
        actual_f = actual_dict[fid]

        # owner, ships, from_planet_id は厳密一致
        assert sim_f[1] == actual_f[1], (
            f"Step {step_idx}, fleet {fid}: owner mismatch"
        )
        assert sim_f[6] == actual_f[6], (
            f"Step {step_idx}, fleet {fid}: ships mismatch"
        )
        assert sim_f[5] == actual_f[5], (
            f"Step {step_idx}, fleet {fid}: from_planet_id mismatch"
        )

        # x, y, angle は浮動小数点
        assert abs(sim_f[2] - actual_f[2]) < tolerance, (
            f"Step {step_idx}, fleet {fid}: x mismatch "
            f"sim={sim_f[2]}, actual={actual_f[2]}"
        )
        assert abs(sim_f[3] - actual_f[3]) < tolerance, (
            f"Step {step_idx}, fleet {fid}: y mismatch"
        )
        assert abs(sim_f[4] - actual_f[4]) < tolerance, (
            f"Step {step_idx}, fleet {fid}: angle mismatch "
            f"sim={sim_f[4]}, actual={actual_f[4]}"
        )


def _comet_group_get(group: Any, key: str, default: Any = None) -> Any:
    """comet グループが dict / namespace のどちらでも読む。"""
    if isinstance(group, dict):
        return group.get(key, default)
    return getattr(group, key, default)


def assert_comet_planet_ids_equal(sim_ids, actual_ids, step_idx):
    """comet_planet_ids の列が一致するか検証する。"""
    a = list(sim_ids or [])
    b = list(actual_ids or [])
    assert a == b, f"Step {step_idx}: comet_planet_ids mismatch sim={a} actual={b}"


def assert_comets_equal(sim_comets, actual_comets, step_idx, tolerance=1e-6):
    """comets 配列（グループごとの paths 等）が一致するか検証する。"""
    assert len(sim_comets) == len(actual_comets), (
        f"Step {step_idx}: comets length mismatch sim={len(sim_comets)} "
        f"actual={len(actual_comets)}"
    )
    for gi, (sg, ag) in enumerate(zip(sim_comets, actual_comets)):
        assert _comet_group_get(sg, "planet_ids") == list(
            _comet_group_get(ag, "planet_ids", [])
        ), f"Step {step_idx} group {gi}: planet_ids mismatch"
        assert _comet_group_get(sg, "path_index") == _comet_group_get(
            ag, "path_index"
        ), f"Step {step_idx} group {gi}: path_index mismatch"
        spaths: List[Any] = _comet_group_get(sg, "paths", [])
        apaths: List[Any] = _comet_group_get(ag, "paths", [])
        assert len(spaths) == len(apaths), (
            f"Step {step_idx} group {gi}: paths outer len mismatch"
        )
        for pi, (pp_s, pp_a) in enumerate(zip(spaths, apaths)):
            assert len(pp_s) == len(pp_a), (
                f"Step {step_idx} group {gi} path {pi}: point count mismatch"
            )
            for ti, (pt_s, pt_a) in enumerate(zip(pp_s, pp_a)):
                assert abs(float(pt_s[0]) - float(pt_a[0])) < tolerance, (
                    f"Step {step_idx} group {gi} path {pi} pt {ti}: x mismatch"
                )
                assert abs(float(pt_s[1]) - float(pt_a[1])) < tolerance, (
                    f"Step {step_idx} group {gi} path {pi} pt {ti}: y mismatch"
                )


def find_first_comet_spawn_step(env_steps):
    """最初のコメット出現が記録された env.steps のインデックスを返す。

    comet_planet_ids が非空の最初の観測を探す。
    """
    for step_idx, step_data in enumerate(env_steps):
        obs = step_data[0].observation
        if observation_get(obs, "comet_planet_ids", []):
            return step_idx
    return len(env_steps)


def _run_orbit_wars_prefix(num_env_step_calls: int) -> Tuple[Any, Any]:
    """random seed=0 で orbit_wars を num_env_step_calls 回 env.step する。"""
    random.seed(0)
    env = make("orbit_wars", debug=False)
    env.reset(2)
    runner = env._Environment__agent_runner(["random", "random"])
    for _ in range(num_env_step_calls):
        assert not env.done, "試合がプレフィックス完了前に終了した"
        actions, logs = runner.act()
        env.step(actions, logs)
    return env, runner


def test_golden_short_match_with_random():
    """短い試合でシミュレータの1ターン進行が公式と一致するか検証する。"""
    random.seed(0)
    env = make("orbit_wars", debug=False)
    env.run(["random", "random"])

    num_players = 2

    # コメット出現前のステップのみ検証（spawn 境界は別テスト）
    spawn_step = find_first_comet_spawn_step(env.steps)
    print(f"First comet spawn at step {spawn_step}")

    max_step = min(spawn_step, len(env.steps) - 1, 30)

    for step_idx in range(_GOLDEN_LOOP_START, max_step):
        prev_obs = env.steps[step_idx - 1][0].observation
        # 適用アクションは遷移「後」の steps[step_idx] に記録される（kaggle core の仕様）
        prev_actions = get_actions_from_step(env.steps[step_idx], num_players)

        sim_state = GameState.from_observation(prev_obs, num_players=num_players)
        _apply_kaggle_configuration_meta(sim_state, env)
        sim_state = step(sim_state, prev_actions)

        actual_obs = env.steps[step_idx][0].observation
        sim_obs = sim_state.to_observation()

        assert_planets_equal(
            sim_obs["planets"],
            observation_get(actual_obs, "planets", []),
            step_idx,
        )
        assert_fleets_equal(
            sim_obs["fleets"],
            observation_get(actual_obs, "fleets", []),
            step_idx,
        )


def test_golden_first_few_steps():
    """最初の5ステップだけを厳密に検証する。

    試合開始直後はコメットがなく、シンプルなケースで動作確認。
    """
    random.seed(0)
    env = make("orbit_wars", debug=False)
    env.run(["random", "random"])

    num_players = 2

    for step_idx in range(_GOLDEN_LOOP_START, min(_GOLDEN_LOOP_START + 5, len(env.steps))):
        prev_obs = env.steps[step_idx - 1][0].observation
        prev_actions = get_actions_from_step(env.steps[step_idx], num_players)

        sim_state = GameState.from_observation(prev_obs, num_players=num_players)
        _apply_kaggle_configuration_meta(sim_state, env)
        sim_state = step(sim_state, prev_actions)

        actual_obs = env.steps[step_idx][0].observation
        sim_obs = sim_state.to_observation()

        assert_planets_equal(
            sim_obs["planets"],
            observation_get(actual_obs, "planets", []),
            step_idx,
        )
        assert_fleets_equal(
            sim_obs["fleets"],
            observation_get(actual_obs, "fleets", []),
            step_idx,
        )


def test_golden_comet_spawn_at_scheduled_steps():
    """コメット出現ターン（observation.step が 50,150,...）で公式と完全一致するか検証する。

    グローバル random はエージェント act() で消費されるため、同一ターンでは
    act 直後の getstate → シミュレータ step → setstate → env.step の順で揃える。
    """
    for game_step in _COMET_GOLDEN_GAME_STEPS:
        n_prefix = game_step - 1
        env, runner = _run_orbit_wars_prefix(n_prefix)
        prev_obs = env.steps[game_step - 1][0].observation
        assert observation_get(prev_obs, "step") == game_step - 1

        actions, logs = runner.act()
        rng_after_act = random.getstate()

        sim_state = GameState.from_observation(prev_obs, num_players=2)
        _apply_kaggle_configuration_meta(sim_state, env)
        sim_state = step(sim_state, actions)

        random.setstate(rng_after_act)
        env.step(actions, logs)

        actual_obs = env.steps[game_step][0].observation
        assert observation_get(actual_obs, "step") == game_step
        sim_obs = sim_state.to_observation()

        assert_comet_planet_ids_equal(
            sim_obs["comet_planet_ids"],
            observation_get(actual_obs, "comet_planet_ids", []),
            game_step,
        )
        assert_comets_equal(
            sim_obs["comets"],
            observation_get(actual_obs, "comets", []),
            game_step,
        )
        assert_planets_equal(
            sim_obs["planets"],
            observation_get(actual_obs, "planets", []),
            game_step,
        )
        assert_fleets_equal(
            sim_obs["fleets"],
            observation_get(actual_obs, "fleets", []),
            game_step,
        )
