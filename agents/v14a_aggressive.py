"""
Orbit Wars - v14a Aggressive (Top10 統計ベース)

v8h_combined をベースに、リプレイ分析（300試合）で見えた勝者の傾向
（手数・連携射撃・大艦隊）へ寄せた攻撃指向の強化版。

主な変更（v8h から）:
  - 連携射撃の参加しきい値を緩和（同ターゲット已有り時 5 → 3 艦）
  - 攻撃割当で自軍惑星を艦数の多い順にスキャン
  - 第2パスで短距離・既に手が付いたターゲットへ追加連携（同一ターン）
  - 可用艦が十分なとき「大艦隊」として最低 80 艦規模を狙う（条件付き）

保持した v8h の要点:
  - 太陽回避（SUN_SAFETY）、象限ボーナス無効（4人でも 1.0）
  - 序盤積極化（80 ターン / 5 惑星まで）
  - リード射撃（_get_safe_future_pos）
  - 自軍公転惑星間の艦隊スキップ（同一設計）
"""

import math
from collections import defaultdict
from typing import NamedTuple

# kaggle_environments に依存しない（LightZero 側で kaggle を外しても動く）
class Planet(NamedTuple):
    """惑星1行 [id, owner, x, y, radius, ships, production] と同じ7フィールド。"""

    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int

TOTAL_TURNS           = 500
SUN_X, SUN_Y          = 50.0, 50.0
ROTATION_RADIUS_LIMIT = 50.0
SUN_RADIUS            = 10.0
SUN_SAFETY            = 12.0
BOARD_SIZE            = 100.0
THREAT_LOOKAHEAD      = 200
EVACUATION_THRESHOLD  = 5

MAX_CONCURRENT_ATTACKERS_PER_TARGET = 4
CONCURRENT_ROUND2_MAX_DIST = 50.0
CONCURRENT_ROUND2_MIN_AVAILABLE = 15


def fleet_speed(ships: float) -> float:
    if ships < 1.0:
        return 1.0
    if ships >= 1000.0:
        return 6.0
    ratio = math.log(ships) / math.log(1000.0)
    return 1.0 + 5.0 * (ratio ** 1.5)


def _segment_to_point_distance(
    ax: float, ay: float,
    bx: float, by: float,
    px: float, py: float,
) -> float:
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _is_reachable(mine_x: float, mine_y: float, target_x: float, target_y: float) -> bool:
    """太陽を遮らずに直接到達できるかを判定（現在位置で判定）。"""
    sun_dist = _segment_to_point_distance(mine_x, mine_y, target_x, target_y, SUN_X, SUN_Y)
    return sun_dist >= SUN_RADIUS + 0.5


def _compute_simple_score(target: Planet, distance: float, my_player: int) -> float:
    """シンプルな価値関数: production と距離のみ。"""
    if target.owner == my_player:
        return 0.0
    weight = 1.0 if target.owner == -1 else 0.7
    return target.production * weight / (1.0 + distance / 30.0)


def _is_early_or_small(current_step: int, num_my_planets: int) -> bool:
    """序盤積極化を80ターン、5惑星まで延長（v8h と同一）。"""
    return current_step < 80 or num_my_planets <= 5


def _should_attack(
    mine_ships: float,
    cost: float,
    current_step: int,
    num_my_planets: int,
    has_concurrent_attack: bool,
) -> bool:
    """v14a: 連携射撃を積極的に（同ターゲット已有り時 5→3）。"""
    if _is_early_or_small(current_step, num_my_planets):
        return mine_ships >= cost + 1

    if has_concurrent_attack:
        return mine_ships >= 3

    if mine_ships < cost * 1.2:
        return False
    if cost < mine_ships * 0.2:
        return False
    return True


def _should_send_huge_fleet(mine_available: float, target_ships: int) -> bool:
    """保有可用艦が十分かつ目標に対して余力が大きいときだけ大艦隊送出を検討する。"""
    if mine_available < 100:
        return False
    return mine_available >= target_ships * 1.5


def _detect_struggling_start(my_planets: list, current_step: int) -> bool:
    """初期配置が不利かを判定。最大の自軍惑星の production が 2 以下なら不利。"""
    if current_step >= 100 or not my_planets:
        return False
    max_prod = max(p.production for p in my_planets)
    return max_prod <= 2


def _compute_comet_score(
    comet: Planet,
    distance: float,
    ttf: float,
    remaining_life: int,
) -> float:
    effective_life = remaining_life - ttf
    if effective_life <= 0:
        return 0.0
    cost = comet.ships + 1.0
    if cost <= 0:
        return 0.0
    gain = effective_life * 1.0
    return (gain / cost) / (1.0 + distance / 30.0)


def _comet_remaining_life(comet_id: int, comet_info: dict) -> int:
    if comet_id not in comet_info:
        return 0
    path, idx = comet_info[comet_id]
    return max(0, len(path) - idx)


def _predict_comet_position(
    comet_id: int,
    ttf: int,
    comet_info: dict,
) -> tuple[float, float] | None:
    if comet_id not in comet_info:
        return None
    path, idx = comet_info[comet_id]
    future_idx = idx + int(ttf)
    if future_idx >= len(path):
        return None
    p = path[future_idx]
    return p[0], p[1]


def _predict_target_position(
    mine_x: float, mine_y: float,
    t: Planet, orbital_radius: float,
    is_orbiting: bool, angular_velocity: float,
    ships_needed: float,
) -> tuple[float, float]:
    speed = fleet_speed(ships_needed)
    if not is_orbiting or angular_velocity == 0.0:
        return t.x, t.y
    dist = math.hypot(mine_x - t.x, mine_y - t.y)
    ttf  = dist / speed
    future_x, future_y = t.x, t.y
    for _ in range(2):
        ca = math.atan2(t.y - SUN_Y, t.x - SUN_X)
        fa = ca + angular_velocity * ttf
        future_x = SUN_X + orbital_radius * math.cos(fa)
        future_y = SUN_Y + orbital_radius * math.sin(fa)
        ttf = math.hypot(mine_x - future_x, mine_y - future_y) / speed
    return future_x, future_y


def _get_comet_future_pos(
    mine_x: float, mine_y: float,
    comet_id: int,
    ships_needed: float,
    comet_info: dict,
) -> tuple[float, float] | None:
    if comet_id not in comet_info:
        return None
    path, idx = comet_info[comet_id]
    if idx >= len(path):
        return None

    p0 = path[idx]
    cx, cy = p0[0], p0[1]
    speed = fleet_speed(ships_needed)
    dist = math.hypot(mine_x - cx, mine_y - cy)
    ttf = int(dist / speed)

    fx, fy = cx, cy

    for _ in range(2):
        future_idx = idx + ttf
        if future_idx >= len(path):
            return None
        p = path[future_idx]
        fx, fy = p[0], p[1]
        new_dist = math.hypot(mine_x - fx, mine_y - fy)
        new_ttf = int(new_dist / speed)
        if new_ttf == ttf:
            break
        ttf = new_ttf

    return fx, fy


def _get_safe_future_pos(
    mine_x: float, mine_y: float,
    t: Planet, angular_velocity: float, ships_needed: float,
    comet_ids: set[int] = None,
    comet_info: dict = None,
) -> tuple[bool, float, float]:
    if comet_ids is not None and comet_info is not None and t.id in comet_ids:
        result = _get_comet_future_pos(mine_x, mine_y, t.id, ships_needed, comet_info)
        if result is None:
            return False, t.x, t.y
        fx, fy = result
    else:
        orb    = math.hypot(t.x - SUN_X, t.y - SUN_Y)
        is_orb = (orb + t.radius < ROTATION_RADIUS_LIMIT)
        fx, fy = _predict_target_position(
            mine_x, mine_y, t, orb, is_orb, angular_velocity, ships_needed
        )

    safe = (_segment_to_point_distance(mine_x, mine_y, fx, fy, SUN_X, SUN_Y) >= SUN_SAFETY)
    return safe, fx, fy


def _predict_my_planet_position(
    mp: Planet,
    t: int,
    angular_velocity: float,
    comet_ids: set[int],
    comet_info: dict,
) -> tuple[float, float] | None:
    if mp.id in comet_ids:
        result = _predict_comet_position(mp.id, t, comet_info)
        return result

    orb = math.hypot(mp.x - SUN_X, mp.y - SUN_Y)
    is_orbiting = (orb + mp.radius < ROTATION_RADIUS_LIMIT)

    if is_orbiting and angular_velocity != 0.0:
        current_angle = math.atan2(mp.y - SUN_Y, mp.x - SUN_X)
        future_angle = current_angle + angular_velocity * t
        future_x = SUN_X + orb * math.cos(future_angle)
        future_y = SUN_Y + orb * math.sin(future_angle)
        return future_x, future_y

    return mp.x, mp.y


def _find_targeted_planet(
    fleet: list,
    my_planets: list[Planet],
    angular_velocity: float,
    comet_ids: set[int],
    comet_info: dict,
) -> tuple[Planet | None, int | None]:
    fx, fy  = fleet[2], fleet[3]
    angle   = fleet[4]
    speed   = fleet_speed(fleet[6])
    cos_a   = math.cos(angle)
    sin_a   = math.sin(angle)

    for t in range(1, THREAT_LOOKAHEAD + 1):
        sx = fx + cos_a * speed * t
        sy = fy + sin_a * speed * t
        if not (0 <= sx <= BOARD_SIZE and 0 <= sy <= BOARD_SIZE):
            break
        for mp in my_planets:
            mp_future = _predict_my_planet_position(
                mp, t, angular_velocity, comet_ids, comet_info
            )
            if mp_future is None:
                continue
            dist = math.hypot(sx - mp_future[0], sy - mp_future[1])
            if dist < mp.radius + 1.0:
                return mp, t

    return None, None


def _compute_threat_multiplayer(
    my_planet: Planet,
    incoming: list[tuple[list, int]],
) -> float:
    """同じ owner の ships を合算した上で、最大 vs 第二最大の差分を脅威量とする。"""
    if not incoming:
        return 0.0

    incoming_sorted = sorted(incoming, key=lambda x: x[1])

    cumulative_by_player: dict = defaultdict(float)
    max_net_threat = 0.0

    for fleet, ttf in incoming_sorted:
        owner = fleet[1]
        cumulative_by_player[owner] += fleet[6]

        attack_amounts = sorted(cumulative_by_player.values(), reverse=True)
        if len(attack_amounts) >= 2:
            net_attack = attack_amounts[0] - attack_amounts[1]
        elif len(attack_amounts) == 1:
            net_attack = attack_amounts[0]
        else:
            net_attack = 0.0

        production_buffer = my_planet.production * ttf
        net_threat = net_attack - production_buffer
        if net_threat > max_net_threat:
            max_net_threat = net_threat

    return max_net_threat


def _compute_aggression_factor(
    my_player: int,
    num_players: int,
    current_step: int,
    totals: dict,
    my_planets: list,
) -> float:
    """4人対戦時のみ作動。2人対戦では常に 1.0 を返す。"""
    if num_players <= 2:
        return 1.0

    if _detect_struggling_start(my_planets, current_step):
        early_game_factor = 1.5
    elif current_step < 50:
        early_game_factor = 0.4
    elif current_step < 100:
        early_game_factor = 0.4 + 0.6 * (current_step - 50) / 50
    else:
        early_game_factor = 1.0

    my_total = totals.get(my_player, 0)
    others = [v for k, v in totals.items() if k != my_player]

    if not others:
        return early_game_factor

    other_max = max(others)

    if my_total > other_max * 1.2:
        relative_factor = 0.3
    elif my_total > other_max * 1.05:
        relative_factor = 0.6
    elif my_total < sum(others) / max(1, len(others)) * 0.7:
        relative_factor = 1.2
    else:
        relative_factor = 1.0

    return early_game_factor * relative_factor


def _compute_target_bonus(
    target: Planet,
    totals: dict,
    my_player: int,
) -> float:
    """敵プレイヤーごとの強さに応じたターゲット選好ボーナス。"""
    if target.owner == -1:
        return 1.0
    if target.owner == my_player:
        return 0.0

    enemy_totals = {k: v for k, v in totals.items() if k != my_player and k != -1}
    if not enemy_totals:
        return 1.0

    target_owner_total = enemy_totals.get(target.owner, 0)
    weakest = min(enemy_totals.values())
    strongest = max(enemy_totals.values())

    if strongest <= weakest:
        return 1.0

    rank = (target_owner_total - weakest) / (strongest - weakest)
    return 1.5 - 0.8 * rank  # 1.5 (最弱) → 0.7 (最強)


def _detect_my_quadrant_groups(my_planets: list) -> tuple:
    """自軍の主象限と公転惑星の軌道半径セットを返す。

    Returns:
        my_quadrant: 静的惑星の主象限（0-3）または None
        my_orbital_radii: 自軍公転惑星の軌道半径のセット（丸めて使う）
    """
    if not my_planets:
        return None, set()

    static_weights = [0.0, 0.0, 0.0, 0.0]
    orbital_radii = set()

    for p in my_planets:
        orb = math.hypot(p.x - SUN_X, p.y - SUN_Y)
        is_orbiting = (orb + p.radius < ROTATION_RADIUS_LIMIT)
        if is_orbiting:
            orbital_radii.add(round(orb, 1))
        else:
            q = (1 if p.x >= 50 else 0) + (2 if p.y >= 50 else 0)
            static_weights[q] += p.production

    my_quadrant = static_weights.index(max(static_weights)) if max(static_weights) > 0 else None
    return my_quadrant, orbital_radii


def _quadrant_strategy_bonus(
    target: Planet,
    my_quadrant: int | None,
    my_orbital_radii: set,
    num_players: int,
) -> float:
    """象限ボーナス完全無効化（v8h と同一）。引数は呼び出し互換のため保持。"""
    return 1.0


def _planet_by_id(planets: list[Planet]) -> dict[int, Planet]:
    return {p.id: p for p in planets}


def _compute_min_garrison(threat: float) -> int:
    if threat <= 0:
        return 0
    return int(threat) + 1


def agent(obs):
    moves = []

    player           = obs.get("player", 0)       if isinstance(obs, dict) else obs.player
    current_step     = obs.get("step",   0)        if isinstance(obs, dict) else obs.step
    raw_planets      = obs.get("planets", [])      if isinstance(obs, dict) else obs.planets
    raw_fleets       = obs.get("fleets",  [])      if isinstance(obs, dict) else obs.fleets
    angular_velocity = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) \
                       else getattr(obs, "angular_velocity", 0.0)

    comet_ids = set(
        obs.get("comet_planet_ids", []) if isinstance(obs, dict)
        else getattr(obs, "comet_planet_ids", [])
    )

    raw_comets = obs.get("comets", []) if isinstance(obs, dict) else obs.comets

    comet_info: dict[int, tuple[list, int]] = {}
    for group in raw_comets:
        g_planet_ids = group["planet_ids"] if isinstance(group, dict) else group.planet_ids
        g_paths      = group["paths"]       if isinstance(group, dict) else group.paths
        g_idx        = group["path_index"]  if isinstance(group, dict) else group.path_index
        for i, comet_id in enumerate(g_planet_ids):
            comet_info[comet_id] = (g_paths[i], g_idx)

    planets    = [Planet(*p) for p in raw_planets]
    planet_by_id = _planet_by_id(planets)
    my_planets = [p for p in planets if p.owner == player]
    targets    = [p for p in planets if p.owner != player]

    if not targets or not my_planets:
        return moves

    all_owners: set[int] = set()
    for p in raw_planets:
        if p[1] != -1:
            all_owners.add(p[1])
    for f in raw_fleets:
        all_owners.add(f[1])
    num_players = max(2, max(all_owners) + 1) if all_owners else 2

    totals: dict[int, float] = defaultdict(float)
    for p in raw_planets:
        if p[1] != -1:
            totals[p[1]] += p[5]
    for f in raw_fleets:
        totals[f[1]] += f[6]

    aggression_factor = _compute_aggression_factor(
        player, num_players, current_step, totals, my_planets
    )

    my_quadrant, my_orbital_radii = _detect_my_quadrant_groups(my_planets)

    # ------------------------------------------------------------------
    # フェーズ 1: 敵艦隊の来襲予測 → 各自軍惑星の最低駐留兵力を計算
    # ------------------------------------------------------------------
    my_planets_for_defense = [p for p in my_planets if p.id not in comet_ids]

    threats_by_planet: dict[int, list] = defaultdict(list)
    for fleet in raw_fleets:
        if fleet[1] == player:
            continue
        targeted, ttf = _find_targeted_planet(
            fleet, my_planets_for_defense, angular_velocity, comet_ids, comet_info
        )
        if targeted is not None:
            threats_by_planet[targeted.id].append((fleet, ttf))

    available: dict[int, float] = {}
    for mp in my_planets:
        if mp.id in comet_ids:
            available[mp.id] = mp.ships
        else:
            threat = _compute_threat_multiplayer(mp, threats_by_planet.get(mp.id, []))
            min_garrison = _compute_min_garrison(threat)
            available[mp.id] = max(0.0, mp.ships - min_garrison)

    # ------------------------------------------------------------------
    # フェーズ 2: コメットからの駐留艦退避
    # ------------------------------------------------------------------
    non_comet_havens = [p for p in my_planets if p.id not in comet_ids]

    for mp in my_planets:
        if mp.id not in comet_ids:
            continue
        remaining_life = _comet_remaining_life(mp.id, comet_info)
        ships_to_evac = int(available[mp.id])
        if remaining_life > EVACUATION_THRESHOLD or ships_to_evac < 1:
            continue
        if not non_comet_havens:
            continue

        haven = min(
            non_comet_havens,
            key=lambda p: math.hypot(mp.x - p.x, mp.y - p.y),
        )

        safe, fx, fy = _get_safe_future_pos(
            mp.x, mp.y, haven, angular_velocity, ships_to_evac,
            comet_ids=comet_ids, comet_info=comet_info,
        )
        if not safe:
            continue

        angle = math.atan2(fy - mp.y, fx - mp.x)
        moves.append([mp.id, angle, ships_to_evac])
        available[mp.id] = 0

    # ------------------------------------------------------------------
    # フェーズ 3: 攻撃割当（第1パス: 最良ターゲット / 第2パス: 追加連携 + 大艦隊）
    # ------------------------------------------------------------------
    num_my_planets = len(my_planets)

    allow_concurrent = (num_players <= 2)
    already_attacking_targets: set[int] = set()
    attacked_targets: set[int] = set()

    CONCURRENT_DISCOUNT = 0.5

    moves_round1: list[list] = []
    meta_round1: list[tuple[int, int]] = []

    for mine in sorted(my_planets, key=lambda p: -p.ships):
        best_target = None
        best_score  = -1.0

        for t in targets:
            if not allow_concurrent and t.id in attacked_targets:
                continue

            if not _is_reachable(mine.x, mine.y, t.x, t.y):
                continue

            dist  = math.hypot(mine.x - t.x, mine.y - t.y)
            cost  = t.ships + 1.0 if t.owner == -1 else t.ships * 1.2 + 1.0
            speed = fleet_speed(cost)
            ttf   = dist / speed

            if t.id in comet_ids:
                remaining_life = _comet_remaining_life(t.id, comet_info)
                if remaining_life - ttf <= 0:
                    continue
                base_score = _compute_comet_score(t, dist, ttf, remaining_life)
            else:
                base_score = _compute_simple_score(t, dist, player)

            if base_score <= 0:
                continue

            target_bonus = _compute_target_bonus(t, totals, player)
            qs_bonus     = _quadrant_strategy_bonus(t, my_quadrant, my_orbital_radii, num_players)
            score = base_score * aggression_factor * target_bonus * qs_bonus

            if allow_concurrent and t.id in already_attacking_targets:
                score *= CONCURRENT_DISCOUNT

            if score > best_score:
                best_score  = score
                best_target = t

        if best_target is None:
            continue

        ships_needed = int(best_target.ships) + 1 if best_target.owner == -1 \
                       else int(best_target.ships * 1.2) + 1
        has_concurrent = allow_concurrent and (best_target.id in already_attacking_targets)

        if not _should_attack(available[mine.id], ships_needed, current_step, num_my_planets, has_concurrent):
            continue

        ships_to_send = ships_needed if not has_concurrent else min(ships_needed, int(available[mine.id]))
        if _should_send_huge_fleet(available[mine.id], best_target.ships):
            ships_to_send = max(ships_to_send, 80)
            ships_to_send = min(ships_to_send, mine.ships - 5)
        ships_to_send = min(ships_to_send, int(available[mine.id]))
        if ships_to_send < 1:
            continue

        safe, future_x, future_y = _get_safe_future_pos(
            mine.x, mine.y, best_target, angular_velocity, ships_to_send,
            comet_ids=comet_ids, comet_info=comet_info,
        )
        if not safe:
            continue

        angle = math.atan2(future_y - mine.y, future_x - mine.x)
        moves_round1.append([mine.id, angle, ships_to_send])
        meta_round1.append((mine.id, best_target.id))
        available[mine.id] -= ships_to_send
        if allow_concurrent:
            already_attacking_targets.add(best_target.id)
        else:
            attacked_targets.add(best_target.id)

    moves_round2: list[list] = []
    if allow_concurrent:
        attack_counts: dict[int, int] = defaultdict(int)
        for _pid, tid in meta_round1:
            attack_counts[tid] += 1

        used_in_r1 = {pid for pid, _ in meta_round1}

        for mine in my_planets:
            if mine.id in used_in_r1:
                continue
            if available[mine.id] < CONCURRENT_ROUND2_MIN_AVAILABLE:
                continue

            for target_id, count in sorted(
                attack_counts.items(),
                key=lambda kv: (-kv[1], kv[0]),
            ):
                if not (1 <= count <= 2):
                    continue
                if attack_counts[target_id] >= MAX_CONCURRENT_ATTACKERS_PER_TARGET:
                    continue

                target = planet_by_id.get(target_id)
                if target is None or target.owner == player:
                    continue
                if not _is_reachable(mine.x, mine.y, target.x, target.y):
                    continue

                dist = math.hypot(mine.x - target.x, mine.y - target.y)
                if dist >= CONCURRENT_ROUND2_MAX_DIST:
                    continue

                ships_needed_r2 = int(target.ships * 0.5) + 5
                if not _should_attack(
                    available[mine.id],
                    ships_needed_r2,
                    current_step,
                    num_my_planets,
                    True,
                ):
                    continue

                ships_to_send = min(int(available[mine.id]), max(1, ships_needed_r2))
                ships_to_send = max(1, ships_to_send)

                safe, future_x, future_y = _get_safe_future_pos(
                    mine.x, mine.y, target, angular_velocity, ships_to_send,
                    comet_ids=comet_ids, comet_info=comet_info,
                )
                if not safe:
                    continue

                angle = math.atan2(future_y - mine.y, future_x - mine.x)
                moves_round2.append([mine.id, angle, ships_to_send])
                available[mine.id] -= ships_to_send
                attack_counts[target_id] += 1
                break

    moves.extend(moves_round1)
    moves.extend(moves_round2)
    return moves
