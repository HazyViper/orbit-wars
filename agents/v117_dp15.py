"""
Orbit Wars - v117_dp15（ヒューリスティック系列の最終形）

v115_iw2 をベースに DIST_PENALTY を 25 → 15 に調整した変種。
ローカル評価: 2P 55.3%（vs v115）/ 4P 1位率 28.6%。

系譜と主な構成要素:
  [A] Indirect Wealth（間接的価値）ボーナス — v114/v115 由来
      IW(i) = Σ_j ( production_j² * (DIAM - dist(i,j)) ) / DIAM
      高生産惑星が近くに多い惑星ほど「将来の拡張拠点」として価値が高い、
      という考え方。IW を 0-1 正規化して全ターゲット（中立+敵）のスコアに乗算。
      ※ このアイデアは Halite 3 の公開ボット oddshrimp（4.3, Base.hs）の
        indWealth 関数を参考に、線形重みを production² に改良して移植したもの。
  [B] 終盤ガリソン削減（v113 から継承）
  [C] DIST_PENALTY = 15（本バージョンの変更点）
      攻撃スコアの距離減衰 production * weight / (1 + distance / DIST_PENALTY)
      を鋭くし、近距離ターゲットをより優先する。
"""

import math
from collections import defaultdict
from typing import NamedTuple

class Planet(NamedTuple):
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

# ---- v114 新規パラメータ ----
BOARD_DIAM  = 141.421  # sqrt(100^2+100^2)、oddshrimp の diam 相当
IW_FACTOR   = 1.0

# ---- v105 継承パラメータ（変更なし）----
ATTACK_THRESHOLD      = 1.05
DIST_PENALTY          = 15.0
ENEMY_WEIGHT          = 0.7
PROXIMITY_BONUS       = 0.5
PROXIMITY_DIST        = 25.0
ROI_NEUTRAL           = 50.0
ROI_ENEMY             = 500.0
NEUTRAL_ROI_FILTER    = 20
GARRISON_BONUS_FACTOR = 50
CONCURRENT_DISCOUNT   = 0.5
SHIPS_NEEDED_INIT     = 1.15
SHIPS_NEEDED_MARGIN   = 1.05


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
    sun_dist = _segment_to_point_distance(mine_x, mine_y, target_x, target_y, SUN_X, SUN_Y)
    return sun_dist >= SUN_RADIUS + 0.5


def _compute_simple_score(
    target: Planet, distance: float, my_player: int, current_step: int = 0
) -> float:
    if target.owner == my_player:
        return 0.0
    if target.owner == -1:
        if current_step >= 150 and (target.ships + 1) > target.production * NEUTRAL_ROI_FILTER:
            return 0.0
        weight = 1.0
    else:
        weight = ENEMY_WEIGHT
        if distance < PROXIMITY_DIST:
            weight *= 1.0 + PROXIMITY_BONUS * (1.0 - distance / PROXIMITY_DIST)
    return target.production * weight / (1.0 + distance / DIST_PENALTY)


def _estimate_ships_needed(
    target: Planet,
    distance: float,
    my_player: int,
) -> int:
    if target.owner == -1 or target.owner == my_player:
        return int(target.ships) + 1

    ships = int(target.ships * SHIPS_NEEDED_INIT) + 1
    prev = 0
    for _ in range(5):
        speed = fleet_speed(ships)
        ttf = max(1, int(distance / speed))
        garrison = target.ships + target.production * ttf
        new_ships = int(garrison * SHIPS_NEEDED_MARGIN) + 1
        if new_ships == ships:
            return ships
        if new_ships == prev:
            return max(ships, new_ships)
        prev = ships
        ships = new_ships
    return ships


def _is_early_or_small(current_step: int, num_my_planets: int) -> bool:
    return current_step < 120 or num_my_planets <= 6


def _should_attack(
    mine_ships: float,
    cost: float,
    current_step: int,
    num_my_planets: int,
    has_concurrent_attack: bool,
) -> bool:
    if _is_early_or_small(current_step, num_my_planets):
        return mine_ships >= cost + 1

    if has_concurrent_attack:
        return mine_ships >= 5

    if mine_ships < cost * ATTACK_THRESHOLD:
        return False
    return True


def _detect_struggling_start(my_planets: list, current_step: int) -> bool:
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
    ca = math.atan2(t.y - SUN_Y, t.x - SUN_X)
    dist = math.hypot(mine_x - t.x, mine_y - t.y)
    ttf = int(dist / speed)
    future_x, future_y = t.x, t.y
    for _ in range(6):
        fa = ca + angular_velocity * ttf
        future_x = SUN_X + orbital_radius * math.cos(fa)
        future_y = SUN_Y + orbital_radius * math.sin(fa)
        new_ttf = int(math.hypot(mine_x - future_x, mine_y - future_y) / speed)
        if new_ttf == ttf:
            break
        ttf = new_ttf
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
    if not incoming:
        return 0.0

    incoming_sorted = sorted(incoming, key=lambda x: x[1])

    cumulative_enemy = 0.0
    max_net_threat = 0.0

    for fleet, ttf in incoming_sorted:
        cumulative_enemy += fleet[6]
        production_buffer = my_planet.production * ttf
        net_threat = cumulative_enemy - production_buffer
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
    return 1.5 - 0.8 * rank


def _detect_my_quadrant_groups(my_planets: list) -> tuple:
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
    return 1.0


def _compute_orbit_approach_bonus(
    mine_x: float, mine_y: float,
    t: Planet,
    angular_velocity: float,
) -> float:
    orb = math.hypot(t.x - SUN_X, t.y - SUN_Y)
    if orb + t.radius >= ROTATION_RADIUS_LIMIT or angular_velocity == 0.0:
        return 1.0

    theta = math.atan2(t.y - SUN_Y, t.x - SUN_X)
    vel_x = -orb * angular_velocity * math.sin(theta)
    vel_y =  orb * angular_velocity * math.cos(theta)

    dx = mine_x - t.x
    dy = mine_y - t.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 1.0

    closing_speed = (vel_x * dx + vel_y * dy) / dist

    max_speed = orb * abs(angular_velocity)
    if max_speed < 1e-6:
        return 1.0

    closing_ratio = max(-1.0, min(1.0, closing_speed / max_speed))
    return 1.0 + 0.3 * closing_ratio


def _compute_min_garrison(threat: float) -> int:
    if threat <= 0:
        return 0
    return int(threat) + 1


def _compute_cumulative_surplus(
    my_planet: Planet,
    enemy_incoming: list[tuple[list, int]],
    friendly_incoming: list[tuple[list, int]],
) -> int:
    if not enemy_incoming and not friendly_incoming:
        return 0

    max_ttf = max(
        max((ttf for _, ttf in enemy_incoming), default=0),
        max((ttf for _, ttf in friendly_incoming), default=0),
    )
    max_ttf = min(max_ttf, THREAT_LOOKAHEAD)

    net_arr: dict[int, int] = defaultdict(int)
    for fleet, ttf in enemy_incoming:
        if 0 < ttf <= max_ttf:
            net_arr[ttf] -= int(fleet[6])
    for fleet, ttf in friendly_incoming:
        if 0 < ttf <= max_ttf:
            net_arr[ttf] += int(fleet[6])

    cumulative = 0
    min_surplus = float("inf")
    for t in range(1, max_ttf + 1):
        cumulative += my_planet.production + net_arr.get(t, 0)
        if cumulative < min_surplus:
            min_surplus = cumulative

    if min_surplus >= 1:
        return 0
    return int(1 - min_surplus)


def _compute_indirect_wealth(planets: list, comet_ids: set) -> dict[int, float]:
    """oddshrimp4.3 indWealth 関数の直接翻訳（Base.hs 420-425行目）。

    IW(i) = Σ_j [ production_j * max(0, DIAM - dist(i, j)) ] / DIAM

    高生産惑星の近くにある惑星ほど「戦略的ハブ」として価値が高い。
    結果は正規化済み（max=1.0）の dict を返す。

    計算量: O(N^2)。N<=20程度なので毎ターン呼び出し可能。
    """
    iw: dict[int, float] = {}

    live_planets = [p for p in planets if p.id not in comet_ids]

    for pi in live_planets:
        total = 0.0
        for pj in live_planets:
            if pj.id == pi.id:
                continue
            d = math.hypot(pi.x - pj.x, pi.y - pj.y)
            total += (pj.production ** 2) * max(0.0, BOARD_DIAM - d)  # prod²で高生産惑星を強調
        iw[pi.id] = total / BOARD_DIAM

    # 0-1 正規化
    if iw:
        max_val = max(iw.values())
        if max_val > 0:
            for k in iw:
                iw[k] /= max_val

    return iw


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
    # フェーズ 1: 敵艦隊の来襲予測 + 自軍援軍の追跡
    # ------------------------------------------------------------------
    my_planets_for_defense = [p for p in my_planets if p.id not in comet_ids]

    threats_by_planet: dict[int, list] = defaultdict(list)
    friendly_by_planet: dict[int, list] = defaultdict(list)

    for fleet in raw_fleets:
        if fleet[1] == player:
            targeted, ttf = _find_targeted_planet(
                fleet, my_planets_for_defense, angular_velocity, comet_ids, comet_info
            )
            if targeted is not None:
                friendly_by_planet[targeted.id].append((fleet, ttf))
        else:
            targeted, ttf = _find_targeted_planet(
                fleet, my_planets_for_defense, angular_velocity, comet_ids, comet_info
            )
            if targeted is not None:
                threats_by_planet[targeted.id].append((fleet, ttf))

    available: dict[int, float] = {}
    min_garrison_by_planet: dict[int, int] = {}
    for mp in my_planets:
        if mp.id in comet_ids:
            available[mp.id] = mp.ships
            min_garrison_by_planet[mp.id] = 0
        else:
            min_garrison = _compute_cumulative_surplus(
                mp,
                threats_by_planet.get(mp.id, []),
                friendly_by_planet.get(mp.id, []),
            )
            min_garrison_by_planet[mp.id] = min_garrison
            available[mp.id] = max(0.0, mp.ships - min_garrison)

    # ------------------------------------------------------------------
    # フェーズ 1.5: 防衛強化
    # ------------------------------------------------------------------
    endangered: list[tuple[Planet, int, int]] = []
    for mp in my_planets_for_defense:
        min_g = min_garrison_by_planet.get(mp.id, 0)
        deficit = min_g - mp.ships
        if deficit <= 0:
            continue
        incoming = threats_by_planet.get(mp.id, [])
        if not incoming:
            continue
        min_enemy_ttf = min(ttf for _, ttf in incoming)
        endangered.append((mp, deficit, min_enemy_ttf))

    endangered.sort(key=lambda x: -(x[0].production * x[1]))

    for mp, deficit, min_enemy_ttf in endangered:
        helpers = sorted(
            [
                p for p in my_planets
                if p.id != mp.id
                and p.id not in comet_ids
                and available.get(p.id, 0.0) > 5
            ],
            key=lambda p: math.hypot(p.x - mp.x, p.y - mp.y),
        )
        for helper in helpers:
            send = min(
                int(available[helper.id]) - 5,
                deficit + mp.production * 3,
            )
            if send < 1:
                continue
            dist_to_mp = math.hypot(helper.x - mp.x, helper.y - mp.y)
            rough_ttf = dist_to_mp / fleet_speed(send)
            mp_future = _predict_my_planet_position(
                mp, int(rough_ttf), angular_velocity, comet_ids, comet_info
            )
            if mp_future is not None:
                dist_to_mp = math.hypot(helper.x - mp_future[0], helper.y - mp_future[1])
            aid_ttf = dist_to_mp / fleet_speed(send)
            if aid_ttf > min_enemy_ttf:
                continue
            safe, fx, fy = _get_safe_future_pos(
                helper.x, helper.y, mp, angular_velocity, send,
                comet_ids=comet_ids, comet_info=comet_info,
            )
            if not safe:
                continue
            angle = math.atan2(fy - helper.y, fx - helper.x)
            moves.append([helper.id, angle, send])
            available[helper.id] -= send
            break

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
    # [v114-A] Indirect Wealth 事前計算（Phase 2.7 / 3 で使用）
    # oddshrimp4.3: IW(i) = Σ_j(prod_j * (DIAM - dist)) / DIAM
    # ------------------------------------------------------------------
    iw_map: dict[int, float] = _compute_indirect_wealth(planets, comet_ids)

    # ------------------------------------------------------------------
    # フェーズ 2.8: 合流攻撃（2P のみ）
    # ------------------------------------------------------------------
    num_my_planets = len(my_planets)
    enemy_planet_counts_pre: dict[int, int] = defaultdict(int)
    for p in planets:
        if p.owner != -1 and p.owner != player:
            enemy_planet_counts_pre[p.owner] += 1
    max_enemy_planet_count_pre = max(enemy_planet_counts_pre.values()) if enemy_planet_counts_pre else 0
    expansion_gap_pre = max_enemy_planet_count_pre - num_my_planets
    if expansion_gap_pre > 0:
        max_bonus_pre = 1.5 if current_step < 150 else 1.2
        expansion_bonus = min(max_bonus_pre, 1.0 + expansion_gap_pre * 0.1)
    else:
        expansion_bonus = 1.0

    merge_executed: set[int] = set()
    merge_attacked_targets: set[int] = set()

    if num_players <= 2:
        planet_natural_target: dict[int, tuple[Planet, float, int] | None] = {}
        for mine in my_planets:
            if mine.id in comet_ids or min_garrison_by_planet.get(mine.id, 0) > 0:
                planet_natural_target[mine.id] = None
                continue
            best_t, best_sc, best_dist = None, -1.0, 0.0
            for t in targets:
                if t.id in comet_ids or not _is_reachable(mine.x, mine.y, t.x, t.y):
                    continue
                dist = math.hypot(mine.x - t.x, mine.y - t.y)
                base = _compute_simple_score(t, dist, player, current_step)
                if base <= 0:
                    continue
                phase27_ships = _estimate_ships_needed(t, dist, player)
                phase27_roi = math.sqrt(ROI_ENEMY / phase27_ships) if phase27_ships > 1 and t.owner != -1 else 1.0
                phase27_orbit = _compute_orbit_approach_bonus(mine.x, mine.y, t, angular_velocity)
                phase27_neutral = expansion_bonus if t.owner == -1 else 1.0
                sc = base * aggression_factor * _compute_target_bonus(t, totals, player) * phase27_roi * phase27_orbit * phase27_neutral
                # [v114-A] Indirect Wealth ボーナス
                iw_val = iw_map.get(t.id, 0.0)
                sc *= (1.0 + IW_FACTOR * iw_val)
                if sc > best_sc:
                    best_sc = sc
                    best_t = t
                    best_dist = dist
            if best_t is None:
                planet_natural_target[mine.id] = None
            else:
                needed = _estimate_ships_needed(best_t, best_dist, player)
                planet_natural_target[mine.id] = (best_t, best_sc, needed)

        from collections import defaultdict as _dd
        frustrated_by_target: dict[int, list[Planet]] = _dd(list)
        for mine in my_planets:
            info = planet_natural_target.get(mine.id)
            if info is None:
                continue
            best_t, _, needed = info
            if best_t.owner == -1 or best_t.owner == player:
                continue
            if best_t.ships < 20:
                continue
            avail = available.get(mine.id, 0.0)
            if avail < needed and avail >= 10:
                frustrated_by_target[best_t.id].append(mine)

        for t_id, planet_list in frustrated_by_target.items():
            if len(planet_list) < 2:
                continue
            t = next((x for x in targets if x.id == t_id), None)
            if t is None:
                continue
            min_dist = min(math.hypot(p.x - t.x, p.y - t.y) for p in planet_list)
            ships_needed = _estimate_ships_needed(t, min_dist, player)

            if any(
                available.get(p.id, 0.0) >= _estimate_ships_needed(
                    t, math.hypot(p.x - t.x, p.y - t.y), player
                )
                for p in my_planets if p.id not in comet_ids
            ):
                continue

            planet_list.sort(key=lambda p: -available.get(p.id, 0.0))
            a1, a2 = planet_list[0], planet_list[1]

            if available.get(a1.id, 0.0) + available.get(a2.id, 0.0) < ships_needed:
                continue

            attack_moves: list[tuple[int, float, int]] = []
            joint_ok = True
            for attacker in [a1, a2]:
                send = int(available[attacker.id])
                if send < 5:
                    joint_ok = False
                    break
                safe, fx, fy = _get_safe_future_pos(
                    attacker.x, attacker.y, t, angular_velocity, send,
                    comet_ids=comet_ids, comet_info=comet_info,
                )
                if not safe:
                    joint_ok = False
                    break
                angle = math.atan2(fy - attacker.y, fx - attacker.x)
                attack_moves.append((attacker.id, angle, send))

            if joint_ok:
                for pid, angle, send in attack_moves:
                    moves.append([pid, angle, send])
                    available[pid] = 0
                    merge_executed.add(pid)
                merge_attacked_targets.add(t.id)

    # ------------------------------------------------------------------
    # フェーズ 3: 攻撃割当
    # ------------------------------------------------------------------
    allow_concurrent = (num_players <= 2)
    already_attacking_targets: set[int] = merge_attacked_targets
    attacked_targets: set[int] = merge_attacked_targets.copy()

    for mine in my_planets:
        if mine.id in merge_executed:
            continue

        best_target = None
        best_score  = -1.0
        best_dist   = 0.0

        for t in targets:
            if not allow_concurrent and t.id in attacked_targets:
                continue

            if not _is_reachable(mine.x, mine.y, t.x, t.y):
                continue

            dist = math.hypot(mine.x - t.x, mine.y - t.y)

            if t.id in comet_ids:
                continue
            base_score = _compute_simple_score(t, dist, player, current_step)

            if base_score <= 0:
                continue

            target_bonus = _compute_target_bonus(t, totals, player)
            qs_bonus     = _quadrant_strategy_bonus(t, my_quadrant, my_orbital_radii, num_players)
            orbit_bonus  = _compute_orbit_approach_bonus(mine.x, mine.y, t, angular_velocity) \
                           if num_players <= 2 else 1.0
            neutral_bonus = expansion_bonus if t.owner == -1 else 1.0
            score = base_score * aggression_factor * target_bonus * qs_bonus * orbit_bonus * neutral_bonus

            ships_est = _estimate_ships_needed(t, dist, player)
            if ships_est > 1:
                roi_const = ROI_NEUTRAL if t.owner == -1 else ROI_ENEMY
                score *= math.sqrt(roi_const / ships_est)

            # [v114-A] Indirect Wealth ボーナス（全ターゲット共通）
            iw_val = iw_map.get(t.id, 0.0)
            score *= (1.0 + IW_FACTOR * iw_val)

            if allow_concurrent and t.id in already_attacking_targets:
                score *= CONCURRENT_DISCOUNT

            if score > best_score:
                best_score  = score
                best_target = t
                best_dist   = dist

        if best_target is None:
            continue

        ships_needed = _estimate_ships_needed(best_target, best_dist, player)
        has_concurrent = allow_concurrent and (best_target.id in already_attacking_targets)

        if not _should_attack(available[mine.id], ships_needed, current_step, num_my_planets, has_concurrent):
            continue

        # [v113-A] 終盤ガリソン削減: 残余ターンを上限に garrison を制限
        ttf_for_garrison = best_dist / fleet_speed(max(1, ships_needed))
        remaining_after_capture = max(0, TOTAL_TURNS - current_step - int(ttf_for_garrison))
        effective_garrison_factor = min(GARRISON_BONUS_FACTOR, remaining_after_capture)
        garrison_bonus = min(
            int(best_target.production * effective_garrison_factor),
            int(available[mine.id]) - ships_needed,
        )
        garrison_bonus = max(0, garrison_bonus)

        if has_concurrent:
            ships_to_send = min(ships_needed, int(available[mine.id]))
        else:
            ships_to_send = ships_needed + garrison_bonus
        if ships_to_send < 1:
            continue

        safe, future_x, future_y = _get_safe_future_pos(
            mine.x, mine.y, best_target, angular_velocity, ships_to_send,
            comet_ids=comet_ids, comet_info=comet_info,
        )
        if not safe:
            continue

        angle = math.atan2(future_y - mine.y, future_x - mine.x)
        moves.append([mine.id, angle, ships_to_send])
        available[mine.id] -= ships_to_send
        if allow_concurrent:
            already_attacking_targets.add(best_target.id)
        else:
            attacked_targets.add(best_target.id)

    return moves
