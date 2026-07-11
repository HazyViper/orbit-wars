"""Orbit Wars のターン処理ロジック（純関数）。

公式 orbit_wars.py の interpreter 関数を参考に、各処理を独立した
純関数として実装する。state は破壊的に変更しない。
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

from .state import CometGroup, FleetState, GameState, PlanetState
from .utils import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    distance,
    fleet_speed,
    point_to_segment_distance,
)


def generate_comet_paths(
    initial_planets: List[List[Any]],
    angular_velocity: float,
    spawn_step: int,
    comet_planet_ids: Optional[List[int]] = None,
    comet_speed: float = 4.0,
    rng: Optional[Any] = None,
) -> Optional[List[List[List[float]]]]:
    """太陽系外コメット用の 4 象限対称パスを生成する（公式 orbit_wars.py L244-381 相当）。

    rng に random.Random を渡すと、その列のみで幾何サンプルを行う（kaggle 新版の
    comet_rng と同じ分離）。None のときは従来どおりモジュール random を使う。
    """
    if rng is None:
        rng = random
    if comet_planet_ids is None:
        comet_pid_set: set = set()
    else:
        comet_pid_set = set(comet_planet_ids)
    for _ in range(300):
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e**2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        dense = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        paths = [
            [[x, y] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - x, BOARD_SIZE - y] for x, y in visible],
        ]

        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in comet_pid_set:
                continue
            pr = distance((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cx, cy),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cx, BOARD_SIZE - cy),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx**2 + dy**2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None


def _spawn_comet_groups(state: GameState) -> GameState:
    """指定ターンの直前にコメット 4 体グループを追加する（公式 L441-477）。

    条件は (state.step + 1) in comet_spawn_steps。成功時のみ planets / comets を更新する。
    """
    new_state = state.copy()
    if (new_state.step + 1) not in new_state.comet_spawn_steps:
        return new_state

    episode_seed = int(new_state.episode_seed) or 0
    comet_rng = random.Random(
        f"orbit_wars-comet-{episode_seed}-{int(new_state.step + 1)}"
    )

    initial_lists = [p.to_list() for p in new_state.initial_planets]
    comet_paths = generate_comet_paths(
        initial_lists,
        new_state.angular_velocity,
        new_state.step + 1,
        new_state.comet_planet_ids,
        new_state.comet_speed,
        rng=comet_rng,
    )
    if not comet_paths:
        return new_state

    max_id = max(p.id for p in new_state.planets)
    next_id = max_id + 1
    comet_ships = min(
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
        comet_rng.randint(1, 99),
    )
    group = CometGroup(planet_ids=[], paths=comet_paths, path_index=-1)
    for i, p_path in enumerate(comet_paths):
        pid = next_id + i
        group.planet_ids.append(pid)
        new_state.comet_planet_ids.append(pid)
        planet = PlanetState(
            id=pid,
            owner=-1,
            x=-99.0,
            y=-99.0,
            radius=COMET_RADIUS,
            ships=comet_ships,
            production=COMET_PRODUCTION,
        )
        new_state.planets.append(planet)
        new_state.initial_planets.append(
            PlanetState(
                id=planet.id,
                owner=planet.owner,
                x=planet.x,
                y=planet.y,
                radius=planet.radius,
                ships=planet.ships,
                production=planet.production,
            )
        )
    new_state.comets.append(group)
    return new_state


def expire_comets_pre(state: GameState) -> GameState:
    """ターン頭で path_index >= len(path) のコメットを削除する。

    公式 orbit_wars.py L419-439 に対応。

    各 comet group について:
    - planet_ids[i] が path_index >= len(paths[i]) なら expired
    - expired なコメットは planets, initial_planets, comet_planet_ids
      から除外
    - すべての planet_ids が expired した group は comets から除外
    """
    new_state = state.copy()

    expired_pids = set()
    for group in new_state.comets:
        idx = group.path_index
        for i, pid in enumerate(group.planet_ids):
            if idx >= len(group.paths[i]):
                expired_pids.add(pid)

    if not expired_pids:
        return new_state

    new_state.planets = [p for p in new_state.planets if p.id not in expired_pids]
    new_state.initial_planets = [
        p for p in new_state.initial_planets if p.id not in expired_pids
    ]
    new_state.comet_planet_ids = [
        pid for pid in new_state.comet_planet_ids if pid not in expired_pids
    ]
    for group in new_state.comets:
        group.planet_ids = [pid for pid in group.planet_ids if pid not in expired_pids]
    new_state.comets = [g for g in new_state.comets if g.planet_ids]

    return new_state


def launch_fleets(state: GameState, actions: List[List[Any]]) -> GameState:
    """各プレイヤーのアクションから艦隊を発射する。

    公式 orbit_wars.py L479-512（process_moves）に対応。

    Args:
        state: 現在の状態
        actions: actions[player_id] = [[from_id, angle, ships], ...]

    Returns:
        新しい状態（艦隊が追加され、惑星の ships が減算済み）
    """
    new_state = state.copy()

    for player_id, action in enumerate(actions):
        if not action or not isinstance(action, list):
            continue
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)

            from_planet = next(
                (p for p in new_state.planets if p.id == from_id), None
            )
            if from_planet is None:
                continue
            if from_planet.owner != player_id:
                continue
            if ships <= 0 or from_planet.ships < ships:
                continue

            from_planet.ships -= ships
            start_x = from_planet.x + math.cos(angle) * (from_planet.radius + 0.1)
            start_y = from_planet.y + math.sin(angle) * (from_planet.radius + 0.1)

            new_fleet = FleetState(
                id=new_state.next_fleet_id,
                owner=player_id,
                x=start_x,
                y=start_y,
                angle=angle,
                from_planet_id=from_id,
                ships=ships,
            )
            new_state.fleets.append(new_fleet)
            new_state.next_fleet_id += 1

    return new_state


def produce_ships(state: GameState) -> GameState:
    """所有惑星の艦数を production だけ増やす。

    公式 orbit_wars.py L514-517 に対応。中立 (-1) は対象外。
    """
    new_state = state.copy()
    for planet in new_state.planets:
        if planet.owner != -1:
            planet.ships += planet.production
    return new_state


def move_fleets(
    state: GameState, max_speed: float = 6.0
) -> Tuple[GameState, Dict[int, List[FleetState]]]:
    """艦隊を移動させ、境界・太陽・惑星衝突を判定する。

    公式 orbit_wars.py L519-551 に対応。

    Returns:
        (新しい状態, combat_lists)
        combat_lists: planet_id -> その惑星へ衝突した艦隊のリスト（FleetState）
    """
    new_state = state.copy()

    fleets_to_remove_ids = set()
    combat_lists = {p.id: [] for p in new_state.planets}

    for fleet in new_state.fleets:
        angle = fleet.angle
        ships = fleet.ships
        speed = fleet_speed(ships, max_speed)

        old_pos = (fleet.x, fleet.y)
        fleet.x += math.cos(angle) * speed
        fleet.y += math.sin(angle) * speed
        new_pos = (fleet.x, fleet.y)

        if not (0 <= fleet.x <= BOARD_SIZE and 0 <= fleet.y <= BOARD_SIZE):
            fleets_to_remove_ids.add(fleet.id)
            continue

        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove_ids.add(fleet.id)
            continue

        for planet in new_state.planets:
            planet_pos = (planet.x, planet.y)
            if point_to_segment_distance(planet_pos, old_pos, new_pos) < planet.radius:
                combat_lists[planet.id].append(fleet)
                fleets_to_remove_ids.add(fleet.id)
                break

    new_state.fleets = [f for f in new_state.fleets if f.id not in fleets_to_remove_ids]

    return new_state, combat_lists


def resolve_combat(
    state: GameState,
    combat_lists: Dict[int, List[FleetState]],
) -> GameState:
    """戦闘を解決する。

    公式 orbit_wars.py L630-669 に対応。

    各惑星に着弾した艦隊について:
    1. プレイヤーごとに艦数を合算
    2. 最大の攻撃力 vs 2位の攻撃力で生存数を計算
    3. 同点なら全滅、勝者の艦は守備と戦闘
    4. 守備を超えれば惑星陥落
    """
    new_state = state.copy()

    for pid, planet_fleets in combat_lists.items():
        if not planet_fleets:
            continue

        planet = next((p for p in new_state.planets if p.id == pid), None)
        if planet is None:
            continue

        player_ships = {}
        for fleet in planet_fleets:
            owner = fleet.owner
            player_ships[owner] = player_ships.get(owner, 0) + fleet.ships

        if not player_ships:
            continue

        sorted_players = sorted(
            player_ships.items(), key=lambda item: item[1], reverse=True
        )
        top_player, top_ships = sorted_players[0]

        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships

            if top_ships == second_ships:
                survivor_ships = 0

            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships

        if survivor_ships > 0:
            if planet.owner == survivor_owner:
                planet.ships += survivor_ships
            else:
                planet.ships -= survivor_ships
                if planet.ships < 0:
                    planet.owner = survivor_owner
                    planet.ships = abs(planet.ships)

    return new_state


def sweep_fleets(
    planet: PlanetState,
    old_pos: Tuple[float, float],
    new_pos: Tuple[float, float],
    fleets: List[FleetState],
    fleets_to_remove_ids: set,
    combat_lists: Dict[int, List[FleetState]],
) -> None:
    """惑星の移動経路上の艦隊を捕捉する（公転・コメット移動の共通処理）。

    公式 orbit_wars.py L559-570 に対応。

    Args:
        planet: 移動する惑星（半径で判定）
        old_pos: 移動前の中心座標 (x, y)
        new_pos: 移動後の中心座標 (x, y)
        fleets: 全艦隊リスト（参照のみ；要素は変更しない）
        fleets_to_remove_ids: 削除対象の艦隊 ID（捕捉時に追加される）
        combat_lists: 惑星 ID → その惑星に引っかかった艦隊のリスト
    """
    # 静止しているときは線分が退化するので判定不要
    if old_pos == new_pos:
        return

    for fleet in fleets:
        # 既に他理由で削除予定の艦隊は二重登録しない（公式は同一リスト参照で判定）
        if fleet.id in fleets_to_remove_ids:
            continue
        # 艦隊位置から惑星の「移動線分」までの距離が惑星半径未満なら捕捉（連続衝突）
        if (
            point_to_segment_distance((fleet.x, fleet.y), old_pos, new_pos)
            < planet.radius
        ):
            combat_lists[planet.id].append(fleet)
            fleets_to_remove_ids.add(fleet.id)


def move_planets(
    state: GameState,
    combat_lists: Dict[int, List[FleetState]],
) -> GameState:
    """公転惑星を回転させる。コメット惑星は対象外。

    公式 orbit_wars.py L572-590 に対応。

    Args:
        state: 現在のゲーム状態
        combat_lists: 艦隊移動フェーズなどで作られた辞書を引き続き更新する

    Returns:
        公転後の状態（掃引で除去された艦隊は fleets から欠ける）
    """
    new_state = state.copy()

    angular_velocity = new_state.angular_velocity
    step = new_state.step
    # コメットとしてマークされた惑星は軌道運動しない（別関数でパス移動）
    comet_pid_set = set(new_state.comet_planet_ids)
    # 初期配置から軌道半径・初期角度を決める（現在位置ではなく公式どおり initial）
    initial_by_id = {p.id: p for p in new_state.initial_planets}

    fleets_to_remove_ids = set()

    for planet in new_state.planets:
        if planet.id in comet_pid_set:
            continue
        initial_p = initial_by_id.get(planet.id)
        if initial_p is None:
            continue

        dx = initial_p.x - CENTER
        dy = initial_p.y - CENTER
        r = math.sqrt(dx**2 + dy**2)
        old_pos = (planet.x, planet.y)

        # 軌道半径 + 惑星半径が閾値未満なら太陽周りに角速度で回転（静的惑星は座標更新なし）
        if r + planet.radius < ROTATION_RADIUS_LIMIT:
            initial_angle = math.atan2(dy, dx)
            current_angle = initial_angle + angular_velocity * step
            planet.x = CENTER + r * math.cos(current_angle)
            planet.y = CENTER + r * math.sin(current_angle)

        new_pos = (planet.x, planet.y)

        # 静的惑星でも呼ぶ（old_pos == new_pos なら sweep_fleets は即 return）
        sweep_fleets(
            planet,
            old_pos,
            new_pos,
            new_state.fleets,
            fleets_to_remove_ids,
            combat_lists,
        )

    new_state.fleets = [
        f for f in new_state.fleets if f.id not in fleets_to_remove_ids
    ]

    return new_state


def move_comets(
    state: GameState,
    combat_lists: Dict[int, List[FleetState]],
) -> Tuple[GameState, List[int]]:
    """コメットを事前計算パスに沿って進め、終端を過ぎた ID を報告する。

    公式 orbit_wars.py L592-610 に対応。

    Args:
        state: 現在のゲーム状態
        combat_lists: 捕捉艦隊を蓄積する辞書（更新される）

    Returns:
        (更新後状態, expired_comet_pids)。後者はパス終端に達した惑星 ID のリスト。
    """
    new_state = state.copy()

    expired_pids = []
    fleets_to_remove_ids = set()

    for group in new_state.comets:
        # 公式どおりターンごとにインデックスを 1 進める（負の初期値もあり得る）
        group.path_index += 1
        idx = group.path_index

        for i, pid in enumerate(group.planet_ids):
            planet = next((p for p in new_state.planets if p.id == pid), None)
            if planet is None:
                continue

            p_path = group.paths[i]
            if idx >= len(p_path):
                # パス配列の外＝このコメットはこれ以上移動できないので失効候補
                expired_pids.append(pid)
            else:
                old_pos = (planet.x, planet.y)
                planet.x = p_path[idx][0]
                planet.y = p_path[idx][1]
                # 画面外プレースホルダからの初回配置では掃引しない（不正捕捉防止）
                if old_pos[0] >= 0:
                    sweep_fleets(
                        planet,
                        old_pos,
                        (planet.x, planet.y),
                        new_state.fleets,
                        fleets_to_remove_ids,
                        combat_lists,
                    )

    new_state.fleets = [
        f for f in new_state.fleets if f.id not in fleets_to_remove_ids
    ]

    return new_state, expired_pids


def expire_comets_post(state: GameState, expired_pids: List[int]) -> GameState:
    """move_comets で失効したコメット惑星を状態から除去する。

    公式 orbit_wars.py L612-626 に対応。
    expire_comets_pre と同じ削除ロジックだが、失効 ID は呼び出し側から渡す。
    """
    if not expired_pids:
        return state.copy()

    new_state = state.copy()
    expired_set = set(expired_pids)

    new_state.planets = [p for p in new_state.planets if p.id not in expired_set]
    new_state.initial_planets = [
        p for p in new_state.initial_planets if p.id not in expired_set
    ]
    new_state.comet_planet_ids = [
        pid for pid in new_state.comet_planet_ids if pid not in expired_set
    ]
    for group in new_state.comets:
        group.planet_ids = [
            pid for pid in group.planet_ids if pid not in expired_set
        ]
    new_state.comets = [g for g in new_state.comets if g.planet_ids]

    return new_state


def step(
    state: GameState,
    actions: List[List[Any]],
    max_speed: float = 6.0,
) -> GameState:
    """1ターン分の処理を実行する。

    公式 orbit_wars.py interpreter のターン処理順序を再現する。

    Args:
        state: 現在の状態（このターン頭の観測に対応）
        actions: actions[player_id] = [[from_id, angle, ships], ...]
        max_speed: 艦隊の最大速度

    Returns:
        次ターンの状態
    """
    # Phase 1: コメット失効（前半）
    state = expire_comets_pre(state)

    # Phase 2: コメット出現（指定ステップのターン頭）
    state = _spawn_comet_groups(state)

    # Phase 3: 艦隊発進
    state = launch_fleets(state, actions)

    # Phase 4: 生産
    state = produce_ships(state)

    # Phase 5: 艦隊移動
    state, combat_lists = move_fleets(state, max_speed)

    # Phase 6a: 公転惑星の回転
    state = move_planets(state, combat_lists)

    # Phase 6b: コメット移動
    state, expired_pids = move_comets(state, combat_lists)

    # Phase 6c: コメット失効（後半）
    state = expire_comets_post(state, expired_pids)

    # Phase 7: 戦闘解決
    state = resolve_combat(state, combat_lists)

    # step を進める（kaggle のフレームワークが interpreter 後に +1 するのと揃える）
    state.step += 1

    return state
