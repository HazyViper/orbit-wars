"""turn_logic.py の単体テスト。"""

import math

from simulator.state import CometGroup, FleetState, GameState, PlanetState
from simulator.turn_logic import (
    expire_comets_post,
    expire_comets_pre,
    launch_fleets,
    move_comets,
    move_fleets,
    move_planets,
    produce_ships,
    resolve_combat,
    sweep_fleets,
)
from simulator.utils import CENTER


# === expire_comets_pre のテスト ===


def test_expire_comets_pre_no_expired():
    """全 path_index が path 範囲内なら何も削除されない。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=0, x=50, y=50, radius=1, ships=10, production=1)
        ],
        comets=[
            CometGroup(
                planet_ids=[0],
                paths=[[(50, 50), (51, 50), (52, 50)]],
                path_index=1,
            )
        ],
        comet_planet_ids=[0],
        initial_planets=[
            PlanetState(id=0, owner=0, x=50, y=50, radius=1, ships=10, production=1)
        ],
    )
    new_state = expire_comets_pre(state)
    assert len(new_state.planets) == 1
    assert len(new_state.comets) == 1
    assert new_state.comet_planet_ids == [0]


def test_expire_comets_pre_one_expired():
    """path_index >= len(paths[i]) のコメットが削除される。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=-1, x=50, y=50, radius=1, ships=10, production=1)
        ],
        comets=[
            CometGroup(
                planet_ids=[0],
                paths=[[(50, 50), (51, 50)]],
                path_index=2,
            )
        ],
        comet_planet_ids=[0],
        initial_planets=[
            PlanetState(id=0, owner=-1, x=50, y=50, radius=1, ships=10, production=1)
        ],
    )
    new_state = expire_comets_pre(state)
    assert len(new_state.planets) == 0
    assert len(new_state.comets) == 0
    assert new_state.comet_planet_ids == []


# === launch_fleets のテスト ===


def test_launch_fleets_basic():
    """自軍の所有惑星から艦隊を発射する。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0, owner=0, x=50, y=50, radius=1.0, ships=100, production=1
            )
        ],
        next_fleet_id=0,
    )
    actions = [
        [[0, 0.0, 50]],
        [],
    ]
    new_state = launch_fleets(state, actions)
    assert len(new_state.fleets) == 1
    assert new_state.planets[0].ships == 50
    assert new_state.fleets[0].ships == 50
    assert new_state.next_fleet_id == 1


def test_launch_fleets_not_owned():
    """所有していない惑星からは発射できない。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0, owner=1, x=50, y=50, radius=1.0, ships=100, production=1
            )
        ],
        next_fleet_id=0,
    )
    actions = [
        [[0, 0.0, 50]],
        [],
    ]
    new_state = launch_fleets(state, actions)
    assert len(new_state.fleets) == 0
    assert new_state.planets[0].ships == 100


def test_launch_fleets_insufficient_ships():
    """艦数が不足している場合は発射できない。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0, owner=0, x=50, y=50, radius=1.0, ships=10, production=1
            )
        ],
        next_fleet_id=0,
    )
    actions = [
        [[0, 0.0, 50]],
        [],
    ]
    new_state = launch_fleets(state, actions)
    assert len(new_state.fleets) == 0
    assert new_state.planets[0].ships == 10


def test_launch_fleets_zero_ships():
    """0 以下の ships は発射できない。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0, owner=0, x=50, y=50, radius=1.0, ships=10, production=1
            )
        ],
        next_fleet_id=0,
    )
    actions = [
        [[0, 0.0, 0]],
        [],
    ]
    new_state = launch_fleets(state, actions)
    assert len(new_state.fleets) == 0


# === produce_ships のテスト ===


def test_produce_ships_owned():
    """所有惑星の ships が production だけ増える。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=0, x=50, y=50, radius=1, ships=100, production=3),
            PlanetState(id=1, owner=1, x=60, y=60, radius=1, ships=50, production=2),
        ]
    )
    new_state = produce_ships(state)
    assert new_state.planets[0].ships == 103
    assert new_state.planets[1].ships == 52


def test_produce_ships_neutral():
    """中立惑星 (owner=-1) は production しない。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0, owner=-1, x=50, y=50, radius=1, ships=100, production=3
            )
        ]
    )
    new_state = produce_ships(state)
    assert new_state.planets[0].ships == 100


# === move_fleets のテスト ===


def test_move_fleets_simple_motion():
    """艦隊が直線的に移動する。"""
    state = GameState(
        fleets=[
            FleetState(
                id=0,
                owner=0,
                x=10.0,
                y=50.0,
                angle=0.0,
                from_planet_id=0,
                ships=1,
            )
        ]
    )
    new_state, _combat = move_fleets(state)
    assert abs(new_state.fleets[0].x - 11.0) < 1e-9
    assert abs(new_state.fleets[0].y - 50.0) < 1e-9


def test_move_fleets_out_of_bounds():
    """境界外に出た艦隊は削除される。"""
    state = GameState(
        fleets=[
            FleetState(
                id=0,
                owner=0,
                x=99.5,
                y=50.0,
                angle=0.0,
                from_planet_id=0,
                ships=1,
            )
        ]
    )
    new_state, _combat = move_fleets(state)
    assert len(new_state.fleets) == 0


def test_move_fleets_sun_collision():
    """太陽中心 (CENTER, CENTER) から見て線分が SUN_RADIUS 内を通れば消滅する。

    連続判定（点と線分の距離）は、1ターンだけ「端」の外側にいても、
    移動線分が円ディスク SUN_RADIUS と交われば消える。
    （50, 44.5）から +y で 1 マス移動すると、太陽 (50, 50) 半径 10 内を横切る。
    """
    state = GameState(
        fleets=[
            FleetState(
                id=0,
                owner=0,
                x=50.0,
                y=44.5,
                angle=math.pi / 2,
                from_planet_id=0,
                ships=1,
            )
        ]
    )
    new_state, _combat = move_fleets(state)
    assert len(new_state.fleets) == 0


def test_move_fleets_planet_collision():
    """惑星に着弾した艦隊は combat_lists に追加される。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0,
                owner=-1,
                x=20.0,
                y=50.0,
                radius=1.0,
                ships=10,
                production=1,
            )
        ],
        fleets=[
            FleetState(
                id=0,
                owner=0,
                x=15.0,
                y=50.0,
                angle=0.0,
                from_planet_id=99,
                ships=1,
            )
        ],
    )
    new_state, combat = move_fleets(state)
    assert len(new_state.fleets) == 1
    assert combat[0] == []

    state2 = GameState(
        planets=[
            PlanetState(
                id=0,
                owner=-1,
                x=20.0,
                y=50.0,
                radius=1.0,
                ships=10,
                production=1,
            )
        ],
        fleets=[
            FleetState(
                id=0,
                owner=0,
                x=18.5,
                y=50.0,
                angle=0.0,
                from_planet_id=99,
                ships=1,
            )
        ],
    )
    new_state2, combat2 = move_fleets(state2)
    assert len(new_state2.fleets) == 0
    assert len(combat2[0]) == 1


# === resolve_combat のテスト ===


def test_resolve_combat_single_attacker_capture():
    """1 player の攻撃で中立惑星を陥落させる。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0,
                owner=-1,
                x=20.0,
                y=50.0,
                radius=1.0,
                ships=5,
                production=1,
            )
        ]
    )
    attacker = FleetState(
        id=0, owner=0, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=10
    )
    combat_lists = {0: [attacker]}
    new_state = resolve_combat(state, combat_lists)

    assert new_state.planets[0].owner == 0
    assert new_state.planets[0].ships == 5


def test_resolve_combat_single_attacker_insufficient():
    """守備を超えない攻撃は陥落しない。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0,
                owner=-1,
                x=20.0,
                y=50.0,
                radius=1.0,
                ships=20,
                production=1,
            )
        ]
    )
    attacker = FleetState(
        id=0, owner=0, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=10
    )
    combat_lists = {0: [attacker]}
    new_state = resolve_combat(state, combat_lists)

    assert new_state.planets[0].owner == -1
    assert new_state.planets[0].ships == 10


def test_resolve_combat_two_attackers_decisive():
    """2 attackers で 1 位が勝ち、守備と戦闘。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0,
                owner=-1,
                x=20.0,
                y=50.0,
                radius=1.0,
                ships=2,
                production=1,
            )
        ]
    )
    attacker_a = FleetState(
        id=0, owner=0, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=10
    )
    attacker_b = FleetState(
        id=1, owner=1, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=6
    )
    combat_lists = {0: [attacker_a, attacker_b]}
    new_state = resolve_combat(state, combat_lists)

    assert new_state.planets[0].owner == 0
    assert new_state.planets[0].ships == 2


def test_resolve_combat_tie_destroys_all():
    """1 位と 2 位が同点なら全滅。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0,
                owner=-1,
                x=20.0,
                y=50.0,
                radius=1.0,
                ships=5,
                production=1,
            )
        ]
    )
    attacker_a = FleetState(
        id=0, owner=0, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=10
    )
    attacker_b = FleetState(
        id=1, owner=1, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=10
    )
    combat_lists = {0: [attacker_a, attacker_b]}
    new_state = resolve_combat(state, combat_lists)

    assert new_state.planets[0].owner == -1
    assert new_state.planets[0].ships == 5


def test_resolve_combat_friendly_arrival():
    """自軍からの艦隊は守備に加算される。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0,
                owner=0,
                x=20.0,
                y=50.0,
                radius=1.0,
                ships=5,
                production=1,
            )
        ]
    )
    attacker = FleetState(
        id=0, owner=0, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=10
    )
    combat_lists = {0: [attacker]}
    new_state = resolve_combat(state, combat_lists)

    assert new_state.planets[0].owner == 0
    assert new_state.planets[0].ships == 15


def test_resolve_combat_three_way_third_destroyed():
    """3 player 戦闘で 3 位以下の艦は完全消滅（戦闘に寄与しない）。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0,
                owner=-1,
                x=20.0,
                y=50.0,
                radius=1.0,
                ships=2,
                production=1,
            )
        ]
    )
    attacker_a = FleetState(
        id=0, owner=0, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=100
    )
    attacker_b = FleetState(
        id=1, owner=1, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=60
    )
    attacker_c = FleetState(
        id=2, owner=2, x=20.0, y=50.0, angle=0.0, from_planet_id=99, ships=30
    )
    combat_lists = {0: [attacker_a, attacker_b, attacker_c]}
    new_state = resolve_combat(state, combat_lists)

    assert new_state.planets[0].owner == 0
    assert new_state.planets[0].ships == 38


# === sweep_fleets のテスト ===


def test_sweep_fleets_no_motion():
    """old_pos == new_pos なら何もしない。"""
    planet = PlanetState(
        id=0, owner=0, x=50, y=50, radius=2.0, ships=10, production=1
    )
    fleet = FleetState(
        id=0, owner=1, x=50, y=50, angle=0.0, from_planet_id=99, ships=5
    )
    fleets_to_remove = set()
    combat_lists = {0: []}

    sweep_fleets(
        planet, (50, 50), (50, 50), [fleet], fleets_to_remove, combat_lists
    )

    assert combat_lists[0] == []
    assert fleet.id not in fleets_to_remove


def test_sweep_fleets_catches_fleet():
    """惑星の移動経路上に艦隊があれば捕捉する。"""
    planet = PlanetState(
        id=0, owner=0, x=10, y=50, radius=2.0, ships=10, production=1
    )
    fleet = FleetState(
        id=0, owner=1, x=15, y=50, angle=0.0, from_planet_id=99, ships=5
    )
    fleets_to_remove = set()
    combat_lists = {0: []}

    sweep_fleets(
        planet, (10, 50), (20, 50), [fleet], fleets_to_remove, combat_lists
    )

    assert len(combat_lists[0]) == 1
    assert 0 in fleets_to_remove


def test_sweep_fleets_skip_already_removed():
    """既に削除予定の艦隊は捕捉しない。"""
    planet = PlanetState(
        id=0, owner=0, x=10, y=50, radius=2.0, ships=10, production=1
    )
    fleet = FleetState(
        id=0, owner=1, x=15, y=50, angle=0.0, from_planet_id=99, ships=5
    )
    fleets_to_remove = {0}
    combat_lists = {0: []}

    sweep_fleets(
        planet, (10, 50), (20, 50), [fleet], fleets_to_remove, combat_lists
    )

    assert combat_lists[0] == []


# === move_planets のテスト ===


def test_move_planets_static_planet():
    """静的惑星 (r + radius >= ROTATION_RADIUS_LIMIT) は座標が変わらない。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=0, x=99, y=50, radius=1, ships=10, production=1)
        ],
        initial_planets=[
            PlanetState(id=0, owner=0, x=99, y=50, radius=1, ships=10, production=1)
        ],
        angular_velocity=0.05,
        step=10,
    )
    combat_lists = {0: []}
    new_state = move_planets(state, combat_lists)

    assert abs(new_state.planets[0].x - 99) < 1e-9
    assert abs(new_state.planets[0].y - 50) < 1e-9


def test_move_planets_orbiting_planet():
    """公転惑星 (r + radius < 50) は angular_velocity * step で回転する。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=0, x=70, y=50, radius=1, ships=10, production=1)
        ],
        initial_planets=[
            PlanetState(id=0, owner=0, x=70, y=50, radius=1, ships=10, production=1)
        ],
        angular_velocity=math.pi / 2,
        step=1,
    )
    combat_lists = {0: []}
    new_state = move_planets(state, combat_lists)

    assert abs(new_state.planets[0].x - 50) < 1e-9
    assert abs(new_state.planets[0].y - 70) < 1e-9


def test_move_planets_skips_comets():
    """comet_planet_ids にあるものは公転処理から除外される。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=0, x=70, y=50, radius=1, ships=10, production=1)
        ],
        initial_planets=[
            PlanetState(id=0, owner=0, x=70, y=50, radius=1, ships=10, production=1)
        ],
        comet_planet_ids=[0],
        angular_velocity=math.pi / 2,
        step=1,
    )
    combat_lists = {0: []}
    new_state = move_planets(state, combat_lists)

    assert new_state.planets[0].x == 70
    assert new_state.planets[0].y == 50


def test_move_planets_sweeps_fleets():
    """公転で動いた惑星が移動線分上の艦隊を捕捉する。"""
    angular_velocity = 0.1
    step = 1
    r_orbit = 20.0
    old_x, old_y = 70.0, 50.0
    current_angle = angular_velocity * step
    new_x = CENTER + r_orbit * math.cos(current_angle)
    new_y = CENTER + r_orbit * math.sin(current_angle)
    mid_x = (old_x + new_x) / 2
    mid_y = (old_y + new_y) / 2

    state = GameState(
        planets=[
            PlanetState(
                id=0, owner=0, x=old_x, y=old_y, radius=2, ships=10, production=1
            )
        ],
        initial_planets=[
            PlanetState(
                id=0, owner=0, x=old_x, y=old_y, radius=2, ships=10, production=1
            )
        ],
        fleets=[
            FleetState(
                id=0,
                owner=1,
                x=mid_x,
                y=mid_y,
                angle=0.0,
                from_planet_id=99,
                ships=5,
            )
        ],
        angular_velocity=angular_velocity,
        step=step,
    )
    combat_lists = {0: []}
    new_state = move_planets(state, combat_lists)

    assert len(combat_lists[0]) == 1
    assert len(new_state.fleets) == 0


# === move_comets のテスト ===


def test_move_comets_advance_path():
    """path_index が +1 され、planet が paths[idx] に移動する。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=-1, x=10, y=10, radius=1, ships=5, production=1)
        ],
        comets=[
            CometGroup(
                planet_ids=[0],
                paths=[[(10, 10), (15, 15), (20, 20), (25, 25)]],
                path_index=0,
            )
        ],
        comet_planet_ids=[0],
    )
    combat_lists = {0: []}
    new_state, expired = move_comets(state, combat_lists)

    assert new_state.comets[0].path_index == 1
    assert new_state.planets[0].x == 15
    assert new_state.planets[0].y == 15
    assert expired == []


def test_move_comets_expired_at_end():
    """path 終端に達したコメットは expired として返される。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=-1, x=20, y=20, radius=1, ships=5, production=1)
        ],
        comets=[
            CometGroup(
                planet_ids=[0],
                paths=[[(10, 10), (15, 15), (20, 20)]],
                path_index=2,
            )
        ],
        comet_planet_ids=[0],
    )
    combat_lists = {0: []}
    new_state, expired = move_comets(state, combat_lists)

    assert expired == [0]


def test_move_comets_skip_sweep_first_placement():
    """初回配置（old_pos が画面外）では sweep_fleets を呼ばない。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=-1, x=-1, y=-1, radius=1, ships=5, production=1)
        ],
        comets=[
            CometGroup(
                planet_ids=[0],
                paths=[[(50, 50), (55, 50)]],
                path_index=-1,
            )
        ],
        comet_planet_ids=[0],
        fleets=[
            FleetState(
                id=0,
                owner=1,
                x=20,
                y=-0.5,
                angle=0.0,
                from_planet_id=99,
                ships=5,
            )
        ],
    )
    combat_lists = {0: []}
    new_state, expired = move_comets(state, combat_lists)

    assert combat_lists[0] == []
    assert len(new_state.fleets) == 1
    assert expired == []


# === expire_comets_post のテスト ===


def test_expire_comets_post_basic():
    """expired_pids にあるコメットを削除する。"""
    state = GameState(
        planets=[
            PlanetState(
                id=0, owner=-1, x=50, y=50, radius=1, ships=10, production=1
            ),
            PlanetState(id=1, owner=0, x=70, y=50, radius=1, ships=20, production=2),
        ],
        comets=[
            CometGroup(
                planet_ids=[0],
                paths=[[(50, 50)]],
                path_index=0,
            )
        ],
        comet_planet_ids=[0],
        initial_planets=[
            PlanetState(
                id=0, owner=-1, x=50, y=50, radius=1, ships=10, production=1
            ),
        ],
    )
    new_state = expire_comets_post(state, [0])

    assert len(new_state.planets) == 1
    assert new_state.planets[0].id == 1
    assert new_state.comet_planet_ids == []
    assert len(new_state.comets) == 0


def test_expire_comets_post_empty():
    """expired_pids が空ならコピーのみ返す（構造は同等）。"""
    state = GameState(
        planets=[
            PlanetState(id=0, owner=0, x=50, y=50, radius=1, ships=10, production=1)
        ]
    )
    new_state = expire_comets_post(state, [])

    assert len(new_state.planets) == 1
    assert new_state.planets[0].id == state.planets[0].id
