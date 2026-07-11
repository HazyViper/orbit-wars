"""state と utils の単体テスト。"""

from simulator.state import FleetState, GameState, PlanetState
from simulator.utils import distance, fleet_speed, point_to_segment_distance


def test_planet_roundtrip():
    """PlanetState の list 変換が完全可逆であることを確認。"""
    original = [1, 0, 50.0, 30.0, 1.0, 100, 3]
    p = PlanetState.from_list(original)
    assert p.to_list() == original


def test_fleet_roundtrip():
    """FleetState の list 変換が完全可逆であることを確認。"""
    original = [5, 0, 25.5, 30.0, 1.5708, 1, 50]
    f = FleetState.from_list(original)
    assert f.to_list() == original


def test_distance():
    assert distance((0, 0), (3, 4)) == 5.0
    assert distance((50, 50), (50, 50)) == 0.0


def test_point_to_segment_distance_endpoint():
    """点が線分の端点と一致する場合は 0。"""
    d = point_to_segment_distance((0, 0), (0, 0), (10, 0))
    assert d == 0.0


def test_point_to_segment_distance_perpendicular():
    """点から線分への垂直距離。"""
    d = point_to_segment_distance((5, 5), (0, 0), (10, 0))
    assert d == 5.0


def test_point_to_segment_distance_outside():
    """点が線分の外側（端点よりも先）にある場合は端点までの距離。"""
    d = point_to_segment_distance((15, 0), (0, 0), (10, 0))
    assert d == 5.0


def test_fleet_speed_one_ship():
    """1 ship は速度 1.0。"""
    assert abs(fleet_speed(1) - 1.0) < 1e-9


def test_fleet_speed_max():
    """1000 ships で最大速度。"""
    assert abs(fleet_speed(1000) - 6.0) < 1e-9


def test_fleet_speed_monotonic():
    """艦数増加で速度も単調増加。"""
    for i in range(2, 999):
        assert fleet_speed(i) <= fleet_speed(i + 1)


def test_game_state_observation_roundtrip():
    """GameState の observation 変換が完全可逆であることを確認。"""
    obs = {
        "planets": [[0, 0, 50.0, 50.0, 1.0, 100, 3]],
        "fleets": [[0, 0, 25.0, 50.0, 0.0, 0, 50]],
        "comets": [],
        "comet_planet_ids": [],
        "initial_planets": [[0, 0, 50.0, 50.0, 1.0, 100, 3]],
        "angular_velocity": 0.05,
        "step": 10,
        "next_fleet_id": 1,
    }
    state = GameState.from_observation(obs, num_players=2)
    out = state.to_observation(player=0)
    out.pop("player", None)  # to_observation で追加されるだけなので除去して比較
    obs.pop("player", None)

    assert out["planets"] == obs["planets"]
    assert out["fleets"] == obs["fleets"]
    assert out["angular_velocity"] == obs["angular_velocity"]
    assert out["step"] == obs["step"]


def test_game_state_copy():
    """GameState.copy() が deep copy であることを確認。"""
    state1 = GameState(
        planets=[
            PlanetState(
                id=0,
                owner=0,
                x=50.0,
                y=50.0,
                radius=1.0,
                ships=100,
                production=3,
            )
        ]
    )
    state2 = state1.copy()
    state2.planets[0].ships = 200

    assert state1.planets[0].ships == 100
    assert state2.planets[0].ships == 200
