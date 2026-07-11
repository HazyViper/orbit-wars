"""幾何計算ユーティリティ。orbit_wars.py の関数・定数を再現する。"""
import math
from typing import Tuple

# orbit_wars.py L17-L27 と同一（MIN/MAX はシミュ레ータ基底では未使用のため省略）
BOARD_SIZE = 100.0
CENTER = BOARD_SIZE / 2.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
PLANET_CLEARANCE = 7
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450]


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """2 点間ユークリッド距離（orbit_wars.py L30-L31 と同一）。"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_segment_distance(
    p: Tuple[float, float],
    v: Tuple[float, float],
    w: Tuple[float, float],
) -> float:
    """点 p から線分 v-w までの最短距離（orbit_wars.py L34-L43 と同一）。

    Minimum distance from point p to line segment v-w.
    """
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v)
    t = max(
        0,
        min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2),
    )
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)


def fleet_speed(ships: int, max_speed: float = 6.0) -> float:
    """orbit_wars.py L528-L529 と同一の速度式（既定 max_speed は shipSpeed と同じ 6）。

    speed = min( 1 + (max_speed - 1) * (log(ships) / log(1000)) ** 1.5 , max_speed )
    ships <= 0 は公式で未定義だがユーティリティとして 0 を返す。
    """
    if ships <= 0:
        return 0.0
    speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    return min(speed, max_speed)
