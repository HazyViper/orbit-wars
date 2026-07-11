"""Orbit Wars シミュレータパッケージ（状態表現・幾何ユーティリティ）。"""

from .state import CometGroup, FleetState, GameState, PlanetState
from .utils import (
    BOARD_SIZE,
    CENTER,
    COMET_PRODUCTION,
    COMET_RADIUS,
    COMET_SPAWN_STEPS,
    PLANET_CLEARANCE,
    ROTATION_RADIUS_LIMIT,
    SUN_RADIUS,
    distance,
    fleet_speed,
    point_to_segment_distance,
)

__all__ = [
    "BOARD_SIZE",
    "CENTER",
    "COMET_PRODUCTION",
    "COMET_RADIUS",
    "COMET_SPAWN_STEPS",
    "CometGroup",
    "FleetState",
    "GameState",
    "PlanetState",
    "PLANET_CLEARANCE",
    "ROTATION_RADIUS_LIMIT",
    "SUN_RADIUS",
    "distance",
    "fleet_speed",
    "point_to_segment_distance",
]
