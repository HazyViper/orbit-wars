"""GameState を player 視点でスコアリングする。"""
from __future__ import annotations

from typing import Any, List

from .state import GameState


def evaluate_state(state: GameState, player_id: int) -> float:
    """プレイヤー視点で状態の良さを数値化する。

    自軍の戦力（惑星 ships + 生産能力 + 飛行中艦）から、
    敵の戦力を引いた値を返す。
    """
    score = 0.0

    for p in state.planets:
        if p.owner == player_id:
            score += p.ships * 1.0
            score += p.production * 50.0
        elif p.owner != -1:
            score -= p.ships * 0.5
            score -= p.production * 25.0

    for f in state.fleets:
        if f.owner == player_id:
            score += f.ships * 0.8
        else:
            score -= f.ships * 0.4

    return score


def simulate_with_action(
    state: GameState,
    action: List[Any],
    player_id: int,
) -> GameState:
    """1手だけ発進した場合の1ターン後の状態を返す。

    他プレイヤーは何もしないと仮定する（v10 のシンプルモデル）。
    """
    return simulate_with_moves(state, [action], player_id)


def simulate_with_moves(
    state: GameState,
    moves: List[List[Any]],
    player_id: int,
) -> GameState:
    """同一プレイヤーの複数発進をまとめて適用した1ターン後の状態を返す。"""
    from .turn_logic import step

    actions: List[List[Any]] = [[] for _ in range(state.num_players)]
    actions[player_id] = list(moves)

    return step(state.copy(), actions)


def evaluate_no_op(state: GameState, player_id: int) -> float:
    """何もしない場合の1ターン後のスコアを返す（baseline）。"""
    from .turn_logic import step

    actions: List[List[Any]] = [[] for _ in range(state.num_players)]
    no_op_state = step(state.copy(), actions)
    return evaluate_state(no_op_state, player_id)
