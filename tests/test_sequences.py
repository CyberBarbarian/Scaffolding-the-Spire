from __future__ import annotations

import pytest

from sts2_bench.sequences import execute_card_sequence


def _combat_state(*, second_card: dict, enemies: list[dict] | None = None) -> dict:
    return {
        "decision": "combat_play",
        "energy": 3,
        "hand": [
            {
                "index": 0,
                "instance_id": "setup",
                "name": "Setup",
                "energy_cost": 1,
                "target_type": "Self",
                "can_play": True,
            },
            second_card,
        ],
        "enemies": enemies or [],
    }


@pytest.mark.parametrize(
    ("next_state", "second_step", "reason_fragment"),
    [
        (
            _combat_state(second_card={}),
            {"card_index": 1},
            "card index 1 is no longer in hand",
        ),
        (
            _combat_state(
                second_card={
                    "index": 1,
                    "instance_id": "follow-up",
                    "name": "Follow Up",
                    "energy_cost": 1,
                    "target_type": "Self",
                    "can_play": False,
                }
            ),
            {"card_index": 1},
            "Follow Up cannot be played now",
        ),
        (
            _combat_state(
                second_card={
                    "index": 1,
                    "instance_id": "follow-up",
                    "name": "Follow Up",
                    "energy_cost": 1,
                    "target_type": "AnyEnemy",
                    "can_play": True,
                },
                enemies=[
                    {"index": 0, "instance_id": "enemy-a", "name": "A", "hp": 10},
                    {"index": 1, "instance_id": "enemy-b", "name": "B", "hp": 10},
                ],
            ),
            {"card_index": 1},
            "Follow Up needs a target",
        ),
        (
            _combat_state(
                second_card={
                    "index": 1,
                    "instance_id": "follow-up",
                    "name": "Follow Up",
                    "energy_cost": 1,
                    "target_type": "AnyEnemy",
                    "can_play": True,
                },
                enemies=[],
            ),
            {"card_index": 1, "target_index": 0},
            "enemy target 0 is not available",
        ),
    ],
)
def test_sequence_early_stop_preserves_executed_command_prefix(
    next_state: dict,
    second_step: dict,
    reason_fragment: str,
) -> None:
    initial = _combat_state(
        second_card={
            "index": 1,
            "instance_id": "follow-up",
            "name": "Follow Up",
            "energy_cost": 1,
            "target_type": next_state["hand"][1].get("target_type", "Self"),
            "can_play": True,
        },
        enemies=[{"index": 0, "instance_id": "enemy-a", "name": "A", "hp": 10}],
    )
    sent: list[dict] = []

    def send(command: dict) -> dict:
        sent.append(command)
        return next_state

    result = execute_card_sequence(
        initial,
        [{"card_index": 0}, second_step],
        send,
    )

    assert result.steps_executed == 1
    assert reason_fragment in str(result.stopped_reason)
    assert result.commands_executed == sent == [
        {"cmd": "action", "action": "play_card", "args": {"card_index": 0}}
    ]
