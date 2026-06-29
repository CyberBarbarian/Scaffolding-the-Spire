"""Lightweight RL-style environment wrapper.

This is intentionally dependency-free.  If Gymnasium is installed, another
adapter can wrap this class and expose formal spaces.  The important contract
for STS2 is the dynamic legal-action list and action mask.
"""

from __future__ import annotations

from typing import Any

from .actions import LegalAction, build_legal_actions
from .context import build_last_action_result, compact_state
from .process import Sts2Process
from .seeds import DEFAULT_TEST_SEED
from .sequences import execute_card_sequence


class Sts2Env:
    """Minimal RL environment over one ``Sts2Process``."""

    def __init__(
        self,
        *,
        character: str = "Ironclad",
        ascension: int = 0,
        lang: str = "en",
        max_actions: int = 512,
        allow_repeat_views: bool = False,
        process: Sts2Process | None = None,
    ) -> None:
        self.character = character
        self.ascension = ascension
        self.lang = lang
        self.max_actions = max_actions
        self.allow_repeat_views = allow_repeat_views
        self.process = process or Sts2Process()
        self.state: dict[str, Any] | None = None
        self.legal_actions: list[LegalAction] = []

    def close(self) -> None:
        self.process.close()

    def reset(self, *, seed: str | None = DEFAULT_TEST_SEED) -> tuple[dict[str, Any], dict[str, Any]]:
        self.process.start()
        self.state = self.process.start_run(
            character=self.character,
            ascension=self.ascension,
            seed=seed,
            lang=self.lang,
        )
        self.legal_actions = build_legal_actions(
            self.state,
            allow_repeat_views=self.allow_repeat_views,
            allow_custom_sequences=False,
        )
        return self.observation(), self.info()

    def step(self, action_index: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Call reset before step")

        if action_index < 0 or action_index >= len(self.legal_actions):
            return self.observation(), -5.0, False, False, {**self.info(), "invalid_action": True}

        old_state = self.state
        action = self.legal_actions[action_index]
        if action.command.get("cmd") == "bench_view":
            self.state = self._apply_view_action(action.command)
            self.legal_actions = build_legal_actions(
                self.state,
                allow_repeat_views=self.allow_repeat_views,
                allow_custom_sequences=False,
            )
            return self.observation(), 0.0, False, False, self.info()
        if action.command.get("cmd") == "bench_sequence":
            sequence_result = execute_card_sequence(self.state, action.command.get("sequence") or [], self.process.send)
            next_state = sequence_result.state
            if next_state.get("type") == "error":
                self.state = next_state
                self.legal_actions = []
                return self.observation(), -5.0, True, False, {**self.info(), "engine_error": True}
            self.state = {
                **next_state,
                "last_action_result": build_last_action_result(
                    old_state,
                    next_state,
                    action_label=action.label,
                    action_kind=action.kind,
                ),
            }
            terminated = self.state.get("decision") == "game_over"
            reward = shaped_reward(old_state, self.state, invalid=False)
            self.legal_actions = build_legal_actions(
                self.state,
                allow_repeat_views=self.allow_repeat_views,
                allow_custom_sequences=False,
            )
            return self.observation(), reward, terminated, False, {
                **self.info(),
                "sequence_steps": sequence_result.steps_executed,
                "sequence_stop": sequence_result.stopped_reason,
            }

        next_state = self.process.send(action.command)
        if next_state.get("type") == "error":
            self.state = next_state
            self.legal_actions = []
            return self.observation(), -5.0, True, False, {**self.info(), "engine_error": True}
        self.state = {
            **next_state,
            "last_action_result": build_last_action_result(
                old_state,
                next_state,
                action_label=action.label,
                action_kind=action.kind,
            ),
        }
        terminated = self.state.get("decision") == "game_over"
        reward = shaped_reward(old_state, self.state, invalid=False)
        self.legal_actions = build_legal_actions(
            self.state,
            allow_repeat_views=self.allow_repeat_views,
            allow_custom_sequences=False,
        )
        return self.observation(), reward, terminated, False, self.info()

    def _apply_view_action(self, command: dict[str, Any]) -> dict[str, Any]:
        assert self.state is not None
        view = command.get("view")
        if view == "deck":
            return {**self.state, "view_deck": True}
        if view == "map":
            map_state = self.process.send({"cmd": "get_map"})
            if map_state.get("type") == "map":
                return {**self.state, "view_map": True, "full_map": map_state}
            return {**self.state, "view_map": True, "view_map_error": map_state}
        if view in {"draw", "discard", "exhaust"}:
            return {**self.state, f"view_{view}_pile": True}
        return self.state

    def observation(self) -> dict[str, Any]:
        return compact_state(self.state or {})

    def action_mask(self) -> list[int]:
        mask = [0] * self.max_actions
        for idx in range(min(len(self.legal_actions), self.max_actions)):
            mask[idx] = 1
        return mask

    def info(self) -> dict[str, Any]:
        return {
            "legal_actions": [action.to_log_dict() for action in self.legal_actions],
            "action_mask": self.action_mask(),
        }


def shaped_reward(old_state: dict[str, Any], new_state: dict[str, Any], *, invalid: bool = False) -> float:
    if invalid:
        return -5.0

    reward = -0.01
    old_player = old_state.get("player") or {}
    new_player = new_state.get("player") or {}
    old_ctx = old_state.get("context") or {}
    new_ctx = new_state.get("context") or {}

    old_floor = int(old_state.get("floor") or old_ctx.get("floor") or 0)
    new_floor = int(new_state.get("floor") or new_ctx.get("floor") or old_floor)
    if new_floor > old_floor:
        reward += 5.0 * (new_floor - old_floor)

    old_hp = old_player.get("hp")
    new_hp = new_player.get("hp")
    if isinstance(old_hp, (int, float)) and isinstance(new_hp, (int, float)):
        reward += 0.2 * (new_hp - old_hp)

    old_gold = old_player.get("gold")
    new_gold = new_player.get("gold")
    if isinstance(old_gold, (int, float)) and isinstance(new_gold, (int, float)) and new_gold > old_gold:
        reward += 0.02 * (new_gold - old_gold)

    if new_state.get("decision") == "game_over":
        reward += 100.0 if new_state.get("victory") else -100.0

    if old_state.get("decision") == "combat_play" and new_state.get("decision") in {
        "combat_reward",
        "card_reward",
        "map_select",
    }:
        reward += 3.0

    return reward
