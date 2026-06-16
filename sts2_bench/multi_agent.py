"""V1 coordinator scaffold over CA, OCC, RM, and RP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import LegalAction
from .agents import Agent, FallbackMode, ModelActionError, first_non_view_action
from .routing import route_state
from .run_context import RunContextStore, render_out_of_combat_context


@dataclass
class OutOfCombatContextAgent:
    """Prompt wrapper that gives OCC a structured RM/RP slice."""

    inner: Agent
    store: RunContextStore
    history_window: int = 6
    history: list[dict[str, str]] = field(default_factory=list)

    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        base_prompt = self.inner.build_prompt(state, legal_actions)
        return "\n\n".join(
            [
                base_prompt,
                render_out_of_combat_context(self.store, state),
                _render_occ_history(self.history),
            ]
        )

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt: str | None = None,
    ) -> tuple[LegalAction, dict[str, Any]]:
        action, meta = self.inner.choose(state, legal_actions, prompt=prompt if prompt is not None else self.build_prompt(state, legal_actions))
        meta = dict(meta)
        meta["occ_context_mode"] = "reference_turn"
        meta["occ_history_turns"] = len(self.history)
        return action, meta

    def reset_episode(self) -> None:
        self.history.clear()
        reset = getattr(self.inner, "reset_episode", None)
        if callable(reset):
            reset()

    def receive_external_feedback(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        feedback: str,
        legal_actions: list[LegalAction],
    ) -> None:
        receive = getattr(self.inner, "receive_external_feedback", None)
        if callable(receive):
            receive(state, action, meta, feedback, legal_actions)

    def record_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        self._remember_transition(state, action, meta, next_state)
        record = getattr(self.inner, "record_transition", None)
        if callable(record):
            record(state, action, meta, next_state)

    def _remember_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None,
    ) -> None:
        if action.command.get("cmd") == "bench_view":
            return
        parsed = meta.get("parsed") if isinstance(meta, dict) else None
        reason = _one_line(parsed.get("reason")) if isinstance(parsed, dict) else ""
        context = state.get("context") or {}
        entry = {
            "decision": _one_line(state.get("decision")),
            "floor": _one_line(state.get("floor") or context.get("floor")),
            "action": _one_line(action.label),
            "reason": reason,
            "outcome": _one_line("; ".join(_transition_changes(next_state))),
        }
        self.history.append(entry)
        if self.history_window > 0 and len(self.history) > self.history_window:
            del self.history[: len(self.history) - self.history_window]


@dataclass
class CoordinatorAgent:
    """Single public agent that delegates to CA or OCC by route_state()."""

    ca_agent: Agent
    occ_agent: Agent
    store: RunContextStore = field(default_factory=RunContextStore)
    fallback_mode: FallbackMode = "error"

    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        route = route_state(state)
        if route.active_agent == "ca":
            return self.ca_agent.build_prompt(state, legal_actions)
        if route.active_agent == "occ":
            return self.occ_agent.build_prompt(state, legal_actions)
        return ""

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt: str | None = None,
    ) -> tuple[LegalAction, dict[str, Any]]:
        if not legal_actions:
            raise ValueError("No legal actions available")

        route = route_state(state)
        if route.active_agent == "none":
            raise ValueError("Terminal state has no active agent")

        delegate = self.ca_agent if route.active_agent == "ca" else self.occ_agent
        action, meta = delegate.choose(
            state,
            legal_actions,
            prompt=prompt if prompt is not None else delegate.build_prompt(state, legal_actions),
        )
        meta = dict(meta)
        action, stale_meta = _guard_action(action, legal_actions, fallback_mode=self.fallback_mode)
        meta.update(stale_meta)
        meta["active_agent"] = route.active_agent
        meta["route_reason"] = route.reason
        meta["run_context_hash"] = self.store.snapshot_hash()

        parsed = meta.get("parsed")
        if isinstance(parsed, dict):
            patch_summary = self._apply_patch(parsed, route.active_agent, state, action)
        else:
            patch_summary = self._empty_patch_summary()

        automatic_summary = self._apply_automatic_fact(parsed if isinstance(parsed, dict) else {}, route.active_agent, state, action)
        patch_summary = _merge_patch_summaries(patch_summary, automatic_summary)
        if patch_summary["applied"] or "error" in patch_summary:
            meta["run_context_patch"] = patch_summary
            meta["run_context_hash"] = self.store.snapshot_hash()

        return action, meta

    def reset_episode(self) -> None:
        self.store.reset()
        for agent in (self.ca_agent, self.occ_agent):
            reset = getattr(agent, "reset_episode", None)
            if callable(reset):
                reset()

    def receive_external_feedback(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        feedback: str,
        legal_actions: list[LegalAction],
    ) -> None:
        active_agent = meta.get("active_agent")
        delegate = self.ca_agent if active_agent == "ca" else self.occ_agent if active_agent == "occ" else None
        if delegate is None:
            route = route_state(state)
            delegate = self.ca_agent if route.active_agent == "ca" else self.occ_agent
        receive = getattr(delegate, "receive_external_feedback", None)
        if callable(receive):
            receive(state, action, meta, feedback, legal_actions)

    def record_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        active_agent = meta.get("active_agent")
        delegate = self.ca_agent if active_agent == "ca" else self.occ_agent if active_agent == "occ" else None
        if delegate is None:
            route = route_state(state)
            delegate = self.ca_agent if route.active_agent == "ca" else self.occ_agent
        record = getattr(delegate, "record_transition", None)
        if callable(record):
            record(state, action, meta, next_state)

    def _apply_patch(
        self,
        parsed: dict[str, Any],
        owner: str,
        state: dict[str, Any],
        action: LegalAction,
    ) -> dict[str, Any]:
        try:
            return self.store.apply_agent_patch(parsed, owner=owner, state=state, action=action)
        except ValueError as exc:
            return {
                "applied": False,
                "error": str(exc),
                "snapshot_hash": self.store.snapshot_hash(),
            }

    def _apply_automatic_fact(
        self,
        parsed: dict[str, Any],
        owner: str,
        state: dict[str, Any],
        action: LegalAction,
    ) -> dict[str, Any]:
        if owner != "occ" or action.command.get("cmd") == "bench_view":
            return self._empty_patch_summary()
        reason = _one_line(parsed.get("reason"))
        text = reason or action.label
        return self.store.apply_agent_patch(
            {
                "run_memory_facts": [
                    {
                        "kind": "decision_summary",
                        "text": text,
                    }
                ]
            },
            owner=owner,
            state=state,
            action=action,
        )

    def _empty_patch_summary(self) -> dict[str, Any]:
        return {
            "applied": False,
            "run_plan_updates": [],
            "facts_added": 0,
            "mandates_added": 0,
            "combat_reports_added": 0,
            "snapshot_hash": self.store.snapshot_hash(),
        }


def _guard_action(
    action: LegalAction,
    legal_actions: list[LegalAction],
    *,
    fallback_mode: FallbackMode,
) -> tuple[LegalAction, dict[str, Any]]:
    if action in legal_actions:
        return action, {}
    if fallback_mode == "error":
        raise ModelActionError("delegate returned an action outside the current legal action list")
    fallback = first_non_view_action(legal_actions)
    return fallback, {
        "fallback": True,
        "error": "delegate returned an action outside the current legal action list",
        "fallback_action_id": fallback.action_id,
    }


def _merge_patch_summaries(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "applied": bool(first.get("applied") or second.get("applied")),
        "run_plan_updates": [
            *list(first.get("run_plan_updates") or []),
            *list(second.get("run_plan_updates") or []),
        ],
        "facts_added": int(first.get("facts_added") or 0) + int(second.get("facts_added") or 0),
        "mandates_added": int(first.get("mandates_added") or 0) + int(second.get("mandates_added") or 0),
        "combat_reports_added": int(first.get("combat_reports_added") or 0)
        + int(second.get("combat_reports_added") or 0),
        "snapshot_hash": second.get("snapshot_hash") or first.get("snapshot_hash"),
    }
    if "error" in first:
        merged["error"] = first["error"]
    if "error" in second:
        merged["error"] = second["error"]
    return merged


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _transition_changes(next_state: dict[str, Any] | None) -> list[str]:
    if not isinstance(next_state, dict):
        return []
    result = next_state.get("last_action_result")
    if not isinstance(result, dict):
        return []
    return [str(item) for item in result.get("changes") or []][:4]


def _render_occ_history(history: list[dict[str, str]]) -> str:
    lines = [
        "OCC short turn history:",
        "These entries are reference only; always re-evaluate the current state and choose only a current legal action_id.",
    ]
    if not history:
        lines.append("- (empty)")
        return "\n".join(lines)
    for entry in history:
        parts = []
        floor = entry.get("floor")
        if floor:
            parts.append(f"F{floor}")
        decision = entry.get("decision")
        if decision:
            parts.append(decision)
        action = entry.get("action")
        if action:
            parts.append(f"action={action}")
        reason = entry.get("reason")
        if reason:
            parts.append(f"reason={reason}")
        outcome = entry.get("outcome")
        if outcome:
            parts.append(f"outcome={outcome}")
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines)
