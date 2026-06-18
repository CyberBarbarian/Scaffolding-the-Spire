"""V1 coordinator scaffold over CA, OCC, RM, and RP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any

from .actions import LegalAction
from .agents import Agent, FallbackMode, ModelActionError, first_non_view_action
from .evidence import build_evidence_bundle, render_evidence_context, render_reward_decision_context
from .posterior import RewardPosteriorEvaluation
from .routing import route_state
from .run_context import RunContextStore, render_out_of_combat_context


@dataclass
class OutOfCombatContextAgent:
    """Prompt wrapper that gives OCC a structured RM/RP slice."""

    inner: Agent
    store: RunContextStore
    history_window: int = 6
    reward_posterior_evaluator: Any | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    _last_reward_context: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        base_prompt = self.inner.build_prompt(state, legal_actions)
        evidence_bundle = build_evidence_bundle(state, legal_actions, store=self.store)
        if _is_card_reward_decision(state):
            evidence_bundle, posterior_meta = self._attach_reward_posterior_evaluation(
                state,
                legal_actions,
                evidence_bundle,
            )
            self._last_reward_context = {
                "base_evidence_bundle_hash": build_evidence_bundle(state, legal_actions, store=self.store).bundle_hash(),
                "evidence_bundle": evidence_bundle,
                "posterior_meta": posterior_meta,
            }
            return "\n\n".join(
                [
                    base_prompt,
                    render_reward_decision_context(evidence_bundle),
                ]
            )
        return "\n\n".join(
            [
                base_prompt,
                render_evidence_context(evidence_bundle),
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
        chosen_prompt = prompt if prompt is not None else self.build_prompt(state, legal_actions)
        action, meta = self.inner.choose(state, legal_actions, prompt=chosen_prompt)
        meta = dict(meta)
        evidence_bundle = build_evidence_bundle(state, legal_actions, store=self.store)
        meta["evidence_bundle_hash"] = evidence_bundle.bundle_hash()
        meta["deck_summary_visible"] = evidence_bundle.deck_summary_visible
        meta["full_deck_details_visible"] = evidence_bundle.full_deck_details_visible
        meta["deck_details_visible"] = evidence_bundle.deck_details_visible
        meta["skip_action_visible"] = evidence_bundle.skip_action_visible
        meta["occ_context_mode"] = "history_free_reward" if _is_card_reward_decision(state) else "reference_turn"
        meta["occ_history_turns"] = len(self.history)
        if _is_card_reward_decision(state):
            self._attach_reward_posterior_meta(meta, evidence_bundle)
        return action, meta

    def reset_episode(self) -> None:
        self.history.clear()
        self._last_reward_context.clear()
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

    def _attach_reward_posterior_evaluation(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        evidence_bundle,
    ):
        if self.reward_posterior_evaluator is None:
            return evidence_bundle, {}
        evaluation, posterior_meta = self.reward_posterior_evaluator.evaluate(state, legal_actions, evidence_bundle)
        evaluation_data = _reward_posterior_evaluation_data(evaluation)
        posterior_meta = dict(posterior_meta or {})
        posterior_meta.setdefault("reward_posterior_evaluator", "unknown")
        posterior_meta.setdefault("reward_posterior_evaluation_hash", _reward_posterior_evaluation_hash(evaluation_data))
        enriched_bundle = replace(evidence_bundle, reward_posterior_evaluation=evaluation_data)
        return enriched_bundle, {
            **posterior_meta,
            "reward_posterior_evaluation_visible": True,
            "reward_posterior_evaluation": evaluation_data,
        }

    def _attach_reward_posterior_meta(self, meta: dict[str, Any], evidence_bundle) -> None:
        context = self._last_reward_context
        if context.get("base_evidence_bundle_hash") != evidence_bundle.bundle_hash():
            if self.reward_posterior_evaluator is None:
                meta["reward_posterior_evaluation_visible"] = False
            return
        posterior_meta = context.get("posterior_meta")
        if isinstance(posterior_meta, dict):
            meta.update(posterior_meta)


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
        if patch_summary["applied"] or "error" in patch_summary or patch_summary.get("card_reward_history_free"):
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
        self._record_combat_fact_transition(state, action, meta, next_state)
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
        if owner == "occ" and _is_card_reward_decision(state):
            summary = self._empty_patch_summary()
            summary["card_reward_history_free"] = True
            summary["ignored_patch_keys"] = sorted(_present_context_patch_keys(parsed))
            return summary
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
        if owner != "occ" or action.command.get("cmd") == "bench_view" or _is_card_reward_decision(state):
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

    def _record_combat_fact_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None,
    ) -> None:
        if not _is_combat_decision(state) or not isinstance(next_state, dict):
            return
        parsed = meta.get("parsed") if isinstance(meta, dict) else None
        notes = _combat_notes_from_parsed(parsed)
        if _is_combat_decision(next_state):
            return
        report = _combat_fact_report(state, next_state, notes)
        if report is None:
            return
        self.store.apply_agent_patch(
            {"combat_reports": [report]},
            owner="ca",
            state=state,
            action=action,
        )


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
    if first.get("card_reward_history_free") or second.get("card_reward_history_free"):
        merged["card_reward_history_free"] = True
        merged["ignored_patch_keys"] = [
            *list(first.get("ignored_patch_keys") or []),
            *list(second.get("ignored_patch_keys") or []),
        ]
    return merged


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _reward_posterior_evaluation_data(evaluation: Any) -> dict[str, Any]:
    if isinstance(evaluation, RewardPosteriorEvaluation):
        return evaluation.to_dict()
    to_dict = getattr(evaluation, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
    else:
        data = evaluation
    if not isinstance(data, dict):
        raise TypeError("reward posterior evaluator must return a dict-like evaluation")
    return json.loads(json.dumps(data, ensure_ascii=False, default=str))


def _reward_posterior_evaluation_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _transition_changes(next_state: dict[str, Any] | None) -> list[str]:
    if not isinstance(next_state, dict):
        return []
    result = next_state.get("last_action_result")
    if not isinstance(result, dict):
        return []
    return [str(item) for item in result.get("changes") or []][:4]


def _is_card_reward_decision(state: dict[str, Any]) -> bool:
    return state.get("decision") == "card_reward"


def _present_context_patch_keys(parsed: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    plan_patch = parsed.get("run_plan_patch")
    if isinstance(plan_patch, dict):
        keys.update(f"run_plan_patch.{key}" for key in plan_patch)
    if parsed.get("run_memory_facts"):
        keys.add("run_memory_facts")
    if parsed.get("combat_reports"):
        keys.add("combat_reports")
    return keys


def _is_combat_decision(state: dict[str, Any]) -> bool:
    if state.get("decision") == "combat_play":
        return True
    return state.get("decision") == "card_select" and isinstance(state.get("combat"), dict)


def _combat_notes_from_parsed(parsed: Any) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    raw_notes = parsed.get("combat_notes")
    if raw_notes is None:
        return []
    values = raw_notes if isinstance(raw_notes, list) else [raw_notes]
    return [_one_line(value) for value in values if _one_line(value)]


def _combat_fact_report(
    state: dict[str, Any],
    next_state: dict[str, Any],
    ca_observations: list[str],
) -> dict[str, Any] | None:
    player_before = state.get("player") or {}
    player_after = next_state.get("player") or {}
    hp_before = player_before.get("hp")
    hp_after = player_after.get("hp")
    hp_delta = None
    if isinstance(hp_before, (int, float)) and isinstance(hp_after, (int, float)):
        hp_delta = int(hp_after) - int(hp_before)
    result = next_state.get("last_action_result") if isinstance(next_state.get("last_action_result"), dict) else {}
    context = state.get("context") or {}
    return {
        "kind": "combat_fact_board",
        "text": _combat_fact_text(hp_before, hp_after, hp_delta, state, next_state),
        "act": state.get("act") or context.get("act"),
        "floor": state.get("floor") or context.get("floor"),
        "room_type": context.get("room_type"),
        "final_round": state.get("round"),
        "hp_before_final_transition": hp_before,
        "hp_after_combat": hp_after,
        "hp_delta_final_transition": hp_delta,
        "next_decision": next_state.get("decision"),
        "transition_changes": [str(item) for item in result.get("changes") or []][:8],
        "ca_observations": ca_observations,
        "source": "engine_transition_plus_ca_notes",
    }


def _combat_fact_text(
    hp_before: Any,
    hp_after: Any,
    hp_delta: int | None,
    state: dict[str, Any],
    next_state: dict[str, Any],
) -> str:
    parts = [
        f"combat ended into {next_state.get('decision')}",
        f"final_round={state.get('round', '?')}",
    ]
    if hp_delta is not None:
        parts.append(f"final_transition_hp={hp_before}->{hp_after} ({hp_delta})")
    return "; ".join(parts)


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
