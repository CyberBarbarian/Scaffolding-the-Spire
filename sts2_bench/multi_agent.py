"""V1 coordinator scaffold over CA, OCC, RM, and RP."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any

from .actions import LegalAction
from .agents import Agent, FallbackMode, ModelActionError, first_non_view_action
from .evidence import build_evidence_bundle, render_evidence_context, render_reward_decision_context
from .posterior import RewardPosteriorEvaluation
from .routing import route_state
from .run_context import PLAN_PATCH_KEYS, RunContextStore, render_out_of_combat_context


@dataclass
class RoomProcessLedger:
    """Short-lived factual process board for the current room."""

    max_actions: int = 12
    room_key: tuple[Any, Any, Any] | None = None
    actions: list[dict[str, Any]] = field(default_factory=list)
    inspected_card_rewards: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.room_key = None
        self.actions.clear()
        self.inspected_card_rewards.clear()

    def record_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        if action.command.get("cmd") == "bench_view":
            return

        current_key = _room_key(state)
        next_key = _room_key(next_state) if isinstance(next_state, dict) else current_key
        if current_key is None:
            if next_key is None:
                self.reset()
                return
            self._ensure_room(next_key)
            self._append_action(state, action, meta, next_state)
            self._record_card_reward_transition(state, action, next_state)
            return

        self._ensure_room(current_key)
        self._append_action(state, action, meta, next_state)
        self._record_card_reward_transition(state, action, next_state)
        if next_key is None or next_key != current_key:
            self.reset()

    def snapshot_for_state(self, state: dict[str, Any]) -> dict[str, Any]:
        key = _room_key(state)
        if key is None or key != self.room_key:
            return {}
        if not self.actions and not self.inspected_card_rewards:
            return {}
        return {
            "room_key": _room_key_payload(key),
            "actions": [dict(item) for item in self.actions],
            "inspected_card_rewards": [dict(item) for item in self.inspected_card_rewards],
            "contract": {
                "purpose": "room-local factual history only",
                "authoritative_source": "current state and current legal_actions",
                "do_not_reuse_stored_indices": True,
            },
        }

    def _ensure_room(self, key: tuple[Any, Any, Any]) -> None:
        if self.room_key == key:
            return
        self.room_key = key
        self.actions.clear()
        self.inspected_card_rewards.clear()

    def _append_action(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None,
    ) -> None:
        parsed = meta.get("parsed") if isinstance(meta, dict) else None
        reason = _one_line(parsed.get("reason")) if isinstance(parsed, dict) else ""
        entry: dict[str, Any] = {
            "decision": _one_line(state.get("decision")),
            "action_kind": _one_line(action.kind),
            "action_label": _stable_room_action_label(action.label),
        }
        if state.get("round") is not None:
            entry["round"] = state.get("round")
        if reason:
            entry["reason"] = reason
        outcome = _transition_changes(next_state)
        if outcome:
            entry["outcome"] = outcome
        self.actions.append(entry)
        if self.max_actions > 0 and len(self.actions) > self.max_actions:
            del self.actions[: len(self.actions) - self.max_actions]

    def _record_card_reward_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        next_state: dict[str, Any] | None,
    ) -> None:
        if action.kind == "combat_reward" and isinstance(next_state, dict) and next_state.get("decision") == "card_reward":
            candidates = _card_names_from_state(next_state)
            entry = self._upsert_card_reward(candidates)
            entry["source_action"] = _stable_room_action_label(action.label)
            entry["candidates"] = candidates
            entry["status"] = "revealed"
            entry["last_decision"] = "revealed_candidates"
            return

        if state.get("decision") == "card_reward" and action.kind == "card_reward_skip":
            candidates = _card_names_from_state(state)
            entry = self._upsert_card_reward(candidates)
            entry["candidates"] = candidates
            entry["status"] = "skipped_visible"
            entry["last_decision"] = "skipped_visible_candidates"
            entry["still_available_after_skip"] = _card_reward_still_available(next_state)
            return

        if state.get("decision") == "card_reward" and action.kind == "card_reward":
            candidates = _card_names_from_state(state)
            entry = self._upsert_card_reward(candidates)
            entry["candidates"] = candidates
            entry["status"] = "picked"
            entry["last_decision"] = "picked_visible_candidate"
            selected = _selected_card_name(action, state)
            if selected:
                entry["selected_card"] = selected
            return

        if action.kind == "combat_reward_skip" and _label_mentions_card_reward(action.label):
            entry = self._upsert_card_reward([])
            entry["source_action"] = _stable_room_action_label(action.label)
            entry["candidates"] = []
            entry["status"] = "skipped_unopened"
            entry["last_decision"] = "skipped_unopened_card_reward"

    def _upsert_card_reward(self, candidates: list[str]) -> dict[str, Any]:
        if candidates:
            candidate_key = tuple(candidates)
            for item in self.inspected_card_rewards:
                if tuple(item.get("candidates") or []) == candidate_key:
                    return item
        if self.inspected_card_rewards:
            last = self.inspected_card_rewards[-1]
            if candidates or not last.get("candidates"):
                return last
        item: dict[str, Any] = {}
        self.inspected_card_rewards.append(item)
        return item


@dataclass
class OutOfCombatContextAgent:
    """Prompt wrapper that gives OCC a structured RM/RP slice."""

    inner: Agent
    store: RunContextStore
    history_window: int = 6
    reward_posterior_evaluator: Any | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    room_process: RoomProcessLedger = field(default_factory=RoomProcessLedger)
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
                    _render_room_process_context(self.room_process, state),
                ]
            )
        if state.get("decision") == "map_select":
            return "\n\n".join(
                [
                    base_prompt,
                    _render_route_occ_context(evidence_bundle, self.store, state),
                ]
            )
        return "\n\n".join(
            [
                base_prompt,
                render_evidence_context(evidence_bundle),
                render_out_of_combat_context(self.store, state),
                _render_room_process_context(self.room_process, state),
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
        meta["active_agent"] = "occ"
        evidence_bundle = build_evidence_bundle(state, legal_actions, store=self.store)
        meta["evidence_bundle_hash"] = evidence_bundle.bundle_hash()
        meta["deck_summary_visible"] = evidence_bundle.deck_summary_visible
        meta["full_deck_details_visible"] = evidence_bundle.full_deck_details_visible
        meta["deck_details_visible"] = evidence_bundle.deck_details_visible
        meta["skip_action_visible"] = evidence_bundle.skip_action_visible
        if _is_card_reward_decision(state):
            meta["occ_context_mode"] = "history_free_reward"
        elif state.get("decision") == "map_select":
            meta["occ_context_mode"] = "structured_route"
        else:
            meta["occ_context_mode"] = "room_process"
        meta["occ_history_turns"] = len(self.history)
        room_process_snapshot = self.room_process.snapshot_for_state(state)
        meta["room_process_visible"] = bool(room_process_snapshot)
        meta["room_process_action_count"] = len(room_process_snapshot.get("actions") or [])
        meta["room_process_inspected_card_reward_count"] = len(room_process_snapshot.get("inspected_card_rewards") or [])
        if _is_card_reward_decision(state):
            self._attach_reward_posterior_meta(meta, evidence_bundle)
        return action, meta

    def reset_episode(self) -> None:
        self.history.clear()
        self.room_process.reset()
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
        self.record_room_transition(state, action, meta, next_state)
        self._remember_transition(state, action, meta, next_state)
        record = getattr(self.inner, "record_transition", None)
        if callable(record):
            record(state, action, meta, next_state)

    def record_room_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        self.room_process.record_transition(state, action, meta, next_state)

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
        self._record_route_step_transition(state, action, meta, next_state)
        self._record_combat_fact_transition(state, action, meta, next_state)
        active_agent = meta.get("active_agent")
        delegate = self.ca_agent if active_agent == "ca" else self.occ_agent if active_agent == "occ" else None
        if delegate is None:
            route = route_state(state)
            active_agent = route.active_agent
            delegate = self.ca_agent if route.active_agent == "ca" else self.occ_agent
        if active_agent != "occ":
            self._record_occ_room_process_transition(state, action, meta, next_state)
        record = getattr(delegate, "record_transition", None)
        if callable(record):
            record(state, action, meta, next_state)

    def _record_route_step_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None,
    ) -> None:
        if state.get("decision") != "map_select" or action.kind != "map" or not isinstance(next_state, dict):
            return
        step = _route_step_report(state, action, meta, next_state)
        if step is not None:
            self.store.run_memory.route_steps.append(step)

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

    def _record_occ_room_process_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None,
    ) -> None:
        record = getattr(self.occ_agent, "record_room_transition", None)
        if callable(record):
            record(state, action, meta, next_state)


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


def _render_room_process_context(ledger: RoomProcessLedger, state: dict[str, Any]) -> str:
    snapshot = ledger.snapshot_for_state(state)
    if not snapshot:
        return ""
    return "Room process context JSON:\n" + json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _room_key(state: dict[str, Any] | None) -> tuple[Any, Any, Any] | None:
    if not isinstance(state, dict):
        return None
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    decision = state.get("decision")
    room_type = state.get("room_type") or context.get("room_type") or _room_type_from_decision(decision)
    if decision == "map_select" or room_type in (None, "", "Map"):
        return None
    floor = state.get("floor") or context.get("floor")
    if floor in (None, ""):
        return None
    act = state.get("act") or context.get("act")
    return (act, floor, room_type)


def _room_key_payload(key: tuple[Any, Any, Any]) -> dict[str, Any]:
    act, floor, room_type = key
    return {"act": act, "floor": floor, "room_type": room_type}


def _room_type_from_decision(decision: Any) -> str | None:
    return {
        "combat_play": "Combat",
        "combat_reward": "CombatReward",
        "card_reward": "CombatReward",
        "event_choice": "Event",
        "event_result": "Event",
        "shop": "Shop",
        "fake_merchant_shop": "Shop",
        "rest_site": "RestSite",
        "treasure": "Treasure",
        "crystal_sphere": "CrystalSphere",
        "bundle_select": "BundleSelect",
    }.get(str(decision or ""))


def _stable_room_action_label(label: Any) -> str:
    text = _one_line(label)
    if not text:
        return ""
    text = re.sub(r"\b(skip unopened card reward) \d+\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(open card reward) \d+:", r"\1:", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(card reward) \d+\b", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(skip reward) \d+:", r"\1:", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(pick|play|buy|remove|upgrade|transform) card \d+:", r"\1 card:", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(select cards?) [0-9,\s]+:", r"\1:", text, flags=re.IGNORECASE)
    return text


def _card_names_from_state(state: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for card in state.get("cards") or []:
        name = ""
        if isinstance(card, dict):
            name = _one_line(card.get("name") or card.get("id"))
        elif card is not None:
            name = _one_line(card)
        if name:
            names.append(name)
    return names


def _selected_card_name(action: LegalAction, state: dict[str, Any]) -> str:
    label = _one_line(action.label)
    if ":" in label:
        selected = _one_line(label.split(":", 1)[1])
        if selected:
            return selected
    args = action.command.get("args") if isinstance(action.command.get("args"), dict) else {}
    raw_index = args.get("card_index")
    for card in state.get("cards") or []:
        if isinstance(card, dict) and raw_index is not None and card.get("index") == raw_index:
            return _one_line(card.get("name") or card.get("id"))
    return ""


def _card_reward_still_available(next_state: dict[str, Any] | None) -> bool:
    if not isinstance(next_state, dict) or next_state.get("decision") != "combat_reward":
        return False
    for reward in next_state.get("rewards") or []:
        if not isinstance(reward, dict):
            continue
        if str(reward.get("kind") or "").lower() == "card_reward":
            return True
    reward_count = next_state.get("reward_count")
    return isinstance(reward_count, int) and reward_count > 0


def _label_mentions_card_reward(label: Any) -> bool:
    text = _one_line(label).lower()
    return "card_reward" in text or "card reward" in text


def _render_route_occ_context(evidence_bundle, store: RunContextStore, state: dict[str, Any]) -> str:
    context = store.context_slice(state)
    run_memory = context["run_memory"]
    deck_summary: dict[str, Any] = {"visible": evidence_bundle.deck_summary_visible}
    if isinstance(evidence_bundle.deck_summary, dict):
        deck_summary.update(evidence_bundle.deck_summary)
    payload = {
        "decision": state.get("decision"),
        "snapshot_hash": context["snapshot_hash"],
        "run_plan": context["run_plan"],
        "run_memory": {
            "pending_mandates": run_memory.get("pending_mandates") or [],
            "recent_facts": run_memory.get("recent_facts") or [],
        },
        "route_process": _route_process_for_prompt(run_memory.get("recent_route_steps") or []),
        "hp_loss_trajectory": _hp_loss_trajectory_for_prompt(run_memory.get("recent_combat_reports") or [], state),
        "deck_summary": deck_summary,
        "evidence_summary": {
            "decision_type": evidence_bundle.decision_type,
            "legal_action_count": len(evidence_bundle.candidate_evidence),
            "missing_information": list(evidence_bundle.missing_information),
            "feedback_context": [dict(item) for item in evidence_bundle.feedback_context],
            "deck_summary_visible": evidence_bundle.deck_summary_visible,
            "full_deck_details_visible": evidence_bundle.full_deck_details_visible,
        },
        "output_contract": {
            "required_keys": [
                "action_id",
                "selected_action_label",
                "reason",
                "evidence_refs",
                "run_plan_patch",
                "run_memory_facts",
            ],
            "allowed_run_plan_patch_keys": sorted(PLAN_PATCH_KEYS),
            "empty_patch": {"run_plan_patch": {}, "run_memory_facts": []},
        },
    }
    return "Route OCC context JSON:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _route_process_for_prompt(route_steps: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [_route_step_for_prompt(step) for step in route_steps if isinstance(step, dict)]
    visited_room_types = _unique_nonempty(
        (step.get("selected") or {}).get("type") for step in steps if isinstance(step.get("selected"), dict)
    )
    return {
        "steps": steps,
        "summary": {
            "route_steps": len(steps),
            "visited_room_types": visited_room_types,
            "last_selected": steps[-1].get("selected") if steps else None,
        },
    }


def _route_step_for_prompt(step: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "act",
        "floor",
        "selected",
        "hp_at_selection",
        "max_hp",
        "gold_at_selection",
        "deck_size_at_selection",
        "selected_matches_active_plan",
        "route_plan_event_type",
        "next_decision",
    )
    return {key: step.get(key) for key in keys if key in step}


def _hp_loss_trajectory_for_prompt(combat_reports: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    segments = []
    for report in combat_reports:
        if not isinstance(report, dict):
            continue
        before = _number_or_none(report.get("hp_before_final_transition"))
        after = _number_or_none(report.get("hp_after_combat"))
        delta = _number_or_none(report.get("hp_delta_final_transition"))
        if delta is None and before is not None and after is not None:
            delta = after - before
        hp_loss = max(0, int(-delta)) if delta is not None else None
        segment = {
            "act": report.get("act"),
            "floor": report.get("floor"),
            "room_type": report.get("room_type"),
            "hp_before": before,
            "hp_after": after,
            "hp_delta": delta,
            "hp_loss": hp_loss,
            "final_round": report.get("final_round"),
        }
        segments.append(segment)

    losses = [int(segment["hp_loss"]) for segment in segments if isinstance(segment.get("hp_loss"), int)]
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    current_hp = _number_or_none(player.get("hp"))
    max_hp = _number_or_none(player.get("max_hp"))
    hp_ratio = None
    if current_hp is not None and max_hp:
        hp_ratio = round(float(current_hp) / float(max_hp), 4)
    return {
        "segments": segments,
        "summary": {
            "recent_combats": len(segments),
            "total_hp_loss": sum(losses),
            "average_hp_loss": round(sum(losses) / len(losses), 2) if losses else 0,
            "largest_hp_loss": max(losses) if losses else 0,
            "current_hp": current_hp,
            "max_hp": max_hp,
            "hp_ratio": hp_ratio,
        },
    }


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


def _route_step_report(
    state: dict[str, Any],
    action: LegalAction,
    meta: dict[str, Any],
    next_state: dict[str, Any],
) -> dict[str, Any] | None:
    selected = _selected_map_node(state, action)
    if selected is None:
        return None
    context = state.get("context") or {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    next_context = next_state.get("context") if isinstance(next_state.get("context"), dict) else {}
    parsed = meta.get("parsed") if isinstance(meta, dict) and isinstance(meta.get("parsed"), dict) else {}
    route_plan_event = parsed.get("route_plan_event") if isinstance(parsed.get("route_plan_event"), dict) else {}
    active_route_plan = state.get("active_route_plan") if isinstance(state.get("active_route_plan"), dict) else {}
    expected_next = active_route_plan.get("expected_next") if isinstance(active_route_plan.get("expected_next"), dict) else None
    return {
        "kind": "route_step",
        "act": state.get("act") or context.get("act"),
        "floor": state.get("floor") or context.get("floor"),
        "selected": selected,
        "action_id": action.action_id,
        "action_label": action.label,
        "hp_at_selection": player.get("hp"),
        "max_hp": player.get("max_hp"),
        "gold_at_selection": player.get("gold"),
        "deck_size_at_selection": player.get("deck_size"),
        "selected_matches_active_plan": _same_route_node(selected, expected_next) if expected_next else None,
        "route_plan_event_type": route_plan_event.get("type"),
        "next_decision": next_state.get("decision"),
        "next_floor": next_state.get("floor") or next_context.get("floor"),
    }


def _selected_map_node(state: dict[str, Any], action: LegalAction) -> dict[str, Any] | None:
    args = action.command.get("args") if isinstance(action.command.get("args"), dict) else {}
    col = args.get("col")
    row = args.get("row")
    if col is None or row is None:
        return None
    selected: dict[str, Any] = {"col": col, "row": row}
    for choice in state.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        if choice.get("col") == col and choice.get("row") == row:
            selected["type"] = choice.get("type")
            break
    return selected


def _same_route_node(left: dict[str, Any], right: dict[str, Any] | None) -> bool:
    if not isinstance(right, dict):
        return False
    for key in ("col", "row", "type"):
        if key in left and key in right and left.get(key) != right.get(key):
            return False
    return left.get("col") == right.get("col") and left.get("row") == right.get("row")


def _unique_nonempty(values) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value in (None, "") or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


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
