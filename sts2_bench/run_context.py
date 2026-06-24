"""Structured per-run memory for the coordinator scaffold."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .actions import LegalAction


PLAN_PATCH_KEYS = {
    "route_plan",
    "build_plan",
    "resource_policy",
    "open_threads",
    "delayed_effects",
    "pending_mandates",
}


@dataclass
class RunPlan:
    route_plan: str = ""
    build_plan: str = ""
    resource_policy: str = ""
    open_threads: list[str] = field(default_factory=list)
    delayed_effects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_plan": self.route_plan,
            "build_plan": self.build_plan,
            "resource_policy": self.resource_policy,
            "open_threads": list(self.open_threads),
            "delayed_effects": list(self.delayed_effects),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunPlan":
        return cls(
            route_plan=str(data.get("route_plan") or ""),
            build_plan=str(data.get("build_plan") or ""),
            resource_policy=str(data.get("resource_policy") or ""),
            open_threads=[str(item) for item in data.get("open_threads") or []],
            delayed_effects=[str(item) for item in data.get("delayed_effects") or []],
        )


@dataclass
class RunMemory:
    facts: list[dict[str, Any]] = field(default_factory=list)
    combat_reports: list[dict[str, Any]] = field(default_factory=list)
    pending_mandates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": [dict(item) for item in self.facts],
            "combat_reports": [dict(item) for item in self.combat_reports],
            "pending_mandates": [dict(item) for item in self.pending_mandates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMemory":
        return cls(
            facts=[dict(item) for item in data.get("facts") or [] if isinstance(item, dict)],
            combat_reports=[dict(item) for item in data.get("combat_reports") or [] if isinstance(item, dict)],
            pending_mandates=[dict(item) for item in data.get("pending_mandates") or [] if isinstance(item, dict)],
        )


@dataclass
class RunContextStore:
    """Per-run RM/RP store owned by the deterministic coordinator."""

    run_plan: RunPlan = field(default_factory=RunPlan)
    run_memory: RunMemory = field(default_factory=RunMemory)

    def reset(self) -> None:
        self.run_plan = RunPlan()
        self.run_memory = RunMemory()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_plan": self.run_plan.to_dict(),
            "run_memory": self.run_memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunContextStore":
        return cls(
            run_plan=RunPlan.from_dict(data.get("run_plan") or {}),
            run_memory=RunMemory.from_dict(data.get("run_memory") or {}),
        )

    def snapshot_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def context_slice(self, state: dict[str, Any], *, max_facts: int = 8) -> dict[str, Any]:
        return {
            "decision": state.get("decision"),
            "run_plan": self.run_plan.to_dict(),
            "run_memory": {
                "recent_facts": self.run_memory.facts[-max_facts:],
                "pending_mandates": list(self.run_memory.pending_mandates),
                "recent_combat_reports": self.run_memory.combat_reports[-max_facts:],
            },
            "snapshot_hash": self.snapshot_hash(),
        }

    def apply_agent_patch(
        self,
        parsed: dict[str, Any],
        *,
        owner: str,
        state: dict[str, Any],
        action: LegalAction,
        run_plan_route_plan_enabled: bool = True,
    ) -> dict[str, Any]:
        """Validate and apply an agent-supplied RM/RP patch.

        Unknown keys raise so unit tests and debug runs catch schema drift.  The
        coordinator catches this at runtime and records the rejected patch
        instead of mutating store state.
        """

        if not isinstance(parsed, dict):
            return self._summary(False, [], 0, 0, 0)

        plan_patch = parsed.get("run_plan_patch")
        facts = parsed.get("run_memory_facts")
        reports = parsed.get("combat_reports")
        has_patch = plan_patch is not None or facts is not None or reports is not None
        if not has_patch:
            return self._summary(False, [], 0, 0, 0)

        plan_updates: list[str] = []
        facts_added = 0
        mandates_added = 0
        combat_reports_added = 0

        ignored_plan_updates: list[str] = []
        if plan_patch is not None:
            if not isinstance(plan_patch, dict):
                raise ValueError("run_plan_patch must be an object")
            if not run_plan_route_plan_enabled and "route_plan" in plan_patch:
                plan_patch = {key: value for key, value in plan_patch.items() if key != "route_plan"}
                ignored_plan_updates.append("route_plan")
            unsupported = sorted(set(plan_patch) - PLAN_PATCH_KEYS)
            if unsupported:
                raise ValueError(f"Unsupported run_plan_patch keys: {', '.join(unsupported)}")
            plan_updates, mandates_added = self._apply_plan_patch(plan_patch, owner, state, action)

        for fact in _normalize_entries(facts, default_kind="decision_summary"):
            self.run_memory.facts.append(self._decorate_entry(fact, owner, state, action))
            facts_added += 1

        for report in _normalize_entries(reports, default_kind="combat_report"):
            self.run_memory.combat_reports.append(self._decorate_entry(report, owner, state, action))
            combat_reports_added += 1

        summary = self._summary(True, plan_updates, facts_added, mandates_added, combat_reports_added)
        if ignored_plan_updates:
            summary["ignored_run_plan_updates"] = ignored_plan_updates
        return summary

    def _apply_plan_patch(
        self,
        patch: dict[str, Any],
        owner: str,
        state: dict[str, Any],
        action: LegalAction,
    ) -> tuple[list[str], int]:
        updates: list[str] = []
        mandates_added = 0
        for key in ("route_plan", "build_plan", "resource_policy"):
            if key in patch:
                setattr(self.run_plan, key, _one_line(patch.get(key)))
                updates.append(key)

        for key in ("open_threads", "delayed_effects"):
            if key in patch:
                values = _string_list(patch.get(key), key)
                setattr(self.run_plan, key, values)
                updates.append(key)

        if "pending_mandates" in patch:
            mandates = _normalize_entries(patch.get("pending_mandates"), default_kind="mandate")
            for mandate in mandates:
                self.run_memory.pending_mandates.append(self._decorate_entry(mandate, owner, state, action))
                mandates_added += 1
            updates.append("pending_mandates")

        return updates, mandates_added

    def _decorate_entry(
        self,
        entry: dict[str, Any],
        owner: str,
        state: dict[str, Any],
        action: LegalAction,
    ) -> dict[str, Any]:
        context = state.get("context") or {}
        decorated = dict(entry)
        decorated.setdefault("owner", owner)
        decorated.setdefault("decision", state.get("decision"))
        decorated.setdefault("act", state.get("act") or context.get("act"))
        decorated.setdefault("floor", state.get("floor") or context.get("floor"))
        decorated.setdefault("action_kind", action.kind)
        decorated.setdefault("action_label", action.label)
        return decorated

    def _summary(
        self,
        applied: bool,
        plan_updates: list[str],
        facts_added: int,
        mandates_added: int,
        combat_reports_added: int,
    ) -> dict[str, Any]:
        return {
            "applied": applied,
            "run_plan_updates": plan_updates,
            "facts_added": facts_added,
            "mandates_added": mandates_added,
            "combat_reports_added": combat_reports_added,
            "snapshot_hash": self.snapshot_hash(),
        }


def render_out_of_combat_context(
    store: RunContextStore,
    state: dict[str, Any],
    *,
    run_plan_route_plan_enabled: bool = True,
) -> str:
    context = store.context_slice(state)
    plan = context["run_plan"]
    memory = context["run_memory"]
    plan_text_fields = ["build_plan", "resource_policy"]
    if run_plan_route_plan_enabled:
        plan_text_fields.insert(0, "route_plan")
    allowed_patch_keys = [*plan_text_fields, "open_threads", "delayed_effects", "pending_mandates"]
    map_event_update_keys = ["resource_policy", "open_threads", "delayed_effects", "pending_mandates"]
    if run_plan_route_plan_enabled:
        map_event_update_keys.insert(0, "route_plan")
    lines = [
        "Out-of-combat controller context:",
        f"snapshot_hash: {context['snapshot_hash']}",
        "RunPlan:",
    ]
    for key in plan_text_fields:
        value = plan.get(key) or "(empty)"
        lines.append(f"- {key}: {value}")
    for key in ("open_threads", "delayed_effects"):
        values = plan.get(key) or []
        lines.append(f"- {key}:")
        if values:
            lines.extend(f"  - {item}" for item in values)
        else:
            lines.append("  - (empty)")

    lines.append("RunMemory:")
    for key in ("pending_mandates", "recent_facts", "recent_combat_reports"):
        values = memory.get(key) or []
        lines.append(f"- {key}:")
        if values:
            for item in values:
                lines.append("  - " + json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        else:
            lines.append("  - (empty)")

    lines.extend(
        [
            "",
            "OCC output contract:",
            "Return one legal action_id, selected_action_label, reason, evidence_refs, run_plan_patch, run_memory_facts, and no_context_change_reason.",
            'Use "run_plan_patch": {} and "run_memory_facts": [] when nothing should change.',
            "When run_plan_patch is empty for a real OCC non-view action, include no_context_change_reason instead of inventing a plan update.",
            f"For map/event choices, update {', '.join(map_event_update_keys[:-1])}, or {map_event_update_keys[-1]} as relevant.",
            "For shops, update resource_policy as relevant. Card reward decisions use a separate history-free reward context and do not update build_plan.",
            f"Allowed run_plan_patch keys: {', '.join(allowed_patch_keys)}.",
            "Do not mutate RM/RP directly; the coordinator validates and commits patches.",
        ]
    )
    return "\n".join(lines)


def _normalize_entries(value: Any, *, default_kind: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    entries: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            entry = dict(item)
            entry.setdefault("kind", default_kind)
            entries.append(entry)
        elif item is not None:
            entries.append({"kind": default_kind, "text": _one_line(item)})
    return entries


def _string_list(value: Any, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_one_line(value)]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a string or list of strings")
    return [_one_line(item) for item in value if _one_line(item)]


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
