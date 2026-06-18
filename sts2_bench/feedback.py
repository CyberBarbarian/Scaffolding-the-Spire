"""Unified feedback events for runner validation retries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .actions import LegalAction
from .agents import Agent


@dataclass(frozen=True)
class FeedbackEvent:
    kind: str
    message: str
    retry: int
    retryable: bool
    code: str | None = None
    step: int | None = None
    decision: str | None = None
    action_id: int | None = None
    action_label: str | None = None
    validator: str | None = None
    evidence_bundle_hash: str | None = None
    max_retries: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "message": self.message,
            "retry": self.retry,
            "retryable": self.retryable,
        }
        for key in (
            "code",
            "step",
            "decision",
            "action_id",
            "action_label",
            "validator",
            "evidence_bundle_hash",
            "max_retries",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.details:
            data["details"] = dict(self.details)
        return data


class FeedbackBus:
    """Per-decision feedback lifecycle for runner validation retries."""

    def __init__(
        self,
        *,
        step: int,
        decision: Any,
        evidence_bundle_hash: str | None = None,
        retry_limits: dict[str, int] | None = None,
    ) -> None:
        self.step = step
        self.decision = str(decision) if decision is not None else None
        self.evidence_bundle_hash = evidence_bundle_hash
        self.retry_limits = dict(retry_limits or {})
        self.events: list[FeedbackEvent] = []
        self._retry_counts: dict[str, int] = {}

    def retry_count(self, kind: str) -> int:
        return int(self._retry_counts.get(kind) or 0)

    def meta_events(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def make_event(
        self,
        *,
        kind: str,
        message: str,
        action: LegalAction,
        validator: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> FeedbackEvent:
        retry = self.retry_count(kind)
        max_retries = int(self.retry_limits.get(kind) or 0)
        return FeedbackEvent(
            kind=kind,
            code=code,
            message=message,
            retry=retry,
            retryable=retry < max_retries,
            step=self.step,
            decision=self.decision,
            action_id=action.action_id,
            action_label=action.label,
            validator=validator,
            evidence_bundle_hash=self.evidence_bundle_hash,
            max_retries=max_retries,
            details=dict(details or {}),
        )

    def trace_entry(
        self,
        feedback_event: FeedbackEvent,
        *,
        entry_type: str,
        reason: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "type": entry_type,
            "step": self.step,
            "retry": feedback_event.retry,
            "retryable": feedback_event.retryable,
            "feedback_event": feedback_event.to_dict(),
        }
        if feedback_event.code:
            entry["code"] = feedback_event.code
        if reason is not None:
            entry["reason"] = reason
        for key, value in extra.items():
            if value is not None:
                entry[key] = value
        return entry

    def dispatch_retry(
        self,
        *,
        agent: Agent,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        event: FeedbackEvent,
        legal_actions: list[LegalAction],
        result_extra: dict[str, Any] | None = None,
        result_extra_key: str | None = None,
    ) -> bool:
        if not event.retryable:
            return False
        self._retry_counts[event.kind] = event.retry + 1
        self.events.append(event)
        if result_extra is not None and result_extra_key:
            result_extra[result_extra_key] = int(result_extra.get(result_extra_key) or 0) + 1
        send_feedback_event(agent, state, action, meta, event, legal_actions)
        return True


def send_feedback_event(
    agent: Agent,
    state: dict[str, Any],
    action: LegalAction,
    meta: dict[str, Any],
    event: FeedbackEvent,
    legal_actions: list[LegalAction],
) -> None:
    receive = getattr(agent, "receive_external_feedback", None)
    if callable(receive):
        receive(state, action, meta, event.message, legal_actions)
