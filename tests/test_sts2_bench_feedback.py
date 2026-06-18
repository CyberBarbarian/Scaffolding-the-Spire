from sts2_bench.actions import LegalAction
from sts2_bench.feedback import FeedbackBus


class _FeedbackAgent:
    def __init__(self):
        self.feedback = []

    def receive_external_feedback(self, state, action, meta, feedback, legal_actions):
        self.feedback.append(
            {
                "state": state,
                "action": action,
                "meta": meta,
                "feedback": feedback,
                "legal_actions": legal_actions,
            }
        )


def test_feedback_bus_builds_enriched_event_and_dispatches_retry():
    action = LegalAction(
        action_id=3,
        label="skip card reward",
        command={"cmd": "action", "action": "skip_card_reward"},
        kind="card_reward_skip",
    )
    agent = _FeedbackAgent()
    state = {"decision": "card_reward"}
    meta = {"agent": "script"}
    result_extra = {}
    bus = FeedbackBus(
        step=7,
        decision="card_reward",
        evidence_bundle_hash="abc123",
        retry_limits={"evidence_validation": 2},
    )

    event = bus.make_event(
        kind="evidence_validation",
        validator="evidence_refs",
        code="invalid_evidence_refs",
        message="bad evidence refs",
        action=action,
        details={"reason": "candidate 999 is not present"},
    )

    assert event.to_dict() == {
        "kind": "evidence_validation",
        "message": "bad evidence refs",
        "retry": 0,
        "retryable": True,
        "code": "invalid_evidence_refs",
        "step": 7,
        "decision": "card_reward",
        "action_id": 3,
        "action_label": "skip card reward",
        "validator": "evidence_refs",
        "evidence_bundle_hash": "abc123",
        "max_retries": 2,
        "details": {"reason": "candidate 999 is not present"},
    }

    retried = bus.dispatch_retry(
        agent=agent,
        state=state,
        action=action,
        meta=meta,
        event=event,
        legal_actions=[action],
        result_extra=result_extra,
        result_extra_key="evidence_validation_retry_count",
    )

    assert retried is True
    assert bus.retry_count("evidence_validation") == 1
    assert bus.meta_events() == [event.to_dict()]
    assert result_extra["evidence_validation_retry_count"] == 1
    assert agent.feedback[0]["feedback"] == "bad evidence refs"


def test_feedback_bus_reports_exhausted_retry_without_dispatching():
    action = LegalAction(
        action_id=0,
        label="end turn",
        command={"cmd": "action", "action": "end_turn"},
        kind="combat_end_turn",
    )
    agent = _FeedbackAgent()
    bus = FeedbackBus(step=2, decision="combat_play", retry_limits={"custom_sequence_validation": 0})

    event = bus.make_event(
        kind="custom_sequence_validation",
        validator="custom_sequence",
        message="sequence failed",
        action=action,
    )
    retried = bus.dispatch_retry(
        agent=agent,
        state={"decision": "combat_play"},
        action=action,
        meta={},
        event=event,
        legal_actions=[action],
    )

    assert event.retry == 0
    assert event.retryable is False
    assert retried is False
    assert bus.meta_events() == []
    assert agent.feedback == []
