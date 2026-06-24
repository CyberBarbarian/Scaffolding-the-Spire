import json

import pytest

from sts2_bench.actions import LegalAction, build_legal_actions
from sts2_bench.evidence import (
    CandidateEvidence,
    EvidenceBundle,
    build_evidence_bundle,
    render_evidence_context,
    validate_action_reason,
    validate_occ_context_contract,
    validate_no_policy_fields,
)
from sts2_bench.runner import JsonlLogger, run_one


def _card_reward_state(*, view_deck=False):
    player = {
        "hp": 70,
        "max_hp": 80,
        "gold": 99,
        "deck_size": 5,
        "deck": [
            {"id": "CARD.STRIKE", "name": "Strike", "type": "Attack", "cost": 1, "description": "Deal 6 damage."},
            {"id": "CARD.STRIKE", "name": "Strike", "type": "Attack", "cost": 1, "description": "Deal 6 damage."},
            {
                "id": "CARD.STRIKE",
                "name": "Strike",
                "type": "Attack",
                "cost": 1,
                "upgrade_level": 1,
                "description": "Deal 9 damage.",
            },
            {"id": "CARD.DEFEND", "name": "Defend", "type": "Skill", "cost": 1, "description": "Gain 5 Block."},
            {"id": "CARD.BASH", "name": "Bash", "type": "Attack", "cost": 2, "description": "Deal 8 damage. Apply 2 Vulnerable."},
        ],
    }
    return {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 3, "room_type": "Monster"},
        "player": player,
        "cards": [
            {"index": 0, "name": "Twin Strike", "type": "Attack", "cost": 1},
            {"index": 1, "name": "Shrug It Off", "type": "Skill", "cost": 1},
        ],
        "can_skip": True,
        "view_deck": view_deck,
    }


def test_reward_evidence_includes_skip_card_reward_as_peer_candidate():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)

    bundle = build_evidence_bundle(state, legal_actions)

    labels = [candidate["label"] for candidate in bundle.to_dict()["candidate_evidence"]]
    kinds = [candidate["kind"] for candidate in bundle.to_dict()["candidate_evidence"]]
    assert "pick card 0: Twin Strike" in labels
    assert "pick card 1: Shrug It Off" in labels
    assert "skip card reward" in labels
    assert "card_reward_skip" in kinds
    skip = next(candidate for candidate in bundle.to_dict()["candidate_evidence"] if candidate["kind"] == "card_reward_skip")
    assert "keeps deck size unchanged" in " ".join(skip["evidence_for"])
    assert bundle.skip_action_visible is True


def test_combat_reward_evidence_describes_unopened_card_reward_actions():
    state = {
        "decision": "combat_reward",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {"hp": 64, "max_hp": 80, "gold": 118, "deck_size": 8},
        "rewards": [{"index": 0, "kind": "card_reward", "can_claim": True, "can_skip": True}],
    }
    legal_actions = build_legal_actions(state)

    bundle = build_evidence_bundle(state, legal_actions)
    candidates = {candidate["label"]: candidate for candidate in bundle.to_dict()["candidate_evidence"]}

    open_candidate = candidates["open card reward 0: reveal offered cards"]
    skip_candidate = candidates["skip unopened card reward 0"]
    assert "reveals offered card candidates" in " ".join(open_candidate["evidence_for"])
    assert "does not add a card before the next card_reward decision" in " ".join(open_candidate["evidence_for"])
    assert "leaves the card_reward pack unopened" in " ".join(skip_candidate["evidence_against"])
    assert "keeps deck size unchanged" not in " ".join(skip_candidate["evidence_for"])


def test_card_reward_posterior_audit_describes_post_action_deck_without_policy_fields():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)

    bundle = build_evidence_bundle(state, legal_actions)
    data = bundle.to_dict()

    audit = data["posterior_audit"]
    assert audit["mode"] == "posterior_candidate_audit_no_policy"
    audits_by_label = {item["label"]: item for item in audit["candidate_audits"]}
    pick = audits_by_label["pick card 1: Shrug It Off"]
    skip = audits_by_label["skip card reward"]
    assert pick["current_deck_size"] == 5
    assert pick["post_action_deck_size"] == 6
    assert pick["deck_change"] == "add_visible_card"
    assert pick["added_card"]["name"] == "Shrug It Off"
    assert pick["existing_copies_before"] == 0
    assert skip["post_action_deck_size"] == 5
    assert skip["deck_change"] == "unchanged"
    assert "score" not in json.dumps(audit, ensure_ascii=False).lower()
    validate_no_policy_fields(data)


def test_deck_context_exposes_grouped_deck_summary_without_view_deck():
    hidden = build_evidence_bundle(_card_reward_state(view_deck=False), build_legal_actions(_card_reward_state()))
    visible = build_evidence_bundle(_card_reward_state(view_deck=True), build_legal_actions(_card_reward_state(view_deck=True)))

    hidden_data = hidden.to_dict()
    visible_data = visible.to_dict()
    strike = next(
        group
        for group in hidden_data["deck_summary"]["groups"]
        if group["name"] == "Strike" and group["upgraded"] == 0
    )
    upgraded_strike = next(
        group
        for group in hidden_data["deck_summary"]["groups"]
        if group["name"] == "Strike" and group["upgraded"] == 1
    )

    assert hidden_data["deck_summary_visible"] is True
    assert hidden_data["full_deck_details_visible"] is False
    assert hidden_data["deck_summary"]["representation"] == "complete_grouped_inventory"
    assert hidden_data["deck_summary"]["deck_size"] == 5
    assert strike["count"] == 2
    assert upgraded_strike["count"] == 1
    assert "Defend" in json.dumps(hidden_data["deck_summary"], ensure_ascii=False)
    assert "Full deck composition is not visible unless view deck has been chosen." not in hidden_data["missing_information"]
    assert hidden.deck_details_visible is False
    assert visible_data["deck_summary_visible"] is True
    assert visible_data["full_deck_details_visible"] is True
    assert visible.deck_details_visible is True


def test_evidence_schema_rejects_policy_fields_recursively():
    with pytest.raises(ValueError, match="Policy field is not allowed"):
        validate_no_policy_fields(
            {
                "candidate_evidence": [
                    {
                        "action_id": 7,
                        "label": "skip card reward",
                        "score": 0.95,
                    }
                ]
            }
        )


def test_render_evidence_context_states_advisory_contract_and_no_scores():
    bundle = EvidenceBundle(
        decision_type="card_reward",
        state_facts=["deck_details_visible=false"],
        candidate_evidence=[
            CandidateEvidence(
                action_id=5,
                label="skip card reward",
                kind="card_reward_skip",
                evidence_for=["keeps deck size unchanged"],
                evidence_against=["does not add any offered card"],
            )
        ],
    )

    rendered = render_evidence_context(bundle)

    assert "Evidence context" in rendered
    assert "advisory only" in rendered
    assert "do not score, rank, recommend, or choose actions" in rendered
    assert "skip card reward" in rendered


def test_occ_empty_run_plan_patch_without_no_context_reason_is_not_a_hard_validation_error():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    skip = next(action for action in legal_actions if action.kind == "card_reward_skip")
    card_pick = next(action for action in legal_actions if action.kind == "card_reward")
    view = next(action for action in legal_actions if action.command.get("cmd") == "bench_view")
    event = LegalAction(
        action_id=99,
        label="choose option 0: test",
        command={"cmd": "action", "action": "choose", "args": {"index": 0}},
        kind="event",
    )
    map_action = LegalAction(
        action_id=100,
        label="go to Monster at col=1 row=1",
        command={"cmd": "action", "action": "select_map_node", "args": {"col": 1, "row": 1}},
        kind="map",
    )

    assert (
        validate_occ_context_contract(
            {"run_plan_patch": {}, "no_context_change_reason": "Skipping does not change the current route plan."},
            event,
            active_agent="occ",
        )
        is None
    )
    assert validate_occ_context_contract({"run_plan_patch": {}}, event, active_agent="occ") is None
    assert validate_occ_context_contract({"run_plan_patch": {}}, card_pick, active_agent="occ") is None
    assert validate_occ_context_contract({"run_plan_patch": {}}, skip, active_agent="occ") is None
    assert validate_occ_context_contract({"run_plan_patch": {}}, map_action, active_agent="occ") is None
    assert validate_occ_context_contract({"run_plan_patch": {}}, view, active_agent="occ") is None
    assert validate_occ_context_contract({"run_plan_patch": {}}, skip, active_agent="ca") is None


def test_action_reason_required_for_real_actions_but_not_view_actions():
    state = _card_reward_state()
    legal_actions = build_legal_actions(state)
    skip = next(action for action in legal_actions if action.kind == "card_reward_skip")
    view = next(action for action in legal_actions if action.command.get("cmd") == "bench_view")

    assert validate_action_reason({}, skip) == "reason must be a non-empty string for the selected action"
    assert validate_action_reason({"reason": "Skip because none of the offered cards improves the deck."}, skip) is None
    assert validate_action_reason({}, view) is None


class _EvidenceRefScriptAgent:
    def __init__(self):
        self.calls = 0
        self.feedback = []

    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        self.calls += 1
        skip = next(action for action in legal_actions if action.kind == "card_reward_skip")
        evidence_ref = 999 if self.calls == 1 else skip.action_id
        return skip, {
            "agent": "evidence_ref_script",
            "parsed": {
                "action_id": skip.action_id,
                "selected_action_label": skip.label,
                "reason": "Use evidence refs to justify skipping.",
                "evidence_refs": [evidence_ref],
                "run_plan_patch": {},
                "run_memory_facts": [],
                "no_context_change_reason": "Skipping does not change the current run plan.",
            },
        }

    def receive_external_feedback(self, state, action, meta, feedback, legal_actions):
        self.feedback.append(feedback)


class _MissingReasonThenReasonAgent:
    def __init__(self):
        self.calls = 0
        self.feedback = []

    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        self.calls += 1
        skip = next(action for action in legal_actions if action.kind == "card_reward_skip")
        parsed = {
            "action_id": skip.action_id,
            "selected_action_label": skip.label,
            "evidence_refs": [skip.action_id],
            "run_plan_patch": {},
            "run_memory_facts": [],
        }
        if self.calls > 1:
            parsed["reason"] = "Skip because the current deck is better unchanged."
        return skip, {"agent": "missing_reason_script", "parsed": parsed}

    def receive_external_feedback(self, state, action, meta, feedback, legal_actions):
        self.feedback.append(feedback)


class _CardRewardProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return _card_reward_state()

    def send(self, command):
        self.sent.append(command)
        return {"decision": "game_over", "victory": False, "player": {"hp": 70, "max_hp": 80}}

    def close(self):
        pass


def test_runner_reasks_when_evidence_refs_do_not_match_candidate_actions(tmp_path):
    agent = _EvidenceRefScriptAgent()
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(
            agent=agent,
            seed="evidence-ref",
            process=_CardRewardProcess(),
            logger=logger,
            max_steps=2,
        )
    finally:
        logger.close()

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    action_rows = [row for row in rows if row.get("type") == "action"]

    assert result.error is None
    assert agent.calls == 2
    assert len(agent.feedback) == 1
    assert "evidence_refs" in agent.feedback[0]
    error_row = next(row for row in rows if row.get("type") == "evidence_validation_error")
    feedback_event = error_row["feedback_event"]
    assert feedback_event["kind"] == "evidence_validation"
    assert feedback_event["code"] == "invalid_evidence_refs"
    assert feedback_event["step"] == 0
    assert feedback_event["decision"] == "card_reward"
    assert feedback_event["action_id"] == action_rows[-1]["action"]["action_id"]
    assert feedback_event["action_label"] == "skip card reward"
    assert feedback_event["validator"] == "evidence_refs"
    assert feedback_event["evidence_bundle_hash"] == action_rows[-1]["agent_meta"]["evidence_bundle_hash"]
    assert feedback_event["max_retries"] == 2
    assert "reason" in feedback_event["details"]
    assert action_rows[-1]["agent_meta"]["feedback_events"][0] == feedback_event
    assert action_rows[-1]["agent_meta"]["parsed"]["action_id"] == action_rows[-1]["action"]["action_id"]


def test_runner_reasks_when_real_action_omits_reason(tmp_path):
    agent = _MissingReasonThenReasonAgent()
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)
    try:
        result = run_one(
            agent=agent,
            seed="missing-reason",
            process=_CardRewardProcess(),
            logger=logger,
            max_steps=2,
        )
    finally:
        logger.close()

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    error_row = next(row for row in rows if row.get("type") == "evidence_validation_error")
    feedback_event = error_row["feedback_event"]

    assert result.error is None
    assert agent.calls == 2
    assert len(agent.feedback) == 1
    assert "reason must be a non-empty string" in agent.feedback[0]
    assert feedback_event["validator"] == "action_reason"
    assert feedback_event["code"] == "missing_action_reason"
