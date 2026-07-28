"""Evidence-only context assembly for pure-LLM policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from .actions import LegalAction, card_select_combination_count
from .context import card_cost_label, description_of, name
from .run_context import RunContextStore


POLICY_FIELDS = {
    "score",
    "rank",
    "recommendation",
    "recommended_action",
    "preferred_action",
    "best_action",
    "selected_action",
    "selected_action_label",
    "chosen_action",
    "should_choose",
    "should_pick",
    "should_skip",
}


@dataclass(frozen=True)
class CandidateEvidence:
    action_id: int
    label: str
    kind: str
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    requires_information: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "label": self.label,
            "kind": self.kind,
            "evidence_for": list(self.evidence_for),
            "evidence_against": list(self.evidence_against),
            "requires_information": list(self.requires_information),
        }


@dataclass(frozen=True)
class EvidenceBundle:
    decision_type: str
    state_facts: list[str] = field(default_factory=list)
    candidate_evidence: list[CandidateEvidence] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    memory_slice: list[str] = field(default_factory=list)
    feedback_context: list[dict[str, Any]] = field(default_factory=list)
    deck_summary: dict[str, Any] | None = None
    deck_summary_visible: bool = False
    full_deck_details_visible: bool = False
    deck_details_visible: bool = False
    skip_action_visible: bool = False
    posterior_audit: dict[str, Any] | None = None
    reward_posterior_evaluation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "decision_type": self.decision_type,
            "state_facts": list(self.state_facts),
            "candidate_evidence": [candidate.to_dict() for candidate in self.candidate_evidence],
            "missing_information": list(self.missing_information),
            "memory_slice": list(self.memory_slice),
            "feedback_context": [dict(item) for item in self.feedback_context],
            "deck_summary_visible": self.deck_summary_visible,
            "full_deck_details_visible": self.full_deck_details_visible,
            "deck_details_visible": self.deck_details_visible,
            "skip_action_visible": self.skip_action_visible,
        }
        if self.deck_summary is not None:
            data["deck_summary"] = dict(self.deck_summary)
        if self.posterior_audit is not None:
            data["posterior_audit"] = dict(self.posterior_audit)
        if self.reward_posterior_evaluation is not None:
            data["reward_posterior_evaluation"] = dict(self.reward_posterior_evaluation)
        validate_no_policy_fields(data)
        return data

    def summary_dict(self) -> dict[str, Any]:
        deck_group_count = 0
        if isinstance(self.deck_summary, dict):
            deck_group_count = len(self.deck_summary.get("groups") or [])
        return {
            "decision_type": self.decision_type,
            "candidate_count": len(self.candidate_evidence),
            "state_fact_count": len(self.state_facts),
            "missing_information_count": len(self.missing_information),
            "memory_slice_count": len(self.memory_slice),
            "feedback_context_count": len(self.feedback_context),
            "deck_summary_visible": self.deck_summary_visible,
            "full_deck_details_visible": self.full_deck_details_visible,
            "deck_group_count": deck_group_count,
            "posterior_audit_visible": self.posterior_audit is not None,
            "reward_posterior_evaluation_visible": self.reward_posterior_evaluation is not None,
        }

    def bundle_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def candidate_action_ids(self) -> set[int]:
        return {candidate.action_id for candidate in self.candidate_evidence}


class EvidenceModule(Protocol):
    def build(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[str, Any]:
        """Return an evidence fragment without policy fields."""


class DeckBuildContextModule:
    """Expose the public deck inventory as grouped facts without choosing actions."""

    def build(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[str, Any]:
        del legal_actions
        state_facts: list[str] = []
        missing_information: list[str] = []
        deck_summary = _add_deck_context(state, state_facts, missing_information)
        return {
            "state_facts": state_facts,
            "missing_information": missing_information,
            "deck_summary": deck_summary,
            "deck_summary_visible": deck_summary is not None,
            "full_deck_details_visible": bool(state.get("view_deck")),
            "deck_details_visible": bool(state.get("view_deck")),
        }


class RecentTransitionEvidenceModule:
    """Expose recent observed transitions without inventing strategy."""

    def build(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[str, Any]:
        del legal_actions
        state_facts: list[str] = []
        _add_last_action_context(state, state_facts)
        return {"state_facts": state_facts}


class RewardEvidenceModule:
    def build(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[str, Any]:
        candidates = _reward_candidates(legal_actions)
        return {
            "state_facts": [f"reward_candidate_count={len(candidates)}"],
            "candidate_evidence": candidates,
            "posterior_audit": _reward_posterior_audit(state, legal_actions),
        }


class ShopEvidenceModule:
    def build(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[str, Any]:
        candidates = _shop_candidates(state, legal_actions)
        player = state.get("player") or {}
        gold = player.get("gold")
        return {
            "state_facts": [
                f"gold={gold if gold is not None else '?'}",
                f"shop_candidate_count={len(candidates)}",
            ],
            "candidate_evidence": candidates,
        }


class RouteEvidenceModule:
    def build(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[str, Any]:
        missing_information: list[str] = []
        candidates = _route_candidates(state, legal_actions, missing_information)
        return {
            "state_facts": [f"route_candidate_count={len(candidates)}"],
            "candidate_evidence": candidates,
            "missing_information": missing_information,
        }


class CardSelectEvidenceModule:
    """Describe the exact subset rule without repeating every legal combination."""

    def build(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[str, Any]:
        min_select = int(state.get("min_select", 1) or 0)
        max_select = int(state.get("max_select", min_select) or min_select)
        nonempty_min_select = max(1, min_select)
        non_selection_candidates = [
            _candidate_from_action(action)
            for action in legal_actions
            if action.kind not in {"card_select", "card_select_space"}
            and action.command.get("cmd") != "bench_view"
        ]
        cards = state.get("cards") or []
        candidate_indices = [card.get("index") for card in cards if isinstance(card, dict)]
        candidate_payload = json.dumps(cards, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        candidate_digest = hashlib.sha256(candidate_payload.encode("utf-8")).hexdigest()
        return {
            "state_facts": [
                f"card_select_candidate_count={len(cards)}",
                "card_select_candidate_indices="
                + json.dumps(candidate_indices, ensure_ascii=False, separators=(",", ":")),
                f"card_select_candidate_digest_sha256={candidate_digest}",
                f"card_select_min_select={min_select}",
                f"card_select_max_select={max_select}",
                f"card_select_nonempty_min_select={nonempty_min_select}",
                f"card_select_can_skip={str(any(action.kind == 'card_select_skip' for action in legal_actions)).lower()}",
                "card_select_legal_combination_count="
                + str(card_select_combination_count(len(cards), min_select, max_select)),
                (
                    "card_select_legal_selection_rule=every non-empty distinct subset of current candidate indices "
                    "whose size is between nonempty_min_select and max_select; empty selects the separate skip action "
                    "only when card_select_can_skip=true"
                ),
                "card_select_complete_candidate_objects_visible_in_legal_actions=true",
            ],
            "candidate_evidence": non_selection_candidates,
        }


class GenericEvidenceModule:
    def build(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[str, Any]:
        del state
        return {"candidate_evidence": _generic_candidates(legal_actions)}


def validate_no_policy_fields(value: Any, *, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in POLICY_FIELDS:
                location = f" at {path}.{key}" if path else f" at {key}"
                raise ValueError(f"Policy field is not allowed in evidence{location}: {key}")
            child_path = f"{path}.{key}" if path else str(key)
            validate_no_policy_fields(item, path=child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_no_policy_fields(item, path=f"{path}[{index}]")


def build_evidence_bundle(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    store: RunContextStore | None = None,
    feedback_events: list[dict[str, Any]] | None = None,
) -> EvidenceBundle:
    decision = str(state.get("decision") or state.get("type") or "?")
    state_facts: list[str] = []
    candidates: list[CandidateEvidence] = []
    missing_information: list[str] = []
    extra: dict[str, Any] = {}

    for module in _modules_for_state(state):
        _merge_fragment(
            module.build(state, legal_actions),
            state_facts=state_facts,
            candidates=candidates,
            missing_information=missing_information,
            extra=extra,
        )

    memory_slice = _memory_slice(store, state)
    feedback_context = [dict(item) for item in feedback_events or []]
    deck_summary = extra.get("deck_summary") if isinstance(extra.get("deck_summary"), dict) else None
    posterior_audit = extra.get("posterior_audit") if isinstance(extra.get("posterior_audit"), dict) else None
    deck_summary_visible = bool(extra.get("deck_summary_visible"))
    full_deck_details_visible = bool(extra.get("full_deck_details_visible"))
    deck_details_visible = full_deck_details_visible
    skip_action_visible = any("skip" in action.kind or "skip" in action.label.lower() for action in legal_actions)

    bundle = EvidenceBundle(
        decision_type=decision,
        state_facts=state_facts,
        candidate_evidence=candidates,
        missing_information=missing_information,
        memory_slice=memory_slice,
        feedback_context=feedback_context,
        deck_summary=deck_summary,
        deck_summary_visible=deck_summary_visible,
        full_deck_details_visible=full_deck_details_visible,
        deck_details_visible=deck_details_visible,
        skip_action_visible=skip_action_visible,
        posterior_audit=posterior_audit,
    )
    bundle.to_dict()
    return bundle


def render_evidence_context(bundle: EvidenceBundle) -> str:
    return "\n".join(
        [
            "Evidence context (advisory only):",
            "Current game state and current legal actions override this evidence.",
            "Evidence modules do not score, rank, recommend, or choose actions; the LLM policy must use the current decision's declared legal selection protocol.",
            json.dumps(bundle.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ]
    )


def render_reward_decision_context(
    bundle: EvidenceBundle,
    *,
    posterior_enabled: bool = True,
) -> str:
    if posterior_enabled:
        lines = [
            "Reward decision context: history-free current facts only.",
            "Use the current state, current legal actions, complete grouped deck inventory, posterior audit, and any recent combat facts that are present.",
            "When deck_summary_visible is true, the complete grouped deck inventory is already included; do not choose view deck merely to inspect deck composition already shown.",
            "Bayesian-style posterior protocol: treat the unchanged current deck as the baseline, then compare each pick as a post-action deck state conditioned only on current visible evidence.",
            "Taking a card requires a current-evidence marginal improvement over skip. If no offered card materially improves the current deck, choose skip.",
            "The reason must compare the selected action against skip or the unchanged deck baseline, and must identify the current visible evidence that makes the selected post-action deck better.",
            "Reward posterior deck-state analysis, when present, is a separate evidence-only comparison of candidate deck states. It is not a recommendation and must not override current legal actions.",
            "Do not use prior reward reasons, OCC history, or build plans. Do not assume future archetypes, future rests, or future support cards.",
            "Treat skip card reward as a peer legal action. Compare the current deck against each candidate post-action deck state.",
        ]
        visible_bundle = bundle
    else:
        lines = [
            "Card reward decision context: current facts only.",
            "Use the current state, current legal actions, and complete grouped deck inventory.",
            "When deck_summary_visible is true, the complete grouped deck inventory is already included; do not choose view deck merely to inspect deck composition already shown.",
            "Do not use prior reward reasons, OCC history, or build plans. Do not assume future archetypes, future rests, or future support cards.",
            "Treat skip card reward as a peer legal action when it is currently legal.",
            "Current deck inventory JSON:",
            json.dumps(
                {
                    "deck_summary": dict(bundle.deck_summary)
                    if isinstance(bundle.deck_summary, dict)
                    else None,
                    "deck_summary_visible": bundle.deck_summary_visible,
                    "skip_action_visible": bundle.skip_action_visible,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
        return "\n".join(lines)
    return "\n".join([*lines, render_evidence_context(visible_bundle)])


def validate_evidence_refs(parsed: dict[str, Any], bundle: EvidenceBundle) -> str | None:
    raw_refs = parsed.get("evidence_refs")
    if raw_refs is None:
        return None
    if not isinstance(raw_refs, list):
        return "evidence_refs must be a list of candidate action ids or objects with action_id"
    valid_ids = bundle.candidate_action_ids()
    invalid: list[Any] = []
    for ref in raw_refs:
        action_id = ref.get("action_id") if isinstance(ref, dict) else ref
        try:
            normalized = int(action_id)
        except (TypeError, ValueError):
            invalid.append(action_id)
            continue
        if normalized not in valid_ids:
            invalid.append(action_id)
    if invalid:
        return f"evidence_refs contains action ids not present in current candidate_evidence: {invalid}"
    return None


def validate_action_reason(parsed: dict[str, Any] | None, action: LegalAction) -> str | None:
    if action.command.get("cmd") == "bench_view":
        return None
    if not isinstance(parsed, dict):
        return "reason must be a non-empty string for the selected action"
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return "reason must be a non-empty string for the selected action"
    return None


def validate_occ_context_contract(
    parsed: dict[str, Any],
    action: LegalAction,
    *,
    active_agent: str | None,
) -> str | None:
    return None


def evidence_ref_feedback(error: str) -> str:
    return (
        f"Your previous response was invalid: {error}. "
        "Choose again from the current legal actions. Keep evidence_refs tied to current candidate_evidence action_ids."
    )


def _modules_for_state(state: dict[str, Any]) -> list[EvidenceModule]:
    modules: list[EvidenceModule] = [DeckBuildContextModule(), RecentTransitionEvidenceModule()]
    decision = state.get("decision")
    if decision in {"card_reward", "combat_reward", "treasure"}:
        modules.append(RewardEvidenceModule())
    elif decision in {"shop", "fake_merchant_shop"}:
        modules.append(ShopEvidenceModule())
    elif decision == "map_select":
        modules.append(RouteEvidenceModule())
    elif decision == "card_select":
        modules.append(CardSelectEvidenceModule())
    else:
        modules.append(GenericEvidenceModule())
    return modules


def _merge_fragment(
    fragment: dict[str, Any],
    *,
    state_facts: list[str],
    candidates: list[CandidateEvidence],
    missing_information: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    validate_no_policy_fields(_fragment_for_validation(fragment))
    state_facts.extend(str(item) for item in fragment.get("state_facts") or [])
    missing_information.extend(str(item) for item in fragment.get("missing_information") or [])
    for item in fragment.get("candidate_evidence") or []:
        if not isinstance(item, CandidateEvidence):
            raise TypeError("candidate_evidence entries must be CandidateEvidence")
        candidates.append(item)
    if extra is not None:
        for key in (
            "deck_summary",
            "deck_summary_visible",
            "full_deck_details_visible",
            "deck_details_visible",
            "posterior_audit",
        ):
            if key in fragment:
                extra[key] = fragment[key]


def _fragment_for_validation(fragment: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(fragment)
    if "candidate_evidence" in normalized:
        normalized["candidate_evidence"] = [
            item.to_dict() if isinstance(item, CandidateEvidence) else item
            for item in normalized.get("candidate_evidence") or []
        ]
    return normalized


def _add_deck_context(
    state: dict[str, Any],
    state_facts: list[str],
    missing_information: list[str],
) -> dict[str, Any] | None:
    player = state.get("player") or {}
    deck_size = player.get("deck_size")
    state_facts.append(f"deck_size={deck_size if deck_size is not None else '?'}")
    deck = player.get("deck")
    if not isinstance(deck, list):
        state_facts.append("deck_summary_visible=false")
        state_facts.append(f"full_deck_details_visible={str(bool(state.get('view_deck'))).lower()}")
        missing_information.append("Deck list is not present in the current state.")
        return None

    deck_summary = _grouped_deck_inventory(deck, deck_size=deck_size)
    state_facts.append("deck_summary_visible=true")
    state_facts.append(f"deck_summary_groups={len(deck_summary['groups'])}")
    state_facts.append(f"full_deck_details_visible={str(bool(state.get('view_deck'))).lower()}")
    if state.get("view_deck"):
        state_facts.append("deck_details_visible=true")
    else:
        state_facts.append("deck_details_visible=false")
    return deck_summary


def _grouped_deck_inventory(deck: list[dict[str, Any]], *, deck_size: Any = None) -> dict[str, Any]:
    groups_by_key: dict[str, dict[str, Any]] = {}
    for card in deck:
        if not isinstance(card, dict):
            continue
        group = _deck_group_item(card)
        key_payload = {key: value for key, value in group.items() if key != "count"}
        key = json.dumps(key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        existing = groups_by_key.get(key)
        if existing is None:
            groups_by_key[key] = group
        else:
            existing["count"] += 1
    groups = sorted(
        groups_by_key.values(),
        key=lambda item: (
            str(item.get("name", "")),
            int(item.get("upgraded", 0) or 0),
            str(item.get("cost", "")),
            str(item.get("id", "")),
        ),
    )
    return {
        "representation": "complete_grouped_inventory",
        "deck_size": deck_size if deck_size is not None else len(deck),
        "grouping_key": "id,name,upgraded,type,cost,description,enchantment,affliction,keywords",
        "groups": groups,
    }


def _deck_group_item(card: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": name(card.get("name")),
        "count": 1,
        "upgraded": _upgrade_level(card),
        "type": card.get("type"),
        "cost": card_cost_label(card),
    }
    for key in ("id", "rarity", "target_type", "star_cost"):
        value = card.get(key)
        if value not in (None, "", []):
            item[key] = value
    description = description_of(card)
    if description:
        item["description"] = description
    keywords = card.get("keywords") or []
    if keywords:
        item["keywords"] = sorted(str(keyword) for keyword in keywords)
    enchantment = card.get("enchantment")
    if enchantment:
        item["enchantment"] = name(enchantment)
    enchantment_description = _one_line(card.get("enchantment_description"))
    if enchantment_description:
        item["enchantment_description"] = enchantment_description
    affliction = card.get("affliction")
    if affliction:
        item["affliction"] = name(affliction)
    affliction_description = _one_line(card.get("affliction_description"))
    if affliction_description:
        item["affliction_description"] = affliction_description
    return item


def _upgrade_level(card: dict[str, Any]) -> int:
    value = card.get("upgrade_level", card.get("upgraded", 0))
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _add_last_action_context(state: dict[str, Any], state_facts: list[str]) -> None:
    result = state.get("last_action_result")
    if not isinstance(result, dict):
        return
    action_label = result.get("action_label")
    if action_label:
        state_facts.append(f"last_action={action_label}")
    for change in (result.get("changes") or [])[:4]:
        state_facts.append(f"last_action_change={change}")


def _reward_candidates(legal_actions: list[LegalAction]) -> list[CandidateEvidence]:
    result: list[CandidateEvidence] = []
    for action in legal_actions:
        if action.command.get("cmd") == "bench_view":
            continue
        label = action.label.lower()
        if "skip" in action.kind or "skip" in label:
            if action.kind == "combat_reward_skip" and _label_mentions_card_reward(action.label):
                result.append(
                    CandidateEvidence(
                        action_id=action.action_id,
                        label=action.label,
                        kind=action.kind,
                        evidence_for=["skip is a current legal shortcut"],
                        evidence_against=[
                            "leaves the card_reward pack unopened",
                            "offered card candidates are not visible until the card_reward is opened/revealed",
                        ],
                    )
                )
                continue
            result.append(
                CandidateEvidence(
                    action_id=action.action_id,
                    label=action.label,
                    kind=action.kind,
                    evidence_for=["skip is a current legal action", "keeps deck size unchanged"],
                    evidence_against=["does not add any offered card or reward"],
                )
            )
        elif action.kind == "combat_reward" and _label_mentions_card_reward(action.label):
            result.append(
                CandidateEvidence(
                    action_id=action.action_id,
                    label=action.label,
                    kind=action.kind,
                    evidence_for=[
                        "reveals offered card candidates",
                        "does not add a card before the next card_reward decision",
                    ],
                    evidence_against=["requires a follow-up pick or skip decision after candidates are visible"],
                )
            )
        elif action.kind.startswith("card_reward") or action.kind in {"combat_reward", "treasure"}:
            result.append(
                CandidateEvidence(
                    action_id=action.action_id,
                    label=action.label,
                    kind=action.kind,
                    evidence_for=["claims the visible offered reward"],
                    evidence_against=["changes deck, relics, potions, gold, or reward state according to the action"],
                )
            )
        else:
            result.append(_candidate_from_action(action))
    return result


def _label_mentions_card_reward(label: str) -> bool:
    normalized = label.lower().replace("_", " ")
    return "card reward" in normalized


def _reward_posterior_audit(state: dict[str, Any], legal_actions: list[LegalAction]) -> dict[str, Any] | None:
    if state.get("decision") != "card_reward":
        return None
    player = state.get("player") or {}
    deck = player.get("deck") if isinstance(player.get("deck"), list) else []
    deck_size = player.get("deck_size")
    current_size = int(deck_size) if isinstance(deck_size, (int, float)) else len(deck)
    offered_cards = {
        card.get("index"): card
        for card in state.get("cards") or []
        if isinstance(card, dict)
    }
    candidate_audits: list[dict[str, Any]] = []
    for action in legal_actions:
        if action.command.get("cmd") == "bench_view":
            continue
        audit: dict[str, Any] = {
            "action_id": action.action_id,
            "label": action.label,
            "kind": action.kind,
            "current_deck_size": current_size,
            "audit_questions": [
                "What changes between the current deck and the post-action deck?",
                "Does the action add immediate value from currently visible facts?",
                "Does the action rely on support that is not currently visible?",
                "Is deck dilution or redundancy relevant compared with skip?",
            ],
        }
        if action.kind == "card_reward_skip":
            audit.update(
                {
                    "post_action_deck_size": current_size,
                    "deck_change": "unchanged",
                    "added_card": None,
                    "existing_copies_before": None,
                }
            )
        elif action.kind == "card_reward":
            args = action.command.get("args") or {}
            card = offered_cards.get(args.get("card_index"))
            card_name = name(card.get("name")) if isinstance(card, dict) else _card_name_from_action_label(action.label)
            audit.update(
                {
                    "post_action_deck_size": current_size + 1,
                    "deck_change": "add_visible_card",
                    "added_card": _deck_group_item(card) if isinstance(card, dict) else {"name": card_name},
                    "existing_copies_before": _count_card_name(deck, card_name),
                }
            )
        else:
            audit.update(
                {
                    "post_action_deck_size": current_size,
                    "deck_change": "non_card_reward_action",
                }
            )
        candidate_audits.append(audit)
    return {
        "mode": "posterior_candidate_audit_no_policy",
        "candidate_audits": candidate_audits,
        "contract": [
            "This audit describes post-action deck facts and reasoning checks only.",
            "It does not assign preference, ordering, or action choice.",
            "Future support should not be assumed unless it is visible in the current state.",
        ],
    }


def _shop_candidates(state: dict[str, Any], legal_actions: list[LegalAction]) -> list[CandidateEvidence]:
    player = state.get("player") or {}
    gold = player.get("gold")
    result = []
    for action in legal_actions:
        if action.command.get("cmd") == "bench_view":
            continue
        if action.kind == "shop_leave" or action.kind == "fake_merchant_leave":
            result.append(
                CandidateEvidence(
                    action_id=action.action_id,
                    label=action.label,
                    kind=action.kind,
                    evidence_for=["leaves the shop without spending more gold"],
                    evidence_against=["forgoes remaining stocked shop options"],
                )
            )
        elif action.kind.startswith("shop_buy") or action.kind.startswith("fake_merchant_buy"):
            result.append(
                CandidateEvidence(
                    action_id=action.action_id,
                    label=action.label,
                    kind=action.kind,
                    evidence_for=[f"purchase is currently legal with gold={gold}"],
                    evidence_against=["spends current gold"],
                )
            )
        elif action.kind == "shop_remove_card":
            result.append(
                CandidateEvidence(
                    action_id=action.action_id,
                    label=action.label,
                    kind=action.kind,
                    evidence_for=[f"card removal option is currently legal with gold={gold}"],
                    evidence_against=["spends current gold and opens a card selection flow"],
                )
            )
        elif action.kind == "fake_merchant_throw_foul_potion":
            result.append(
                CandidateEvidence(
                    action_id=action.action_id,
                    label=action.label,
                    kind=action.kind,
                    evidence_for=["event-specific fake merchant action is currently legal"],
                    evidence_against=["uses the event action instead of buying or leaving"],
                )
            )
        else:
            result.append(_candidate_from_action(action))
    return result


def _route_candidates(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    missing_information: list[str],
) -> list[CandidateEvidence]:
    if not isinstance(state.get("full_map"), dict):
        missing_information.append("Full map snapshot is not visible unless view map has been chosen or include_full_map is enabled.")
    result: list[CandidateEvidence] = []
    for action in legal_actions:
        if action.command.get("cmd") == "bench_view":
            continue
        args = (action.command.get("args") or {}) if isinstance(action.command, dict) else {}
        coords = ""
        if "col" in args and "row" in args:
            coords = f" at col={args.get('col')} row={args.get('row')}"
        result.append(
            CandidateEvidence(
                action_id=action.action_id,
                label=action.label,
                kind=action.kind,
                evidence_for=[f"legal current map choice{coords}"],
                evidence_against=["commits to the selected next node"],
            )
        )
    return result


def _generic_candidates(legal_actions: list[LegalAction]) -> list[CandidateEvidence]:
    return [_candidate_from_action(action) for action in legal_actions if action.command.get("cmd") != "bench_view"]


def _candidate_from_action(action: LegalAction) -> CandidateEvidence:
    return CandidateEvidence(
        action_id=action.action_id,
        label=action.label,
        kind=action.kind,
        evidence_for=["current legal action"],
        evidence_against=[],
    )


def _one_line(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def _memory_slice(store: RunContextStore | None, state: dict[str, Any]) -> list[str]:
    if store is None:
        return []
    context = store.context_slice(state)
    memory = context.get("run_memory") or {}
    values: list[str] = []
    keys = ("recent_combat_reports",) if state.get("decision") == "card_reward" else (
        "pending_mandates",
        "recent_facts",
        "recent_combat_reports",
    )
    for key in keys:
        for item in memory.get(key) or []:
            if isinstance(item, dict):
                text = item.get("text") or item.get("reason") or json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                values.append(f"{key}: {text}")
            else:
                values.append(f"{key}: {name(item)}")
    return values


def _count_card_name(deck: list[dict[str, Any]], card_name: str) -> int:
    return sum(1 for card in deck if isinstance(card, dict) and name(card.get("name")) == card_name)


def _card_name_from_action_label(label: str) -> str:
    if ":" in label:
        return _one_line(label.rsplit(":", 1)[1])
    return _one_line(label)
