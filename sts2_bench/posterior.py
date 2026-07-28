"""LLM posterior deck-state analysis for card reward decisions."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .actions import LegalAction
from .agents import ModelActionError, _is_transport_error, _transport_retry_delay, parse_json_object
from .context import render_state_text
from .evidence import EvidenceBundle, validate_no_policy_fields
from .game_rules import game_rules_meta, render_game_rules


POSTERIOR_POLICY_FIELDS = {
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

POSTERIOR_FINAL_MODE = "reward_posterior_deck_analysis"
POSTERIOR_SLOT_FILL_MODE = "reward_posterior_slot_fill"
POSTERIOR_FILLABLE_FIELDS = (
    "deck_delta",
    "immediate_current_deck_effect",
    "redundancy_or_dilution_risks",
    "visible_support",
    "missing_or_uncertain_support",
    "comparison_to_unchanged_deck",
    "uncertainty",
)


class ChatCompletionClient(Protocol):
    model: str

    def _chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return an OpenAI-compatible chat completion response."""


@dataclass(frozen=True)
class RewardPosteriorEvaluation:
    data: dict[str, Any]
    prompt: str = ""
    raw_response: str = ""
    usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = json.loads(json.dumps(self.data, ensure_ascii=False, default=str))
        validate_reward_posterior_policy_fields(payload)
        return payload

    def summary_dict(self) -> dict[str, Any]:
        data = self.to_dict()
        analyses = data.get("candidate_deck_analyses")
        return {
            "mode": data.get("mode"),
            "candidate_analysis_count": len(analyses) if isinstance(analyses, list) else 0,
            "global_uncertainty_count": len(data.get("global_uncertainties") or []),
            "hash": self.evaluation_hash(),
        }

    def evaluation_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class LLMRewardPosteriorEvaluator:
    """Ask an LLM to compare post-action deck states without choosing an action."""

    def __init__(self, chat_client: ChatCompletionClient, *, max_retries: int = 1) -> None:
        self.chat_client = chat_client
        self.max_retries = max_retries

    def build_prompt(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        evidence_bundle: EvidenceBundle,
    ) -> str:
        code_owned_skeleton = build_reward_posterior_skeleton(evidence_bundle)
        rule_block = render_game_rules(state)
        update_targets = [
            {
                "action_id": candidate.action_id,
                "label": candidate.label,
                "kind": candidate.kind,
            }
            for candidate in evidence_bundle.candidate_evidence
            if not is_code_generated_candidate(candidate)
        ]
        schema = {
            "mode": POSTERIOR_SLOT_FILL_MODE,
            "candidate_analysis_updates": [
                {
                    "action_id": "<one required non-skip candidate action_id>",
                    "deck_delta": "<what changes in the deck after this action>",
                    "immediate_current_deck_effect": "<effect on the currently visible deck only>",
                    "redundancy_or_dilution_risks": "<redundancy/dilution risks versus the unchanged deck>",
                    "visible_support": "<support already visible in the current deck/relic/facts>",
                    "missing_or_uncertain_support": "<support not currently visible or uncertain>",
                    "comparison_to_unchanged_deck": "<deck-state comparison against skip baseline>",
                    "uncertainty": "<remaining uncertainty, no action advice>",
                }
            ],
            "global_uncertainties": ["<uncertainty shared by multiple candidate deck states>"],
        }
        return "\n".join(
            [
                "Reward posterior deck-state evaluator.",
                "Do not choose a game action. Do not recommend, rank, score, or prefer any action.",
                "Only compare the current unchanged deck against each candidate post-action deck state.",
                "Use only currently visible evidence. Do not assume future archetype support, future shops, future relics, or future card rewards.",
                "Code owns the final evaluation structure: mode, baseline_analysis, candidate_deck_analyses, action_id, label, and kind.",
                "Do not output candidate_deck_analyses, labels, kinds, scores, ranks, recommendations, or selected actions.",
                "Return only slot-fill text for candidate_analysis_updates. Unsupported keys will be rejected.",
                "Skip is generated by code as the unchanged-deck candidate; do not include card_reward_skip in candidate_analysis_updates.",
                "",
                rule_block,
                "Current authoritative game state:",
                render_state_text(state),
                "",
                "Code-owned final candidate_deck_analyses skeleton:",
                json.dumps(code_owned_skeleton["candidate_deck_analyses"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "Required non-skip candidate_analysis_updates targets:",
                json.dumps(update_targets, ensure_ascii=False, separators=(",", ":")),
                "Include every required non-skip target above exactly once. Fill every schema text field for each update.",
                "Return only valid JSON. Forbidden keys anywhere in the JSON include: "
                + ", ".join(sorted(POSTERIOR_POLICY_FIELDS)),
                "",
                "Current legal actions:",
                json.dumps([action.to_prompt_dict() for action in legal_actions], ensure_ascii=False, separators=(",", ":")),
                "",
                "Evidence bundle:",
                json.dumps(evidence_bundle.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "",
                "Required JSON schema:",
                json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            ]
        )

    def evaluate(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        evidence_bundle: EvidenceBundle,
    ) -> tuple[RewardPosteriorEvaluation, dict[str, Any]]:
        prompt = self.build_prompt(state, legal_actions, evidence_bundle)
        messages = [
            {
                "role": "system",
                "content": "You analyze deck-state deltas for card rewards. Return only valid JSON and never choose an action.",
            },
            {"role": "user", "content": prompt},
        ]
        last_error: str | None = None
        previous_errors: list[str] = []
        previous_raw_responses: list[str] = []
        request_audits: list[dict[str, Any]] = []
        begin_request = getattr(self.chat_client, "_begin_model_request", None)
        complete_request = getattr(self.chat_client, "_complete_model_request", None)
        fail_request = getattr(self.chat_client, "_fail_model_request", None)
        formal_audit_enabled = getattr(self.chat_client, "_formal_provider_audit_enabled", lambda: False)()
        if formal_audit_enabled and not all(
            callable(item) for item in (begin_request, complete_request, fail_request)
        ):
            raise ModelActionError("Reward posterior evaluator requires formal provider audit hooks")
        rule_meta = game_rules_meta(state, prompt=prompt, projection_enabled=True)
        if not rule_meta.get("game_rule_exposure_valid"):
            raise ModelActionError("Reward posterior evaluator rules-v2 exposure is invalid")
        for attempt in range(self.max_retries + 1):
            started = time.time()
            text: str | None = None
            try:
                if callable(begin_request):
                    begin_request(request_audits, messages)
                response = self.chat_client._chat(messages)
                elapsed = time.time() - started
                if callable(complete_request):
                    complete_request(request_audits, response, elapsed_sec=elapsed)
                text = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(text)
                data = build_reward_posterior_evaluation(parsed, evidence_bundle)
                evaluation = RewardPosteriorEvaluation(
                    data=data,
                    prompt=prompt,
                    raw_response=text,
                    usage=response.get("usage"),
                )
                meta = {
                    "reward_posterior_evaluator": "llm",
                    "reward_posterior_model": getattr(self.chat_client, "model", None),
                    "reward_posterior_attempt": attempt,
                    "reward_posterior_elapsed_sec": elapsed,
                    "reward_posterior_prompt_chars": len(prompt),
                    "reward_posterior_evaluation_hash": evaluation.evaluation_hash(),
                    "reward_posterior_usage": response.get("usage"),
                    "reward_posterior_model_request_audits": request_audits,
                    **{
                        f"reward_posterior_{key}": value
                        for key, value in rule_meta.items()
                    },
                }
                if previous_errors:
                    meta["reward_posterior_previous_errors"] = list(previous_errors)
                if previous_raw_responses:
                    meta["reward_posterior_previous_raw_responses"] = list(previous_raw_responses)
                meta["reward_posterior_code_owned_candidate_ids"] = sorted(evidence_bundle.candidate_action_ids())
                code_owned_skip_ids = [
                    candidate.action_id
                    for candidate in evidence_bundle.candidate_evidence
                    if is_code_generated_candidate(candidate)
                ]
                if code_owned_skip_ids:
                    meta["reward_posterior_code_owned_skip_candidate_ids"] = code_owned_skip_ids
                return evaluation, meta
            except Exception as exc:  # noqa: BLE001 - convert evaluator failure into model action error
                if callable(fail_request):
                    fail_request(request_audits, exc, elapsed_sec=time.time() - started)
                last_error = str(exc)
                previous_errors.append(last_error)
                if text is not None:
                    previous_raw_responses.append(text)
                if _is_transport_error(exc):
                    if attempt < self.max_retries:
                        time.sleep(_transport_retry_delay(attempt, exc))
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous posterior deck analysis was invalid: {last_error}. "
                            "Return only JSON matching the slot-fill schema. Provide candidate_analysis_updates for every non-skip target exactly once. "
                            "Do not output candidate_deck_analyses, labels, kinds, score, rank, recommendation, preferred_action, best_action, selected_action, or any action advice."
                        ),
                    }
                )
        raise ModelActionError(f"Reward posterior evaluator failed after {self.max_retries + 1} attempt(s): {last_error}")


def build_reward_posterior_evaluation(data: dict[str, Any], evidence_bundle: EvidenceBundle) -> dict[str, Any]:
    validate_reward_posterior_slot_fill(data, evidence_bundle)
    evaluation = build_reward_posterior_skeleton(evidence_bundle)
    analyses_by_id = {
        int(item["action_id"]): item
        for item in evaluation["candidate_deck_analyses"]
        if isinstance(item, dict)
    }
    for update in data.get("candidate_analysis_updates") or []:
        action_id = int(update["action_id"])
        target = analyses_by_id[action_id]
        for field in POSTERIOR_FILLABLE_FIELDS:
            target[field] = str(update[field]).strip()
    evaluation["global_uncertainties"] = normalize_global_uncertainties(data.get("global_uncertainties"))
    validate_reward_posterior_evaluation(evaluation, evidence_bundle)
    return evaluation


def build_reward_posterior_skeleton(evidence_bundle: EvidenceBundle) -> dict[str, Any]:
    analyses: list[dict[str, Any]] = []
    for candidate in evidence_bundle.candidate_evidence:
        if is_code_generated_candidate(candidate):
            fillable = code_owned_skip_candidate_analysis()
        else:
            fillable = {field: "" for field in POSTERIOR_FILLABLE_FIELDS}
        analyses.append(
            {
                "action_id": candidate.action_id,
                "label": candidate.label,
                "kind": candidate.kind,
                **fillable,
            }
        )
    return {
        "mode": POSTERIOR_FINAL_MODE,
        "baseline_analysis": "The unchanged current deck is the code-owned skip-card-reward baseline.",
        "candidate_deck_analyses": analyses,
        "global_uncertainties": [],
    }


def is_code_generated_candidate(candidate: Any) -> bool:
    return getattr(candidate, "kind", None) == "card_reward_skip"


def code_owned_skip_candidate_analysis() -> dict[str, str]:
    return {
        "deck_delta": "unchanged current deck; no card is added",
        "immediate_current_deck_effect": "the visible deck remains exactly as shown before the card reward",
        "redundancy_or_dilution_risks": "no new redundancy or dilution is introduced",
        "visible_support": "the current visible deck is the complete support for this unchanged baseline candidate",
        "missing_or_uncertain_support": "future rewards, shops, relics, and archetype support remain unknown",
        "comparison_to_unchanged_deck": "identical to the unchanged-deck baseline",
        "uncertainty": "none for the immediate deck delta; future outcomes remain unknown",
    }


def validate_reward_posterior_slot_fill(data: dict[str, Any], evidence_bundle: EvidenceBundle) -> None:
    if not isinstance(data, dict):
        raise ValueError("reward posterior slot fill must be a JSON object")
    validate_reward_posterior_policy_fields(data)
    allowed_top_keys = {"mode", "candidate_analysis_updates", "global_uncertainties"}
    extra_top_keys = sorted(set(data) - allowed_top_keys)
    if extra_top_keys:
        raise ValueError(f"reward posterior slot fill contains unsupported top-level keys: {extra_top_keys}")
    if data.get("mode") != POSTERIOR_SLOT_FILL_MODE:
        raise ValueError(f"reward posterior slot fill mode must be {POSTERIOR_SLOT_FILL_MODE}")
    updates = data.get("candidate_analysis_updates")
    if not isinstance(updates, list):
        raise ValueError("candidate_analysis_updates must be a list")
    by_id = {
        candidate.action_id: candidate
        for candidate in evidence_bundle.candidate_evidence
        if not is_code_generated_candidate(candidate)
    }
    expected_ids = set(by_id)
    actual_ids: set[int] = set()
    allowed_update_keys = {"action_id", *POSTERIOR_FILLABLE_FIELDS}
    for update in updates:
        if not isinstance(update, dict):
            raise ValueError("candidate_analysis_updates entries must be objects")
        extra_update_keys = sorted(set(update) - allowed_update_keys)
        if extra_update_keys:
            raise ValueError(f"candidate_analysis_updates entries contain unsupported keys: {extra_update_keys}")
        try:
            action_id = int(update.get("action_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate_analysis_updates entries must include integer action_id") from exc
        if action_id not in by_id:
            raise ValueError(f"candidate_analysis_updates contains unknown or code-owned action_id: {action_id}")
        if action_id in actual_ids:
            raise ValueError(f"candidate_analysis_updates duplicates action_id: {action_id}")
        actual_ids.add(action_id)
        missing_fields = [
            field
            for field in POSTERIOR_FILLABLE_FIELDS
            if not isinstance(update.get(field), str) or not update.get(field, "").strip()
        ]
        if missing_fields:
            raise ValueError(f"candidate_analysis_updates action_id {action_id} is missing text fields: {missing_fields}")
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "candidate_analysis_updates must cover every non-skip candidate exactly once; "
            f"missing={missing} extra={extra}"
        )
    normalize_global_uncertainties(data.get("global_uncertainties"))


def normalize_global_uncertainties(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("global_uncertainties must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("global_uncertainties entries must be strings")
        text = item.strip()
        if text:
            result.append(text)
    return result


def validate_reward_posterior_evaluation(data: dict[str, Any], evidence_bundle: EvidenceBundle) -> None:
    if not isinstance(data, dict):
        raise ValueError("reward posterior evaluation must be a JSON object")
    validate_reward_posterior_policy_fields(data)
    if data.get("mode") != POSTERIOR_FINAL_MODE:
        raise ValueError(f"reward posterior evaluation mode must be {POSTERIOR_FINAL_MODE}")
    analyses = data.get("candidate_deck_analyses")
    if not isinstance(analyses, list):
        raise ValueError("candidate_deck_analyses must be a list")
    by_id = {candidate.action_id: candidate for candidate in evidence_bundle.candidate_evidence}
    expected_ids = set(by_id)
    actual_ids: set[int] = set()
    for item in analyses:
        if not isinstance(item, dict):
            raise ValueError("candidate_deck_analyses entries must be objects")
        try:
            action_id = int(item.get("action_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate_deck_analyses entries must include integer action_id") from exc
        if action_id not in by_id:
            raise ValueError(f"candidate_deck_analyses contains unknown action_id: {action_id}")
        label = item.get("label")
        if label != by_id[action_id].label:
            raise ValueError(
                "candidate_deck_analyses label must exactly match current candidate_evidence: "
                f"action_id {action_id} is {by_id[action_id].label!r}, got {label!r}"
            )
        kind = item.get("kind")
        if kind != by_id[action_id].kind:
            raise ValueError(
                "candidate_deck_analyses kind must exactly match current candidate_evidence: "
                f"action_id {action_id} is {by_id[action_id].kind!r}, got {kind!r}"
            )
        if action_id in actual_ids:
            raise ValueError(f"candidate_deck_analyses duplicates action_id: {action_id}")
        actual_ids.add(action_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "candidate_deck_analyses must cover every candidate_evidence action_id exactly once; "
            f"missing={missing} extra={extra}"
        )


def validate_reward_posterior_policy_fields(value: Any, *, path: str = "") -> None:
    validate_no_policy_fields(value, path=path)
    if isinstance(value, dict):
        for key, item in value.items():
            if key in POSTERIOR_POLICY_FIELDS:
                location = f" at {path}.{key}" if path else f" at {key}"
                raise ValueError(f"Policy field is not allowed in reward posterior evaluation{location}: {key}")
            child_path = f"{path}.{key}" if path else str(key)
            validate_reward_posterior_policy_fields(item, path=child_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_reward_posterior_policy_fields(item, path=f"{path}[{index}]")
