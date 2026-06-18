"""LLM posterior deck-state analysis for card reward decisions."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .actions import LegalAction
from .agents import ModelActionError, parse_json_object
from .evidence import EvidenceBundle, validate_no_policy_fields


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
        del state
        schema = {
            "mode": "reward_posterior_deck_analysis",
            "baseline_analysis": "<what the unchanged current deck means as the skip baseline>",
            "candidate_deck_analyses": [
                {
                    "action_id": "<candidate_evidence action_id>",
                    "label": "<exact candidate_evidence label>",
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
                "Every candidate_deck_analyses item must correspond to one candidate_evidence action_id and copy its label exactly.",
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
        for attempt in range(self.max_retries + 1):
            started = time.time()
            try:
                response = self.chat_client._chat(messages)
                elapsed = time.time() - started
                text = response["choices"][0]["message"]["content"]
                parsed = parse_json_object(text)
                validate_reward_posterior_evaluation(parsed, evidence_bundle)
                evaluation = RewardPosteriorEvaluation(
                    data=parsed,
                    prompt=prompt,
                    raw_response=text,
                    usage=response.get("usage"),
                )
                return evaluation, {
                    "reward_posterior_evaluator": "llm",
                    "reward_posterior_model": getattr(self.chat_client, "model", None),
                    "reward_posterior_attempt": attempt,
                    "reward_posterior_elapsed_sec": elapsed,
                    "reward_posterior_prompt_chars": len(prompt),
                    "reward_posterior_evaluation_hash": evaluation.evaluation_hash(),
                    "reward_posterior_usage": response.get("usage"),
                }
            except Exception as exc:  # noqa: BLE001 - convert evaluator failure into model action error
                last_error = str(exc)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous posterior deck analysis was invalid: {last_error}. "
                            "Return only JSON matching the schema. Analyze every current candidate_evidence action_id exactly once. "
                            "Do not include score, rank, recommendation, preferred_action, best_action, selected_action, or any action advice."
                        ),
                    }
                )
        raise ModelActionError(f"Reward posterior evaluator failed after {self.max_retries + 1} attempt(s): {last_error}")


def validate_reward_posterior_evaluation(data: dict[str, Any], evidence_bundle: EvidenceBundle) -> None:
    if not isinstance(data, dict):
        raise ValueError("reward posterior evaluation must be a JSON object")
    validate_reward_posterior_policy_fields(data)
    if data.get("mode") != "reward_posterior_deck_analysis":
        raise ValueError("reward posterior evaluation mode must be reward_posterior_deck_analysis")
    analyses = data.get("candidate_deck_analyses")
    if not isinstance(analyses, list):
        raise ValueError("candidate_deck_analyses must be a list")
    by_id = {candidate.action_id: candidate.label for candidate in evidence_bundle.candidate_evidence}
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
        if label != by_id[action_id]:
            raise ValueError(
                "candidate_deck_analyses label must exactly match current candidate_evidence: "
                f"action_id {action_id} is {by_id[action_id]!r}, got {label!r}"
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
