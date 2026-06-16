"""OpenAI-compatible agent with a small frozen experience-memory layer."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..actions import LegalAction
from ..agents import ConversationMode, OpenAICompatAgent, parse_json_object


ExperienceMemoryMode = Literal["frozen", "observe"]
ExperienceAutoLearnTrigger = Literal["act_end", "act_end_or_death"]
EXPERIENCE_SECTION_HEADER = (
    "Experience notes, advisory only. Current game state and legal actions override these notes:"
)
DECISION_PHASE_TAGS = {
    "combat_play",
    "combat_reward",
    "card_reward",
    "bundle_select",
    "card_select",
    "event_choice",
    "map_select",
    "shop",
    "rest",
    "game_over",
}
COMBAT_MEMORY_LEVELS = {"combat", "encounter"}
GENERIC_RETRIEVAL_TAGS = {
    "act",
    "combat",
    "combat_play",
    "encounter",
    "episode",
    "event",
    "game",
    "global",
    "ironclad",
    "map",
    "monster",
    "playbook",
    "reward",
    "risk",
    "rule",
}
CARD_CONDITION_TAGS = {
    "bloodletting",
    "flex_potion",
    "fysh_oil",
    "inflame",
    "spot_weakness",
}


@dataclass(frozen=True)
class ExperienceMemoryEntry:
    id: str
    memory_type: str
    trigger: str
    lesson: str
    recommendation: str
    anti_pattern: str = ""
    counterfactual: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    retrieval_hints: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperienceMemoryEntry":
        entry_id = str(payload.get("id") or payload.get("memory_id") or "").strip()
        if not entry_id:
            raise ValueError("experience memory entry requires id")
        memory_type = str(payload.get("memory_type") or payload.get("type") or "rule").strip()
        if memory_type not in {"rule", "risk", "episode", "playbook", "counterfactual"}:
            raise ValueError(f"invalid experience memory_type: {memory_type}")
        return cls(
            id=entry_id,
            memory_type=memory_type,
            trigger=_one_line(payload.get("trigger")),
            lesson=_one_line(payload.get("lesson")),
            recommendation=_one_line(payload.get("recommendation")),
            anti_pattern=_one_line(payload.get("anti_pattern")),
            counterfactual=_one_line(payload.get("counterfactual")),
            scope=dict(payload.get("scope") or {}),
            quality=dict(payload.get("quality") or {}),
            retrieval_hints=dict(payload.get("retrieval_hints") or {}),
        )

    @property
    def confidence(self) -> float:
        value = self.quality.get("confidence", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @property
    def priority(self) -> float:
        value = self.retrieval_hints.get("priority", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @property
    def keywords(self) -> set[str]:
        raw_keywords = self.retrieval_hints.get("keywords") or []
        if not isinstance(raw_keywords, list):
            return set()
        return {_normalize_tag(keyword) for keyword in raw_keywords if _normalize_tag(keyword)}


@dataclass
class StateFeatures:
    character: str
    ascension: int
    decision: str
    act: int | None
    floor: int | None
    room_type: str
    boss: str
    enemy_names: list[str]
    card_names: list[str]
    relic_names: list[str]
    potion_names: list[str]
    action_kinds: list[str]
    hp_bucket: str
    hp_ratio: float | None
    deck_size_bucket: str
    round_number: int | None
    incoming_damage: int
    has_attack_intent: bool
    has_attack_card: bool
    has_block_card: bool
    tags: set[str]


@dataclass
class RetrievalResult:
    entries: list[ExperienceMemoryEntry]
    rendered_text: str
    rendered_ids: list[str]
    rendered_chars: int
    store_digest: str
    store_path: str

    def meta(self, *, mode: ExperienceMemoryMode) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": mode,
            "store_path": self.store_path,
            "store_digest": self.store_digest,
            "retrieved_ids": list(self.rendered_ids),
            "rendered_chars": self.rendered_chars,
            "entry_count": len(self.entries),
        }


class ExperienceMemoryRetriever:
    """Tiny JSONL retriever for frozen, human-reviewed experience notes."""

    def __init__(
        self,
        *,
        store_path: str | Path,
        top_k: int = 5,
        token_budget: int = 700,
        min_confidence: float = 0.35,
        include_global: bool = True,
        allow_cross_character: bool = False,
        allow_cross_ascension: bool = True,
    ) -> None:
        self.store_path = Path(store_path)
        self.top_k = top_k
        self.token_budget = token_budget
        self.min_confidence = min_confidence
        self.include_global = include_global
        self.allow_cross_character = allow_cross_character
        self.allow_cross_ascension = allow_cross_ascension
        self.entries = self._load_entries()
        self.store_digest = experience_store_digest(self.store_path)

    def retrieve(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        character: str,
        ascension: int,
    ) -> RetrievalResult:
        features = build_state_features(
            state,
            legal_actions,
            character=character,
            ascension=ascension,
        )
        scored: list[tuple[float, ExperienceMemoryEntry]] = []
        for entry in self.entries:
            if not self._scope_allows(entry, features):
                continue
            score = self._score(entry, features)
            if score <= 0:
                continue
            scored.append((score, entry))

        scored.sort(key=lambda item: (-item[0], -item[1].confidence, item[1].id))
        selected = [entry for _score, entry in scored[: max(0, self.top_k)]]
        rendered = render_experience_notes(selected, token_budget=self.token_budget)
        rendered_ids = [entry.id for entry in selected if entry.id in rendered]
        return RetrievalResult(
            entries=selected,
            rendered_text=rendered,
            rendered_ids=rendered_ids,
            rendered_chars=len(rendered),
            store_digest=self.store_digest,
            store_path=str(self.store_path),
        )

    def _load_entries(self) -> list[ExperienceMemoryEntry]:
        if not self.store_path.is_file():
            return []
        entries: list[ExperienceMemoryEntry] = []
        for line_number, raw_line in enumerate(self.store_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{self.store_path}:{line_number} must be a JSON object")
            if payload.get("status", "active") != "active":
                continue
            entry = ExperienceMemoryEntry.from_dict(payload)
            if entry.confidence < self.min_confidence:
                continue
            entries.append(entry)
        return entries

    def _scope_allows(self, entry: ExperienceMemoryEntry, features: StateFeatures) -> bool:
        scope = entry.scope
        level = str(scope.get("level") or "").lower()
        if level == "global" and not self.include_global:
            return False

        entry_character = str(scope.get("character") or "").strip()
        if entry_character and entry_character != features.character and not self.allow_cross_character:
            return False

        if not self.allow_cross_ascension:
            min_asc = _int_or_none(scope.get("ascension_min"))
            max_asc = _int_or_none(scope.get("ascension_max"))
            if min_asc is not None and features.ascension < min_asc:
                return False
            if max_asc is not None and features.ascension > max_asc:
                return False

        scope_act = _int_or_none(scope.get("act"))
        if scope_act is not None and features.act is not None and scope_act != features.act:
            return False

        phase = _normalize_tag(scope.get("phase"))
        decision = _normalize_tag(features.decision)
        if not _phase_allows(phase, features):
            return False

        if decision != "combat_play":
            if level in COMBAT_MEMORY_LEVELS:
                return False
            if "combat_play" in entry.keywords:
                return False

        entry_enemy = _normalize_tag(scope.get("enemy") or scope.get("encounter"))
        enemy_tags = {_normalize_tag(enemy) for enemy in features.enemy_names}
        if level in COMBAT_MEMORY_LEVELS and entry_enemy and enemy_tags and not _enemy_scope_matches(entry_enemy, enemy_tags):
            return False
        if not _entry_conditions_allow(entry, features):
            return False
        if level == "global" and not _global_entry_has_context_match(entry, features):
            return False

        return True

    def _score(self, entry: ExperienceMemoryEntry, features: StateFeatures) -> float:
        score = 0.0
        scope = entry.scope
        entry_boss = _normalize_tag(scope.get("boss"))
        entry_enemy = _normalize_tag(scope.get("enemy") or scope.get("encounter"))
        boss_tag = _normalize_tag(features.boss)
        enemy_tags = {_normalize_tag(enemy) for enemy in features.enemy_names}
        if entry_boss and entry_boss == boss_tag:
            score += 2.0
        if entry_enemy and _enemy_scope_matches(entry_enemy, enemy_tags):
            score += 2.0

        if scope.get("character") == features.character:
            score += 1.2
        if _int_or_none(scope.get("act")) == features.act and features.act is not None:
            score += 0.8

        phase = _normalize_tag(scope.get("phase"))
        if phase and phase == _normalize_tag(features.decision):
            score += 0.7

        overlap = entry.keywords.intersection(features.tags)
        score += min(len(overlap), 5) * 0.45

        if str(scope.get("level") or "").lower() == "global" and self.include_global:
            score += 0.35

        score += entry.confidence * 0.8
        score += entry.priority * 0.3
        return score


@dataclass
class ExperienceOpenAICompatAgent(OpenAICompatAgent):
    """OpenAI-compatible policy with frozen experience retrieval and compact event tracing."""

    experience_memory_mode: ExperienceMemoryMode = "frozen"
    experience_memory_store_path: str = "sts2_bench/experience/memory/active.jsonl"
    experience_memory_top_k: int = 5
    experience_memory_token_budget: int = 700
    experience_memory_min_confidence: float = 0.35
    experience_memory_include_global: bool = True
    experience_memory_allow_cross_character: bool = False
    experience_memory_allow_cross_ascension: bool = True
    experience_event_trace_path: str = ""
    experience_auto_learn_enabled: bool = False
    experience_auto_learn_trigger: ExperienceAutoLearnTrigger = "act_end"
    experience_auto_learn_candidate_path: str = "sts2_bench/experience/memory/candidates.jsonl"
    experience_auto_learn_max_candidates_per_span: int = 2
    experience_auto_learn_max_events_per_span: int = 40
    experience_auto_learn_min_confidence: float = 0.35
    experience_auto_learn_promote: bool = True
    experience_auto_learn_promote_on_death: bool = False
    experience_retriever: ExperienceMemoryRetriever = field(init=False)
    _experience_cache_key: str = field(default="", init=False)
    _experience_cache: RetrievalResult | None = field(default=None, init=False)
    _episode_seed: str = field(default="", init=False)
    _episode_character: str = field(default="Ironclad", init=False)
    _episode_ascension: int = field(default=0, init=False)
    _event_step: int = field(default=0, init=False)
    _episode_events: list[dict[str, Any]] = field(default_factory=list, init=False)
    _auto_learned_acts: set[int] = field(default_factory=set, init=False)
    _auto_learn_records: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.experience_memory_mode not in {"frozen", "observe"}:
            raise ValueError(f"Unsupported experience_memory_mode: {self.experience_memory_mode}")
        if self.conversation_mode != "single_turn":
            raise ValueError("experience memory MVP currently supports only single_turn context mode")
        if self.experience_auto_learn_trigger not in {"act_end", "act_end_or_death"}:
            raise ValueError(f"Unsupported experience_auto_learn_trigger: {self.experience_auto_learn_trigger}")
        self.refresh_experience_memory()

    def start_episode(self, *, seed: str, character: str, ascension: int) -> None:
        super().reset_episode()
        self._episode_seed = seed
        self._episode_character = character
        self._episode_ascension = ascension
        self._event_step = 0
        self._episode_events.clear()
        self._experience_cache_key = ""
        self._experience_cache = None
        self._auto_learned_acts.clear()
        self._auto_learn_records.clear()

    def reset_episode(self) -> None:
        super().reset_episode()
        self._event_step = 0
        self._episode_events.clear()
        self._experience_cache_key = ""
        self._experience_cache = None
        self._auto_learned_acts.clear()
        self._auto_learn_records.clear()

    def build_prompt(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> str:
        base_prompt = super().build_prompt(state, legal_actions)
        retrieval = self._retrieve(state, legal_actions)
        return inject_experience_section(base_prompt, retrieval.rendered_text)

    def choose(
        self,
        state: dict[str, Any],
        legal_actions: list[LegalAction],
        *,
        prompt: str | None = None,
    ) -> tuple[LegalAction, dict[str, Any]]:
        retrieval = self._retrieve(state, legal_actions)
        if prompt is None:
            prompt = inject_experience_section(super().build_prompt(state, legal_actions), retrieval.rendered_text)
        elif EXPERIENCE_SECTION_HEADER not in prompt:
            prompt = inject_experience_section(prompt, retrieval.rendered_text)

        action, meta = super().choose(state, legal_actions, prompt=prompt)
        meta["experience_memory"] = {
            **retrieval.meta(mode=self.experience_memory_mode),
            "used_memory_ids": _valid_used_memory_ids(meta.get("parsed"), retrieval.rendered_ids),
        }
        return action, meta

    def record_transition(
        self,
        state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        next_state: dict[str, Any] | None = None,
    ) -> None:
        super().record_transition(state, action, meta, next_state)
        if self.experience_event_trace_path or self.experience_auto_learn_enabled:
            event = build_compact_experience_event(
                state,
                action,
                meta,
                next_state,
                seed=self._episode_seed,
                character=self._episode_character,
                ascension=self._episode_ascension,
                step=self._event_step,
            )
            self._episode_events.append(event)
            if self.experience_event_trace_path:
                self._append_event(event)
            self._event_step += 1
            self._maybe_auto_learn_after_transition(state, next_state)

    def refresh_experience_memory(self) -> None:
        self.experience_retriever = ExperienceMemoryRetriever(
            store_path=self.experience_memory_store_path,
            top_k=self.experience_memory_top_k,
            token_budget=self.experience_memory_token_budget,
            min_confidence=self.experience_memory_min_confidence,
            include_global=self.experience_memory_include_global,
            allow_cross_character=self.experience_memory_allow_cross_character,
            allow_cross_ascension=self.experience_memory_allow_cross_ascension,
        )
        self._experience_cache_key = ""
        self._experience_cache = None

    def pop_auto_learn_records(self) -> list[dict[str, Any]]:
        records = list(self._auto_learn_records)
        self._auto_learn_records.clear()
        return records

    def _retrieve(self, state: dict[str, Any], legal_actions: list[LegalAction]) -> RetrievalResult:
        cache_key = _retrieval_cache_key(state, legal_actions, self._episode_character, self._episode_ascension)
        if self._experience_cache_key == cache_key and self._experience_cache is not None:
            return self._experience_cache
        retrieval = self.experience_retriever.retrieve(
            state,
            legal_actions,
            character=self._episode_character,
            ascension=self._episode_ascension,
        )
        self._experience_cache_key = cache_key
        self._experience_cache = retrieval
        return retrieval

    def _append_event(self, event: dict[str, Any]) -> None:
        path = Path(self.experience_event_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _maybe_auto_learn_after_transition(self, state: dict[str, Any], next_state: dict[str, Any] | None) -> None:
        if not self.experience_auto_learn_enabled:
            return
        if not isinstance(next_state, dict):
            return
        previous_act = _state_int(state, "act")
        next_act = _state_int(next_state, "act")
        if previous_act is None:
            return
        if next_act is not None and next_act > previous_act:
            self._auto_learn_act_once(previous_act, cause="act_end")
            return
        if self.experience_auto_learn_trigger == "act_end_or_death" and next_state.get("decision") == "game_over":
            self._auto_learn_act_once(previous_act, cause="death")

    def _auto_learn_act_once(self, act: int, *, cause: str) -> None:
        if act in self._auto_learned_acts:
            return
        self._auto_learned_acts.add(act)
        self._auto_learn_completed_act(act, cause=cause)

    def _auto_learn_completed_act(self, act: int, *, cause: str) -> None:
        from .reflection import learn_from_events

        trace_path = Path(self.experience_event_trace_path) if self.experience_event_trace_path else None
        promote = self.experience_auto_learn_promote and (
            cause != "death" or self.experience_auto_learn_promote_on_death
        )
        try:
            events = [
                event
                for event in self._episode_events
                if event.get("seed") == self._episode_seed and _int_or_none(event.get("act")) == act
            ]
            summary = learn_from_events(
                events,
                trace_path=f"{trace_path or '<episode_events>'}#seed={self._episode_seed}:act={act}:cause={cause}",
                candidate_out=self.experience_auto_learn_candidate_path,
                active_path=self.experience_memory_store_path,
                chat_fn=self._reflection_chat,
                max_candidates_per_span=self.experience_auto_learn_max_candidates_per_span,
                max_events_per_span=self.experience_auto_learn_max_events_per_span,
                promote=promote,
                min_confidence=self.experience_auto_learn_min_confidence,
            )
            if promote:
                self.refresh_experience_memory()
            self._auto_learn_records.append(
                {
                    "enabled": True,
                    "trigger": self.experience_auto_learn_trigger,
                    "cause": cause,
                    "act": act,
                    "promote": promote,
                    "status": "ok",
                    "event_count": len(events),
                    "summary": summary,
                }
            )
        except Exception as exc:  # noqa: BLE001 - learning failure should not abort the run.
            self._auto_learn_records.append(
                {
                    "enabled": True,
                    "trigger": self.experience_auto_learn_trigger,
                    "cause": cause,
                    "act": act,
                    "promote": promote,
                    "status": "error",
                    "error": str(exc),
                }
            )

    def _reflection_chat(self, messages: list[dict[str, str]]) -> str:
        request_messages = list(messages)
        last_error = ""
        for _attempt in range(self.max_retries + 1):
            try:
                response = self._chat(request_messages)
                text = response["choices"][0]["message"]["content"]
                parse_json_object(text)
                return text
            except Exception as exc:  # noqa: BLE001 - retry with a schema correction message.
                last_error = str(exc)
                request_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response was invalid: {last_error}. "
                            "Return only valid JSON matching the requested schema."
                        ),
                    }
                )
        raise ValueError(f"auto-learn reflection response stayed invalid: {last_error}")


def build_state_features(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    *,
    character: str,
    ascension: int,
) -> StateFeatures:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    boss = context.get("boss") if isinstance(context.get("boss"), dict) else {}
    boss_name = _name(boss.get("name")) if boss else ""
    enemy_names = [_name(enemy.get("name")) for enemy in state.get("enemies") or [] if enemy.get("hp", 0) > 0]
    hand_cards = [card for card in state.get("hand") or [] if isinstance(card, dict)]
    choice_cards = [card for card in state.get("cards") or [] if isinstance(card, dict)]
    cards = hand_cards + choice_cards
    card_names = [_name(card.get("name")) for card in cards]
    relic_names = [_name(relic.get("name")) for relic in player.get("relics") or []]
    potion_names = [_name(potion.get("name")) for potion in player.get("potions") or [] if potion]
    decision = str(state.get("decision") or "")
    room_type = str(context.get("room_type") or "")
    act = _state_int(state, "act")
    floor = _state_int(state, "floor")
    round_number = _state_int(state, "round")
    hp_ratio = _ratio_value(player.get("hp"), player.get("max_hp"))
    incoming = _incoming_damage(state)
    has_attack_card = any(_card_has_stat(card, "damage") for card in hand_cards)
    has_block_card = any(_card_has_stat(card, "block") for card in hand_cards)
    tags = {
        _normalize_tag(tag)
        for tag in [
            character,
            decision,
            room_type,
            boss_name,
            *enemy_names,
            *card_names,
            *relic_names,
            *potion_names,
            *[action.kind for action in legal_actions],
            *[action.label for action in legal_actions],
        ]
        if _normalize_tag(tag)
    }
    for card in cards:
        card_type = _normalize_tag(card.get("type"))
        if card_type:
            tags.add(card_type)
    if has_attack_card:
        tags.update({"attack", "attack_card", "damage"})
    if has_block_card:
        tags.update({"block", "block_card"})
    if hp_ratio is not None:
        if hp_ratio > 0.75:
            tags.add("high_hp")
        elif hp_ratio <= 0.25:
            tags.update({"low_hp", "critical_hp"})
        elif hp_ratio <= 0.50:
            tags.add("low_hp")
    if incoming > 0:
        tags.update({"incoming_attack", "enemy_attacking"})
    else:
        tags.update({"no_incoming_attack", "safe_turn"})
    if round_number == 1:
        tags.update({"first_turn", "turn1", "turn_1"})
    elif round_number is not None and round_number <= 2:
        tags.add("early_turn")
    for enemy in state.get("enemies") or []:
        if not isinstance(enemy, dict) or enemy.get("hp", 0) <= 0:
            continue
        for intent in enemy.get("intents") or []:
            if isinstance(intent, dict):
                intent_tag = _normalize_tag(intent.get("type"))
                if intent_tag:
                    tags.add(intent_tag)
        for power in enemy.get("powers") or []:
            if isinstance(power, dict):
                power_tag = _normalize_tag(power.get("name"))
                if power_tag:
                    tags.add(power_tag)
    for power in state.get("player_powers") or []:
        if isinstance(power, dict):
            power_tag = _normalize_tag(power.get("name"))
            if power_tag:
                tags.add(power_tag)
    if act is not None:
        tags.add(f"act{act}")
    return StateFeatures(
        character=character,
        ascension=ascension,
        decision=decision,
        act=act,
        floor=floor,
        room_type=room_type,
        boss=boss_name,
        enemy_names=enemy_names,
        card_names=card_names,
        relic_names=relic_names,
        potion_names=potion_names,
        action_kinds=[action.kind for action in legal_actions],
        hp_bucket=_ratio_bucket(player.get("hp"), player.get("max_hp")),
        hp_ratio=hp_ratio,
        deck_size_bucket=_deck_size_bucket(player.get("deck_size")),
        round_number=round_number,
        incoming_damage=incoming,
        has_attack_intent=incoming > 0,
        has_attack_card=has_attack_card,
        has_block_card=has_block_card,
        tags=tags,
    )


def render_experience_notes(entries: list[ExperienceMemoryEntry], *, token_budget: int) -> str:
    char_budget = max(0, token_budget * 4)
    lines: list[str] = []
    used = 0
    for entry in entries:
        note_parts = [
            f"- [{entry.memory_type}/id={entry.id}/conf={entry.confidence:.2f}]",
        ]
        if entry.trigger:
            note_parts.append(f"Trigger: {entry.trigger}")
        if entry.recommendation:
            note_parts.append(f"Recommendation: {entry.recommendation}")
        elif entry.lesson:
            note_parts.append(f"Lesson: {entry.lesson}")
        if entry.anti_pattern:
            note_parts.append(f"Avoid: {entry.anti_pattern}")
        if entry.counterfactual:
            note_parts.append(f"Boundary: {entry.counterfactual}")
        line = " ".join(note_parts)
        if char_budget and lines and used + len(line) + 1 > char_budget:
            break
        if char_budget and not lines and len(line) > char_budget:
            line = line[: max(0, char_budget - 3)].rstrip() + "..."
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def inject_experience_section(prompt: str, rendered_text: str) -> str:
    if not rendered_text:
        return prompt
    section = "\n".join([EXPERIENCE_SECTION_HEADER, rendered_text])
    marker = "\nGame state:"
    if marker in prompt:
        return prompt.replace(marker, "\n" + section + "\n" + marker.lstrip(), 1)
    return prompt + "\n\n" + section


def build_compact_experience_event(
    state: dict[str, Any],
    action: LegalAction,
    meta: dict[str, Any],
    next_state: dict[str, Any] | None,
    *,
    seed: str,
    character: str,
    ascension: int,
    step: int,
) -> dict[str, Any]:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    player_before = state.get("player") if isinstance(state.get("player"), dict) else {}
    player_after = next_state.get("player") if isinstance(next_state, dict) and isinstance(next_state.get("player"), dict) else {}
    result = next_state.get("last_action_result") if isinstance(next_state, dict) else None
    parsed = meta.get("parsed") if isinstance(meta.get("parsed"), dict) else {}
    experience_meta = meta.get("experience_memory") if isinstance(meta.get("experience_memory"), dict) else {}

    return {
        "seed": seed,
        "step": step,
        "character": character,
        "ascension": ascension,
        "act": state.get("act") or context.get("act"),
        "floor": state.get("floor") or context.get("floor"),
        "decision": state.get("decision"),
        "room_type": context.get("room_type"),
        "encounter": _encounter_name(state),
        "hp_bucket_before": _ratio_bucket(player_before.get("hp"), player_before.get("max_hp")),
        "hp_bucket_after": _ratio_bucket(player_after.get("hp"), player_after.get("max_hp")),
        "deck_size_bucket": _deck_size_bucket(player_before.get("deck_size")),
        "action": {
            "kind": action.kind,
            "label": action.label,
        },
        "outcome": {
            "decision_after": next_state.get("decision") if isinstance(next_state, dict) else None,
            "hp_delta": _numeric_delta(player_before.get("hp"), player_after.get("hp")),
            "block_delta": _numeric_delta(player_before.get("block"), player_after.get("block")),
            "energy_delta": _numeric_delta(state.get("energy"), next_state.get("energy") if isinstance(next_state, dict) else None),
            "enemy_hp_delta": _enemy_hp_delta(state, next_state),
            "important_changes": _important_changes(result),
        },
        "agent": {
            "reason": _one_line(parsed.get("reason") or meta.get("reason")),
            "fallback": bool(meta.get("fallback")),
            "retrieved_memory_ids": experience_meta.get("retrieved_ids", []),
            "used_memory_ids": experience_meta.get("used_memory_ids", []),
        },
    }


def experience_store_digest(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_used_memory_ids(parsed: Any, retrieved_ids: list[str]) -> list[str]:
    if not isinstance(parsed, dict):
        return []
    raw_ids = parsed.get("used_memory_ids")
    if not isinstance(raw_ids, list):
        return []
    allowed = set(retrieved_ids)
    used: list[str] = []
    for raw_id in raw_ids:
        memory_id = str(raw_id)
        if memory_id in allowed and memory_id not in used:
            used.append(memory_id)
    return used


def _retrieval_cache_key(
    state: dict[str, Any],
    legal_actions: list[LegalAction],
    character: str,
    ascension: int,
) -> str:
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    payload = {
        "character": character,
        "ascension": ascension,
        "decision": state.get("decision"),
        "act": state.get("act") or context.get("act"),
        "floor": state.get("floor") or context.get("floor"),
        "round": state.get("round"),
        "actions": [action.label for action in legal_actions],
        "hand": [_name(card.get("name")) for card in state.get("hand") or []],
        "cards": [_name(card.get("name")) for card in state.get("cards") or []],
        "enemies": [_name(enemy.get("name")) for enemy in state.get("enemies") or [] if enemy.get("hp", 0) > 0],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _phase_allows(phase: str, features: StateFeatures) -> bool:
    if not phase:
        return True
    decision = _normalize_tag(features.decision)
    if phase in DECISION_PHASE_TAGS:
        return phase == decision
    if decision != "combat_play":
        return False

    if phase in {"combat", "combat_turn", "combat_action"}:
        return True
    if phase in {"first_turn", "turn1", "turn_1"}:
        return features.round_number == 1
    if phase in {"early_turn", "early_combat"}:
        return features.round_number is not None and features.round_number <= 2
    if phase in {"stunned_turn", "stun_turn"}:
        return features.incoming_damage == 0 and ("stun" in features.tags or "stunned" in features.tags)
    return False


def _entry_conditions_allow(entry: ExperienceMemoryEntry, features: StateFeatures) -> bool:
    apply_text = _condition_text(entry.trigger, entry.lesson, entry.recommendation, " ".join(sorted(entry.keywords)))
    boundary_text = _condition_text(entry.counterfactual)

    if _requires_high_hp(apply_text) and not _hp_ratio_at_least(features.hp_ratio, _high_hp_threshold(apply_text)):
        return False
    if _requires_low_hp(apply_text) and not _hp_ratio_at_most(features.hp_ratio, 0.50):
        return False
    if _blocks_low_hp(boundary_text) and _hp_ratio_at_most(features.hp_ratio, 0.50):
        return False

    if _requires_safe_turn(apply_text) and features.incoming_damage > 0:
        return False
    if _blocks_not_stunned(boundary_text) and features.incoming_damage > 0:
        return False
    if _requires_first_turn(apply_text) and features.round_number != 1:
        return False

    if _requires_attack_cards(boundary_text) and not features.has_attack_card:
        return False

    mentioned_card_tags = {tag for tag in CARD_CONDITION_TAGS if tag in apply_text}
    if mentioned_card_tags and not mentioned_card_tags.intersection(features.tags):
        return False

    return True


def _global_entry_has_context_match(entry: ExperienceMemoryEntry, features: StateFeatures) -> bool:
    keywords = entry.keywords - GENERIC_RETRIEVAL_TAGS
    if not keywords:
        return False
    return bool(keywords.intersection(features.tags))


def _condition_text(*values: Any) -> str:
    return _normalize_tag(" ".join(_one_line(value) for value in values if _one_line(value)))


def _requires_high_hp(text: str) -> bool:
    return bool(re.search(r"(high_hp|above_?50|above_?75|hp_above|safely_above|above_?50_percent)", text))


def _requires_low_hp(text: str) -> bool:
    return bool(re.search(r"(low_hp|critical_hp|below_?50|hp_below)", text))


def _blocks_low_hp(text: str) -> bool:
    return bool(re.search(r"(do_not_apply.*low_hp|do_not_use.*below_?50|below_?50|low_hp|hp_is_low)", text))


def _high_hp_threshold(text: str) -> float:
    if re.search(r"(above_?75|75_percent|75)", text):
        return 0.75
    return 0.50


def _requires_safe_turn(text: str) -> bool:
    return bool(re.search(r"(safe_turn|not_attacking|no_incoming_attack|non_attack|stunned|stun_turn)", text))


def _blocks_not_stunned(text: str) -> bool:
    return bool(re.search(r"(enemy_is_not_stunned|not_stunned|enemy_not_stunned)", text))


def _requires_first_turn(text: str) -> bool:
    return bool(re.search(r"(first_turn|turn_?1|turn_one)", text))


def _requires_attack_cards(text: str) -> bool:
    return bool(re.search(r"(lack_sufficient_attacks|lack_attacks|no_attacks|sufficient_attacks|attack_cards)", text))


def _hp_ratio_at_least(ratio: float | None, threshold: float) -> bool:
    return ratio is not None and ratio > threshold


def _hp_ratio_at_most(ratio: float | None, threshold: float) -> bool:
    return ratio is not None and ratio <= threshold


def _state_int(state: dict[str, Any], key: str) -> int | None:
    value = state.get(key)
    if value is None:
        context = state.get("context") if isinstance(state.get("context"), dict) else {}
        value = context.get(key)
    return _int_or_none(value)


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric_delta(before: Any, after: Any) -> int | float | None:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def _enemy_hp_delta(state: dict[str, Any], next_state: dict[str, Any] | None) -> int | None:
    if not isinstance(next_state, dict):
        return None
    if state.get("decision") != "combat_play" or next_state.get("decision") != "combat_play":
        return None

    before = _enemy_hp_by_index(state)
    after = _enemy_hp_by_index(next_state)
    if not before or not after:
        return None
    if before.keys().isdisjoint(after.keys()):
        return None
    return int(sum(after.get(index, 0) - hp for index, hp in before.items()))


def _important_changes(result: Any) -> list[str]:
    if not isinstance(result, dict):
        return []
    return [_one_line(change) for change in (result.get("changes") or [])[:8] if _one_line(change)]


def _encounter_name(state: dict[str, Any]) -> str:
    enemies = [_name(enemy.get("name")) for enemy in state.get("enemies") or [] if enemy.get("hp", 0) > 0]
    if enemies:
        return ", ".join(enemies[:3])
    context = state.get("context") if isinstance(state.get("context"), dict) else {}
    boss = context.get("boss") if isinstance(context.get("boss"), dict) else {}
    return _name(boss.get("name")) if boss else ""


def _ratio_bucket(value: Any, maximum: Any) -> str:
    ratio = _ratio_value(value, maximum)
    if ratio is None:
        return "unknown"
    if ratio <= 0.25:
        return "0.00-0.25"
    if ratio <= 0.50:
        return "0.25-0.50"
    if ratio <= 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def _ratio_value(value: Any, maximum: Any) -> float | None:
    if not isinstance(value, (int, float)) or not isinstance(maximum, (int, float)) or maximum <= 0:
        return None
    return float(value) / float(maximum)


def _incoming_damage(state: dict[str, Any]) -> int:
    total = 0
    for enemy in state.get("enemies") or []:
        if not isinstance(enemy, dict) or enemy.get("hp", 0) <= 0:
            continue
        for intent in enemy.get("intents") or []:
            if not isinstance(intent, dict) or intent.get("type") not in {"Attack", "DeathBlow"}:
                continue
            damage = intent.get("damage")
            hits = intent.get("hits", 1) or 1
            if isinstance(damage, (int, float)) and isinstance(hits, (int, float)):
                total += int(damage) * int(hits)
    return total


def _card_has_stat(card: dict[str, Any], stat_name: str) -> bool:
    stats = card.get("stats") if isinstance(card.get("stats"), dict) else {}
    value = stats.get(stat_name)
    return isinstance(value, (int, float)) and value > 0


def _deck_size_bucket(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    if value <= 12:
        return "0-12"
    if value <= 16:
        return "13-16"
    if value <= 20:
        return "17-20"
    if value <= 25:
        return "21-25"
    return "26+"


def _normalize_tag(value: Any) -> str:
    return re.sub(r"[^a-z0-9_\-]+", "_", _one_line(value).lower()).strip("_")


def _enemy_scope_matches(entry_enemy: str, enemy_tags: set[str]) -> bool:
    if not entry_enemy:
        return True
    if not enemy_tags:
        return False
    if entry_enemy in enemy_tags:
        return True
    return any(tag and (tag in entry_enemy or entry_enemy in tag) for tag in enemy_tags)


def _enemy_hp_by_index(state: dict[str, Any]) -> dict[int, int]:
    enemies: dict[int, int] = {}
    for enemy in state.get("enemies") or []:
        index = enemy.get("index")
        hp = enemy.get("hp")
        if isinstance(index, (int, float)) and isinstance(hp, (int, float)):
            enemies[int(index)] = int(hp)
    return enemies


def _name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name") or value.get("id")
    return _one_line(value)


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
