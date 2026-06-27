"""Per-combat recording helpers for full-game runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import LegalAction
from .context import compact_state


COMBAT_DECISIONS = {"combat_play", "card_select"}


@dataclass
class _ActiveCombat:
    combat_id: str
    directory: Path
    step: int
    decision: str
    act: int | None
    floor: int | None
    room_type: str | None
    entry_hp: int | None
    max_hp: int | None
    entry_state: dict[str, Any]


class CombatRecorder:
    """Write per-combat artifacts that can be replayed or rendered later."""

    def __init__(
        self,
        *,
        root: Path | str,
        run_id: str,
        condition: str | None = None,
        seed: str | None = None,
        character: str | None = None,
        ascension: int | None = None,
        lang: str | None = None,
        save_entries: bool = True,
    ) -> None:
        self.root = Path(root)
        self.run_id = _safe_id(run_id)
        self.condition = condition
        self.seed = seed
        self.character = character
        self.ascension = ascension
        self.lang = lang
        self.save_entries = save_entries
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        self._index = self._load_index()
        self._counter = self._initial_counter()
        self._active: _ActiveCombat | None = None

    def before_decision(
        self,
        process: Any,
        state: dict[str, Any],
        step: int,
        action_tape: list[dict[str, Any]],
    ) -> None:
        if self._active is not None or not _is_combat_state(state):
            return
        self._start_combat(process=process, state=state, step=step, action_tape=action_tape)

    def after_action(
        self,
        *,
        step: int,
        old_state: dict[str, Any],
        action: LegalAction,
        meta: dict[str, Any],
        commands: list[dict[str, Any]],
        next_state: dict[str, Any],
    ) -> None:
        if self._active is None:
            if not _is_combat_state(old_state):
                return
            self._start_combat(process=None, state=old_state, step=step, action_tape=[])
        active = self._active
        if active is None:
            return

        action_row: dict[str, Any] = {
            "type": "combat_action",
            "combat_id": active.combat_id,
            "step": step,
            "decision": old_state.get("decision"),
            "action": _action_summary(action),
            "commands": [_copy_json(command) for command in commands],
            "agent_meta": _copy_json(meta),
            "next_decision": next_state.get("decision"),
            "next_hp": _hp(next_state),
        }
        if len(commands) == 1:
            action_row["command"] = _copy_json(commands[0])
        _append_jsonl(active.directory / "room.jsonl", action_row)
        for command in commands:
            _append_jsonl(
                active.directory / "commands.jsonl",
                {
                    "type": "command",
                    "step": step,
                    "decision": old_state.get("decision"),
                    "command": _copy_json(command),
                },
            )

        if not _is_combat_state(next_state):
            self._finish_combat(next_state)

    def finalize(self, state: dict[str, Any] | None = None, *, reason: str = "finalize") -> None:
        if self._active is None:
            return
        self._finish_combat(state or self._active.entry_state, reason=reason)

    def _start_combat(
        self,
        *,
        process: Any,
        state: dict[str, Any],
        step: int,
        action_tape: list[dict[str, Any]],
    ) -> None:
        self._counter += 1
        context = _context(state)
        act = _int_or_none(state.get("act") or context.get("act"))
        floor = _int_or_none(state.get("floor") or context.get("floor"))
        room_type = str(context.get("room_type") or state.get("room_type") or "Combat")
        suffix = f"A{act or 0}F{floor or 0}_{room_type}"
        combat_id = _safe_id(f"{self.run_id}_combat_{self._counter:03d}_{suffix}")
        directory = self.root / combat_id
        directory.mkdir(parents=True, exist_ok=True)
        active = _ActiveCombat(
            combat_id=combat_id,
            directory=directory,
            step=step,
            decision=str(state.get("decision") or ""),
            act=act,
            floor=floor,
            room_type=room_type,
            entry_hp=_hp(state),
            max_hp=_max_hp(state),
            entry_state=_copy_json(state),
        )
        self._active = active

        save = self._write_entry_save(process, directory)
        snapshot = {
            "type": "combat_entry_snapshot",
            "version": 1,
            "combat_id": combat_id,
            "run": self._run_metadata(),
            "entry": {
                "step": step,
                "decision": state.get("decision"),
                "act": act,
                "floor": floor,
                "room_type": room_type,
                "hp": active.entry_hp,
                "max_hp": active.max_hp,
                "state": compact_state(state),
                "enemies": _copy_json(state.get("enemies") or (state.get("combat") or {}).get("enemies") or []),
                "action_tape_length": len(action_tape),
                "prefix_commands": _copy_json(action_tape),
            },
            "save": save,
        }
        _write_json(directory / "entry_snapshot.json", snapshot)
        _append_jsonl(
            directory / "room.jsonl",
            {
                "type": "combat_entry",
                "combat_id": combat_id,
                "run": self._run_metadata(),
                "step": step,
                "decision": state.get("decision"),
                "act": act,
                "floor": floor,
                "room_type": room_type,
                "entry_hp": active.entry_hp,
                "max_hp": active.max_hp,
            },
        )

    def _finish_combat(self, state: dict[str, Any], *, reason: str = "combat_exit") -> None:
        active = self._active
        if active is None:
            return
        exit_hp = _hp(state)
        hp_loss = active.entry_hp - exit_hp if active.entry_hp is not None and exit_hp is not None else None
        terminal_death = state.get("decision") == "game_over" and state.get("victory") is not True
        row = {
            "combat_id": active.combat_id,
            "run_id": self.run_id,
            "condition": self.condition,
            "seed": self.seed,
            "character": self.character,
            "ascension": self.ascension,
            "lang": self.lang,
            "act": active.act,
            "floor": active.floor,
            "room_type": active.room_type,
            "entry_step": active.step,
            "entry_decision": active.decision,
            "entry_hp": active.entry_hp,
            "max_hp": active.max_hp,
            "exit_decision": state.get("decision"),
            "exit_hp": exit_hp,
            "hp_loss": hp_loss,
            "terminal_death": terminal_death,
            "reason": reason,
            "directory": str(active.directory),
            "entry_snapshot": str(active.directory / "entry_snapshot.json"),
            "room_log": str(active.directory / "room.jsonl"),
            "commands_log": str(active.directory / "commands.jsonl"),
        }
        _append_jsonl(
            active.directory / "room.jsonl",
            {
                "type": "combat_exit",
                "combat_id": active.combat_id,
                "step": None,
                "decision": state.get("decision"),
                "exit_hp": exit_hp,
                "hp_loss": hp_loss,
                "terminal_death": terminal_death,
                "reason": reason,
            },
        )
        self._index["combats"].append(row)
        _write_json(self._index_path, self._index)
        self._active = None

    def _write_entry_save(self, process: Any, directory: Path) -> dict[str, Any]:
        save_path = directory / "entry.save"
        if not self.save_entries or process is None:
            return {"attempted": False, "success": False, "path": str(save_path)}
        try:
            result = process.send({"cmd": "write_continue_save", "path": str(save_path)})
        except Exception as exc:  # noqa: BLE001 - recorder must not break a run.
            return {"attempted": True, "success": False, "path": str(save_path), "error": str(exc)}
        if isinstance(result, dict):
            return {
                "attempted": True,
                "success": bool(result.get("success")),
                "path": str(result.get("path") or save_path),
                "result": _copy_json(result),
            }
        return {"attempted": True, "success": False, "path": str(save_path), "result": _copy_json(result)}

    def _run_metadata(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "condition": self.condition,
            "seed": self.seed,
            "character": self.character,
            "ascension": self.ascension,
            "lang": self.lang,
        }

    def _load_index(self) -> dict[str, Any]:
        if self._index_path.is_file():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            if isinstance(data, dict) and isinstance(data.get("combats"), list):
                return data
        return {"type": "combat_record_index", "version": 1, "combats": []}

    def _initial_counter(self) -> int:
        prefix = f"{self.run_id}_combat_"
        counter = 0
        for row in self._index.get("combats") or []:
            combat_id = row.get("combat_id") if isinstance(row, dict) else None
            if not isinstance(combat_id, str) or not combat_id.startswith(prefix):
                continue
            match = re.search(r"_combat_(\d+)_", combat_id)
            if match:
                counter = max(counter, int(match.group(1)))
        return counter


def _is_combat_state(state: dict[str, Any]) -> bool:
    decision = state.get("decision")
    if decision == "card_select":
        return isinstance(state.get("combat"), dict)
    return decision in COMBAT_DECISIONS


def _context(state: dict[str, Any]) -> dict[str, Any]:
    context = state.get("context")
    return context if isinstance(context, dict) else {}


def _hp(state: dict[str, Any]) -> int | None:
    player = state.get("player")
    value = player.get("hp") if isinstance(player, dict) else state.get("hp")
    return _int_or_none(value)


def _max_hp(state: dict[str, Any]) -> int | None:
    player = state.get("player")
    value = player.get("max_hp") if isinstance(player, dict) else state.get("max_hp")
    return _int_or_none(value)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _action_summary(action: LegalAction) -> dict[str, Any]:
    return {
        "action_id": action.action_id,
        "kind": action.kind,
        "label": action.label,
    }


def _copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "combat"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
