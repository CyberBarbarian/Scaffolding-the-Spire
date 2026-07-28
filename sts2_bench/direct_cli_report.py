"""Adapt direct-CLI engine telemetry to the existing benchmark report surface."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .trace_analysis import analyze_trace


_CONTROL_COMMANDS = {"start_run", "load_save", "write_continue_save", "quit"}
_ACTION_KINDS = {
    "select_map": "map",
    "select_map_node": "map",
    "play_card": "combat_play_card",
    "end_turn": "combat_end_turn",
    "use_potion": "combat_use_potion",
    "discard_potion": "combat_discard_potion",
    "open_card_reward": "card_reward_open",
    "take_card_reward": "card_reward",
    "select_card_reward": "card_reward",
    "skip_card_reward": "card_reward_skip",
    "skip_reward": "card_reward_skip",
    "choose_event": "event",
    "rest": "rest_site",
    "upgrade_card": "card_select",
    "select_cards": "card_select",
    "remove_card": "card_select",
    "buy_card": "shop_card",
    "buy_relic": "shop_relic",
    "buy_potion": "shop_potion",
}

_SUPPORTED_DECISIONS = {
    "map_select",
    "combat_play",
    "combat_reward",
    "card_reward",
    "treasure",
    "treasure_empty",
    "bundle_select",
    "card_select",
    "shop",
    "fake_merchant_shop",
    "rest_site",
    "crystal_sphere",
    "event_result",
    "event_choice",
    "game_over",
}
_EXPLICITLY_UNSUPPORTED_DECISIONS = {"unrecognized_state", "unknown", "event_blocked"}
DIRECT_CLI_ANALYSIS_FILES = (
    "summary.json",
    "replay.md",
    "deck_and_card_rewards_zh.md",
    "hp_by_action.svg",
    "hp_by_room.svg",
    "map_path.svg",
    "route_hp_replay.svg",
)


def analyze_direct_cli_run(
    engine_trace_path: str | Path,
    benchmark_trace_path: str | Path,
    analysis_dir: str | Path,
    *,
    config: dict[str, Any],
    cli_summary: dict[str, Any],
) -> dict[str, Any]:
    """Convert one raw CLI engine trace and generate the standard trace reports."""

    source = Path(engine_trace_path)
    target = Path(benchmark_trace_path)
    engine_audit = audit_engine_trace(source, config=config, cli_summary=cli_summary)
    rows = convert_engine_trace(source, config=config, cli_summary=cli_summary)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    benchmark_audit = audit_benchmark_trace(target, engine_audit=engine_audit, expected_rows=rows)
    report = analyze_trace(target, analysis_dir)
    return {
        "engine_trace": str(source),
        "benchmark_trace": str(target),
        "analysis_dir": str(Path(analysis_dir)),
        "converted_event_count": len(rows),
        "room_count": len(report.get("rooms") or []),
        "deck_event_count": len((report.get("final_deck") or {}).get("events") or []),
        "engine_trace_sha256": _hash_file(source),
        "benchmark_trace_sha256": _hash_file(target),
        "engine_audit": engine_audit,
        "benchmark_audit": benchmark_audit,
        "analysis_artifact_manifest": build_analysis_artifact_manifest(analysis_dir),
    }


def build_analysis_artifact_manifest(analysis_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Hash the complete deterministic report surface for later validation."""

    root = Path(analysis_dir)
    manifest: dict[str, dict[str, Any]] = {}
    for name in DIRECT_CLI_ANALYSIS_FILES:
        path = root / name
        if path.is_file():
            manifest[name] = {"sha256": _hash_file(path), "bytes": path.stat().st_size}
    return manifest


def audit_engine_trace(
    path: str | Path,
    *,
    config: dict[str, Any],
    cli_summary: dict[str, Any],
) -> dict[str, Any]:
    """Validate that raw CLI telemetry proves one natural, authoritative terminal state."""

    source = Path(path)
    entries = _read_jsonl(source)
    state_errors: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    terminal_states: list[dict[str, Any]] = []
    start_run_commands: list[dict[str, Any]] = []
    terminal_entry_index: int | None = None
    post_terminal_game_actions: list[int] = []
    non_control_actions = 0
    observed_floors: list[int] = []
    observed_floors_by_act: dict[int, int] = {}
    cumulative_floor_components_by_act: dict[int, int] = {}

    for entry_index, entry in enumerate(entries):
        entry_type = entry.get("type")
        data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
        if entry_type == "state":
            state_type = str(data.get("type") or "")
            if state_type in {"error", "unsupported", "unrecognized_state"}:
                state_errors.append({"entry_index": entry_index, "state": data})
            if state_type != "decision":
                continue
            decision = str(data.get("decision") or "")
            decisions.append(data)
            context = data.get("context") if isinstance(data.get("context"), dict) else {}
            observed_floor: int | None = None
            for candidate in (
                context.get("display_floor"),
                context.get("map_floor"),
                context.get("floor"),
                data.get("floor"),
            ):
                if isinstance(candidate, int) and not isinstance(candidate, bool):
                    observed_floor = candidate
                    break
            if observed_floor is not None:
                observed_floors.append(observed_floor)
                act = context.get("act")
                if not isinstance(act, int) or isinstance(act, bool) or act < 1:
                    act = 1
                observed_floors_by_act[act] = max(
                    observed_floors_by_act.get(act, observed_floor),
                    observed_floor,
                )
                if act == 1:
                    cumulative_component = observed_floor
                else:
                    engine_floor = context.get("engine_floor")
                    if isinstance(engine_floor, int) and not isinstance(engine_floor, bool):
                        cumulative_component = engine_floor
                    else:
                        # Later acts expose their first entered room as visible
                        # floor 0, so display_floor + 1 is the room count.
                        cumulative_component = observed_floor + 1
                cumulative_floor_components_by_act[act] = max(
                    cumulative_floor_components_by_act.get(act, cumulative_component),
                    cumulative_component,
                )
            if decision in _EXPLICITLY_UNSUPPORTED_DECISIONS or decision not in _SUPPORTED_DECISIONS:
                unsupported.append({"entry_index": entry_index, "decision": decision, "state": data})
            if decision == "game_over":
                terminal_states.append(data)
                terminal_entry_index = entry_index
            continue
        if entry_type == "action" and terminal_entry_index is not None:
            command = str(data.get("cmd") or "")
            if command not in _CONTROL_COMMANDS:
                post_terminal_game_actions.append(entry_index)
        if entry_type == "action" and data.get("cmd") == "start_run":
            start_run_commands.append(data)
        if entry_type == "action" and str(data.get("cmd") or "") not in _CONTROL_COMMANDS:
            non_control_actions += 1

    termination_class = str(cli_summary.get("termination_class") or "")
    natural_terminal = termination_class == "natural_game_over"
    behavior_terminal = termination_class in {
        "model_forfeit",
        "decision_budget_exhausted",
        "model_protocol_error",
    }
    terminal_state = terminal_states[-1] if terminal_states else (decisions[-1] if behavior_terminal and decisions else {})
    last_decision = decisions[-1] if decisions else {}
    player = terminal_state.get("player") if isinstance(terminal_state.get("player"), dict) else {}
    deck = player.get("deck") if isinstance(player.get("deck"), list) else None
    deck_size = player.get("deck_size")
    deck_complete = (
        isinstance(deck, list)
        and isinstance(deck_size, int)
        and not isinstance(deck_size, bool)
        and deck_size == len(deck)
        and all(isinstance(card, dict) and bool(card.get("id") or card.get("name")) for card in deck)
    )
    expected_victory = cli_summary.get("outcome") == "victory"
    outcome_matches = (
        bool(terminal_state) and bool(terminal_state.get("victory")) == expected_victory
        if natural_terminal
        else behavior_terminal and cli_summary.get("outcome") in {
            "forfeit",
            "decision_budget_exhausted",
            "model_protocol_error",
        }
    )
    if natural_terminal:
        completion_matches = (
            cli_summary.get("completed") is True
            and cli_summary.get("error") is None
            and cli_summary.get("decision_limit_reached") is False
            and cli_summary.get("outcome") in {"victory", "defeat"}
        )
    else:
        completion_matches = (
            behavior_terminal
            and cli_summary.get("completed") is False
            and (termination_class == "decision_budget_exhausted")
            == (cli_summary.get("decision_limit_reached") is True)
        )
    checks = {
        "trace_nonempty": bool(entries),
        "exactly_one_start_run": len(start_run_commands) == 1,
        "start_run_matches_config": len(start_run_commands) == 1
        and start_run_commands[0].get("seed") == config.get("seed")
        and start_run_commands[0].get("character") == config.get("character")
        and start_run_commands[0].get("ascension") == config.get("ascension")
        and start_run_commands[0].get("lang") == config.get("lang"),
        "has_decisions": bool(decisions),
        "no_engine_error_states": not state_errors,
        "no_unsupported_decisions": not unsupported,
        "termination_class_supported": natural_terminal or behavior_terminal,
        "game_over_count_matches_termination": len(terminal_states) == (1 if natural_terminal else 0),
        "last_decision_matches_termination": bool(last_decision)
        and ((last_decision.get("decision") == "game_over") == natural_terminal),
        "no_game_actions_after_terminal": not post_terminal_game_actions,
        "terminal_player_present": bool(player),
        "authoritative_terminal_deck_complete": deck_complete,
        "terminal_outcome_matches_summary": outcome_matches,
        "summary_termination_matches": completion_matches,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "engine_trace_sha256": _hash_file(source),
        "entry_count": len(entries),
        "decision_count": len(decisions),
        "non_control_action_count": non_control_actions,
        # STS2 restarts the visible floor counter at each act.  Act 1's
        # display floor already equals the number of entered map rooms because
        # Neow occupies engine floor 1.  Later acts start their first entered
        # room at display floor 0, so their engine floor is the room count.
        # Summing these per-act components yields cumulative run progress.
        "max_observed_floor": max(observed_floors) if observed_floors else None,
        "max_observed_floor_by_act": {
            str(act): observed_floors_by_act[act]
            for act in sorted(observed_floors_by_act)
        },
        "cumulative_floor_components_by_act": {
            str(act): cumulative_floor_components_by_act[act]
            for act in sorted(cumulative_floor_components_by_act)
        },
        "cumulative_observed_floor": (
            sum(cumulative_floor_components_by_act.values())
            if cumulative_floor_components_by_act
            else None
        ),
        "termination_class": termination_class,
        "start_run_commands": start_run_commands,
        "state_errors": state_errors,
        "unsupported_decisions": unsupported,
        "post_terminal_game_action_entries": post_terminal_game_actions,
        "authoritative_terminal_state": terminal_state or None,
        "authoritative_terminal_state_sha256": _hash_json(terminal_state) if terminal_state else None,
        "authoritative_terminal_deck": deck,
        "authoritative_terminal_deck_sha256": _hash_json(deck) if isinstance(deck, list) else None,
    }


def audit_benchmark_trace(
    path: str | Path,
    *,
    engine_audit: dict[str, Any],
    expected_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind the converted trace terminal row to the audited raw engine terminal state."""

    target = Path(path)
    rows = _read_jsonl(target)
    result_rows = [row for row in rows if row.get("type") == "result"]
    unsupported_rows = [
        {"row_index": index, "row": row}
        for index, row in enumerate(rows)
        if row.get("type") in {"error", "decision_error", "runner_error", "unsupported"}
        or (
            row.get("type") == "decision"
            and (
                str(row.get("decision") or "") in _EXPLICITLY_UNSUPPORTED_DECISIONS
                or str(row.get("decision") or "") not in _SUPPORTED_DECISIONS
            )
        )
    ]
    result = result_rows[-1].get("result") if result_rows and isinstance(result_rows[-1].get("result"), dict) else {}
    extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
    terminal_state = extra.get("authoritative_terminal_state")
    expected_terminal = engine_audit.get("authoritative_terminal_state")
    termination_class = str(engine_audit.get("termination_class") or "")
    natural_terminal = termination_class == "natural_game_over"
    behavior_terminal = termination_class in {
        "model_forfeit",
        "decision_budget_exhausted",
        "model_protocol_error",
    }
    decision_rows = [row for row in rows if row.get("type") == "decision"]
    action_rows = [row for row in rows if row.get("type") == "action"]
    result_termination_valid = result.get("termination_class") == termination_class and (
        (
            natural_terminal
            and result.get("decision") == "game_over"
            and result.get("truncated") is False
            and result.get("error") is None
        )
        or (
            behavior_terminal
            and result.get("decision") != "game_over"
            and (result.get("truncated") is True) == (termination_class == "decision_budget_exhausted")
        )
    )
    checks = {
        "trace_nonempty": bool(rows),
        "exactly_one_result": len(result_rows) == 1,
        "result_is_last_row": bool(rows) and rows[-1].get("type") == "result",
        "no_error_or_unsupported_rows": not unsupported_rows,
        "result_termination_matches_engine": result_termination_valid,
        "decision_count_matches_engine": len(decision_rows) == engine_audit.get("decision_count"),
        "action_count_matches_engine": len(action_rows) == engine_audit.get("non_control_action_count"),
        "rows_match_deterministic_conversion": expected_rows is None or rows == expected_rows,
        "authoritative_terminal_state_matches_engine": isinstance(terminal_state, dict)
        and terminal_state == expected_terminal,
        "authoritative_terminal_hash_matches_engine": extra.get("authoritative_terminal_state_sha256")
        == engine_audit.get("authoritative_terminal_state_sha256"),
        "authoritative_deck_hash_matches_engine": extra.get("authoritative_terminal_deck_sha256")
        == engine_audit.get("authoritative_terminal_deck_sha256"),
    }
    return {
        "valid": bool(engine_audit.get("valid")) and all(checks.values()),
        "checks": checks,
        "benchmark_trace_sha256": _hash_file(target),
        "row_count": len(rows),
        "unsupported_rows": unsupported_rows,
        "result": result,
    }


def convert_engine_trace(
    path: str | Path,
    *,
    config: dict[str, Any],
    cli_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert ``GameLogger`` state/action rows into benchmark trace events."""

    entries = _read_jsonl(Path(path))
    rows: list[dict[str, Any]] = [
        {
            "type": "start",
            "seed": config.get("seed"),
            "character": config.get("character"),
            "ascension": config.get("ascension"),
            "protocol": config.get("protocol"),
        }
    ]
    last_decision: dict[str, Any] | None = None
    last_decision_row: dict[str, Any] | None = None
    game_action_count = 0
    raw_last_decision: dict[str, Any] | None = None
    raw_terminal_state: dict[str, Any] | None = None
    for entry in entries:
        data = entry.get("data")
        if entry.get("type") != "state" or not isinstance(data, dict) or data.get("type") != "decision":
            continue
        raw_last_decision = copy.deepcopy(data)
        if data.get("decision") == "game_over":
            raw_terminal_state = copy.deepcopy(data)

    for index, entry in enumerate(entries):
        entry_type = entry.get("type")
        data = entry.get("data") if isinstance(entry.get("data"), dict) else {}
        if entry_type == "state":
            if data.get("type") == "decision" and data.get("decision"):
                last_decision = data
                last_decision_row = {
                    "type": "decision",
                    "step": entry.get("step"),
                    "decision": data.get("decision"),
                    "state": data,
                    "legal_actions": _legal_actions(data),
                }
                rows.append(last_decision_row)
            elif data.get("type") == "map" and last_decision_row is not None:
                last_decision_row["map_snapshot"] = data
            continue
        if entry_type != "action" or last_decision is None:
            continue

        command = str(data.get("cmd") or "")
        if command in _CONTROL_COMMANDS:
            continue
        following_state = _next_decision_state(entries, index + 1)
        last_result = following_state.get("last_action_result")
        if not isinstance(last_result, dict):
            last_result = {}
        kind, label = _action_identity(data, last_decision, following_state, last_result)
        if following_state and not kind.startswith("view_"):
            synthesized_result = dict(last_result)
            synthesized_result.setdefault("action_kind", kind)
            synthesized_result.setdefault("action_label", label)
            if not synthesized_result.get("changes"):
                deck_changes = _deck_changes(last_decision, following_state)
                if deck_changes:
                    synthesized_result["changes"] = deck_changes
            following_state["last_action_result"] = synthesized_result
        game_action_count += 0 if kind.startswith("view_") else 1
        rows.append(
            {
                "type": "action",
                "step": entry.get("step"),
                "decision": last_decision.get("decision"),
                "action": {
                    "kind": kind,
                    "label": label,
                    "cmd": data,
                },
            }
        )

    # Report-only synthesis above intentionally annotates converted decision
    # rows, but the result closure must use an untouched engine snapshot.
    terminal_state = raw_terminal_state or raw_last_decision or {}
    context = terminal_state.get("context") if isinstance(terminal_state.get("context"), dict) else {}
    player = terminal_state.get("player") if isinstance(terminal_state.get("player"), dict) else {}
    rows.append(
        {
            "type": "result",
            "result": {
                "seed": config.get("seed"),
                "character": config.get("character"),
                "ascension": config.get("ascension"),
                "victory": cli_summary.get("outcome") == "victory",
                "decision": terminal_state.get("decision"),
                "act": context.get("act") or terminal_state.get("act"),
                "floor": context.get("floor") or terminal_state.get("floor"),
                "room_type": context.get("room_type"),
                "hp": player.get("hp"),
                "max_hp": player.get("max_hp"),
                "gold": player.get("gold"),
                "deck_size": player.get("deck_size"),
                "game_action_count": game_action_count,
                "truncated": bool(cli_summary.get("decision_limit_reached")),
                "error": cli_summary.get("error"),
                "termination_class": cli_summary.get("termination_class"),
                "extra": {
                    "authoritative_terminal_state": terminal_state,
                    "authoritative_terminal_state_sha256": _hash_json(terminal_state),
                    "authoritative_terminal_deck": player.get("deck"),
                    "authoritative_terminal_deck_sha256": _hash_json(player.get("deck")),
                },
            },
        }
    )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{line_no}")
        rows.append(row)
    return rows


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _next_decision_state(entries: list[dict[str, Any]], start: int) -> dict[str, Any]:
    for entry in entries[start:]:
        if entry.get("type") != "state":
            continue
        data = entry.get("data")
        if isinstance(data, dict) and data.get("type") == "decision":
            return data
    return {}


def _action_identity(
    command: dict[str, Any],
    state: dict[str, Any],
    following_state: dict[str, Any],
    last_result: dict[str, Any],
) -> tuple[str, str]:
    if command.get("cmd") == "get_map":
        return "view_map", "view map and current position"
    raw_action = str(command.get("action") or command.get("cmd") or "unknown")
    kind = str(last_result.get("action_kind") or _action_kind(raw_action, state, command))
    label = str(last_result.get("action_label") or _action_label(raw_action, state, following_state, command))
    return kind, label


def _action_kind(raw_action: str, state: dict[str, Any], command: dict[str, Any]) -> str:
    decision = str(state.get("decision") or "")
    if decision == "fake_merchant_shop":
        return {
            "buy_relic": "fake_merchant_relic",
            "use_potion": "fake_merchant_foul_potion",
            "leave_room": "fake_merchant_leave",
        }.get(raw_action, raw_action)
    if raw_action == "choose_option":
        return "rest_site" if decision == "rest_site" else "event"
    if raw_action == "claim_reward":
        reward = _indexed_item(state.get("rewards"), _arg_int(command, "reward_index"))
        if isinstance(reward, dict) and reward.get("kind") == "card_reward":
            return "card_reward_open"
        return f"combat_reward_{(reward or {}).get('kind') or 'claim'}"
    if raw_action == "claim_relic":
        return "treasure_relic"
    if raw_action == "leave_room":
        return "leave_room"
    return _ACTION_KINDS.get(raw_action) or raw_action


def _action_label(
    raw_action: str,
    state: dict[str, Any],
    following_state: dict[str, Any],
    command: dict[str, Any],
) -> str:
    if raw_action == "select_map_node":
        args = command.get("args") if isinstance(command.get("args"), dict) else {}
        col, row = args.get("col"), args.get("row")
        selected = next(
            (
                item
                for item in state.get("choices") or []
                if isinstance(item, dict) and item.get("col") == col and item.get("row") == row
            ),
            {},
        )
        room_type = selected.get("type") or _context(following_state).get("room_type") or "room"
        return f"go to {room_type} at col={col} row={row}"
    if raw_action in {"select_card_reward", "take_card_reward"}:
        index = _arg_int(command, "card_index")
        card = _indexed_item(state.get("cards"), index)
        return f"pick card {index}: {_item_name(card, fallback='unknown card')}"
    if raw_action == "select_cards":
        args = command.get("args") if isinstance(command.get("args"), dict) else {}
        raw_indexes = str(args.get("indices") or "")
        indexes = [int(item) for item in raw_indexes.split(",") if item.strip().isdigit()]
        names = [_item_name(_indexed_item(state.get("cards"), index), fallback=f"card {index}") for index in indexes]
        return f"select cards {raw_indexes}: {' + '.join(names)}"
    if raw_action == "choose_option":
        index = _arg_int(command, "option_index")
        option = _indexed_item(state.get("options"), index)
        title = _item_name(option, fallback=f"option {index}")
        return title if state.get("decision") == "rest_site" else f"choose event option {index}: {title}"
    if raw_action == "claim_reward":
        index = _arg_int(command, "reward_index")
        reward = _indexed_item(state.get("rewards"), index)
        kind = (reward or {}).get("kind") if isinstance(reward, dict) else None
        detail = (reward or {}).get("amount") if isinstance(reward, dict) else None
        suffix = f" {detail}" if detail is not None else ""
        return f"claim {kind or 'reward'}{suffix}"
    if raw_action == "claim_relic":
        index = _arg_int(command, "relic_index")
        relic = _indexed_item(state.get("relics"), index)
        return f"claim relic {index}: {_item_name(relic, fallback='unknown relic')}"
    if state.get("decision") == "fake_merchant_shop" and raw_action == "buy_relic":
        index = _arg_int(command, "relic_index")
        relic = _indexed_item(state.get("relics"), index)
        return f"buy fake merchant relic {index}: {_item_name(relic, fallback='unknown relic')}"
    if state.get("decision") == "fake_merchant_shop" and raw_action == "use_potion":
        return "throw foul potion at fake merchant"
    if state.get("decision") == "fake_merchant_shop" and raw_action == "leave_room":
        return "leave fake merchant"
    return _fallback_action_label(command)


def _legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    decision = str(state.get("decision") or "")
    actions: list[dict[str, Any]] = []
    if decision == "map_select":
        for item in state.get("choices") or []:
            if not isinstance(item, dict):
                continue
            col, row = item.get("col"), item.get("row")
            actions.append(
                {
                    "kind": "map",
                    "label": f"go to {item.get('type') or 'room'} at col={col} row={row}",
                    "cmd": {"cmd": "action", "action": "select_map_node", "args": {"col": col, "row": row}},
                }
            )
    elif decision == "card_reward":
        for index, item in enumerate(state.get("cards") or []):
            if not isinstance(item, dict):
                continue
            card_index = item.get("index", index)
            actions.append(
                {
                    "kind": "card_reward",
                    "label": f"pick card {card_index}: {_item_name(item, fallback='unknown card')}",
                    "cmd": {
                        "cmd": "action",
                        "action": "select_card_reward",
                        "args": {"card_index": card_index},
                    },
                }
            )
        if state.get("can_skip"):
            actions.append({"kind": "card_reward_skip", "label": "skip card reward"})
    elif decision == "fake_merchant_shop":
        for relic in state.get("relics") or []:
            if not isinstance(relic, dict) or not relic.get("is_stocked") or not relic.get("can_buy"):
                continue
            index = relic.get("index")
            actions.append(
                {
                    "kind": "fake_merchant_relic",
                    "label": f"buy fake merchant relic {index}: {_item_name(relic, fallback='unknown relic')}",
                    "cmd": {"cmd": "action", "action": "buy_relic", "args": {"relic_index": index}},
                }
            )
        foul_action = state.get("foul_potion_action")
        if state.get("can_throw_foul_potion") and isinstance(foul_action, dict):
            if foul_action.get("action") and foul_action.get("potion_index") is not None:
                actions.append(
                    {
                        "kind": "fake_merchant_foul_potion",
                        "label": "throw foul potion at fake merchant",
                        "cmd": {
                            "cmd": "action",
                            "action": foul_action["action"],
                            "args": {"potion_index": foul_action["potion_index"]},
                        },
                    }
                )
        if state.get("can_leave"):
            actions.append(
                {
                    "kind": "fake_merchant_leave",
                    "label": "leave fake merchant",
                    "cmd": {"cmd": "action", "action": "leave_room"},
                }
            )
    return actions


def _deck_changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    before_deck = Counter(_deck_cards(before))
    after_deck = Counter(_deck_cards(after))
    changes: list[str] = []
    for name, count in sorted((before_deck - after_deck).items()):
        changes.extend([f"deck: -{name}"] * count)
    for name, count in sorted((after_deck - before_deck).items()):
        changes.extend([f"deck: +{name}"] * count)
    return changes


def _deck_cards(state: dict[str, Any]) -> list[str]:
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    cards = player.get("deck") if isinstance(player.get("deck"), list) else []
    result: list[str] = []
    for card in cards:
        if not isinstance(card, dict) or not card.get("name"):
            continue
        result.append(f"{card['name']}{'+' if card.get('upgraded') else ''}")
    return result


def _indexed_item(items: Any, index: int | None) -> dict[str, Any] | None:
    if not isinstance(items, list) or index is None:
        return None
    for position, item in enumerate(items):
        if isinstance(item, dict) and item.get("index", position) == index:
            return item
    return None


def _arg_int(command: dict[str, Any], name: str) -> int | None:
    args = command.get("args") if isinstance(command.get("args"), dict) else {}
    value = args.get(name)
    return value if isinstance(value, int) else None


def _item_name(item: dict[str, Any] | None, *, fallback: str) -> str:
    if not isinstance(item, dict):
        return fallback
    return str(item.get("title") or item.get("name") or item.get("kind") or fallback)


def _context(state: dict[str, Any]) -> dict[str, Any]:
    return state.get("context") if isinstance(state.get("context"), dict) else {}


def _fallback_action_label(command: dict[str, Any]) -> str:
    action = command.get("action") or command.get("cmd") or "unknown"
    args = command.get("args")
    if isinstance(args, dict) and args:
        return f"{action} {json.dumps(args, ensure_ascii=False, sort_keys=True)}"
    return str(action)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate standard reports from a direct-CLI engine trace.")
    parser.add_argument("engine_trace", help="Input GameLogger JSONL trace")
    parser.add_argument("--config", required=True, help="Direct-CLI config.json")
    parser.add_argument("--summary", required=True, help="Direct-CLI summary.json")
    parser.add_argument("--benchmark-trace", required=True, help="Converted benchmark JSONL output")
    parser.add_argument("--out-dir", required=True, help="Standard trace-analysis output directory")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cli_summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    artifacts = analyze_direct_cli_run(
        args.engine_trace,
        args.benchmark_trace,
        args.out_dir,
        config=config,
        cli_summary=cli_summary,
    )
    print(json.dumps(artifacts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
