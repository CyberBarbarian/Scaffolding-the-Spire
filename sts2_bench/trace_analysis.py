"""Offline trace analysis for STS2 benchmark JSONL logs."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .context import MAP_ICONS


LOW_HP_RATIO = 0.60
LARGE_LOSS_HP = 20
LARGE_LOSS_RATIO = 0.25
ROUTE_PLAN_STAGE_COLORS = ["#2166ac", "#b2182b", "#1b9e77", "#7570b3", "#e6ab02"]
TRACE_ANALYSIS_MARKER = ".sts2-trace-analysis-root"
TRACE_ANALYSIS_MARKER_CONTENT = "sts2_trace_analysis_root_v1\n"


def _prepare_trace_analysis_dir(out_path: Path) -> None:
    if out_path.exists():
        if not out_path.is_dir():
            raise FileExistsError(f"Trace analysis output exists and is not a directory: {out_path}")
        entries = list(out_path.iterdir())
        if entries:
            marker = out_path / TRACE_ANALYSIS_MARKER
            try:
                marker_content = marker.read_text(encoding="utf-8")
            except OSError as exc:
                raise FileExistsError(
                    f"Refusing to write trace analysis into a non-empty unowned directory: {out_path}"
                ) from exc
            if marker_content != TRACE_ANALYSIS_MARKER_CONTENT:
                raise FileExistsError(
                    f"Refusing to write trace analysis into a non-empty unowned directory: {out_path}"
                )
    else:
        out_path.mkdir(parents=True, exist_ok=False)
    (out_path / TRACE_ANALYSIS_MARKER).write_text(
        TRACE_ANALYSIS_MARKER_CONTENT,
        encoding="utf-8",
    )


def analyze_trace(
    trace_path: str | Path,
    out_dir: str | Path,
    *,
    low_hp_ratio: float = LOW_HP_RATIO,
    large_loss_hp: int = LARGE_LOSS_HP,
    large_loss_ratio: float = LARGE_LOSS_RATIO,
) -> dict[str, Any]:
    """Analyze one benchmark JSONL trace and write replay artifacts."""

    trace = _load_trace(Path(trace_path))
    out_path = Path(out_dir)
    _prepare_trace_analysis_dir(out_path)

    report = _build_report(
        trace,
        low_hp_ratio=low_hp_ratio,
        large_loss_hp=large_loss_hp,
        large_loss_ratio=large_loss_ratio,
    )
    _write_large_loss_rooms(trace, report, out_path)
    _write_hp_by_action_svg(report, out_path / "hp_by_action.svg")
    _write_hp_by_room_svg(report, out_path / "hp_by_room.svg")
    _write_map_path_svg(report, out_path / "map_path.svg")
    _write_route_hp_replay_svg(report, out_path / "route_hp_replay.svg")
    _write_replay_markdown(report, out_path / "replay.md")
    _write_deck_and_card_rewards_markdown(report, out_path / "deck_and_card_rewards_zh.md")
    (out_path / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def analyze_benchmark_trace(
    trace_path: str | Path,
    out_dir: str | Path,
    *,
    low_hp_ratio: float = LOW_HP_RATIO,
    large_loss_hp: int = LARGE_LOSS_HP,
    large_loss_ratio: float = LARGE_LOSS_RATIO,
) -> dict[str, Any]:
    """Analyze a benchmark trace, splitting multi-run JSONL logs per seed."""

    source = Path(trace_path)
    out_path = Path(out_dir)
    _prepare_trace_analysis_dir(out_path)

    run_paths = _write_split_run_traces(source, out_path)
    if not run_paths:
        report = analyze_trace(
            source,
            out_path,
            low_hp_ratio=low_hp_ratio,
            large_loss_hp=large_loss_hp,
            large_loss_ratio=large_loss_ratio,
        )
        index = {
            "source": str(source),
            "out_dir": str(out_path),
            "run_count": 1,
            "runs": [
                {
                    "seed": (report.get("run") or {}).get("seed"),
                    "trace": str(source),
                    "out_dir": str(out_path),
                    "summary": report.get("run"),
                }
            ],
        }
        (out_path / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return index

    runs = []
    for run_path in run_paths:
        run_out = out_path / run_path.stem
        report = analyze_trace(
            run_path,
            run_out,
            low_hp_ratio=low_hp_ratio,
            large_loss_hp=large_loss_hp,
            large_loss_ratio=large_loss_ratio,
        )
        runs.append(
            {
                "seed": (report.get("run") or {}).get("seed"),
                "trace": str(run_path),
                "out_dir": str(run_out),
                "summary": report.get("run"),
            }
        )

    index = {
        "source": str(source),
        "out_dir": str(out_path),
        "run_count": len(runs),
        "runs": runs,
    }
    (out_path / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


def _write_split_run_traces(source: Path, out_dir: Path) -> list[Path]:
    run_config_lines: list[str] = []
    current_lines: list[str] = []
    run_paths: list[Path] = []
    saw_run_start = False
    run_index = 0

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        obj = json.loads(raw_line)
        event_type = obj.get("type")
        if event_type == "run_config" and not current_lines:
            run_config_lines.append(raw_line)
            continue
        if event_type == "start":
            saw_run_start = True
            current_lines = [*run_config_lines, raw_line]
            continue
        if not current_lines:
            continue
        current_lines.append(raw_line)
        if event_type == "result":
            seed = ((obj.get("result") or {}).get("seed")) or f"run_{run_index:04d}"
            run_path = out_dir / f"{run_index:02d}_{_safe_name(seed)}.jsonl"
            run_path.write_text("\n".join(current_lines) + "\n", encoding="utf-8")
            run_paths.append(run_path)
            current_lines = []
            run_index += 1

    if len(run_paths) <= 1 and not saw_run_start:
        for path in run_paths:
            path.unlink(missing_ok=True)
        return []
    return run_paths


def _load_trace(path: Path) -> dict[str, Any]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    last_decision: dict[str, Any] | None = None
    start: dict[str, Any] | None = None
    result: dict[str, Any] | None = None

    for index, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        obj = json.loads(raw_line)
        typed = {"line_no": index, "obj": obj}
        entries.append(typed)
        event_type = obj.get("type")
        if event_type == "start":
            start = obj
        elif event_type == "decision":
            authoritative_state = obj.get("authoritative_state")
            logged_state = authoritative_state if isinstance(authoritative_state, dict) else obj.get("state") or {}
            last_decision = {**obj, "state": logged_state}
            decisions.append({"line_no": index, "obj": obj, "state": logged_state})
        elif event_type == "action":
            action = obj.get("action") or {}
            record = {
                "line_no": index,
                "obj": obj,
                "step": obj.get("step"),
                "decision": obj.get("decision"),
                "action": action,
                "label": action.get("label"),
                "kind": action.get("kind"),
                "agent_meta": obj.get("agent_meta") or {},
                "before_decision": last_decision,
                "before_state": (last_decision or {}).get("state") or {},
                "legal_actions": (last_decision or {}).get("legal_actions") or [],
                "logged_after_state": obj.get("state_after") if isinstance(obj.get("state_after"), dict) else None,
            }
            actions.append(record)
        elif event_type == "result":
            result = obj.get("result") or {}

    for action in actions:
        action["after_state"] = (
            action.get("logged_after_state")
            or _next_state_after(decisions, action["line_no"])
            or _result_state(result)
        )

    return {
        "path": path,
        "raw_lines": raw_lines,
        "entries": entries,
        "decisions": decisions,
        "actions": actions,
        "start": start or {},
        "result": result or {},
    }


def _next_state_after(decisions: list[dict[str, Any]], line_no: int) -> dict[str, Any] | None:
    for decision in decisions:
        if decision["line_no"] > line_no:
            return decision["state"]
    return None


def _result_state(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
    for key in ("authoritative_terminal_state", "last_observed_state"):
        state = extra.get(key)
        if isinstance(state, dict):
            return state
    return {
        "decision": result.get("decision"),
        "act": result.get("act"),
        "floor": result.get("floor"),
        "context": {
            "act": result.get("act"),
            "floor": result.get("floor"),
            "room_type": result.get("room_type"),
        },
        "player": {
            "hp": result.get("hp"),
            "max_hp": result.get("max_hp"),
            "gold": result.get("gold"),
            "deck_size": result.get("deck_size"),
        },
    }


def _build_report(
    trace: dict[str, Any],
    *,
    low_hp_ratio: float,
    large_loss_hp: int,
    large_loss_ratio: float,
) -> dict[str, Any]:
    actions = trace["actions"]
    rooms = _extract_rooms(actions)
    map_choices = _extract_map_choices(actions)
    card_rewards = _extract_card_rewards(actions)
    bundle_selections = _extract_bundle_selections(actions)
    rest_sites = _extract_rest_sites(actions)
    shops = _extract_shops(actions)
    start = trace.get("start") or {}
    result = trace.get("result") or {}
    final_deck = _extract_final_deck(actions, result, start)
    terminal_state = _result_state(result)
    final_inventory = _terminal_inventory(result, terminal_state)
    route_plan_events = _extract_route_plan_events(trace["entries"])
    route_plan_validation_errors = _extract_route_plan_validation_errors(trace["entries"])
    high_risk_events = _extract_high_risk_events(map_choices, low_hp_ratio=low_hp_ratio)
    hp_timeline = _extract_hp_timeline(actions, rooms)

    for room in rooms:
        max_hp = room.get("entry_max_hp") or room.get("exit_max_hp") or 0
        threshold = max(large_loss_hp, int(max_hp * large_loss_ratio)) if max_hp else large_loss_hp
        room["large_loss"] = bool(room.get("hp_loss", 0) >= threshold or room.get("terminal_death"))
        room["large_loss_threshold"] = threshold

    return {
        "source": str(trace["path"]),
        "run": {
            "seed": result.get("seed") or start.get("seed"),
            "character": result.get("character") or start.get("character"),
            "ascension": result.get("ascension") if "ascension" in result else start.get("ascension"),
            "victory": result.get("victory"),
            "terminal": {
                "decision": result.get("decision"),
                "act": result.get("act"),
                "floor": result.get("floor"),
                "room_type": result.get("room_type"),
                "hp": result.get("hp"),
                "max_hp": result.get("max_hp"),
                "error": result.get("error"),
                "truncated": result.get("truncated"),
            },
        },
        "hp_timeline": hp_timeline,
        "map_choices": map_choices,
        "rooms": rooms,
        "card_rewards": card_rewards,
        "bundle_selections": bundle_selections,
        "final_deck": final_deck,
        "final_inventory": final_inventory,
        "terminal_state": terminal_state,
        "rest_sites": rest_sites,
        "shops": shops,
        "route_plan_events": route_plan_events,
        "route_plan_validation_errors": route_plan_validation_errors,
        "high_risk_events": high_risk_events,
    }


def _terminal_inventory(result: dict[str, Any], terminal_state: dict[str, Any]) -> dict[str, Any]:
    extra = result.get("extra") if isinstance(result.get("extra"), dict) else {}
    authoritative = isinstance(extra.get("authoritative_terminal_state"), dict)
    player = terminal_state.get("player") if isinstance(terminal_state.get("player"), dict) else {}
    return {
        "source": "authoritative_terminal_state" if authoritative else "result_summary",
        "deck_size": player.get("deck_size", result.get("deck_size")),
        "relics": list(player.get("relics") or []),
        "potions": list(player.get("potions") or []),
        "potion_slots": player.get("potion_slots"),
        "potion_empty_slots": player.get("potion_empty_slots"),
    }


def _extract_hp_timeline(actions: list[dict[str, Any]], rooms: list[dict[str, Any]]) -> dict[str, Any]:
    by_action = []
    game_action_index = 0
    first_state = actions[0]["before_state"] if actions else {}
    first_hp = _hp_info(first_state)
    if first_hp["hp"] is not None:
        by_action.append({"game_action_index": 0, **first_hp, **_position_info(first_state)})

    for action in actions:
        if _is_view_action(action):
            continue
        game_action_index += 1
        after_state = action.get("after_state") or {}
        hp = _hp_info(after_state)
        if hp["hp"] is None:
            continue
        by_action.append(
            {
                "game_action_index": game_action_index,
                "step": action.get("step"),
                "action": action.get("label"),
                "action_kind": action.get("kind"),
                **hp,
                **_position_info(after_state),
            }
        )

    by_room = [
        {
            "room_index": room["room_index"],
            "act": room.get("act"),
            "floor": room.get("floor"),
            "room_type": room.get("room_type"),
            "entry_hp": room.get("entry_hp"),
            "exit_hp": room.get("exit_hp"),
            "max_hp": room.get("entry_max_hp") or room.get("exit_max_hp"),
            "hp_loss": room.get("hp_loss"),
            "hp_delta": room.get("hp_delta"),
        }
        for room in rooms
    ]
    return {"by_action": by_action, "by_room": by_room}


def _extract_rooms(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rooms: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for action in actions:
        before = action.get("before_state") or {}
        after = action.get("after_state") or {}
        before_pos = _position_info(before)
        after_pos = _position_info(after)

        if action.get("kind") == "map" and after_pos.get("room_type") not in (None, "Map"):
            if current is not None:
                _close_room(current, before, action)
                rooms.append(current)
            entry_hp = _hp_info(after)
            current = {
                "room_index": len(rooms) + 1,
                "act": after_pos.get("act"),
                "floor": after_pos.get("floor"),
                "room_type": after_pos.get("room_type"),
                "node": _node_from_action(action),
                "entered_by": action.get("label"),
                "entry_step": action.get("step"),
                "entry_line": action.get("line_no"),
                "entry_hp": entry_hp["hp"],
                "entry_max_hp": entry_hp["max_hp"],
                "actions": [],
                "raw_start_line": action.get("line_no"),
            }
            continue

        if current is None:
            continue
        if before_pos.get("room_type") == current.get("room_type"):
            current["actions"].append(_action_summary(action))
        if after_pos.get("room_type") == "Map" or after.get("decision") == "game_over":
            _close_room(current, after, action)
            rooms.append(current)
            current = None

    if current is not None:
        _close_room(current, {}, None)
        rooms.append(current)
    return rooms


def _close_room(room: dict[str, Any], state: dict[str, Any], action: dict[str, Any] | None) -> None:
    exit_hp = _hp_info(state)
    room["exit_step"] = (action or {}).get("step")
    room["exit_line"] = (action or {}).get("line_no")
    room["exit_hp"] = exit_hp["hp"]
    room["exit_max_hp"] = exit_hp["max_hp"]
    if room.get("entry_hp") is not None and room.get("exit_hp") is not None:
        room["hp_delta"] = int(room["exit_hp"]) - int(room["entry_hp"])
        room["hp_loss"] = max(0, int(room["entry_hp"]) - int(room["exit_hp"]))
    else:
        room["hp_delta"] = None
        room["hp_loss"] = 0
    room["terminal_death"] = state.get("decision") == "game_over" and (exit_hp["hp"] == 0)
    room["raw_end_line"] = (action or {}).get("line_no") or room.get("raw_start_line")


def _extract_map_choices(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for action in actions:
        if action.get("decision") != "map_select":
            continue
        if action.get("kind") != "map":
            continue
        before = action.get("before_state") or {}
        before_decision = action.get("before_decision") or {}
        info = _position_info(before)
        hp = _hp_info(before)
        legal_routes = [_legal_map_summary(item) for item in action.get("legal_actions") or [] if item.get("kind") == "map"]
        selected = _legal_map_summary(action.get("action") or {})
        record = {
            "step": action.get("step"),
            "line": action.get("line_no"),
            **info,
            **hp,
            "selected": selected,
            "legal_routes": legal_routes,
            "reason": _reason(action),
        }
        map_snapshot = before_decision.get("map_snapshot")
        if isinstance(map_snapshot, dict) and map_snapshot.get("type") == "map":
            record["map_snapshot"] = map_snapshot
        result.append(record)
    return result


def _extract_route_plan_events(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for entry in entries:
        obj = entry.get("obj") or {}
        if obj.get("type") != "route_plan_event":
            continue
        event = dict(obj)
        event["line"] = entry.get("line_no")
        events.append(event)
    return events


def _extract_route_plan_validation_errors(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for entry in entries:
        obj = entry.get("obj") or {}
        if obj.get("type") != "route_plan_validation_error":
            continue
        error = dict(obj)
        error["line"] = entry.get("line_no")
        errors.append(error)
    return errors


def _extract_card_rewards(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rewards = []
    for action in actions:
        state = action.get("before_state") or {}
        unopened = _is_unopened_combat_card_reward_skip(action)
        visible_card_reward = state.get("decision") == "card_reward" or action.get("kind") in {
            "card_reward",
            "card_reward_skip",
        }
        if not visible_card_reward and not unopened:
            continue
        cards = state.get("cards") or []
        offered = [_card_name(card) for card in cards if _card_name(card)]
        if not offered:
            offered = [
                card
                for card in (_card_name_from_label(item.get("label")) for item in action.get("legal_actions") or [])
                if card
            ]
        selected_card = None if _is_card_reward_skip(action) else _card_name_from_label(action.get("label"))
        hp = _hp_info(state)
        offered_unavailable_reason = "skipped_before_opening_card_reward" if unopened and not offered else None
        rewards.append(
            {
                "step": action.get("step"),
                "act": _position_info(state).get("act"),
                "floor": _position_info(state).get("floor"),
                "room_type": _position_info(state).get("room_type"),
                **hp,
                "offered": offered,
                "offered_zh": [_card_zh_name(card) for card in offered],
                "offered_bilingual": [_card_bilingual_name(card) for card in offered],
                "selected": action.get("label"),
                "selected_card": selected_card,
                "selected_card_zh": _card_zh_name(selected_card) if selected_card else None,
                "selected_card_bilingual": _card_bilingual_name(selected_card) if selected_card else None,
                "skipped": _is_card_reward_skip(action),
                "unopened": unopened,
                "offered_unavailable_reason": offered_unavailable_reason,
            }
        )
    return rewards


def _extract_bundle_selections(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selections = []
    for action in actions:
        state = action.get("before_state") or {}
        if state.get("decision") != "bundle_select" and action.get("kind") != "bundle_select":
            continue

        command = (action.get("action") or {}).get("cmd") or (action.get("obj") or {}).get("command") or {}
        args = command.get("args") if isinstance(command.get("args"), dict) else {}
        bundle_index = args.get("bundle_index")
        if bundle_index is None:
            match = re.search(r"select bundle\s+(\d+)", str(action.get("label") or ""), flags=re.IGNORECASE)
            bundle_index = int(match.group(1)) if match else None

        selected_bundle = None
        for bundle in state.get("bundles") or state.get("options") or []:
            if not isinstance(bundle, dict):
                continue
            if bundle.get("index") == bundle_index or str(bundle.get("index")) == str(bundle_index):
                selected_bundle = bundle
                break
        engine_cards = [
            dict(card)
            for card in (selected_bundle or {}).get("cards") or []
            if isinstance(card, dict)
        ]
        cards = [card_name for card_name in (_card_name(card) for card in engine_cards) if card_name]
        if not cards:
            cards = _bundle_card_names_from_label(action.get("label"))
        selections.append(
            {
                "step": action.get("step"),
                **_position_info(state),
                **_hp_info(state),
                "bundle_index": bundle_index,
                "selected": action.get("label"),
                "cards": cards,
                "cards_zh": [_card_zh_name(card) for card in cards],
                "cards_bilingual": [_card_bilingual_name(card) for card in cards],
                "engine_cards": engine_cards,
            }
        )
    return selections


def _bundle_card_names_from_label(label: Any) -> list[str]:
    match = re.search(r"select bundle\s+\d+\s*:\s*(.+)$", str(label or ""), flags=re.IGNORECASE)
    if not match:
        return []
    return [card.strip() for card in match.group(1).split(",") if card.strip()]


def _extract_final_deck(actions: list[dict[str, Any]], result: dict[str, Any], start: dict[str, Any] | None = None) -> dict[str, Any]:
    initial_deck = _initial_deck_from_first_state(actions)
    initial_source = "trace_initial_state"
    if not initial_deck:
        initial_deck = _initial_deck_from_first_card_select(actions)
        initial_source = "trace_card_select"
    if not initial_deck:
        initial_deck = _starter_deck_for_character((result or {}).get("character") or (start or {}).get("character"))
        initial_source = "starter_deck_fallback" if initial_deck else "unavailable"
    deck = list(initial_deck)
    events: list[dict[str, Any]] = []
    warnings: list[str] = []

    for action in actions:
        changes = _deck_changes_after_action(action)
        if not changes:
            continue
        state = action.get("before_state") or {}
        event_changes = []
        for change in changes:
            card = change["name"]
            if change["operation"] == "remove":
                removed = _remove_card_from_deck(deck, card)
                if not removed:
                    warnings.append(f"Could not remove {card} at step {action.get('step')}")
            elif change["operation"] == "add":
                deck.append(card)
            event_changes.append(_localized_deck_change(change))
        events.append(
            {
                "step": action.get("step"),
                **_position_info(state),
                **_hp_info(state),
                "category": _deck_event_category(action),
                "label": action.get("label"),
                "changes": event_changes,
            }
        )

    terminal_cards = _authoritative_terminal_deck(result)
    if terminal_cards is not None:
        return _authoritative_final_deck_report(
            terminal_cards,
            initial_deck=initial_deck,
            initial_source=initial_source,
            result=result,
            events=events,
            warnings=warnings,
        )

    counts = Counter(deck)
    return {
        "source": "reconstructed_from_trace_deck_changes",
        "initial_source": initial_source,
        "initial_size": len(initial_deck),
        "size": len(deck),
        "result_deck_size": result.get("deck_size"),
        "matches_result_deck_size": result.get("deck_size") in (None, len(deck)),
        "unique_count": len(counts),
        "unmapped": sorted({card for card in deck if not _card_has_zh_mapping(card)}),
        "cards": [_localized_card(card) for card in deck],
        "counts": [
            {**_localized_card(card), "count": count}
            for card, count in sorted(counts.items(), key=lambda item: (_card_zh_name(item[0]), item[0]))
        ],
        "events": events,
        "warnings": warnings,
    }


def _authoritative_terminal_deck(result: dict[str, Any]) -> list[dict[str, Any]] | None:
    terminal = _result_state(result)
    if terminal.get("decision") != "game_over":
        return None
    player = terminal.get("player") if isinstance(terminal.get("player"), dict) else {}
    deck = player.get("deck")
    if not isinstance(deck, list) or not all(isinstance(card, dict) for card in deck):
        return None
    return [dict(card) for card in deck]


def _authoritative_final_deck_report(
    cards: list[dict[str, Any]],
    *,
    initial_deck: list[str],
    initial_source: str,
    result: dict[str, Any],
    events: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    records = [_authoritative_card_record(card) for card in cards]
    counts = Counter((record["name"], record["upgrade_level"]) for record in records)
    result_deck_size = result.get("deck_size")
    if result_deck_size not in (None, len(cards)):
        warnings.append(
            f"Authoritative terminal deck has {len(cards)} cards but result deck_size is {result_deck_size}"
        )
    count_rows = []
    for (card_name, upgrade_level), count in sorted(
        counts.items(),
        key=lambda item: (_card_zh_name(item[0][0]), item[0][0], item[0][1]),
    ):
        record = _localized_card(card_name)
        record["upgrade_level"] = upgrade_level
        record["bilingual"] = _upgraded_bilingual(record["bilingual"], upgrade_level)
        record["count"] = count
        count_rows.append(record)
    return {
        "source": "authoritative_terminal_state",
        "initial_source": initial_source,
        "initial_size": len(initial_deck),
        "size": len(cards),
        "result_deck_size": result_deck_size,
        "matches_result_deck_size": result_deck_size in (None, len(cards)),
        "unique_count": len(counts),
        "unmapped": sorted({record["name"] for record in records if not _card_has_zh_mapping(record["name"])}),
        "cards": records,
        "counts": count_rows,
        "engine_card_objects": cards,
        "events": events,
        "warnings": warnings,
    }


def _authoritative_card_record(card: dict[str, Any]) -> dict[str, Any]:
    card_name = _card_name(card) or "?"
    raw_upgrade = card.get("upgrade_level", card.get("upgraded", 0))
    try:
        upgrade_level = int(raw_upgrade or 0)
    except (TypeError, ValueError):
        upgrade_level = 0
    record = _localized_card(card_name)
    record["upgrade_level"] = upgrade_level
    record["bilingual"] = _upgraded_bilingual(record["bilingual"], upgrade_level)
    record["engine_card"] = card
    return record


def _upgraded_bilingual(label: str, upgrade_level: int) -> str:
    if upgrade_level <= 0:
        return label
    suffix = "+" if upgrade_level == 1 else f"+{upgrade_level}"
    return f"{label}{suffix}"


def _starter_deck_for_character(character: Any) -> list[str]:
    if str(character or "") == "Ironclad":
        return ["Strike"] * 5 + ["Defend"] * 4 + ["Bash"]
    return []


def _initial_deck_from_first_state(actions: list[dict[str, Any]]) -> list[str]:
    """Use the first complete engine-exported deck, independent of character."""

    for action in actions:
        state = action.get("before_state") or {}
        deck = _deck_from_state(state)
        deck_size = _deck_size_info(state)
        if deck and (deck_size is None or len(deck) == deck_size):
            return deck
        if _deck_changes_after_action(action):
            break
    return []


def _initial_deck_from_first_card_select(actions: list[dict[str, Any]]) -> list[str]:
    for action in actions:
        before = action.get("before_state") or {}
        if before.get("decision") != "card_select":
            if _deck_changes_after_action(action):
                break
            continue
        deck_size = _deck_size_info(before)
        deck = _deck_from_state(before)
        if deck and (deck_size is None or len(deck) == deck_size):
            return deck
        deck = _initial_deck_from_card_select_actions(action.get("legal_actions") or [], deck_size=deck_size)
        if deck and (deck_size is None or len(deck) == deck_size):
            return deck
        if _deck_changes_after_action(action):
            break
    return []


def _deck_from_state(state: dict[str, Any]) -> list[str]:
    player = state.get("player") if isinstance(state.get("player"), dict) else {}
    raw_deck = player.get("deck") or state.get("deck")
    if not isinstance(raw_deck, list):
        return []
    return [name for name in (_deck_card_name(card) for card in raw_deck) if name]


def _deck_card_name(card: Any) -> str | None:
    card_name = _card_name(card)
    if card_name is None or not isinstance(card, dict):
        return card_name
    raw_upgrade = card.get("upgrade_level", card.get("upgraded", 0))
    if raw_upgrade is True:
        upgrade_level = 1
    else:
        try:
            upgrade_level = int(raw_upgrade or 0)
        except (TypeError, ValueError):
            upgrade_level = 0
    if upgrade_level <= 0:
        return card_name
    return card_name + ("+" if upgrade_level == 1 else f"+{upgrade_level}")


def _initial_deck_from_card_select_actions(legal_actions: list[dict[str, Any]], *, deck_size: int | None) -> list[str]:
    indexed_names: dict[int, set[str]] = {}
    single_select_names: list[str] = []
    for item in legal_actions:
        if item.get("kind") != "card_select":
            continue
        label = item.get("label")
        cards_by_index = _card_names_by_index_from_multi_select_label(label)
        if cards_by_index:
            for index, name in cards_by_index.items():
                indexed_names.setdefault(index, set()).add(name)
            continue
        card = _card_name_from_label(label)
        if card:
            single_select_names.append(card)

    if indexed_names:
        indexes = sorted(indexed_names)
        expected_indexes = list(range(deck_size)) if deck_size is not None else list(range(indexes[-1] + 1))
        if indexes == expected_indexes and all(len(names) == 1 for names in indexed_names.values()):
            return [next(iter(indexed_names[index])) for index in indexes]

    if single_select_names and (deck_size is None or len(single_select_names) == deck_size):
        return single_select_names
    return []


def _card_names_by_index_from_multi_select_label(label: Any) -> dict[int, str]:
    if not label:
        return {}
    match = re.search(r"select cards\s+(?P<indexes>\d+(?:,\d+)*)\s*:\s*(?P<cards>.+)$", str(label))
    if not match:
        return {}
    try:
        indexes = [int(item) for item in match.group("indexes").split(",")]
    except ValueError:
        return {}
    cards = [item.strip() for item in re.split(r"\s+\+\s+", match.group("cards")) if item.strip()]
    if len(indexes) != len(cards):
        return {}
    return dict(zip(indexes, cards))


def _deck_changes_after_action(action: dict[str, Any]) -> list[dict[str, Any]]:
    after = action.get("after_state") or {}
    last_result = after.get("last_action_result") or {}
    action_label = action.get("label")
    if action_label and last_result.get("action_label") and last_result.get("action_label") != action_label:
        return []
    changes: list[dict[str, Any]] = []
    for raw_change in last_result.get("changes") or []:
        changes.extend(_deck_changes_from_text(raw_change))
    return changes


def _deck_changes_from_text(raw_change: Any) -> list[dict[str, Any]]:
    text = str(raw_change or "")
    if not text.startswith("deck:"):
        return []
    body = text[len("deck:") :].strip()
    replace = re.match(r"^-(?P<old>.+?)\s+\+(?P<new>.+)$", body)
    if replace:
        return [
            *_expand_counted_deck_change("remove", replace.group("old").strip(), text),
            *_expand_counted_deck_change("add", replace.group("new").strip(), text),
        ]
    add = re.match(r"^\+(?P<new>.+)$", body)
    if add:
        return _expand_counted_deck_change("add", add.group("new").strip(), text)
    remove = re.match(r"^-(?P<old>.+)$", body)
    if remove:
        return _expand_counted_deck_change("remove", remove.group("old").strip(), text)
    return []


def _expand_counted_deck_change(operation: str, name: str, raw: str) -> list[dict[str, Any]]:
    names = _split_multi_deck_change_names(operation, name)
    if len(names) > 1:
        changes: list[dict[str, Any]] = []
        for item in names:
            changes.extend(_expand_counted_deck_change(operation, item, raw))
        return changes
    match = re.match(r"^(?P<name>.+?)x(?P<count>[2-9]\d*)$", name)
    if match:
        name = match.group("name").strip()
        count = int(match.group("count"))
    else:
        count = 1
    return [{"operation": operation, "name": name, "raw": raw} for _ in range(count)]


def _split_multi_deck_change_names(operation: str, name: str) -> list[str]:
    marker = "+" if operation == "add" else "-" if operation == "remove" else ""
    if not marker or f" {marker}" not in name:
        return [name.strip()]
    parts = [item.strip() for item in re.split(rf"\s+\{marker}(?=[A-Za-z])", name) if item.strip()]
    return parts or [name.strip()]


def _remove_card_from_deck(deck: list[str], card: str) -> bool:
    try:
        deck.pop(deck.index(card))
        return True
    except ValueError:
        target_base, _ = _split_card_upgrade(card)
        for index, existing in enumerate(deck):
            base, _ = _split_card_upgrade(existing)
            if base == target_base:
                deck.pop(index)
                return True
    return False


def _deck_event_category(action: dict[str, Any]) -> str:
    kind = str(action.get("kind") or "")
    decision = str(action.get("decision") or "")
    room = _position_info(action.get("before_state") or {}).get("room_type")
    if kind == "card_reward":
        return "战斗奖励选牌"
    if kind == "shop_buy_card":
        return "商店购买"
    if kind == "card_select" and (decision == "rest_site" or room == "RestSite"):
        return "营火升级"
    if kind == "card_select":
        return "事件/选择改牌"
    if kind == "event":
        return "事件改牌"
    return "其他"


def _extract_rest_sites(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rest_sites = []
    pending_smith: dict[str, Any] | None = None
    for action in actions:
        if action.get("kind") == "rest_site":
            label = str(action.get("label") or "")
            state = action.get("before_state") or {}
            choice = "Smith" if "Smith" in label else "Rest" if "Rest" in label else label
            record = {
                "step": action.get("step"),
                **_position_info(state),
                **_hp_info(state),
                "choice": choice,
                "label": label,
                "smith_card": None,
            }
            rest_sites.append(record)
            pending_smith = record if choice == "Smith" else None
            continue
        if pending_smith and action.get("kind") == "card_select":
            pending_smith["smith_card"] = _selected_card_from_label(action.get("label"))
            pending_smith = None
        elif action.get("decision") != "card_select":
            pending_smith = None
    return rest_sites


def _extract_shops(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shops: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    pending_removal: dict[str, Any] | None = None

    def flush_pending_removal() -> None:
        nonlocal pending_removal
        if current is None or pending_removal is None:
            return
        current["removals"].append({"label": pending_removal["label"], "card": pending_removal.get("card")})
        pending_removal = None

    def close_current() -> None:
        nonlocal current
        if current is None:
            return
        flush_pending_removal()
        shops.append(current)
        current = None

    for action in actions:
        kind = str(action.get("kind") or "")
        before = action.get("before_state") or {}
        room = _position_info(before).get("room_type")
        in_shop_context = room == "Shop" or kind.startswith("shop_") or (pending_removal is not None and kind == "card_select")
        if not in_shop_context:
            close_current()
            continue
        if current is None:
            current = {
                "act": _position_info(before).get("act"),
                "floor": _position_info(before).get("floor"),
                "room_type": room,
                "entry_hp": _hp_info(before).get("hp"),
                "actions": [],
                "purchases": [],
                "removals": [],
            }
        label = action.get("label")
        if kind == "shop_remove_card":
            flush_pending_removal()
            current["actions"].append(label)
            pending_removal = {"label": label, "action_index": len(current["actions"]) - 1}
            continue
        if pending_removal is not None and kind == "card_select":
            card = _selected_card_from_label(label)
            pending_removal["card"] = card
            if card:
                current["actions"][pending_removal["action_index"]] = f"{pending_removal['label']} -> {card}"
            flush_pending_removal()
            continue
        if kind.startswith("shop_"):
            flush_pending_removal()
            current["actions"].append(label)
            if kind.startswith("shop_buy"):
                current["purchases"].append(label)
    close_current()
    return shops


def _extract_high_risk_events(map_choices: list[dict[str, Any]], *, low_hp_ratio: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, choice in enumerate(map_choices):
        hp_ratio = _safe_ratio(choice.get("hp"), choice.get("max_hp"))
        selected_label = ((choice.get("selected") or {}).get("label") or "")
        legal_labels = [route.get("label") or "" for route in choice.get("legal_routes") or []]
        safe_routes = [label for label in legal_labels if "RestSite" in label or "Shop" in label]

        if "Elite" in selected_label and hp_ratio is not None and hp_ratio <= low_hp_ratio:
            event_type = "low_hp_elite_selected" if len(legal_labels) > 1 else "low_hp_forced_elite"
            events.append(_risk_event(event_type, choice))
        if (
            hp_ratio is not None
            and hp_ratio <= low_hp_ratio
            and safe_routes
            and "RestSite" not in selected_label
            and "Shop" not in selected_label
        ):
            event = _risk_event("low_hp_safe_route_ignored", choice)
            event["safe_routes"] = safe_routes
            events.append(event)

        if "Elite" in selected_label and len(legal_labels) == 1 and hp_ratio is not None and hp_ratio <= low_hp_ratio:
            branch = _nearest_previous_branch(map_choices, index)
            if branch:
                event = _risk_event("low_hp_forced_elite_after_branch", choice)
                event["branch_step"] = branch.get("step")
                event["branch_hp"] = branch.get("hp")
                event["branch_max_hp"] = branch.get("max_hp")
                event["branch_selected"] = branch.get("selected")
                event["branch_legal_routes"] = branch.get("legal_routes")
                event["branch_reason"] = branch.get("reason")
                events.append(event)
    return events


def _nearest_previous_branch(map_choices: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    current = map_choices[index]
    for previous in reversed(map_choices[:index]):
        if previous.get("act") != current.get("act"):
            break
        if len(previous.get("legal_routes") or []) > 1:
            return previous
    return None


def _risk_event(event_type: str, choice: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": event_type,
        "step": choice.get("step"),
        "line": choice.get("line"),
        "act": choice.get("act"),
        "floor": choice.get("floor"),
        "hp": choice.get("hp"),
        "max_hp": choice.get("max_hp"),
        "selected": choice.get("selected"),
        "legal_routes": choice.get("legal_routes"),
        "reason": choice.get("reason"),
    }


def _write_large_loss_rooms(trace: dict[str, Any], report: dict[str, Any], out_dir: Path) -> None:
    target = out_dir / "large_loss_rooms"
    target.mkdir(exist_ok=True)
    raw_lines = trace["raw_lines"]
    for room in report["rooms"]:
        if not room.get("large_loss"):
            continue
        name = f"room_{room['room_index']:03d}_{_safe_name(room.get('room_type'))}"
        start = max(1, int(room.get("raw_start_line") or 1))
        end = max(start, int(room.get("raw_end_line") or start))
        lines = raw_lines[start - 1 : end]
        (target / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (target / f"{name}.md").write_text(_large_loss_room_md(room), encoding="utf-8")


def _large_loss_room_md(room: dict[str, Any]) -> str:
    actions = "\n".join(f"- {item.get('label')}" for item in room.get("actions") or [])
    return (
        f"# Room {room.get('room_index')}: {room.get('room_type')}\n\n"
        f"- Act/Floor: A{room.get('act')} F{room.get('floor')}\n"
        f"- Entry HP: {room.get('entry_hp')}/{room.get('entry_max_hp')}\n"
        f"- Exit HP: {room.get('exit_hp')}/{room.get('exit_max_hp')}\n"
        f"- HP loss: {room.get('hp_loss')}\n"
        f"- Entered by: {room.get('entered_by')}\n\n"
        "## Actions\n\n"
        f"{actions or '- No actions recorded'}\n"
    )


def _node_label(node: dict[str, Any]) -> str:
    node_type = str(node.get("type") or "?")
    col = node.get("col")
    row = node.get("row")
    if col is None or row is None:
        return node_type
    return f"{node_type}({col},{row})"


def _format_node_sequence(nodes: list[dict[str, Any]]) -> str:
    return " -> ".join(_node_label(node) for node in nodes)


def _write_replay_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# STS2 Trace Replay",
        "",
        f"- Source: `{report.get('source')}`",
        f"- Seed: `{(report.get('run') or {}).get('seed')}`",
        "",
        "## Route And Room HP",
        "",
        "| # | Act | Floor | Room | Node | Entry HP | Exit HP | Loss | Entered by |",
        "|---:|---:|---:|---|---|---:|---:|---:|---|",
    ]
    for room in report.get("rooms") or []:
        node = room.get("node") or {}
        node_text = f"({node.get('col')},{node.get('row')})" if node else ""
        lines.append(
            "| {idx} | {act} | {floor} | {room} | {node} | {entry}/{entry_max} | {exit}/{exit_max} | {loss} | {entered} |".format(
                idx=room.get("room_index"),
                act=room.get("act"),
                floor=room.get("floor"),
                room=room.get("room_type"),
                node=node_text,
                entry=room.get("entry_hp"),
                entry_max=room.get("entry_max_hp"),
                exit=room.get("exit_hp"),
                exit_max=room.get("exit_max_hp"),
                loss=room.get("hp_loss"),
                entered=room.get("entered_by"),
            )
        )

    lines.extend(["", "## Map Decisions", ""])
    for choice in report.get("map_choices") or []:
        lines.append(
            f"- step {choice.get('step')} A{choice.get('act')}F{choice.get('floor')} "
            f"HP {choice.get('hp')}/{choice.get('max_hp')}: "
            f"selected `{(choice.get('selected') or {}).get('label')}`"
        )
        legal = ", ".join(f"`{route.get('label')}`" for route in choice.get("legal_routes") or [])
        if legal:
            lines.append(f"  - legal routes: {legal}")
        if choice.get("reason"):
            lines.append(f"  - reason: {choice.get('reason')}")

    lines.extend(["", "## Route plan events", ""])
    for event in report.get("route_plan_events") or []:
        lines.append(
            f"- step {event.get('step')} `{event.get('event_type')}`: "
            f"{event.get('reason') or ''}".rstrip()
        )
        selected = event.get("selected_node") or {}
        if selected:
            lines.append(f"  - selected node: {_node_label(selected)}")
        tags = event.get("reason_tags") or []
        if tags:
            lines.append(f"  - reason tags: {', '.join(str(tag) for tag in tags)}")
        if event.get("route_validity"):
            lines.append(f"  - route validity: {event.get('route_validity')}")
        if event.get("override_used"):
            lines.append("  - override: legal action was not topological in the current map snapshot")
        new_plan = event.get("new_plan") or []
        if new_plan:
            lines.append(f"  - new plan: {_format_node_sequence(new_plan)}")
        active_plan = event.get("active_plan") or {}
        remaining = active_plan.get("remaining") or []
        if remaining:
            lines.append(f"  - active remaining: {_format_node_sequence(remaining)}")

    errors = report.get("route_plan_validation_errors") or []
    if errors:
        lines.extend(["", "## Route plan validation errors", ""])
        for error in errors:
            lines.append(
                f"- step {error.get('step')} `{error.get('code')}`: "
                f"{error.get('message') or error.get('reason') or ''}".rstrip()
            )
            if error.get("retry") is not None:
                lines.append(f"  - retry: {error.get('retry')}")

    lines.extend(["", "## Card Rewards", ""])
    for reward in report.get("card_rewards") or []:
        offered = reward.get("offered_bilingual") or reward.get("offered") or []
        if not offered and reward.get("unopened"):
            offered = ["candidates unavailable; skipped before opening card_reward"]
        selected = reward.get("selected_card_bilingual") or reward.get("selected")
        lines.append(
            f"- step {reward.get('step')} offered {offered} -> `{selected}`"
        )

    lines.extend(["", "## Bundle Selections", ""])
    for selection in report.get("bundle_selections") or []:
        cards = selection.get("cards_bilingual") or selection.get("cards") or []
        lines.append(f"- step {selection.get('step')} bundle {selection.get('bundle_index')}: {cards}")

    lines.extend(["", "## Rest Sites", ""])
    for rest in report.get("rest_sites") or []:
        suffix = f"; smith `{rest.get('smith_card')}`" if rest.get("smith_card") else ""
        lines.append(
            f"- step {rest.get('step')} A{rest.get('act')}F{rest.get('floor')} "
            f"HP {rest.get('hp')}/{rest.get('max_hp')}: {rest.get('choice')}{suffix}"
        )

    lines.extend(["", "## Shops", ""])
    for shop in report.get("shops") or []:
        lines.append(f"- A{shop.get('act')}F{shop.get('floor')}:")
        for item in shop.get("actions") or []:
            lines.append(f"  - {item}")

    lines.extend(["", "## High Risk Route Events", ""])
    for event in report.get("high_risk_events") or []:
        selected = (event.get("selected") or {}).get("label")
        lines.append(
            f"- `{event.get('type')}` at step {event.get('step')} "
            f"A{event.get('act')}F{event.get('floor')} HP {event.get('hp')}/{event.get('max_hp')}: `{selected}`"
        )
        if event.get("branch_selected"):
            lines.append(
                f"  - previous branch step {event.get('branch_step')}: "
                f"`{(event.get('branch_selected') or {}).get('label')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_deck_and_card_rewards_markdown(report: dict[str, Any], path: Path) -> None:
    final_deck = report.get("final_deck") or {}
    card_rewards = report.get("card_rewards") or []
    bundle_selections = report.get("bundle_selections") or []
    authoritative_deck = final_deck.get("source") == "authoritative_terminal_state"
    deck_note = (
        "- 说明: 终局牌组直接读取自 Headless 返回的完整 game_over 状态；选牌和牌组变化事件仍按轨迹整理。"
        if authoritative_deck
        else "- 说明: 轨迹没有权威终局牌组，本文件只能用初始牌组和后续 `deck:` 变更重构。"
    )
    deck_heading = "## 终局牌组（引擎终局状态）" if authoritative_deck else "## 终局牌组（重构）"
    lines = [
        "# 终局牌组与选牌事件（中文）",
        "",
        f"- Source trace: `{report.get('source')}`",
        deck_note,
        "- 中文名来源: `localization_eng/cards.json` title 反查 key，再从 `localization_zhs/cards.json` 读取官方简中 title。",
        "",
        deck_heading,
        "",
        f"- 初始牌组张数: {final_deck.get('initial_size', 0)}",
        f"- 终局牌组张数: {final_deck.get('size', 0)}",
        f"- result deck_size: {final_deck.get('result_deck_size')}",
        f"- 与 result deck_size 对齐: {final_deck.get('matches_result_deck_size')}",
        f"- 终局不同牌名数: {final_deck.get('unique_count', 0)}",
        f"- 未映射牌名: {', '.join(final_deck.get('unmapped') or []) if final_deck.get('unmapped') else '无'}",
        "",
        "| 中文牌名 | English | 升级 | 数量 |",
        "|---|---|---:|---:|",
    ]
    for item in final_deck.get("counts") or []:
        lines.append(
            f"| {item.get('name_zh')} | {item.get('name')} | {item.get('upgrade_level', 0)} | {item.get('count')} |"
        )

    lines.extend(["", "### 终局完整列表", ""])
    for index, card in enumerate(final_deck.get("cards") or [], start=1):
        lines.append(f"{index}. {card.get('bilingual')}")

    lines.extend(
        [
            "",
            "## Card Reward 候选与选择",
            "",
            "| # | Step | 位置 | HP | 候选项（中文 / English） | 选择 |",
            "|---:|---:|---|---:|---|---|",
        ]
    )
    for index, reward in enumerate(card_rewards, start=1):
        pos = f"A{reward.get('act')}F{reward.get('floor')} {reward.get('room_type')}"
        hp_text = f"{reward.get('hp')}/{reward.get('max_hp')}"
        offered = "<br>".join(
            f"{offer_index}. {name}"
            for offer_index, name in enumerate(reward.get("offered_bilingual") or reward.get("offered") or [], start=1)
        )
        if not offered and reward.get("unopened"):
            offered = "未展开 / candidates unavailable"
        selected = reward.get("selected_card_bilingual") or ("跳过 / skip" if reward.get("skipped") else reward.get("selected"))
        if reward.get("unopened"):
            selected = f"{selected}<br>（未打开 card_reward，trace 没有候选牌）"
        lines.append(f"| {index} | {reward.get('step')} | {pos} | {hp_text} | {offered} | **{selected}** |")

    lines.extend(
        [
            "",
            "## Bundle Select 卡牌包获得",
            "",
            "| # | Step | 位置 | HP | Bundle | 获得卡牌（中文 / English） |",
            "|---:|---:|---|---:|---:|---|",
        ]
    )
    for index, selection in enumerate(bundle_selections, start=1):
        pos = f"A{selection.get('act')}F{selection.get('floor')} {selection.get('room_type')}"
        hp_text = f"{selection.get('hp')}/{selection.get('max_hp')}"
        cards = "<br>".join(selection.get("cards_bilingual") or selection.get("cards") or [])
        lines.append(
            f"| {index} | {selection.get('step')} | {pos} | {hp_text} | "
            f"{selection.get('bundle_index')} | {cards} |"
        )

    lines.extend(
        [
            "",
            "## 其他入牌/改牌事件",
            "",
            "| Step | 位置 | 类型 | 动作 | 牌组变化（中文 / English） |",
            "|---:|---|---|---|---|",
        ]
    )
    for event in final_deck.get("events") or []:
        pos = f"A{event.get('act')}F{event.get('floor')} {event.get('room_type')}"
        changes = "<br>".join(_format_deck_change(change) for change in event.get("changes") or [])
        lines.append(
            f"| {event.get('step')} | {pos} | {event.get('category')} | "
            f"`{event.get('label')}` | {changes} |"
        )

    warnings = final_deck.get("warnings") or []
    if warnings:
        lines.extend(["", "## 重构警告", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## 直接可读结论",
            "",
            f"- 本局终局 deck_size 为 {final_deck.get('result_deck_size')}，重构牌组为 {final_deck.get('size')}。",
            "- 已展开的 card reward 候选项优先从 trace state.cards 读取；compact state 缺失时，从 legal action label 恢复。",
            "- 如果在 combat_reward 层直接跳过未展开的 card_reward，trace 不包含候选牌，只能标记为 candidates unavailable。",
        ]
    )
    if card_rewards:
        last_reward = card_rewards[-1]
        offered = "、".join(last_reward.get("offered_bilingual") or last_reward.get("offered") or [])
        if not offered and last_reward.get("unopened"):
            offered = "未展开 / candidates unavailable"
        selected = last_reward.get("selected_card_bilingual") or last_reward.get("selected")
        lines.append(
            f"- 最后一次 card reward 是 step {last_reward.get('step')}，候选为 {offered}，选择 {selected}。"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hp_by_action_svg(report: dict[str, Any], path: Path) -> None:
    items = [
        item
        for item in (report.get("hp_timeline") or {}).get("by_action", [])
        if item.get("game_action_index") is not None and item.get("hp") is not None
    ]
    width, height = 1180, 540
    margin_left, margin_top, margin_bottom = 62, 128, 70
    room_band_y, room_band_label_y = 72, 84
    panel_width, panel_gap = 260, 24
    plot_right = width - panel_width - panel_gap
    if not items:
        path.write_text(_empty_svg(width, height, "HP by game action"), encoding="utf-8")
        return

    clean = [
        {
            **item,
            "x": float(item["game_action_index"]),
            "hp_value": float(item["hp"]),
            "max_hp_value": float(item.get("max_hp") or item["hp"] or 1),
        }
        for item in items
    ]
    min_x = min(item["x"] for item in clean)
    max_x = max(item["x"] for item in clean)
    max_y = max(max(item["hp_value"], item["max_hp_value"]) for item in clean)
    if max_x == min_x:
        max_x += 1
    if max_y <= 0:
        max_y = 1

    def sx(x: float) -> float:
        return margin_left + (x - min_x) / (max_x - min_x) * (plot_right - margin_left)

    def sy(y: float) -> float:
        return height - margin_bottom - y / max_y * (height - margin_top - margin_bottom)

    low_hp = max_y * LOW_HP_RATIO
    max_line = " ".join(f"{sx(item['x']):.1f},{sy(item['max_hp_value']):.1f}" for item in clean)
    min_item = min(clean, key=lambda item: item["hp_value"])
    drop_markers = _action_drop_markers(clean)
    panel_rows = _action_event_rows(clean, drop_markers)
    room_groups = _action_room_groups(clean)

    circles = []
    for item in clean:
        hp = item["hp_value"]
        max_hp = item["max_hp_value"]
        fill = _room_type_color(item.get("room_type"))
        stroke = "#b2182b" if hp <= low_hp else "#ffffff"
        label = _action_point_label(item)
        circles.append(
            f'<circle cx="{sx(item["x"]):.1f}" cy="{sy(hp):.1f}" r="4" fill="{fill}" stroke="{stroke}" stroke-width="1.4">'
            f"<title>{html.escape(label)}</title></circle>"
        )
    action_segments = []
    for prev, item in zip(clean, clean[1:]):
        color = _room_type_color(item.get("room_type") or prev.get("room_type"))
        action_segments.append(
            f'<line class="room-action-segment" x1="{sx(prev["x"]):.1f}" y1="{sy(prev["hp_value"]):.1f}" '
            f'x2="{sx(item["x"]):.1f}" y2="{sy(item["hp_value"]):.1f}" stroke="{color}" stroke-width="3">'
            f"<title>{html.escape(_action_point_label(item))}</title></line>"
        )
    room_bands = []
    for group in room_groups:
        x1 = sx(group["start_x"])
        x2 = sx(group["end_x"])
        width_px = max(4, x2 - x1)
        color = _room_type_color(group.get("room_type"))
        label = f"F{group.get('floor')} {group.get('room_type')}"
        room_bands.append(
            f'<rect class="room-band" x="{x1:.1f}" y="{room_band_y}" width="{width_px:.1f}" height="16" '
            f'fill="{color}" opacity="0.18"><title>{html.escape(label)}</title></rect>'
        )
        if width_px >= 44:
            room_bands.append(
                f'<text class="room-band-label" x="{(x1 + x2) / 2:.1f}" y="{room_band_label_y}" '
                f'text-anchor="middle">{html.escape(_truncate_svg_text(label, 16))}</text>'
            )
    for prev, item, loss in drop_markers:
        x = sx(item["x"])
        y1 = sy(prev["hp_value"])
        y2 = sy(item["hp_value"])
        label_y = min(y1, y2) - 6
        circles.append(
            f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="#d55e00" stroke-width="2">'
            f"<title>HP drop {loss:g}: {html.escape(_action_point_label(item))}</title></line>"
        )
        circles.append(
            f'<text class="drop-label" x="{x + 5:.1f}" y="{label_y:.1f}">-{int(loss)}</text>'
        )

    stats = (
        f"points={len(clean)}; min HP {int(min_item['hp_value'])}/{int(min_item['max_hp_value'])} "
        f"at action {int(min_item['x'])} ({_location_label(min_item)})"
    )
    panel_x = plot_right + panel_gap
    panel_items = [
        f'<text class="panel-title" x="{panel_x + 14}" y="84">Key events</text>',
        f'<text class="panel-muted" x="{panel_x + 14}" y="106">Low HP zone: &lt;= {int(low_hp)}/{int(max_y)}</text>',
    ]
    row_y = 132
    for row in panel_rows[:8]:
        color = row["color"]
        label = _truncate_svg_text(row["label"], 28)
        detail = _truncate_svg_text(row["detail"], 32)
        panel_items.extend(
            [
                f'<circle cx="{panel_x + 18}" cy="{row_y - 5}" r="4" fill="{color}"/>',
                f'<text class="panel-label" x="{panel_x + 30}" y="{row_y}">{html.escape(label)}</text>',
                f'<text class="panel-muted" x="{panel_x + 30}" y="{row_y + 16}">{html.escape(detail)}</text>',
            ]
        )
        row_y += 38
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-plot-top="{margin_top}" data-room-band-y="{room_band_y}">
<style>
text {{ font-family: Arial, sans-serif; font-size: 13px; fill: #222; }}
.axis {{ stroke: #333; stroke-width: 1.2; }}
.grid {{ stroke: #ddd; stroke-width: 1; }}
.hp-line {{ fill: none; stroke: #2166ac; stroke-width: 2.8; }}
.max-line {{ fill: none; stroke: #777; stroke-width: 1.5; stroke-dasharray: 5 4; }}
.room-action-segment {{ stroke-linecap: round; }}
.room-band-label {{ font-size: 10px; fill: #444; }}
.drop-label {{ font-size: 11px; font-weight: 700; fill: #8c2d04; paint-order: stroke; stroke: white; stroke-width: 3px; }}
.panel-title {{ font-size: 14px; font-weight: 700; }}
.panel-label {{ font-size: 12px; font-weight: 700; }}
.panel-muted {{ font-size: 11px; fill: #555; }}
</style>
<rect width="100%" height="100%" fill="#fff"/>
<text x="{margin_left}" y="25" font-size="18" font-weight="700">HP by game action</text>
<text class="chart-summary" x="{margin_left}" y="48" font-size="12">{html.escape(stats)}</text>
<rect x="{margin_left}" y="{sy(low_hp):.1f}" width="{plot_right - margin_left}" height="{height - margin_bottom - sy(low_hp):.1f}" fill="#fddbc7" opacity="0.45"/>
<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{plot_right}" y2="{height - margin_bottom}"/>
<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>
<line class="grid" x1="{margin_left}" y1="{sy(max_y):.1f}" x2="{plot_right}" y2="{sy(max_y):.1f}"/>
<line class="grid" x1="{margin_left}" y1="{sy(low_hp):.1f}" x2="{plot_right}" y2="{sy(low_hp):.1f}"/>
<text x="{margin_left - 8}" y="{sy(max_y):.1f}" text-anchor="end">{max_y:g}</text>
<text x="{margin_left - 8}" y="{height - margin_bottom}" text-anchor="end">0</text>
<text x="{(margin_left + plot_right) / 2:.1f}" y="{height - 22}" text-anchor="middle">game action</text>
<text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle">HP</text>
{chr(10).join(room_bands)}
<polyline class="max-line" points="{max_line}"><title>Max HP</title></polyline>
{chr(10).join(action_segments)}
{chr(10).join(circles)}
<g class="event-panel">
<rect x="{panel_x}" y="58" width="{panel_width}" height="{height - 80}" rx="6" fill="#f8f9fb" stroke="#d5d8de"/>
{chr(10).join(panel_items)}
</g>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_hp_by_room_svg(report: dict[str, Any], path: Path) -> None:
    rooms = [room for room in report.get("rooms") or [] if room.get("room_index") is not None]
    panel_width, panel_gap = 280, 24
    width, height = max(1180, 92 * max(1, len(rooms)) + panel_width + 160), 560
    margin_left, margin_top, margin_bottom = 72, 106, 76
    plot_right = width - panel_width - panel_gap
    hp_rooms = [
        room
        for room in rooms
        if room.get("entry_hp") is not None or room.get("exit_hp") is not None
    ]
    if not hp_rooms:
        path.write_text(_empty_svg(width, height, "Room HP loss"), encoding="utf-8")
        return

    max_y = max(
        value
        for room in hp_rooms
        for value in (
            room.get("entry_hp"),
            room.get("exit_hp"),
            room.get("entry_max_hp"),
            room.get("exit_max_hp"),
        )
        if isinstance(value, (int, float))
    )
    if max_y <= 0:
        max_y = 1

    min_room = min(room["room_index"] for room in rooms)
    max_room = max(room["room_index"] for room in rooms)
    if max_room == min_room:
        max_room += 1

    def sx(room_index: float) -> float:
        return margin_left + (room_index - min_room) / (max_room - min_room) * (plot_right - margin_left)

    def sy(hp: float) -> float:
        return height - margin_bottom - hp / max_y * (height - margin_top - margin_bottom)

    low_hp = max_y * LOW_HP_RATIO
    room_shapes = []
    axis_labels = []
    exit_points = []
    for room in rooms:
        room_index = float(room["room_index"])
        x = sx(room_index)
        room_type = str(room.get("room_type") or "?")
        icon = MAP_ICONS.get(room_type, ".")
        entry = room.get("entry_hp")
        exit_hp = room.get("exit_hp")
        max_hp = room.get("entry_max_hp") or room.get("exit_max_hp")
        loss = room.get("hp_loss") or 0
        severity_class = _loss_severity_class(room)
        severity_color = _loss_severity_color(room)
        type_color = _room_type_color(room_type)
        if entry is not None and exit_hp is not None:
            y_entry = sy(float(entry))
            y_exit = sy(float(exit_hp))
            room_shapes.append(
                f'<line class="room-hp-segment {severity_class}" x1="{x:.1f}" y1="{y_entry:.1f}" x2="{x:.1f}" y2="{y_exit:.1f}" '
                f'stroke="{severity_color}" stroke-width="{6 if room.get("large_loss") else 4}">'
                f"<title>{html.escape(_room_hp_title(room))}</title></line>"
            )
            room_shapes.append(f'<circle cx="{x:.1f}" cy="{y_entry:.1f}" r="4" fill="#fff" stroke="{type_color}" stroke-width="2"/>')
            room_shapes.append(f'<circle cx="{x:.1f}" cy="{y_exit:.1f}" r="5.5" fill="{type_color}" stroke="{severity_color}" stroke-width="2"/>')
            exit_points.append((x, y_exit))
        elif exit_hp is not None:
            y_exit = sy(float(exit_hp))
            room_shapes.append(
                f'<circle cx="{x:.1f}" cy="{y_exit:.1f}" r="5.5" fill="{type_color}" stroke="{severity_color}" stroke-width="2">'
                f"<title>{html.escape(_room_hp_title(room))}</title></circle>"
            )
            exit_points.append((x, y_exit))

        if entry is not None or exit_hp is not None:
            loss_label = f"F{room.get('floor')} {room_type} -{int(loss)}"
            hp_label = f"{entry}->{exit_hp}"
            axis_labels.append(
                f'<line class="loss-leader" x1="{x:.1f}" y1="{margin_top - 24}" x2="{x:.1f}" y2="{sy(float(entry if entry is not None else exit_hp)):.1f}"/>'
            )
            axis_labels.append(
                f'<text class="loss-label" x="{x:.1f}" y="{margin_top - 40}" text-anchor="middle">{html.escape(loss_label)}'
                f"<title>{html.escape(_room_hp_title(room))}</title></text>"
            )
            axis_labels.append(
                f'<text class="hp-transition-label" x="{x:.1f}" y="{margin_top - 24}" text-anchor="middle">{html.escape(hp_label)}</text>'
            )
        axis_labels.append(
            f'<text class="room-index-label" x="{x:.1f}" y="{height - 48}" text-anchor="middle">Room #{int(room_index)}</text>'
        )
        axis_labels.append(
            f'<text class="room-type-label" x="{x:.1f}" y="{height - 32}" text-anchor="middle">{html.escape(icon)} {html.escape(room_type)}</text>'
        )

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in exit_points)
    large_count = sum(1 for room in rooms if room.get("large_loss"))
    total_loss = sum(int(room.get("hp_loss") or 0) for room in rooms)
    stats = f"rooms={len(rooms)}; total loss={total_loss}; large loss rooms={large_count}"
    panel_x = plot_right + panel_gap
    panel_rows = _room_loss_rows(rooms)
    panel_items = [
        f'<text class="panel-title" x="{panel_x + 14}" y="84">Top room losses</text>',
        f'<text class="panel-muted" x="{panel_x + 14}" y="106">Low HP zone: &lt;= {int(low_hp)}/{int(max_y)}</text>',
    ]
    row_y = 132
    if not panel_rows:
        panel_items.append(f'<text class="panel-muted" x="{panel_x + 14}" y="{row_y}">No HP loss recorded</text>')
    for row in panel_rows[:8]:
        color = row["color"]
        label = _truncate_svg_text(row["label"], 28)
        detail = _truncate_svg_text(row["detail"], 32)
        panel_items.extend(
            [
                f'<rect x="{panel_x + 14}" y="{row_y - 13}" width="8" height="8" fill="{color}"/>',
                f'<text class="panel-label" x="{panel_x + 30}" y="{row_y}">{html.escape(label)}</text>',
                f'<text class="panel-muted" x="{panel_x + 30}" y="{row_y + 16}">{html.escape(detail)}</text>',
            ]
        )
        row_y += 38
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, sans-serif; font-size: 13px; fill: #222; }}
.axis {{ stroke: #333; stroke-width: 1.2; }}
.grid {{ stroke: #ddd; stroke-width: 1; }}
.exit-line {{ fill: none; stroke: #2166ac; stroke-width: 2; stroke-dasharray: 4 4; opacity: .7; }}
.room-hp-segment {{ stroke-linecap: round; }}
.loss-leader {{ stroke: #b9bec8; stroke-width: 1; stroke-dasharray: 3 3; }}
.loss-label {{ font-size: 11px; font-weight: 700; fill: #222; paint-order: stroke; stroke: white; stroke-width: 3px; }}
.hp-transition-label {{ font-size: 10px; fill: #555; paint-order: stroke; stroke: white; stroke-width: 3px; }}
.room-index-label {{ font-size: 10px; font-weight: 700; }}
.room-type-label {{ font-size: 10px; fill: #555; }}
.panel-title {{ font-size: 14px; font-weight: 700; }}
.panel-label {{ font-size: 12px; font-weight: 700; }}
.panel-muted {{ font-size: 11px; fill: #555; }}
</style>
<rect width="100%" height="100%" fill="#fff"/>
<text x="{margin_left}" y="25" font-size="18" font-weight="700">Room HP loss</text>
<text x="{margin_left}" y="44" font-size="12">{html.escape(stats)}</text>
<rect x="{margin_left}" y="{sy(low_hp):.1f}" width="{plot_right - margin_left}" height="{height - margin_bottom - sy(low_hp):.1f}" fill="#fddbc7" opacity="0.45"/>
<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{plot_right}" y2="{height - margin_bottom}"/>
<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>
<line class="axis" x1="{plot_right}" y1="{margin_top}" x2="{plot_right}" y2="{height - margin_bottom}"/>
<line class="grid" x1="{margin_left}" y1="{sy(max_y):.1f}" x2="{plot_right}" y2="{sy(max_y):.1f}"/>
<line class="grid" x1="{margin_left}" y1="{sy(low_hp):.1f}" x2="{plot_right}" y2="{sy(low_hp):.1f}"/>
<text x="{margin_left - 8}" y="{sy(max_y):.1f}" text-anchor="end">{max_y:g}</text>
<text x="{margin_left - 8}" y="{height - margin_bottom}" text-anchor="end">0</text>
<text x="{plot_right + 8}" y="{sy(max_y):.1f}">{max_y:g}</text>
<text x="{plot_right + 8}" y="{height - margin_bottom}">0</text>
<text x="20" y="{(margin_top + height - margin_bottom) / 2:.1f}" transform="rotate(-90 20 {(margin_top + height - margin_bottom) / 2:.1f})" text-anchor="middle">HP axis</text>
<text x="{(margin_left + plot_right) / 2:.1f}" y="{height - 12}" text-anchor="middle">room sequence</text>
<polyline class="exit-line" points="{polyline}"/>
{chr(10).join(room_shapes)}
{chr(10).join(axis_labels)}
<g class="room-loss-panel">
<rect x="{panel_x}" y="58" width="{panel_width}" height="{height - 80}" rx="6" fill="#f8f9fb" stroke="#d5d8de"/>
{chr(10).join(panel_items)}
</g>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_route_hp_replay_svg(report: dict[str, Any], path: Path) -> None:
    rooms = [room for room in report.get("rooms") or [] if room.get("room_index") is not None]
    if not rooms:
        path.write_text(_empty_svg(1180, 760, "Map route vs HP route"), encoding="utf-8")
        return

    map_choices = report.get("map_choices") or []
    has_full_map = any(isinstance(choice.get("map_snapshot"), dict) for choice in map_choices)
    room_label_slot_width = _route_hp_label_slot_width(rooms)
    width = max(1280 if has_full_map else 1180, room_label_slot_width * len(rooms) + 160)
    height = 1120 if has_full_map else 820
    margin_left, margin_right = 76, 110
    map_top, map_bottom = (72, 650) if has_full_map else (72, 330)
    hp_top, hp_bottom = (760, 1060) if has_full_map else (430, 760)
    plot_right = width - margin_right

    def sx(room_index: float) -> float:
        if len(rooms) == 1:
            return (margin_left + plot_right) / 2
        min_room = float(rooms[0]["room_index"])
        max_room = float(rooms[-1]["room_index"])
        return margin_left + (room_index - min_room) / (max_room - min_room) * (plot_right - margin_left)

    node_rows = [
        int((room.get("node") or {}).get("row"))
        for room in rooms
        if (room.get("node") or {}).get("row") is not None
    ]
    min_node_row = min(node_rows) if node_rows else 0
    max_node_row = max(node_rows) if node_rows else max(1, len(rooms) - 1)
    if max_node_row == min_node_row:
        max_node_row += 1

    def map_y(room: dict[str, Any], fallback_index: int) -> float:
        node = room.get("node") or {}
        row = node.get("row")
        value = int(row) if row is not None else fallback_index
        return map_bottom - (value - min_node_row) / (max_node_row - min_node_row) * (map_bottom - map_top - 42)

    hp_values = [
        value
        for room in rooms
        for value in (
            room.get("entry_hp"),
            room.get("exit_hp"),
            room.get("entry_max_hp"),
            room.get("exit_max_hp"),
        )
        if isinstance(value, (int, float))
    ]
    max_hp = max(hp_values) if hp_values else 1
    if max_hp <= 0:
        max_hp = 1

    def hp_y(hp: float) -> float:
        return hp_bottom - hp / max_hp * (hp_bottom - hp_top)

    topology_map_context = _route_topology_map_svg(
        map_choices,
        rooms,
        route_plan_events=report.get("route_plan_events") or [],
        margin_left=margin_left,
        plot_right=plot_right,
        map_top=map_top,
        map_bottom=map_bottom,
    )
    same_floor_map_context = None if topology_map_context is not None else _route_same_floor_map_svg(
        map_choices,
        rooms,
        margin_left=margin_left,
        plot_right=plot_right,
        map_top=map_top,
        map_bottom=map_bottom,
    )
    primary_map_context = topology_map_context or same_floor_map_context
    map_points = []
    map_shapes = []
    hp_exit_points = []
    hp_shapes = []
    labels = []
    for index, room in enumerate(rooms):
        room_index = float(room["room_index"])
        x = sx(room_index)
        y = map_y(room, index)
        room_type = str(room.get("room_type") or "?")
        color = _room_type_color(room_type)
        icon = MAP_ICONS.get(room_type, ".")
        node = room.get("node") or {}
        if primary_map_context is None:
            map_points.append((x, y))
            map_shapes.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{color}" stroke="#222" stroke-width="1.4">'
                f"<title>{html.escape(_room_hp_title(room))}</title></circle>"
            )
            map_shapes.append(
                    f'<text class="route-node-small-label" x="{x:.1f}" y="{y - 16:.1f}" text-anchor="middle">'
                    f'F{html.escape(str(room.get("floor")))} {html.escape(icon)}</text>'
                )
            if node.get("col") is not None and node.get("row") is not None:
                map_shapes.append(
                    f'<text class="route-node-coordinate" x="{x:.1f}" y="{y + 28:.1f}" text-anchor="middle">'
                    f'({html.escape(str(node.get("col")))},{html.escape(str(node.get("row")))})</text>'
                )

        entry = room.get("entry_hp")
        exit_hp = room.get("exit_hp")
        loss = int(room.get("hp_loss") or 0)
        severity = _loss_severity_class(room)
        severity_color = _loss_severity_color(room)
        if entry is not None and exit_hp is not None:
            y_entry = hp_y(float(entry))
            y_exit = hp_y(float(exit_hp))
            hp_shapes.append(
                f'<line class="room-hp-segment {severity}" x1="{x:.1f}" y1="{y_entry:.1f}" x2="{x:.1f}" y2="{y_exit:.1f}" '
                f'stroke="{severity_color}" stroke-width="{6 if room.get("large_loss") else 4}">'
                f"<title>{html.escape(_room_hp_title(room))}</title></line>"
            )
            hp_shapes.append(f'<circle cx="{x:.1f}" cy="{y_entry:.1f}" r="5.5" fill="#fff" stroke="{color}" stroke-width="2.4"/>')
            hp_shapes.append(f'<circle cx="{x:.1f}" cy="{y_exit:.1f}" r="7" fill="{color}" stroke="{severity_color}" stroke-width="2.4"/>')
            hp_exit_points.append((x, y_exit))
            labels.append(
                _route_hp_room_label_svg(
                    x=x,
                    y=hp_top - 52,
                    room=room,
                    room_type=room_type,
                    loss=loss,
                    entry=entry,
                    exit_hp=exit_hp,
                )
            )
        labels.append(
            f'<text class="room-index-label" x="{x:.1f}" y="{height - 24}" text-anchor="middle">#{int(room_index)}</text>'
        )

    map_polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in map_points)
    hp_polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in hp_exit_points)
    fallback_map_context = f"""
<rect x="{margin_left}" y="{map_top}" width="{plot_right - margin_left}" height="{map_bottom - map_top}" fill="#fbfbfc" stroke="#d5d8de"/>
<polyline class="map-route" points="{map_polyline}"/>
{chr(10).join(map_shapes)}
"""
    map_context = primary_map_context if primary_map_context is not None else fallback_map_context
    map_heading = (
        "Played map topology"
        if topology_map_context is not None
        else "Same-floor choice map"
        if same_floor_map_context is not None
        else "Selected map route"
    )
    low_hp = max_hp * LOW_HP_RATIO
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, sans-serif; font-size: 15px; fill: #222; }}
.panel-title {{ font-size: 26px; font-weight: 700; }}
.section-title {{ font-size: 19px; font-weight: 700; }}
.axis {{ stroke: #333; stroke-width: 1.2; }}
.grid {{ stroke: #ddd; stroke-width: 1; }}
.map-route {{ fill: none; stroke: #2166ac; stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
.hp-route {{ fill: none; stroke: #2166ac; stroke-width: 2.4; stroke-dasharray: 4 4; }}
.route-topology-edge {{ stroke: #9f9687; stroke-width: 2; opacity: 0.72; }}
.route-topology-path {{ fill: none; stroke: #2166ac; stroke-width: 4; stroke-linejoin: round; stroke-linecap: round; opacity: 0.95; }}
.route-topology-path.route-plan-stage-1, .route-topology-node.selected.route-plan-stage-1 {{ stroke: {ROUTE_PLAN_STAGE_COLORS[0]}; }}
.route-topology-path.route-plan-stage-2, .route-topology-node.selected.route-plan-stage-2 {{ stroke: {ROUTE_PLAN_STAGE_COLORS[1]}; }}
.route-topology-path.route-plan-stage-3, .route-topology-node.selected.route-plan-stage-3 {{ stroke: {ROUTE_PLAN_STAGE_COLORS[2]}; }}
.route-topology-path.route-plan-stage-4, .route-topology-node.selected.route-plan-stage-4 {{ stroke: {ROUTE_PLAN_STAGE_COLORS[3]}; }}
.route-topology-path.route-plan-stage-5, .route-topology-node.selected.route-plan-stage-5 {{ stroke: {ROUTE_PLAN_STAGE_COLORS[4]}; }}
.route-stage-label.route-plan-stage-1, .route-stage-legend.route-plan-stage-1 {{ fill: {ROUTE_PLAN_STAGE_COLORS[0]}; }}
.route-stage-label.route-plan-stage-2, .route-stage-legend.route-plan-stage-2 {{ fill: {ROUTE_PLAN_STAGE_COLORS[1]}; }}
.route-stage-label.route-plan-stage-3, .route-stage-legend.route-plan-stage-3 {{ fill: {ROUTE_PLAN_STAGE_COLORS[2]}; }}
.route-stage-label.route-plan-stage-4, .route-stage-legend.route-plan-stage-4 {{ fill: {ROUTE_PLAN_STAGE_COLORS[3]}; }}
.route-stage-label.route-plan-stage-5, .route-stage-legend.route-plan-stage-5 {{ fill: {ROUTE_PLAN_STAGE_COLORS[4]}; }}
.route-topology-node {{ stroke-width: 1.4; }}
.route-topology-node.selected {{ stroke: #7f0000; stroke-width: 3; }}
.route-topology-node.option {{ stroke: #5f5548; opacity: 0.9; }}
.route-topology-node.current {{ stroke: #1466cc; stroke-width: 2.6; }}
.route-topology-act-label {{ font-size: 18px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 4px; }}
.route-stage-label {{ fill: #111; font-size: 16px; font-weight: 700; paint-order: stroke; stroke: white; stroke-width: 5px; }}
.route-stage-legend {{ font-size: 15px; font-weight: 700; fill: #333; paint-order: stroke; stroke: white; stroke-width: 4px; }}
.replan-connector {{ stroke: #444; stroke-width: 1.4; stroke-dasharray: 3 3; }}
.replan-flag {{ fill: #fff; stroke: #b2182b; stroke-width: 2; }}
.replan-flag-title {{ font-size: 14px; font-weight: 700; fill: #111; }}
.replan-flag-reason {{ font-size: 13px; fill: #333; }}
.route-node-icon {{ font-size: 18px; font-weight: 700; dominant-baseline: central; alignment-baseline: central; pointer-events: none; }}
.route-node-small-label {{ font-size: 15px; fill: #222; paint-order: stroke; stroke: white; stroke-width: 4px; }}
.route-node-coordinate {{ font-size: 14px; fill: #555; paint-order: stroke; stroke: white; stroke-width: 4px; }}
.route-topology-coordinate {{ font-size: 15px; fill: #555; paint-order: stroke; stroke: white; stroke-width: 4px; }}
.same-floor-route-line {{ fill: none; stroke: #2166ac; stroke-width: 3.5; stroke-linejoin: round; stroke-linecap: round; }}
.same-floor-node {{ stroke-width: 1.4; }}
.same-floor-node.selected {{ stroke: #222; stroke-width: 2.2; }}
.same-floor-node.selected.large-loss {{ stroke: #7f0000; stroke-width: 3; }}
.same-floor-node.option {{ stroke: #888; opacity: 0.66; }}
.same-floor-col-guide {{ stroke: #e1e4e8; stroke-width: 1; stroke-dasharray: 3 4; }}
.same-floor-coordinate {{ font-size: 15px; fill: #555; paint-order: stroke; stroke: white; stroke-width: 4px; }}
.selected-room-badge {{ font-size: 18px; font-weight: 700; fill: #111; paint-order: stroke; stroke: white; stroke-width: 5px; }}
.room-hp-segment {{ stroke-linecap: round; }}
.loss-floor-label {{ font-size: 15px; font-weight: 700; fill: #222; paint-order: stroke; stroke: white; stroke-width: 5px; }}
.loss-room-label {{ font-size: 14px; font-weight: 700; fill: #222; paint-order: stroke; stroke: white; stroke-width: 5px; }}
.hp-transition-label {{ font-size: 15px; fill: #555; paint-order: stroke; stroke: white; stroke-width: 5px; }}
.room-index-label {{ font-size: 13px; font-weight: 700; fill: #555; }}
.hp-axis-label {{ font-size: 14px; fill: #555; }}
</style>
<rect width="100%" height="100%" fill="#fff"/>
<text class="panel-title" x="{margin_left}" y="34">Map route vs HP route</text>
<g class="map-route-panel">
<text class="section-title" x="{margin_left}" y="66">{html.escape(map_heading)}</text>
{map_context}
</g>
<g class="hp-route-panel">
<text class="section-title" x="{margin_left}" y="{hp_top - 88}">HP route by room</text>
<rect x="{margin_left}" y="{hp_y(low_hp):.1f}" width="{plot_right - margin_left}" height="{hp_bottom - hp_y(low_hp):.1f}" fill="#fddbc7" opacity="0.45"/>
<line class="axis" x1="{margin_left}" y1="{hp_bottom}" x2="{plot_right}" y2="{hp_bottom}"/>
<line class="axis" x1="{margin_left}" y1="{hp_top}" x2="{margin_left}" y2="{hp_bottom}"/>
<line class="axis" x1="{plot_right}" y1="{hp_top}" x2="{plot_right}" y2="{hp_bottom}"/>
<line class="grid" x1="{margin_left}" y1="{hp_y(max_hp):.1f}" x2="{plot_right}" y2="{hp_y(max_hp):.1f}"/>
<line class="grid" x1="{margin_left}" y1="{hp_y(low_hp):.1f}" x2="{plot_right}" y2="{hp_y(low_hp):.1f}"/>
<text class="hp-axis-label" x="{margin_left - 8}" y="{hp_y(max_hp):.1f}" text-anchor="end">{max_hp:g}</text>
<text class="hp-axis-label" x="{margin_left - 8}" y="{hp_bottom}" text-anchor="end">0</text>
<text class="hp-axis-label" x="{plot_right + 8}" y="{hp_y(max_hp):.1f}">{max_hp:g}</text>
<text class="hp-axis-label" x="{plot_right + 8}" y="{hp_bottom}">0</text>
<polyline class="hp-route" points="{hp_polyline}"/>
{chr(10).join(hp_shapes)}
{chr(10).join(labels)}
</g>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _route_hp_label_slot_width(rooms: list[dict[str, Any]]) -> int:
    width = 126
    for room in rooms:
        room_type = str(room.get("room_type") or "?")
        loss = int(room.get("hp_loss") or 0)
        detail = f"{room_type} -{loss}"
        transition = ""
        if room.get("entry_hp") is not None and room.get("exit_hp") is not None:
            transition = f"{room.get('entry_hp')}->{room.get('exit_hp')}"
        width = max(width, int(max(len(detail) * 8.4 + 32, len(transition) * 7.6 + 28)))
    return width


def _route_hp_room_label_svg(
    *,
    x: float,
    y: float,
    room: dict[str, Any],
    room_type: str,
    loss: int,
    entry: Any,
    exit_hp: Any,
) -> str:
    floor = str(room.get("floor"))
    full_label = f"F{floor} {room_type} -{loss}"
    room_label = f"{room_type} -{loss}"
    hp_transition = f"{entry}->{exit_hp}"
    title = f"{full_label} {hp_transition}"
    room_index = room.get("room_index")
    return (
        f'<g class="hp-room-label" data-room-index="{html.escape(str(room_index))}">'
        f"<title>{html.escape(title)}</title>"
        f'<text class="loss-floor-label" x="{x:.1f}" y="{y:.1f}" text-anchor="middle">F{html.escape(floor)}</text>'
        f'<text class="loss-room-label" x="{x:.1f}" y="{y + 18:.1f}" text-anchor="middle">{html.escape(room_label)}</text>'
        f'<text class="hp-transition-label" x="{x:.1f}" y="{y + 38:.1f}" text-anchor="middle">{html.escape(hp_transition)}</text>'
        "</g>"
    )


def _action_drop_markers(items: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    markers = []
    for prev, item in zip(items, items[1:]):
        loss = prev["hp_value"] - item["hp_value"]
        if loss >= 5:
            markers.append((prev, item, loss))
    return markers


def _route_topology_map_svg(
    map_choices: list[dict[str, Any]],
    rooms: list[dict[str, Any]],
    *,
    route_plan_events: list[dict[str, Any]],
    margin_left: float,
    plot_right: float,
    map_top: float,
    map_bottom: float,
) -> str | None:
    def coord_from_node(node: dict[str, Any] | None) -> tuple[int, int] | None:
        if not node:
            return None
        if node.get("col") is None or node.get("row") is None:
            return None
        return (int(node["col"]), int(node["row"]))

    maps_by_act: dict[Any, dict[str, Any]] = {}
    current_by_act: dict[Any, tuple[int, int]] = {}
    selected_by_act: dict[Any, list[tuple[int, int]]] = {}
    selected_entries_by_act: dict[Any, list[dict[str, Any]]] = {}
    legal_by_act: dict[Any, set[tuple[int, int]]] = {}
    route_plan_changes = _route_plan_stage_changes(route_plan_events)

    for choice in map_choices:
        act = choice.get("act")
        snapshot = choice.get("map_snapshot")
        if isinstance(snapshot, dict) and snapshot.get("type") == "map":
            maps_by_act.setdefault(act, snapshot)
            current_coord = coord_from_node(snapshot.get("current_coord") or {})
            if current_coord is not None:
                current_by_act.setdefault(act, current_coord)
        for route in choice.get("legal_routes") or []:
            coord = coord_from_node(route.get("node") or {})
            if coord is not None:
                legal_by_act.setdefault(act, set()).add(coord)
        selected = coord_from_node((choice.get("selected") or {}).get("node") or {})
        if selected is not None:
            selected_by_act.setdefault(act, [])
            if not selected_by_act[act] or selected_by_act[act][-1] != selected:
                selected_by_act[act].append(selected)
            selected_entries_by_act.setdefault(act, [])
            if not selected_entries_by_act[act] or selected_entries_by_act[act][-1].get("coord") != selected:
                replan = _route_plan_replan_for_step(choice.get("step"), route_plan_changes)
                selected_entries_by_act[act].append(
                    {
                        "coord": selected,
                        "step": choice.get("step"),
                        "stage": _route_plan_stage_for_step(choice.get("step"), route_plan_changes),
                        "replan": replan,
                    }
                )

    if not maps_by_act:
        return None

    room_by_act_coord: dict[tuple[Any, int, int], dict[str, Any]] = {}
    floor_by_act_row: dict[tuple[Any, int], Any] = {}
    played_rows_by_act: dict[Any, list[int]] = {}
    for room in rooms:
        coord = coord_from_node(room.get("node") or {})
        if coord is None:
            continue
        act = room.get("act")
        col, row = coord
        room_by_act_coord[(act, col, row)] = room
        floor_by_act_row.setdefault((act, row), room.get("floor"))
        played_rows_by_act.setdefault(act, []).append(row)
        selected_by_act.setdefault(act, [])
        if coord not in selected_by_act[act]:
            selected_by_act[act].append(coord)
        selected_entries_by_act.setdefault(act, [])
        if not any(entry.get("coord") == coord for entry in selected_entries_by_act[act]):
            selected_entries_by_act[act].append({"coord": coord, "step": room.get("entry_step"), "stage": 0, "replan": None})

    acts = [
        act
        for act in sorted(
            maps_by_act,
            key=lambda item: (item is None, 0 if isinstance(item, int) else 1, item if isinstance(item, int) else str(item)),
        )
        if played_rows_by_act.get(act) or selected_by_act.get(act) or current_by_act.get(act) is not None
    ]
    if not acts:
        return None

    panel_gap = 28
    panel_top = map_top + 52
    panel_bottom = map_bottom - 34
    panel_width = (plot_right - margin_left - panel_gap * max(0, len(acts) - 1)) / max(1, len(acts))
    panel_shapes: list[str] = []
    act_labels: list[str] = []
    edges: list[str] = []
    path_lines: list[str] = []
    node_shapes: list[str] = []
    replan_shapes: list[str] = []
    legend_shapes: list[str] = _route_plan_stage_legend_svg(route_plan_changes, margin_left + 270, map_top - 6)

    for act_index, act in enumerate(acts):
        snapshot = maps_by_act[act]
        node_map = _snapshot_node_map(snapshot)
        if not node_map:
            continue
        selected = selected_by_act.get(act) or []
        current_coord = current_by_act.get(act)
        played_rows = played_rows_by_act.get(act) or [row for _, row in selected]
        if current_coord is not None:
            played_rows.append(current_coord[1])
        if not played_rows:
            continue
        min_row = min(played_rows)
        max_row = max(played_rows)
        visible_node_map = {
            coord: node
            for coord, node in node_map.items()
            if min_row <= coord[1] <= max_row
        }
        if not visible_node_map:
            continue

        min_col = min(col for col, _ in visible_node_map)
        max_col = max(col for col, _ in visible_node_map)
        if max_col == min_col:
            max_col += 1
        if max_row == min_row:
            max_row += 1

        left = margin_left + act_index * (panel_width + panel_gap)
        right = left + panel_width

        def sx_row(row: int, *, left: float = left, right: float = right) -> float:
            return left + 44 + (row - min_row) / (max_row - min_row) * (right - left - 88)

        def sy_col(col: int) -> float:
            node_top = panel_top + 26
            node_bottom = panel_bottom - 26
            return node_top + (col - min_col) / (max_col - min_col) * (node_bottom - node_top)

        act_label = f"Act {act}" if act is not None else "Act ?"
        panel_y = panel_top - 8
        panel_height = panel_bottom - panel_top + 32
        act_label_y = panel_y - 12
        panel_shapes.append(
            f'<rect class="route-topology-panel" fill="none" x="{left:.1f}" y="{panel_y:.1f}" '
            f'width="{panel_width:.1f}" height="{panel_height:.1f}" stroke="#d5d0c6"/>'
        )
        act_labels.append(
            f'<text class="route-topology-act-label" x="{left:.1f}" y="{act_label_y:.1f}">'
            f"{html.escape(act_label)}</text>"
        )

        for (col, row), node in sorted(visible_node_map.items(), key=lambda item: (item[0][1], item[0][0])):
            for child in node.get("children") or []:
                child_coord = coord_from_node(child)
                if child_coord is None or child_coord not in visible_node_map:
                    continue
                edges.append(
                    f'<line class="route-topology-edge" x1="{sx_row(row):.1f}" y1="{sy_col(col):.1f}" '
                    f'x2="{sx_row(child_coord[1]):.1f}" y2="{sy_col(child_coord[0]):.1f}"/>'
                )

        path_entries: list[dict[str, Any]] = []
        if current_coord is not None and current_coord in visible_node_map:
            path_entries.append({"coord": current_coord, "stage": 0})
        for entry in selected_entries_by_act.get(act) or []:
            coord = entry.get("coord")
            if coord in visible_node_map and (not path_entries or path_entries[-1].get("coord") != coord):
                path_entries.append(entry)
        seen_stage_labels: set[int] = set()
        if not route_plan_changes and len(path_entries) >= 2:
            path_points = " ".join(
                f"{sx_row(row):.1f},{sy_col(col):.1f}"
                for entry in path_entries
                for col, row in [entry.get("coord")]
                if (col, row) in visible_node_map
            )
            path_lines.append(f'<polyline class="route-topology-path" points="{path_points}"/>')
        else:
            for previous, entry in zip(path_entries, path_entries[1:]):
                prev_coord = previous.get("coord")
                coord = entry.get("coord")
                if prev_coord not in visible_node_map or coord not in visible_node_map:
                    continue
                prev_col, prev_row = prev_coord
                col, row = coord
                stage = int(entry.get("stage") or 0)
                stage_class = _route_plan_stage_class(stage)
                x1, y1 = sx_row(prev_row), sy_col(prev_col)
                x2, y2 = sx_row(row), sy_col(col)
                path_class = "route-topology-path" if stage <= 0 else f"route-topology-path {stage_class}"
                path_lines.append(
                    f'<line class="{path_class}" data-route-stage="{stage}" '
                    f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}">'
                    f"<title>Route stage {stage}</title></line>"
                )
                if stage > 0 and stage not in seen_stage_labels:
                    seen_stage_labels.add(stage)
                    label_x = (x1 + x2) / 2
                    label_y = (y1 + y2) / 2 - 12
                    path_lines.append(
                        f'<text class="route-stage-label {stage_class}" x="{label_x:.1f}" y="{label_y:.1f}" '
                        f'text-anchor="middle">R{stage}</text>'
                    )

        selected_set = set(selected)
        selected_stage_by_coord = {
            entry["coord"]: int(entry.get("stage") or 0)
            for entry in selected_entries_by_act.get(act) or []
            if entry.get("coord") is not None
        }
        replan_by_coord = {
            entry["coord"]: entry
            for entry in selected_entries_by_act.get(act) or []
            if entry.get("coord") is not None and entry.get("replan")
        }
        legal_set = legal_by_act.get(act) or set()
        for (col, row), node in sorted(visible_node_map.items(), key=lambda item: (item[0][1], item[0][0])):
            room_type = str(node.get("type") or "?")
            room = room_by_act_coord.get((act, col, row))
            is_selected = (col, row) in selected_set or room is not None
            is_current = current_coord == (col, row)
            if is_selected:
                stage = selected_stage_by_coord.get((col, row), 0)
                node_class = "route-topology-node selected" if stage <= 0 else f"route-topology-node selected {_route_plan_stage_class(stage)}"
            elif is_current:
                node_class = "route-topology-node current"
            else:
                node_class = "route-topology-node option"
            radius = 20 if is_selected else 17 if is_current else 14
            floor = floor_by_act_row.get((act, row), row + 1)
            title_parts = [
                f"act={act}",
                f"floor={floor}",
                f"col={col}",
                f"row={row}",
                f"type={room_type}",
            ]
            if room is not None:
                title_parts.append(f"loss={int(room.get('hp_loss') or 0)}")
            if (col, row) in legal_set:
                title_parts.append("legal=true")
            title = " ".join(title_parts)
            x = sx_row(row)
            y = sy_col(col)
            node_shapes.append(
                f'<circle class="{node_class}" cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{_room_type_fill(room_type)}">'
                f"<title>{html.escape(title)}</title></circle>"
            )
            node_shapes.append(
                f'<text class="route-node-icon" x="{x:.1f}" y="{y:.1f}" text-anchor="middle" dominant-baseline="central">'
                f'{html.escape(MAP_ICONS.get(room_type, "."))}</text>'
            )
            if room is not None:
                loss = int(room.get("hp_loss") or 0)
                node_shapes.append(
                    f'<text class="selected-room-badge" x="{x:.1f}" y="{y - radius - 7:.1f}" text-anchor="middle">'
                    f'#{room.get("room_index")} -{loss}</text>'
                )
                node_shapes.append(
                    f'<text class="route-topology-coordinate" x="{x:.1f}" y="{y + radius + 18:.1f}" text-anchor="middle">'
                    f'F{html.escape(str(room.get("floor")))} ({col},{row})</text>'
                )
            replan_entry = replan_by_coord.get((col, row))
            if replan_entry:
                replan_shapes.append(
                    _route_replan_flag_svg(
                        x=x,
                        y=y,
                        radius=radius,
                        stage=int(replan_entry.get("stage") or 0),
                        event=(replan_entry.get("replan") or {}).get("event") or {},
                        panel_left=left,
                        panel_right=right,
                        panel_top=panel_top,
                        panel_bottom=panel_bottom,
                    )
                )

    return f"""
<g class="route-topology-map" data-layout="horizontal" data-x-axis="row" data-y-axis="col">
{chr(10).join(legend_shapes)}
{chr(10).join(panel_shapes)}
{chr(10).join(edges)}
{chr(10).join(path_lines)}
{chr(10).join(node_shapes)}
{chr(10).join(replan_shapes)}
{chr(10).join(act_labels)}
</g>
"""


def _route_same_floor_map_svg(
    map_choices: list[dict[str, Any]],
    rooms: list[dict[str, Any]],
    *,
    margin_left: float,
    plot_right: float,
    map_top: float,
    map_bottom: float,
) -> str | None:
    visited = [room for room in rooms if room.get("room_index") is not None]
    if not visited:
        return None

    def coord_from_node(node: dict[str, Any] | None) -> tuple[int, int] | None:
        if not node:
            return None
        if node.get("col") is None or node.get("row") is None:
            return None
        return (int(node["col"]), int(node["row"]))

    choices_by_act_coord: dict[tuple[Any, int, int], list[dict[str, Any]]] = {}
    choices_by_coord: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for choice in map_choices:
        selected_coord = coord_from_node((choice.get("selected") or {}).get("node") or {})
        if selected_coord is None:
            continue
        act = choice.get("act")
        choices_by_act_coord.setdefault((act, selected_coord[0], selected_coord[1]), []).append(choice)
        choices_by_coord.setdefault(selected_coord, []).append(choice)

    def choice_for_room(room: dict[str, Any]) -> dict[str, Any] | None:
        room_coord = coord_from_node(room.get("node") or {})
        if room_coord is None:
            return None
        act_matches = choices_by_act_coord.get((room.get("act"), room_coord[0], room_coord[1])) or []
        if act_matches:
            return act_matches[0]
        coord_matches = choices_by_coord.get(room_coord) or []
        return coord_matches[0] if coord_matches else None

    min_room = float(visited[0]["room_index"])
    max_room = float(visited[-1]["room_index"])
    entries: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    cols: list[int] = []
    for room in visited:
        selected_coord = coord_from_node(room.get("node") or {})
        if selected_coord is None:
            continue
        selected_col, selected_row = selected_coord
        choice = choice_for_room(room)
        options_by_coord: dict[tuple[int, int], dict[str, Any]] = {}

        def add_option(node: dict[str, Any] | None, *, selected: bool = False, legal: bool = False) -> None:
            option_coord = coord_from_node(node)
            if option_coord is None:
                return
            if option_coord[1] != selected_row:
                return
            col, row = option_coord
            entry = options_by_coord.setdefault(
                option_coord,
                {
                    "col": col,
                    "row": row,
                    "type": None,
                    "selected": False,
                    "legal": False,
                },
            )
            node_type = (node or {}).get("type")
            if entry.get("type") is None and node_type is not None:
                entry["type"] = node_type
            entry["selected"] = bool(entry.get("selected") or selected)
            entry["legal"] = bool(entry.get("legal") or legal)

        snapshot = (choice or {}).get("map_snapshot") if choice else None
        if isinstance(snapshot, dict) and snapshot.get("type") == "map":
            for node in _snapshot_node_map(snapshot).values():
                add_option(node, selected=coord_from_node(node) == selected_coord)
        if choice:
            for route in choice.get("legal_routes") or []:
                add_option(route.get("node") or {}, legal=True)
            add_option((choice.get("selected") or {}).get("node") or {}, selected=True, legal=True)
        add_option(
            {
                "col": selected_col,
                "row": selected_row,
                "type": room.get("room_type"),
            },
            selected=True,
            legal=True,
        )
        for option in options_by_coord.values():
            if option.get("selected") and option.get("type") is None:
                option["type"] = room.get("room_type")
            if option.get("type") is None:
                option["type"] = "?"
            cols.append(int(option["col"]))
        ordered_options = sorted(options_by_coord.values(), key=lambda item: (int(item["col"]), int(item["row"])))
        entries.append((room, ordered_options))

    if not entries:
        return None
    min_col = min(cols) if cols else 0
    max_col = max(cols) if cols else 1
    if max_col == min_col:
        max_col += 1
    inner_top = map_top + 58
    inner_bottom = map_bottom - 28

    def sx(room: dict[str, Any]) -> float:
        if max_room == min_room:
            return (margin_left + plot_right) / 2
        room_index = float(room.get("room_index") or min_room)
        return margin_left + (room_index - min_room) / (max_room - min_room) * (plot_right - margin_left)

    def sy_col(col: int) -> float:
        value = int(col)
        return inner_top + (value - min_col) / (max_col - min_col) * (inner_bottom - inner_top)

    col_guides: list[str] = []
    for col in range(min_col, max_col + 1):
        y = sy_col(col)
        col_guides.append(
            f'<line class="same-floor-col-guide" x1="{margin_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}"/>'
        )
        col_guides.append(
            f'<text class="same-floor-coordinate" x="{margin_left - 8}" y="{y + 3:.1f}" text-anchor="end">col {col}</text>'
        )

    option_shapes: list[str] = []
    selected_points: list[tuple[float, float]] = []
    for room, options in entries:
        x = sx(room)
        loss = int(room.get("hp_loss") or 0)
        for option in options:
            room_type = str(option.get("type") or "?")
            is_selected = bool(option.get("selected"))
            y = sy_col(int(option["col"]))
            if is_selected:
                selected_points.append((x, y))
            radius = 20 if is_selected and room.get("large_loss") else 18 if is_selected else 13
            node_class = "same-floor-node selected" if is_selected else "same-floor-node option"
            if is_selected and room.get("large_loss"):
                node_class += " large-loss"
            title_parts = [
                f"act={room.get('act')}",
                f"floor={room.get('floor')}",
                f"col={option.get('col')}",
                f"row={option.get('row')}",
                f"type={room_type}",
            ]
            if is_selected:
                title_parts.append(f"loss={loss}")
            title = " ".join(title_parts)
            option_shapes.append(
                f'<circle class="{node_class}" cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{_room_type_fill(room_type)}">'
                f"<title>{html.escape(title)}</title></circle>"
            )
            option_shapes.append(
                f'<text class="route-node-icon" x="{x:.1f}" y="{y:.1f}" text-anchor="middle" dominant-baseline="central">'
                f'{html.escape(MAP_ICONS.get(room_type, "."))}</text>'
            )
            if not is_selected:
                continue
            option_shapes.append(
                f'<text class="selected-room-badge" x="{x:.1f}" y="{y - radius - 7:.1f}" text-anchor="middle">'
                f'#{room.get("room_index")} -{loss}</text>'
            )
            option_shapes.append(
                f'<text class="same-floor-coordinate" x="{x:.1f}" y="{y + radius + 18:.1f}" text-anchor="middle">'
                f'F{html.escape(str(room.get("floor")))} ({html.escape(str(option.get("col")))},{html.escape(str(option.get("row")))})</text>'
            )

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in selected_points)
    return f"""
<g class="same-floor-map-route">
<text class="section-title" x="{margin_left}" y="{map_top + 18}">Same-floor route choices</text>
<rect x="{margin_left}" y="{map_top + 28}" width="{plot_right - margin_left}" height="{map_bottom - map_top - 28}" fill="#fbfbfc" stroke="#d5d8de"/>
{chr(10).join(col_guides)}
<polyline class="same-floor-route-line" points="{points}"/>
{chr(10).join(option_shapes)}
</g>
"""


def _route_plan_stage_changes(route_plan_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    stage = 0
    for event in sorted(route_plan_events, key=lambda item: (_route_plan_event_step(item), item.get("line") or 0)):
        event_type = event.get("event_type")
        if event_type not in {"route_plan_created", "route_plan_updated"}:
            continue
        stage += 1
        changes.append({"stage": stage, "step": _route_plan_event_step(event), "event": event})
    return changes


def _route_plan_event_step(event: dict[str, Any]) -> int:
    try:
        return int(event.get("step"))
    except (TypeError, ValueError):
        return 10**9


def _route_plan_stage_for_step(step: Any, changes: list[dict[str, Any]]) -> int:
    try:
        step_value = int(step)
    except (TypeError, ValueError):
        return 0
    stage = 0
    for change in changes:
        if int(change.get("step") or 0) <= step_value:
            stage = int(change.get("stage") or 0)
        else:
            break
    return stage


def _route_plan_replan_for_step(step: Any, changes: list[dict[str, Any]]) -> dict[str, Any] | None:
    try:
        step_value = int(step)
    except (TypeError, ValueError):
        return None
    for change in changes:
        event = change.get("event") or {}
        if int(change.get("step") or 0) == step_value and event.get("event_type") == "route_plan_updated":
            return change
    return None


def _route_plan_stage_class(stage: int) -> str:
    if stage <= 0:
        return "route-plan-stage-0"
    return f"route-plan-stage-{((stage - 1) % len(ROUTE_PLAN_STAGE_COLORS)) + 1}"


def _route_plan_stage_legend_svg(changes: list[dict[str, Any]], x: float, y: float) -> list[str]:
    stages = [int(change.get("stage") or 0) for change in changes if int(change.get("stage") or 0) > 0]
    if not stages:
        return []
    unique_stages = []
    for stage in stages:
        if stage not in unique_stages:
            unique_stages.append(stage)
    parts = [
        f'<text class="route-stage-legend" x="{x:.1f}" y="{y:.1f}">Route stages:</text>',
    ]
    cursor = x + 112
    for stage in unique_stages[:5]:
        stage_class = _route_plan_stage_class(stage)
        parts.append(
            f'<text class="route-stage-legend {stage_class}" x="{cursor:.1f}" y="{y:.1f}">R{stage}</text>'
        )
        cursor += 34
    return parts


def _route_replan_flag_svg(
    *,
    x: float,
    y: float,
    radius: float,
    stage: int,
    event: dict[str, Any],
    panel_left: float,
    panel_right: float,
    panel_top: float,
    panel_bottom: float,
) -> str:
    reason_label = _route_replan_reason_label(event)
    full_reason = _route_replan_full_reason(event)
    flag_width = min(260, max(166, 72 + len(reason_label) * 7))
    flag_height = 50
    flag_x = x + radius + 16
    if flag_x + flag_width > panel_right - 4:
        flag_x = x - radius - 16 - flag_width
    flag_x = max(panel_left + 6, min(flag_x, panel_right - flag_width - 6))
    flag_y = y - radius - flag_height - 24
    if flag_y < panel_top + 6:
        flag_y = y + radius + 18
    if flag_y + flag_height > panel_bottom - 4:
        flag_y = max(panel_top + 6, y - radius - flag_height - 24)
    connector_x = flag_x if flag_x > x else flag_x + flag_width
    connector_y = flag_y + flag_height / 2
    stage_class = _route_plan_stage_class(stage)
    notch = 12
    flag_path = (
        f"M {flag_x:.1f} {flag_y:.1f} "
        f"h {flag_width - notch:.1f} l {notch:.1f} {flag_height / 2:.1f} "
        f"l {-notch:.1f} {flag_height / 2:.1f} h {-flag_width + notch:.1f} z"
    )
    return (
        f'<g class="route-replan-marker" data-route-stage="{stage}">'
        f'<title>{html.escape(full_reason)}</title>'
        f'<line class="replan-connector" x1="{x:.1f}" y1="{y:.1f}" x2="{connector_x:.1f}" y2="{connector_y:.1f}"/>'
        f'<path class="replan-flag {stage_class}" d="{flag_path}"/>'
        f'<text class="replan-flag-title" x="{flag_x + 10:.1f}" y="{flag_y + 19:.1f}">REPLAN R{stage}</text>'
        f'<text class="replan-flag-reason" x="{flag_x + 10:.1f}" y="{flag_y + 38:.1f}">'
        f'{html.escape(_truncate_svg_text(reason_label, 28))}</text>'
        "</g>"
    )


def _route_replan_reason_label(event: dict[str, Any]) -> str:
    tags = [str(tag) for tag in event.get("reason_tags") or [] if str(tag)]
    if tags:
        return ", ".join(tags)
    return str(event.get("reason") or "route changed")


def _route_replan_full_reason(event: dict[str, Any]) -> str:
    reason = str(event.get("reason") or "")
    tags = _route_replan_reason_label(event)
    if reason and tags and reason != tags:
        return f"{tags}: {reason}"
    return reason or tags


def _snapshot_node_map(snapshot: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    node_map: dict[tuple[int, int], dict[str, Any]] = {}
    for row_nodes in snapshot.get("rows") or []:
        for node in row_nodes:
            if node.get("col") is None or node.get("row") is None:
                continue
            node_map[(int(node["col"]), int(node["row"]))] = node
    boss = snapshot.get("boss") or {}
    if boss.get("col") is not None and boss.get("row") is not None:
        node_map[(int(boss["col"]), int(boss["row"]))] = boss
    return node_map


def _action_room_groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for item in items:
        key = (item.get("act"), item.get("floor"), item.get("room_type"))
        if current is None or current.get("key") != key:
            current = {
                "key": key,
                "act": item.get("act"),
                "floor": item.get("floor"),
                "room_type": item.get("room_type"),
                "start_x": item["x"],
                "end_x": item["x"],
            }
            groups.append(current)
            continue
        current["end_x"] = item["x"]
    return groups


def _room_type_color(room_type: Any) -> str:
    colors = {
        "Monster": "#d55e00",
        "Elite": "#b2182b",
        "Boss": "#6a3d9a",
        "RestSite": "#1b9e77",
        "Shop": "#2b8cbe",
        "Treasure": "#d8a600",
        "Event": "#7570b3",
        "Map": "#8c8c8c",
        "Ancient": "#8a7fdb",
    }
    return colors.get(str(room_type), "#666666")


def _room_type_fill(room_type: Any) -> str:
    fills = {
        "Monster": "#f1c27d",
        "Elite": "#d55e5e",
        "Boss": "#6b4c9a",
        "RestSite": "#70ad74",
        "Shop": "#6aaed6",
        "Treasure": "#d8b84e",
        "Event": "#c9c9c9",
        "Unknown": "#c9c9c9",
        "Ancient": "#8a7fdb",
        "Map": "#d0d0d0",
    }
    return fills.get(str(room_type), "#dddddd")


def _loss_severity_class(room: dict[str, Any]) -> str:
    if room.get("large_loss"):
        return "large-loss"
    loss = int(room.get("hp_loss") or 0)
    if loss >= 10:
        return "medium-loss"
    if loss > 0:
        return "small-loss"
    return "no-loss"


def _loss_severity_color(room: dict[str, Any]) -> str:
    severity = _loss_severity_class(room)
    return {
        "large-loss": "#b2182b",
        "medium-loss": "#d55e00",
        "small-loss": "#e6ab02",
        "no-loss": "#1b9e77",
    }[severity]


def _action_event_rows(
    items: list[dict[str, Any]],
    drop_markers: list[tuple[dict[str, Any], dict[str, Any], float]],
) -> list[dict[str, str]]:
    if not items:
        return []
    rows: list[dict[str, str]] = []

    def add(kind: str, item: dict[str, Any], color: str, extra: str = "") -> None:
        detail = f"action {int(item['x'])}; {_location_label(item)}"
        if extra:
            detail = f"{detail}; {extra}"
        rows.append(
            {
                "label": f"{kind}: HP {_hp_label(item)}",
                "detail": detail,
                "color": color,
            }
        )

    first = items[0]
    min_item = min(items, key=lambda item: item["hp_value"])
    last = items[-1]
    add("Start", first, "#2166ac")
    if min_item is not first:
        add("Min", min_item, "#b2182b" if min_item["hp_value"] <= min_item["max_hp_value"] * LOW_HP_RATIO else "#2166ac")
    if last is not first and last is not min_item:
        add("End", last, "#b2182b" if last["hp_value"] <= last["max_hp_value"] * LOW_HP_RATIO else "#2166ac")

    for _prev, item, loss in sorted(drop_markers, key=lambda marker: marker[2], reverse=True)[:5]:
        add(f"Drop -{int(loss)}", item, "#d55e00")

    deduped = []
    seen = set()
    for row in rows:
        key = (row["label"], row["detail"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _room_loss_rows(rooms: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for room in sorted(rooms, key=lambda item: int(item.get("hp_loss") or 0), reverse=True):
        loss = int(room.get("hp_loss") or 0)
        if loss <= 0:
            continue
        room_type = str(room.get("room_type") or "?")
        icon = MAP_ICONS.get(room_type, ".")
        color = "#b2182b" if room.get("large_loss") else "#d55e00" if loss >= 5 else "#2166ac"
        entry = room.get("entry_hp")
        entry_max = room.get("entry_max_hp")
        exit_hp = room.get("exit_hp")
        exit_max = room.get("exit_max_hp")
        status = "; large loss" if room.get("large_loss") else ""
        rows.append(
            {
                "label": f"#{room.get('room_index')} {icon} {room_type}: -{loss}",
                "detail": f"{entry}/{entry_max} -> {exit_hp}/{exit_max}{status}",
                "color": color,
            }
        )
    return rows


def _truncate_svg_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _action_point_label(item: dict[str, Any]) -> str:
    parts = [
        f"action {int(item['x'])}",
        f"HP {_hp_label(item)}",
        _location_label(item),
    ]
    if item.get("action"):
        parts.append(str(item.get("action")))
    if item.get("action_kind"):
        parts.append(str(item.get("action_kind")))
    return " | ".join(part for part in parts if part)


def _hp_label(item: dict[str, Any]) -> str:
    return f"{int(item['hp_value'])}/{int(item['max_hp_value'])}"


def _location_label(item: dict[str, Any]) -> str:
    return f"A{item.get('act')}F{item.get('floor')} {item.get('room_type')}"


def _room_hp_title(room: dict[str, Any]) -> str:
    status = " large loss" if room.get("large_loss") else ""
    return (
        f"Room {room.get('room_index')} A{room.get('act')}F{room.get('floor')} {room.get('room_type')} | "
        f"Entry {room.get('entry_hp')}/{room.get('entry_max_hp')} | "
        f"Exit {room.get('exit_hp')}/{room.get('exit_max_hp')} | "
        f"loss {room.get('hp_loss')}{status}"
    )


def _write_line_svg(points: list[tuple[Any, Any, Any]], path: Path, *, title: str, x_label: str, y_label: str) -> None:
    width, height = 920, 360
    margin_left, margin_right, margin_top, margin_bottom = 56, 24, 42, 48
    clean = [(float(x), float(y), float(max_hp or y or 1)) for x, y, max_hp in points if x is not None and y is not None]
    if not clean:
        path.write_text(_empty_svg(width, height, title), encoding="utf-8")
        return
    min_x = min(x for x, _, _ in clean)
    max_x = max(x for x, _, _ in clean)
    max_y = max(max(hp, max_hp) for _, hp, max_hp in clean)
    if max_x == min_x:
        max_x += 1
    if max_y <= 0:
        max_y = 1

    def sx(x: float) -> float:
        return margin_left + (x - min_x) / (max_x - min_x) * (width - margin_left - margin_right)

    def sy(y: float) -> float:
        return height - margin_bottom - y / max_y * (height - margin_top - margin_bottom)

    coords = [(sx(x), sy(y), x, y) for x, y, _ in clean]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in coords)
    circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5"><title>{x_label} {raw_x:g}: HP {raw_y:g}</title></circle>'
        for x, y, raw_x, raw_y in coords
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, sans-serif; font-size: 13px; fill: #222; }}
.axis {{ stroke: #333; stroke-width: 1.2; }}
.grid {{ stroke: #ddd; stroke-width: 1; }}
polyline {{ fill: none; stroke: #2166ac; stroke-width: 2.5; }}
circle {{ fill: #b2182b; stroke: white; stroke-width: 1.2; }}
</style>
<rect width="100%" height="100%" fill="#fff"/>
<text x="{margin_left}" y="24" font-size="18" font-weight="700">{html.escape(title)}</text>
<line class="axis" x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}"/>
<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}"/>
<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle">{html.escape(x_label)}</text>
<text x="16" y="{height / 2:.1f}" transform="rotate(-90 16 {height / 2:.1f})" text-anchor="middle">{html.escape(y_label)}</text>
<text x="{margin_left - 8}" y="{sy(max_y):.1f}" text-anchor="end">{max_y:g}</text>
<text x="{margin_left - 8}" y="{height - margin_bottom}" text-anchor="end">0</text>
<line class="grid" x1="{margin_left}" y1="{sy(max_y):.1f}" x2="{width - margin_right}" y2="{sy(max_y):.1f}"/>
<polyline points="{polyline}"/>
{circles}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_map_path_svg(report: dict[str, Any], path: Path) -> None:
    map_choices = report.get("map_choices") or []
    if any(isinstance(choice.get("map_snapshot"), dict) for choice in map_choices):
        _write_full_map_path_svg(map_choices, path)
        return

    nodes: dict[tuple[Any, int, int], dict[str, Any]] = {}
    selected_by_act: dict[Any, list[tuple[int, int]]] = {}
    for choice in map_choices:
        act = choice.get("act")
        for route in choice.get("legal_routes") or []:
            node = route.get("node") or {}
            if node.get("col") is not None and node.get("row") is not None:
                key = (act, int(node["col"]), int(node["row"]))
                entry = nodes.setdefault(key, {"type": node.get("type"), "selected": False})
                if entry.get("type") is None and node.get("type") is not None:
                    entry["type"] = node.get("type")
        node = ((choice.get("selected") or {}).get("node") or {})
        if node.get("col") is not None and node.get("row") is not None:
            key = (act, int(node["col"]), int(node["row"]))
            entry = nodes.setdefault(key, {"type": node.get("type"), "selected": False})
            if entry.get("type") is None and node.get("type") is not None:
                entry["type"] = node.get("type")
            entry["selected"] = True
            selected_by_act.setdefault(act, []).append((int(node["col"]), int(node["row"])))

    acts = sorted(
        {key[0] for key in nodes},
        key=lambda item: (item is None, 0 if isinstance(item, int) else 1, item if isinstance(item, int) else str(item)),
    )
    width, height = max(760, 300 * max(1, len(acts))), 620
    if not nodes:
        path.write_text(_empty_svg(width, height, "Map path"), encoding="utf-8")
        return

    panel_gap = 24
    panel_top = 55
    panel_bottom = height - 45
    panel_width = (width - 80 - panel_gap * max(0, len(acts) - 1)) / max(1, len(acts))
    node_shapes = []
    path_lines = []
    for act_index, act in enumerate(acts):
        left = 40 + act_index * (panel_width + panel_gap)
        right = left + panel_width
        act_nodes = {
            (col, row): item
            for (node_act, col, row), item in nodes.items()
            if node_act == act
        }
        min_col = min(col for col, _ in act_nodes)
        max_col = max(col for col, _ in act_nodes)
        min_row = min(row for _, row in act_nodes)
        max_row = max(row for _, row in act_nodes)
        if max_col == min_col:
            max_col += 1
        if max_row == min_row:
            max_row += 1

        def sx(col: int, *, left: float = left, right: float = right) -> float:
            return left + 30 + (col - min_col) / (max_col - min_col) * (right - left - 60)

        def sy(row: int) -> float:
            return panel_bottom - (row - min_row) / (max_row - min_row) * (panel_bottom - panel_top - 35)

        label = f"Act {act}" if act is not None else "Act ?"
        node_shapes.append(f'<text x="{left:.1f}" y="48" font-size="14" font-weight="700">{html.escape(label)}</text>')
        node_shapes.append(
            f'<rect x="{left:.1f}" y="{panel_top - 18:.1f}" width="{panel_width:.1f}" height="{panel_bottom - panel_top + 30:.1f}" '
            f'fill="none" stroke="#d0d0d0" stroke-width="1"/>'
        )
        selected = selected_by_act.get(act) or []
        if selected:
            path_lines.append(
                '<polyline points="{}" fill="none" stroke="#2166ac" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'.format(
                    " ".join(f"{sx(col):.1f},{sy(row):.1f}" for col, row in selected)
                )
            )

        for (col, row), item in sorted(act_nodes.items(), key=lambda pair: (pair[0][1], pair[0][0])):
            chosen = item.get("selected")
            fill = "#b2182b" if chosen else "#d9d9d9"
            stroke = "#7f0000" if chosen else "#777"
            room_type = item.get("type") or "?"
            escaped_type = html.escape(str(room_type))
            node_shapes.append(
                f'<circle cx="{sx(col):.1f}" cy="{sy(row):.1f}" r="{9 if chosen else 7}" fill="{fill}" stroke="{stroke}" stroke-width="1.5">'
                f"<title>act={html.escape(str(act))} col={col} row={row} type={escaped_type}</title></circle>"
            )
            node_shapes.append(f'<text x="{sx(col):.1f}" y="{sy(row) - 13:.1f}" text-anchor="middle">{escaped_type}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text {{ font-family: Arial, sans-serif; font-size: 12px; fill: #222; }}</style>
<rect width="100%" height="100%" fill="#fff"/>
<text x="40" y="28" font-size="18" font-weight="700">Map path</text>
{chr(10).join(path_lines)}
{chr(10).join(node_shapes)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_full_map_path_svg(map_choices: list[dict[str, Any]], path: Path) -> None:
    maps_by_act: dict[Any, dict[str, Any]] = {}
    selected_by_act: dict[Any, list[tuple[int, int]]] = {}
    legal_by_act: dict[Any, set[tuple[int, int]]] = {}

    for choice in map_choices:
        act = choice.get("act")
        snapshot = choice.get("map_snapshot")
        if isinstance(snapshot, dict) and snapshot.get("type") == "map":
            maps_by_act.setdefault(act, snapshot)
            current = snapshot.get("current_coord") or {}
            if current.get("col") is not None and current.get("row") is not None:
                selected_by_act.setdefault(act, [])
                start = (int(current["col"]), int(current["row"]))
                if not selected_by_act[act]:
                    selected_by_act[act].append(start)
        for route in choice.get("legal_routes") or []:
            node = route.get("node") or {}
            if node.get("col") is not None and node.get("row") is not None:
                legal_by_act.setdefault(act, set()).add((int(node["col"]), int(node["row"])))
        selected = ((choice.get("selected") or {}).get("node") or {})
        if selected.get("col") is not None and selected.get("row") is not None:
            selected_by_act.setdefault(act, [])
            coord = (int(selected["col"]), int(selected["row"]))
            if not selected_by_act[act] or selected_by_act[act][-1] != coord:
                selected_by_act[act].append(coord)

    acts = sorted(
        maps_by_act,
        key=lambda item: (item is None, 0 if isinstance(item, int) else 1, item if isinstance(item, int) else str(item)),
    )
    if not acts:
        path.write_text(_empty_svg(760, 620, "Full map path"), encoding="utf-8")
        return

    width = max(820, 360 * len(acts))
    height = 760
    panel_gap = 24
    panel_top = 72
    panel_bottom = height - 72
    panel_width = (width - 80 - panel_gap * max(0, len(acts) - 1)) / max(1, len(acts))
    edges: list[str] = []
    path_lines: list[str] = []
    node_shapes: list[str] = []

    room_fill = {
        "Monster": "#f1c27d",
        "Elite": "#d55e5e",
        "Boss": "#6b4c9a",
        "RestSite": "#70ad74",
        "Shop": "#6aaed6",
        "Treasure": "#d8b84e",
        "Event": "#c9c9c9",
        "Unknown": "#c9c9c9",
        "Ancient": "#8a7fdb",
    }

    for act_index, act in enumerate(acts):
        snapshot = maps_by_act[act]
        rows = snapshot.get("rows") or []
        boss = snapshot.get("boss") or {}
        node_map: dict[tuple[int, int], dict[str, Any]] = {}
        for row_nodes in rows:
            for node in row_nodes:
                if node.get("col") is None or node.get("row") is None:
                    continue
                node_map[(int(node["col"]), int(node["row"]))] = node
        if boss.get("col") is not None and boss.get("row") is not None:
            node_map[(int(boss["col"]), int(boss["row"]))] = boss
        if not node_map:
            continue

        min_col = min(col for col, _ in node_map)
        max_col = max(col for col, _ in node_map)
        min_row = min(row for _, row in node_map)
        max_row = max(row for _, row in node_map)
        if max_col == min_col:
            max_col += 1
        if max_row == min_row:
            max_row += 1

        left = 40 + act_index * (panel_width + panel_gap)
        right = left + panel_width

        def sx(col: int, *, left: float = left, right: float = right) -> float:
            return left + 35 + (col - min_col) / (max_col - min_col) * (right - left - 70)

        def sy(row: int) -> float:
            return panel_bottom - (row - min_row) / (max_row - min_row) * (panel_bottom - panel_top - 35)

        label = f"Act {act}" if act is not None else "Act ?"
        node_shapes.append(f'<text x="{left:.1f}" y="52" font-size="15" font-weight="700">{html.escape(label)}</text>')
        node_shapes.append(
            f'<rect x="{left:.1f}" y="{panel_top - 26:.1f}" width="{panel_width:.1f}" height="{panel_bottom - panel_top + 50:.1f}" '
            f'fill="none" stroke="#d0d0d0" stroke-width="1"/>'
        )

        for (col, row), node in sorted(node_map.items(), key=lambda item: (item[0][1], item[0][0])):
            for child in node.get("children") or []:
                if child.get("col") is None or child.get("row") is None:
                    continue
                child_coord = (int(child["col"]), int(child["row"]))
                if child_coord not in node_map:
                    continue
                edges.append(
                    f'<line class="map-edge" x1="{sx(col):.1f}" y1="{sy(row):.1f}" '
                    f'x2="{sx(child_coord[0]):.1f}" y2="{sy(child_coord[1]):.1f}"/>'
                )

        selected = selected_by_act.get(act) or []
        if selected:
            points = " ".join(f"{sx(col):.1f},{sy(row):.1f}" for col, row in selected if (col, row) in node_map)
            if points:
                path_lines.append(
                    f'<polyline class="selected-path" points="{points}" fill="none" stroke="#2166ac" '
                    f'stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'
                )

        selected_set = set(selected)
        legal_set = legal_by_act.get(act) or set()
        current = snapshot.get("current_coord") or {}
        current_coord = None
        if current.get("col") is not None and current.get("row") is not None:
            current_coord = (int(current["col"]), int(current["row"]))

        for (col, row), node in sorted(node_map.items(), key=lambda item: (item[0][1], item[0][0])):
            room_type = str(node.get("type") or "?")
            fill = room_fill.get(room_type, "#dddddd")
            stroke = "#7f0000" if (col, row) in selected_set else "#1466cc" if (col, row) in legal_set else "#5f5548"
            stroke_width = 3 if (col, row) in selected_set or (col, row) == current_coord else 1.2
            radius = 16 if room_type == "Boss" else 13 if (col, row) in selected_set else 11
            icon = MAP_ICONS.get(room_type, ".")
            title = f"act={act} col={col} row={row} type={room_type}"
            node_shapes.append(
                f'<circle cx="{sx(col):.1f}" cy="{sy(row):.1f}" r="{radius}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}"><title>{html.escape(title)}</title></circle>'
            )
            node_shapes.append(
                f'<text x="{sx(col):.1f}" y="{sy(row) + 4:.1f}" text-anchor="middle" font-weight="700">{html.escape(icon)}</text>'
            )

    legend = " ".join(f"{html.escape(icon)}={html.escape(name)}" for name, icon in MAP_ICONS.items())
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, sans-serif; font-size: 12px; fill: #222; }}
.map-edge {{ stroke: #8b7c67; stroke-width: 2; opacity: 0.72; }}
.selected-path {{ opacity: 0.95; }}
</style>
<rect width="100%" height="100%" fill="#fbf8ef"/>
<text x="40" y="28" font-size="18" font-weight="700">Full map path</text>
{chr(10).join(edges)}
{chr(10).join(path_lines)}
{chr(10).join(node_shapes)}
<text x="40" y="{height - 24}" font-size="12">Legend: {legend}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _empty_svg(width: int, height: int, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<text x="24" y="32" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>
<text x="24" y="62" font-family="Arial, sans-serif" font-size="13">No data available.</text>
</svg>
"""


def _position_info(state: dict[str, Any]) -> dict[str, Any]:
    context = state.get("context") or {}
    return {
        "act": state.get("act") or context.get("act"),
        "floor": state.get("floor") or context.get("floor"),
        "room_type": context.get("room_type") or state.get("room_type"),
    }


def _hp_info(state: dict[str, Any]) -> dict[str, Any]:
    player = state.get("player") or {}
    return {
        "hp": player.get("hp") if player.get("hp") is not None else state.get("hp"),
        "max_hp": player.get("max_hp") if player.get("max_hp") is not None else state.get("max_hp"),
    }


def _deck_size_info(state: dict[str, Any]) -> int | None:
    player = state.get("player") or {}
    value = player.get("deck_size") if player.get("deck_size") is not None else state.get("deck_size")
    return int(value) if isinstance(value, int) else None


def _safe_ratio(hp: Any, max_hp: Any) -> float | None:
    if isinstance(hp, (int, float)) and isinstance(max_hp, (int, float)) and max_hp:
        return float(hp) / float(max_hp)
    return None


def _is_view_action(action: dict[str, Any]) -> bool:
    kind = str(action.get("kind") or "")
    command = (action.get("action") or {}).get("cmd") or {}
    return kind.startswith("view_") or command.get("cmd") == "bench_view"


def _legal_map_summary(item: dict[str, Any]) -> dict[str, Any]:
    label = item.get("label")
    command = item.get("cmd") or item.get("command") or {}
    command_node = _node_from_command(command)
    label_node = _node_from_label(label)
    node = command_node or label_node
    if command_node and label_node:
        node = dict(command_node)
        if node.get("type") is None:
            node["type"] = label_node.get("type")
    return {"label": label, "node": node}


def _node_from_action(action: dict[str, Any]) -> dict[str, Any] | None:
    return _node_from_command((action.get("action") or {}).get("cmd") or {}) or _node_from_label(action.get("label"))


def _node_from_command(command: dict[str, Any]) -> dict[str, Any] | None:
    args = command.get("args") or {}
    col = args.get("col")
    row = args.get("row")
    if col is None or row is None:
        return None
    return {"col": col, "row": row, "type": None}


def _node_from_label(label: Any) -> dict[str, Any] | None:
    if not label:
        return None
    match = re.search(r"go to (?P<type>\w+) at col=(?P<col>-?\d+) row=(?P<row>-?\d+)", str(label))
    if not match:
        return None
    return {
        "col": int(match.group("col")),
        "row": int(match.group("row")),
        "type": match.group("type"),
    }


def _action_summary(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": action.get("step"),
        "line": action.get("line_no"),
        "kind": action.get("kind"),
        "label": action.get("label"),
        "reason": _reason(action),
    }


def _reason(action: dict[str, Any]) -> str | None:
    meta = action.get("agent_meta") or {}
    parsed = meta.get("parsed")
    if isinstance(parsed, dict):
        for key in ("reason", "plan_update"):
            if parsed.get(key):
                return str(parsed[key])
    raw = meta.get("raw_response")
    return str(raw) if raw else None


def _card_name(card: Any) -> str | None:
    if not isinstance(card, dict):
        return None
    name = card.get("name")
    if isinstance(name, dict):
        return str(name.get("en") or name.get("name") or name.get("id") or "?")
    return str(name) if name is not None else None


def _card_name_from_label(label: Any) -> str | None:
    if not label:
        return None
    text = str(label)
    match = re.search(
        r"(?:pick|take|select|buy) card\s+\d+:\s+(.+?)(?:\s+for\s+\d+\s+gold)?$",
        text,
    )
    return match.group(1).strip() if match else None


def _selected_card_from_label(label: Any) -> str | None:
    if not label:
        return None
    return _card_name_from_label(label) or str(label)


def _is_card_reward_skip(action: dict[str, Any]) -> bool:
    return "skip" in str(action.get("kind") or "").lower() or "skip" in str(action.get("label") or "").lower()


def _is_unopened_combat_card_reward_skip(action: dict[str, Any]) -> bool:
    if str(action.get("kind") or "") != "combat_reward_skip":
        return False
    label = str(action.get("label") or "").lower()
    if label.startswith("confirm skipped card reward"):
        return False
    before = action.get("before_state") if isinstance(action.get("before_state"), dict) else {}
    last_result = before.get("last_action_result") if isinstance(before.get("last_action_result"), dict) else {}
    if last_result.get("action_kind") == "card_reward_skip":
        return False
    return "skip unopened card reward" in label or "skip reward" in label and "card_reward" in label


def _localized_card(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "name_zh": _card_zh_name(name),
        "bilingual": _card_bilingual_name(name),
    }


def _localized_deck_change(change: dict[str, Any]) -> dict[str, Any]:
    return {
        **change,
        **_localized_card(change["name"]),
    }


def _format_deck_change(change: dict[str, Any]) -> str:
    prefix = "-" if change.get("operation") == "remove" else "+"
    return f"{prefix}{change.get('bilingual') or _card_bilingual_name(change.get('name'))}"


def _card_bilingual_name(name: str | None) -> str:
    if not name:
        return ""
    zh = _card_zh_name(name)
    return f"{zh} / {name}" if zh != name else name


def _card_zh_name(name: str | None) -> str:
    if not name:
        return ""
    base, plus = _split_card_upgrade(name)
    key = _card_en_to_key().get(base)
    if key:
        return f"{_card_zh_titles().get(key, base)}{plus}"
    return name


def _card_has_zh_mapping(name: str) -> bool:
    base, _ = _split_card_upgrade(name)
    return base in _card_en_to_key()


def _split_card_upgrade(name: str) -> tuple[str, str]:
    base = str(name or "").strip()
    plus = ""
    while base.endswith("+"):
        base = base[:-1]
        plus += "+"
    return base, plus


_CARD_LOCALIZATION_CACHE: tuple[dict[str, str], dict[str, str]] | None = None


def _card_en_to_key() -> dict[str, str]:
    return _card_localization()[0]


def _card_zh_titles() -> dict[str, str]:
    return _card_localization()[1]


def _card_localization() -> tuple[dict[str, str], dict[str, str]]:
    global _CARD_LOCALIZATION_CACHE
    if _CARD_LOCALIZATION_CACHE is not None:
        return _CARD_LOCALIZATION_CACHE
    repo_root = Path(__file__).resolve().parents[1]
    eng_path = repo_root / "localization_eng" / "cards.json"
    zhs_path = repo_root / "localization_zhs" / "cards.json"
    try:
        eng = json.loads(eng_path.read_text(encoding="utf-8-sig"))
        zhs = json.loads(zhs_path.read_text(encoding="utf-8-sig"))
    except OSError:
        _CARD_LOCALIZATION_CACHE = ({}, {})
        return _CARD_LOCALIZATION_CACHE
    en_to_key = {
        value: key[: -len(".title")]
        for key, value in eng.items()
        if key.endswith(".title") and isinstance(value, str)
    }
    zh_titles = {
        key[: -len(".title")]: value
        for key, value in zhs.items()
        if key.endswith(".title") and isinstance(value, str)
    }
    _CARD_LOCALIZATION_CACHE = (en_to_key, zh_titles)
    return _CARD_LOCALIZATION_CACHE


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "room")).strip("_")
    return text or "room"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze an STS2 benchmark JSONL trace.")
    parser.add_argument("trace", help="Input benchmark JSONL trace")
    parser.add_argument("--out-dir", required=True, help="Directory for generated replay artifacts")
    parser.add_argument("--low-hp-ratio", type=float, default=LOW_HP_RATIO)
    parser.add_argument("--large-loss-hp", type=int, default=LARGE_LOSS_HP)
    parser.add_argument("--large-loss-ratio", type=float, default=LARGE_LOSS_RATIO)
    args = parser.parse_args(argv)
    report = analyze_trace(
        args.trace,
        args.out_dir,
        low_hp_ratio=args.low_hp_ratio,
        large_loss_hp=args.large_loss_hp,
        large_loss_ratio=args.large_loss_ratio,
    )
    print(json.dumps({"summary": report["run"], "out_dir": args.out_dir}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
