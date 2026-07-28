import hashlib
import itertools
import http.server
import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest
import sts2_bench.agents as agents_module

from sts2_bench.actions import UnsupportedDecisionError, build_legal_actions
from sts2_bench.agents import (
    ModelActionError,
    OpenAICompatAgent,
    PromptContract,
    RandomAgent,
    build_compact_assistant_message,
    build_prompt_contract,
    parse_json_object,
)
from sts2_bench.multi_agent import CoordinatorAgent, OutOfCombatContextAgent
from sts2_bench.run_benchmark import _redact_argv, load_benchmark_config
from sts2_bench.runner import (
    JsonlLogger,
    RunResult,
    _custom_sequence_feedback,
    _decision_log_entry,
    agent_from_args,
    run_one,
    summarize,
)
from sts2_bench.sequences import normalize_card_sequence_payload


@pytest.mark.parametrize(
    ("state", "expected_action_names"),
    [
        (
            {"decision": "map_select", "choices": [{"col": 0, "row": 1, "type": "Monster"}]},
            {"select_map_node"},
        ),
        (
            {
                "decision": "combat_play",
                "energy": 3,
                "hand": [{"index": 0, "name": "Strike", "cost": 1, "can_play": True}],
                "enemies": [],
                "player": {"potions": []},
            },
            {"play_card", "end_turn"},
        ),
        (
            {
                "decision": "combat_reward",
                "rewards": [{"index": 0, "kind": "potion", "can_claim": True, "can_skip": True}],
                "player": {"potions": []},
            },
            {"claim_reward", "skip_reward"},
        ),
        (
            {"decision": "card_reward", "cards": [{"index": 0, "name": "Card"}], "can_skip": True},
            {"select_card_reward", "skip_card_reward"},
        ),
        (
            {
                "decision": "card_select",
                "cards": [{"index": 0, "name": "Card"}],
                "min_select": 0,
                "max_select": 1,
                "can_skip": True,
            },
            {"select_cards", "skip_select"},
        ),
        (
            {"decision": "bundle_select", "bundles": [{"index": 4, "cards": []}]},
            {"select_bundle"},
        ),
        (
            {
                "decision": "event_choice",
                "options": [
                    {"index": 0, "title": "Open", "is_locked": False},
                    {"index": 1, "title": "Locked", "is_locked": True},
                ],
                "can_leave": True,
            },
            {"choose_option", "leave_room"},
        ),
        (
            {
                "decision": "rest_site",
                "options": [
                    {"index": 0, "title": "Enabled", "is_enabled": True},
                    {"index": 1, "title": "Disabled", "is_enabled": False},
                ],
            },
            {"choose_option"},
        ),
        (
            {
                "decision": "shop",
                "cards": [{"index": 0, "name": "Card", "is_stocked": True, "can_buy": True}],
                "relics": [{"index": 0, "name": "Relic", "is_stocked": True, "can_buy": True}],
                "potions": [{"index": 0, "name": "Potion", "is_stocked": True, "can_buy": True}],
                "card_removal_cost": 50,
                "can_remove_card": True,
            },
            {"buy_card", "buy_relic", "buy_potion", "remove_card", "leave_room"},
        ),
        (
            {
                "decision": "fake_merchant_shop",
                "relics": [{"index": 0, "name": "Relic", "is_stocked": True, "can_buy": True}],
                "can_throw_foul_potion": True,
                "foul_potion_action": {"action": "use_potion", "potion_index": 0},
                "can_leave": True,
            },
            {"buy_relic", "use_potion", "leave_room"},
        ),
        (
            {"decision": "treasure", "relics": [{"index": 0, "name": "Relic"}]},
            {"claim_relic"},
        ),
        (
            {"decision": "treasure", "relics": [], "can_proceed": True},
            {"proceed"},
        ),
        (
            {
                "decision": "crystal_sphere",
                "can_proceed": False,
                "can_use_big_tool": True,
                "can_use_small_tool": True,
                "clickable_cells": [{"x": 1, "y": 2}],
            },
            {"crystal_sphere_set_tool", "crystal_sphere_click_cell"},
        ),
        (
            {"decision": "event_result", "can_proceed": True},
            {"proceed"},
        ),
        (
            {"decision": "game_over"},
            set(),
        ),
    ],
)
def test_complete_decision_action_contract_matrix(state, expected_action_names):
    actions = build_legal_actions(state, allow_card_sequences=False)
    action_names = {
        action.command.get("action")
        for action in actions
        if action.command.get("cmd") == "action"
    }
    assert action_names == expected_action_names


@pytest.mark.parametrize(
    "decision",
    ["event_blocked", "unrecognized_state", "unknown", "treasure_empty"],
)
def test_defensive_decision_states_have_no_implicit_actions(decision):
    with pytest.raises(UnsupportedDecisionError):
        build_legal_actions({"decision": decision})
from sts2_bench.sequence_validation import validate_custom_sequence


def test_event_result_has_explicit_proceed_action():
    actions = build_legal_actions(
        {
            "decision": "event_result",
            "can_proceed": True,
            "context": {"room_type": "Event"},
        }
    )

    assert any(action.kind == "event_result_proceed" for action in actions)


@pytest.mark.parametrize(
    "decision",
    ["unrecognized_state", "unknown", "event_blocked", "future_unhandled_decision"],
)
def test_unhandled_headless_decision_refuses_implicit_proceed(decision):
    with pytest.raises(UnsupportedDecisionError, match="refusing implicit proceed"):
        build_legal_actions({"decision": decision, "context": {"room_type": "Unknown"}})


def test_unknown_map_node_remains_a_legal_map_choice():
    actions = build_legal_actions(
        {
            "decision": "map_select",
            "choices": [{"type": "Unknown", "col": 2, "row": 4}],
        }
    )

    assert any(action.kind == "map" and "Unknown" in action.label for action in actions)


def test_parse_json_object_accepts_trailing_text_after_first_object():
    parsed = parse_json_object('{"action_id": 2, "reason": "take card"}\nExtra rationale.')

    assert parsed == {"action_id": 2, "reason": "take card"}


def test_parse_json_object_accepts_intro_and_fenced_object():
    parsed = parse_json_object(
        'Choice:\n```json\n{"action_id": 1, "reason": "route"}\n```\nDone.'
    )

    assert parsed == {"action_id": 1, "reason": "route"}


def test_openai_http_backends_send_byte_identical_model_request(monkeypatch):
    response_payload = {"choices": [{"message": {"content": "{}"}}]}
    response_bytes = json.dumps(response_payload).encode("utf-8")
    urllib_seen = {}
    session_seen = {}
    messages = [
        {"role": "system", "content": "完整上下文"},
        {"role": "user", "content": "Choose from every legal action."},
    ]
    original_messages = json.loads(json.dumps(messages, ensure_ascii=False))

    class UrlopenResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return response_bytes

    def fake_urlopen(request, timeout):
        urllib_seen.update(
            {
                "url": request.full_url,
                "body": request.data,
                "headers": {key.lower(): value for key, value in request.header_items()},
                "timeout": timeout,
            }
        )
        return UrlopenResponse()

    class SessionResponse:
        status_code = 200
        content = response_bytes

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeSession:
        def post(self, url, *, data, headers, timeout):
            session_seen.update(
                {
                    "url": url,
                    "body": data,
                    "headers": {key.lower(): value for key, value in headers.items()},
                    "timeout": timeout,
                }
            )
            return SessionResponse()

        def close(self):
            return None

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    urllib_agent = OpenAICompatAgent(
        base_url="https://example.invalid/v1",
        model="same-model",
        api_key="same-key",
        temperature=0.0,
        json_response_format=True,
    )
    session_agent = OpenAICompatAgent(
        base_url="https://example.invalid/v1",
        model="same-model",
        api_key="same-key",
        temperature=0.0,
        json_response_format=True,
        http_backend="requests_session",
    )
    session_agent._http_session = FakeSession()

    urllib_result = urllib_agent._chat(messages, timeout_override=17.0)
    session_result = session_agent._chat(messages, timeout_override=17.0)

    assert urllib_result == session_result == response_payload
    assert messages == original_messages
    assert urllib_seen["url"] == session_seen["url"]
    assert urllib_seen["body"] == session_seen["body"]
    assert urllib_seen["headers"] == session_seen["headers"]
    assert urllib_seen["headers"]["user-agent"] == "Mozilla/5.0 (compatible; sts2-agent-bench/1.0)"
    assert urllib_seen["timeout"] == 17.0
    decoded = json.loads(urllib_seen["body"].decode("utf-8"))
    assert decoded == {
        "model": "same-model",
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }


def test_requests_session_backend_reuses_one_http_connection(monkeypatch):
    client_ports = []
    response_body = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            self.rfile.read(content_length)
            client_ports.append(self.client_address[1])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, _format, *_args):
            return None

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = OpenAICompatAgent(
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        model="test",
        http_backend="requests_session",
    )
    try:
        messages = [{"role": "user", "content": "same payload"}]
        agent._chat(messages)
        agent._chat(messages)
    finally:
        agent.close_http_transport()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert len(client_ports) == 2
    assert len(set(client_ports)) == 1


def test_requests_session_worker_reuses_connection_and_preserves_exact_body(monkeypatch):
    client_ports = []
    request_bodies = []
    response_body = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            content_length = int(self.headers["Content-Length"])
            request_bodies.append(self.rfile.read(content_length))
            client_ports.append(self.client_address[1])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, _format, *_args):
            return None

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = OpenAICompatAgent(
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        model="test",
        api_key="secret",
        temperature=0.0,
        json_response_format=True,
        http_backend="requests_session_worker",
    )
    messages = [{"role": "system", "content": "完整卡牌信息"}, {"role": "user", "content": "全部合法动作"}]
    original_messages = json.loads(json.dumps(messages, ensure_ascii=False))
    expected_body = agent._chat_request_parts(messages)[1]
    try:
        agent._chat(messages, timeout_override=10.0)
        agent._chat(messages, timeout_override=10.0)
    finally:
        agent.close_http_transport()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert messages == original_messages
    assert request_bodies == [expected_body, expected_body]
    assert len(client_ports) == 2
    assert len(set(client_ports)) == 1


def test_requests_session_worker_enforces_hard_wall_clock_on_slow_trickle(monkeypatch):
    response_body = json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.close_connection = True
            content_length = int(self.headers["Content-Length"])
            self.rfile.read(content_length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            try:
                for byte in response_body:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, _format, *_args):
            return None

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    agent = OpenAICompatAgent(
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        model="test",
        http_backend="requests_session_worker",
    )
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="hard wall-clock timeout"):
            agent._chat([{"role": "user", "content": "slow"}], timeout_override=1.0)
    finally:
        elapsed = time.monotonic() - started
        agent.close_http_transport()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert 0.7 <= elapsed < 1.3


def test_requests_session_worker_deadline_includes_blocked_ipc_dispatch(monkeypatch):
    release = threading.Event()

    class FakeProcess:
        def terminate(self):
            release.set()

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

        def close(self):
            return None

    class BlockingConnection:
        def send(self, _value):
            release.wait(timeout=5.0)
            raise BrokenPipeError("closed at deadline")

        def close(self):
            release.set()

    process = FakeProcess()
    connection = BlockingConnection()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1/v1",
        model="test",
        http_backend="requests_session_worker",
    )

    def fake_ensure():
        agent._http_worker_process = process
        agent._http_worker_connection = connection
        return process, connection, agent._http_worker_epoch

    monkeypatch.setattr(agent, "_ensure_http_worker", fake_ensure)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="hard wall-clock timeout"):
        agent._chat([{"role": "user", "content": "large request"}], timeout_override=0.1)

    assert time.monotonic() - started < 0.3
    assert release.is_set()


def test_http_worker_epoch_prevents_timed_out_startup_from_overwriting_retry(monkeypatch):
    release_first_start = threading.Event()
    first_start_entered = threading.Event()
    processes = []

    class FakeConnection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self, index):
            self.index = index
            self.alive = False
            self.terminated = False
            self.closed = False

        def start(self):
            if self.index == 0:
                first_start_entered.set()
                release_first_start.wait(timeout=2.0)
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.terminate()

        def join(self, timeout=None):
            return None

        def close(self):
            self.closed = True

    class FakeContext:
        def Pipe(self, duplex=True):
            return FakeConnection(), FakeConnection()

        def Process(self, **_kwargs):
            process = FakeProcess(len(processes))
            processes.append(process)
            return process

    monkeypatch.setattr(agents_module.multiprocessing, "get_context", lambda _mode: FakeContext())
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1/v1",
        model="test",
        http_backend="requests_session_worker",
    )
    first_result = {}

    def start_first_generation():
        try:
            first_result["value"] = agent._ensure_http_worker()
        except Exception as exc:
            first_result["error"] = exc

    first_thread = threading.Thread(target=start_first_generation)
    first_thread.start()
    assert first_start_entered.wait(timeout=1.0)
    agent._force_abort_http_worker()
    second_process, second_connection, second_epoch = agent._ensure_http_worker()
    release_first_start.set()
    first_thread.join(timeout=2.0)

    assert isinstance(first_result.get("error"), urllib.error.URLError)
    assert processes[0].terminated is True
    assert second_process is processes[1]
    assert second_process.terminated is False
    assert agent._http_worker_process is second_process
    assert agent._http_worker_connection is second_connection
    assert agent._http_worker_epoch == second_epoch
    agent._force_abort_http_worker(second_process, second_connection, second_epoch)


def _json_after(prompt: str, marker: str):
    start = prompt.index(marker) + len(marker)
    return json.JSONDecoder().raw_decode(prompt[start:].lstrip())[0]


def test_meta_view_actions_include_combat_piles():
    state = {
        "decision": "combat_play",
        "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
        "hand": [],
        "enemies": [],
    }

    actions = build_legal_actions(state)
    by_kind = {action.kind: action for action in actions}

    assert by_kind["view_deck"].command == {"cmd": "bench_view", "view": "deck"}
    assert by_kind["view_map"].command == {"cmd": "bench_view", "view": "map"}
    assert by_kind["view_draw_pile"].command == {"cmd": "bench_view", "view": "draw"}
    assert by_kind["view_discard_pile"].command == {"cmd": "bench_view", "view": "discard"}
    assert by_kind["view_exhaust_pile"].command == {"cmd": "bench_view", "view": "exhaust"}


def test_meta_view_actions_do_not_repeat_by_default():
    state = {
        "decision": "map_select",
        "view_deck": True,
        "view_draw_pile": True,
        "choices": [],
    }

    actions = build_legal_actions(state)
    kinds = {action.kind for action in actions}

    assert "view_deck" not in kinds
    assert "view_draw_pile" not in kinds
    assert "view_map" in kinds
    assert "view_discard_pile" in kinds
    assert "view_exhaust_pile" in kinds


def test_card_select_requires_multi_card_actions_when_min_select_is_two():
    state = {
        "decision": "card_select",
        "min_select": 2,
        "max_select": 2,
        "cards": [
            {"index": 0, "name": "Bloodletting"},
            {"index": 1, "name": "Setup Strike"},
            {"index": 2, "name": "Sword Boomerang"},
        ],
    }

    actions = build_legal_actions(state)
    select_actions = [action for action in actions if action.kind == "card_select"]

    assert select_actions
    assert {action.command["args"]["indices"] for action in select_actions} == {
        "0,1",
        "0,2",
        "1,2",
    }
    assert all("," in action.command["args"]["indices"] for action in select_actions)
    assert not any(action.label.startswith("select card ") for action in select_actions)


def test_card_select_keeps_single_card_actions_when_min_select_is_one():
    state = {
        "decision": "card_select",
        "min_select": 1,
        "max_select": 1,
        "cards": [
            {"index": 3, "name": "Bash"},
            {"index": 4, "name": "Armaments"},
        ],
    }

    actions = build_legal_actions(state)
    select_actions = [action for action in actions if action.kind == "card_select"]

    assert [action.command["args"]["indices"] for action in select_actions] == ["3", "4"]
    assert [action.label for action in select_actions] == [
        "select card 3: Bash",
        "select card 4: Armaments",
    ]


def test_card_select_single_card_choices_are_not_capped():
    state = {
        "decision": "card_select",
        "min_select": 1,
        "max_select": 1,
        "cards": [{"index": idx, "name": f"Card {idx}"} for idx in range(70)],
    }

    actions = build_legal_actions(state)
    select_actions = [action for action in actions if action.kind == "card_select"]

    assert len(select_actions) == 70
    assert select_actions[-1].command["args"]["indices"] == "69"
    assert select_actions[-1].label == "select card 69: Card 69"


def test_card_select_can_skip_even_when_min_select_is_positive():
    state = {
        "decision": "card_select",
        "min_select": 1,
        "max_select": 1,
        "can_skip": True,
        "cards": [{"index": 0, "name": "Strike"}],
    }

    actions = build_legal_actions(state)

    assert any(action.kind == "card_select_skip" for action in actions)


def test_bundle_select_action_labels_include_pack_card_names():
    state = {
        "decision": "bundle_select",
        "bundles": [
            {
                "index": 0,
                "cards": [
                    {"name": "Setup Strike"},
                    {"name": "Cinder"},
                    {"name": "Ashen Strike"},
                ],
            },
            {
                "index": 1,
                "cards": [
                    {"name": "Iron Wave"},
                    {"name": "Shrug It Off"},
                    {"name": "Dismantle"},
                ],
            },
        ],
    }

    actions = [action for action in build_legal_actions(state) if action.kind == "bundle_select"]

    assert [action.label for action in actions] == [
        "select bundle 0: Setup Strike, Cinder, Ashen Strike",
        "select bundle 1: Iron Wave, Shrug It Off, Dismantle",
    ]


def test_combat_reward_exposes_skip_and_potion_discard_shortcuts():
    state = {
        "decision": "combat_reward",
        "rewards": [
            {"index": 0, "kind": "potion", "can_claim": False, "can_skip": True},
            {"index": 1, "kind": "gold", "amount": 12, "can_claim": True},
            {"index": 2, "kind": "card_reward", "can_claim": True, "can_skip": True},
        ],
        "player": {
            "potions": [
                {"index": 0, "name": "Potion A"},
                {"index": 2, "name": "Potion B"},
            ]
        },
    }

    actions = build_legal_actions(state)
    potion_skip = next(action for action in actions if action.label == "skip reward 0: potion")
    open_card_reward = next(action for action in actions if action.label == "open card reward 2: reveal offered cards")
    skip_unopened_card_reward = next(action for action in actions if action.label == "skip unopened card reward 2")

    assert potion_skip.command == {
        "cmd": "action",
        "action": "skip_reward",
        "args": {"reward_index": 0},
    }
    assert open_card_reward.command == {
        "cmd": "action",
        "action": "claim_reward",
        "args": {"reward_index": 2},
    }
    assert skip_unopened_card_reward.command == {
        "cmd": "action",
        "action": "skip_reward",
        "args": {"reward_index": 2},
    }
    assert not any(
        action.kind == "combat_reward" and action.command["args"]["reward_index"] == 0
        for action in actions
    )
    assert any(action.kind == "combat_reward" and action.command["args"]["reward_index"] == 1 for action in actions)
    assert {
        action.command["args"]["potion_index"]
        for action in actions
        if action.kind == "combat_reward_discard_potion"
    } == {0, 2}


def test_combat_reward_labels_card_reward_cleanup_after_skip():
    state = {
        "decision": "combat_reward",
        "rewards": [{"index": 0, "kind": "card_reward", "can_claim": False, "can_skip": True}],
    }

    actions = build_legal_actions(state)

    cleanup = next(action for action in actions if action.label == "confirm skipped card reward 0")
    assert cleanup.command == {
        "cmd": "action",
        "action": "skip_reward",
        "args": {"reward_index": 0},
    }


def test_combat_reward_blocked_card_reward_claims_only_hide_matching_pack():
    state = {
        "decision": "combat_reward",
        "rewards": [
            {"index": 0, "kind": "card_reward", "can_claim": True, "can_skip": True},
            {"index": 1, "kind": "card_reward", "can_claim": True, "can_skip": True},
        ],
    }

    actions = build_legal_actions(state, blocked_card_reward_claims={0})

    labels = {action.label for action in actions}
    assert "open card reward 0: reveal offered cards" not in labels
    assert "confirm skipped card reward 0" in labels
    assert "open card reward 1: reveal offered cards" in labels
    assert "skip unopened card reward 1" in labels


def test_event_choice_exposes_leave_when_available():
    state = {
        "decision": "event_choice",
        "can_leave": True,
        "options": [{"index": 0, "title": "Search", "is_locked": False}],
    }

    actions = build_legal_actions(state)

    assert any(action.kind == "event" for action in actions)
    leave = next(action for action in actions if action.kind == "event_leave")
    assert leave.command == {"cmd": "action", "action": "leave_room"}


def test_standard_shop_uses_only_engine_exported_affordances():
    state = {
        "decision": "shop",
        "cards": [
            {"index": 0, "name": "Too Expensive", "price": 101, "is_stocked": True, "can_buy": False},
            {"index": 2, "name": "Available", "price": 50, "is_stocked": True, "can_buy": True},
        ],
        "relics": [
            {"index": 0, "name": "Missing Affordance", "cost": 10, "is_stocked": True},
        ],
        "potions": [
            {"index": 0, "name": "Full Slots", "cost": 25, "is_stocked": True, "can_buy": False},
        ],
        "card_removal_cost": 50,
        "can_remove_card": False,
        "player": {"gold": 999},
    }

    actions = build_legal_actions(state)

    shop_actions = [action for action in actions if action.kind.startswith("shop_")]
    assert {action.kind for action in shop_actions} == {"shop_buy_card", "shop_leave"}
    buy = next(action for action in actions if action.kind == "shop_buy_card")
    assert buy.command["args"] == {"card_index": 2}


def test_reward_actions_fail_closed_when_engine_affordance_fields_are_missing():
    combat_actions = build_legal_actions(
        {
            "decision": "combat_reward",
            "rewards": [{"index": 0, "kind": "gold"}],
            "player": {"potions": []},
        }
    )
    card_actions = build_legal_actions(
        {
            "decision": "card_reward",
            "cards": [{"index": 0, "name": "Strike"}],
        }
    )

    assert not any(action.kind in {"combat_reward", "combat_reward_skip"} for action in combat_actions)
    assert not any(action.kind == "card_reward_skip" for action in card_actions)


def test_fake_merchant_shop_exposes_relic_purchase_and_leave_actions():
    state = {
        "decision": "fake_merchant_shop",
        "can_leave": True,
        "relics": [
            {"index": 0, "name": "Bag of Marbles", "cost": 95, "is_stocked": True, "can_buy": True},
            {"index": 1, "name": "Anchor", "cost": 120, "is_stocked": True, "can_buy": False},
            {"index": 2, "name": "Lantern", "cost": 110, "is_stocked": False, "can_buy": False},
        ],
        "player": {"gold": 100},
    }

    actions = build_legal_actions(state)

    buy_actions = [action for action in actions if action.kind == "fake_merchant_buy_relic"]
    assert len(buy_actions) == 1
    assert buy_actions[0].label == "buy fake merchant relic 0: Bag of Marbles for 95 gold"
    assert buy_actions[0].command == {
        "cmd": "action",
        "action": "buy_relic",
        "args": {"relic_index": 0},
    }
    leave = next(action for action in actions if action.kind == "fake_merchant_leave")
    assert leave.command == {"cmd": "action", "action": "leave_room"}


def test_fake_merchant_shop_exposes_foul_potion_action_when_available():
    state = {
        "decision": "fake_merchant_shop",
        "can_throw_foul_potion": True,
        "foul_potion_action": {"action": "use_potion", "potion_index": 2},
        "relics": [],
        "player": {"gold": 0},
    }

    actions = build_legal_actions(state)

    foul_potion = next(action for action in actions if action.kind == "fake_merchant_throw_foul_potion")
    assert foul_potion.command == {
        "cmd": "action",
        "action": "use_potion",
        "args": {"potion_index": 2},
    }


def test_fake_merchant_prompt_preserves_relic_effects_and_foul_potion_facts():
    state = {
        "decision": "fake_merchant_shop",
        "context": {"act": 1, "floor": 5, "room_type": "Shop"},
        "can_leave": True,
        "can_throw_foul_potion": True,
        "foul_potion_action": {"action": "use_potion", "potion_index": 2},
        "relics": [
            {
                "index": 0,
                "id": "RELIC.BAG_OF_MARBLES",
                "name": "Bag of Marbles",
                "cost": 95,
                "is_stocked": True,
                "can_buy": True,
                "description": "At the start of each combat, apply 1 Vulnerable to all enemies.",
                "hover_tips": [
                    {"name": "Vulnerable", "description": "Take 50% more Attack damage."}
                ],
            }
        ],
        "player": {
            "hp": 70,
            "max_hp": 80,
            "gold": 100,
            "deck": [],
            "deck_size": 0,
            "potions": [
                {
                    "index": 2,
                    "name": "Foul Potion",
                    "description": "Can be thrown at the fake merchant.",
                    "target_type": "None",
                }
            ],
        },
    }
    agent = OpenAICompatAgent(base_url="http://127.0.0.1:1", model="test")

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Fake merchant relics (1):" in prompt
    assert "At the start of each combat, apply 1 Vulnerable to all enemies." in prompt
    assert "tip: Vulnerable: Take 50% more Attack damage." in prompt
    assert "can_throw_foul_potion=True" in prompt
    assert '"action":"use_potion"' in prompt
    assert '"potion_index":2' in prompt


def test_crystal_sphere_exposes_tool_and_cell_actions():
    state = {
        "decision": "crystal_sphere",
        "can_proceed": False,
        "can_use_big_tool": True,
        "can_use_small_tool": True,
        "clickable_cells": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    }

    actions = build_legal_actions(state)

    assert {action.command["args"]["tool"] for action in actions if action.kind == "crystal_sphere_tool"} == {
        "big",
        "small",
    }
    assert {
        (action.command["args"]["x"], action.command["args"]["y"])
        for action in actions
        if action.kind == "crystal_sphere_cell"
    } == {(1, 2), (3, 4)}
    assert not any(action.kind == "crystal_sphere" for action in actions)


class _ChooseEndTurnAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "test"}


class _ViewThenEndTurnAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        kind = "combat_end_turn" if state.get("view_deck") else "view_deck"
        action = next(action for action in legal_actions if action.kind == kind)
        return action, {"agent": "test"}


class _RaisingAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        messages = [{"role": "user", "content": str(prompt or "") }]
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        raise ModelActionError(
            "LLM failed after 1 attempt(s): action_id out of range: 99",
            audit_meta={
                "model_request_audits": [
                    {
                        "request_index": 0,
                        "messages": messages,
                        "messages_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            },
        )


class _ErrorProcess:
    def start(self):
        pass

    def start_run(self, **kwargs):
        return {
            "decision": "combat_play",
            "context": {"act": 1, "floor": 1},
            "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
            "energy": 0,
            "hand": [],
            "enemies": [],
        }

    def send(self, command):
        return {"type": "error", "message": "forced engine error"}

    def close(self):
        pass


class _TaggedEngineErrorProcess:
    def __init__(self, *, immediate: bool):
        self.immediate = immediate
        self.sent_commands = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        if self.immediate:
            return {
                "decision": "game_over",
                "victory": False,
                "engine_error": True,
                "engine_error_reason": "initial terminal export failed",
                "context": {"act": 1, "floor": 7},
                "player": {"hp": 0, "max_hp": 80},
            }
        return {
            "decision": "combat_play",
            "context": {"act": 1, "floor": 1},
            "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
            "energy": 0,
            "hand": [],
            "enemies": [],
        }

    def send(self, command):
        self.sent_commands.append(command)
        return {
            "decision": "combat_reward",
            "error": "engine_error",
            "engine_error_reason": "reward export failed",
            "context": {"act": 1, "floor": 1},
            "player": {"hp": 80, "max_hp": 80},
        }

    def close(self):
        pass


class _NonTerminalEndTurnProcess:
    def start(self):
        pass

    def start_run(self, **kwargs):
        return {
            "decision": "combat_play",
            "context": {"act": 1, "floor": 1},
            "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
            "energy": 0,
            "max_energy": 3,
            "hand": [],
            "enemies": [],
        }

    def send(self, command):
        return {
            "decision": "combat_play",
            "context": {"act": 1, "floor": 1},
            "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
            "energy": 3,
            "max_energy": 3,
            "round": 2,
            "hand": [],
            "enemies": [],
        }

    def close(self):
        pass


class _ImmediateTerminalProcess:
    def start(self):
        pass

    def start_run(self, **kwargs):
        return {
            "decision": "game_over",
            "victory": False,
            "context": {"act": 1, "floor": 7, "room_type": "Monster"},
            "player": {
                "hp": 0,
                "max_hp": 80,
                "gold": 42,
                "deck_size": 2,
                "deck": [
                    {"id": "strike", "name": "Strike", "upgrade_level": 0},
                    {"id": "bash", "name": "Bash", "upgrade_level": 1},
                ],
                "relics": [{"id": "burning_blood", "name": "Burning Blood"}],
                "potions": [],
            },
        }

    def close(self):
        pass


class _UnsupportedDecisionProcess:
    def __init__(self, decision="unrecognized_state"):
        self.decision = decision
        self.sent_commands = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return {
            "decision": self.decision,
            "context": {"act": 1, "floor": 1, "room_type": "Unknown"},
            "player": {"hp": 80, "max_hp": 80},
        }

    def send(self, command):
        self.sent_commands.append(command)
        raise AssertionError("unsupported decisions must stop before sending a command")

    def close(self):
        pass


class _EmptyRecognizedDecisionProcess(_UnsupportedDecisionProcess):
    def __init__(self):
        super().__init__(decision="event_result")

    def start_run(self, **kwargs):
        state = super().start_run(**kwargs)
        state["context"]["room_type"] = "Event"
        state["can_proceed"] = False
        state.update(
            {
                "view_deck": True,
                "view_map": True,
                "view_draw_pile": True,
                "view_discard_pile": True,
                "view_exhaust_pile": True,
            }
        )
        return state


def test_runner_stops_after_engine_error_state():
    result = run_one(
        agent=_ChooseEndTurnAgent(),
        seed="test",
        process=_ErrorProcess(),
        max_steps=20,
    )

    assert result.error == "forced engine error"
    assert result.invalid_states == 1
    assert result.steps == 1
    assert not result.truncated


def test_runner_rejects_engine_error_marker_before_accepting_game_over():
    process = _TaggedEngineErrorProcess(immediate=True)

    result = run_one(
        agent=_ChooseEndTurnAgent(),
        seed="tagged-initial",
        process=process,
        max_steps=20,
    )

    assert result.error == "initial terminal export failed"
    assert result.invalid_states == 1
    assert result.decision == "game_over"
    assert process.sent_commands == []


def test_runner_rejects_engine_error_marker_after_action():
    process = _TaggedEngineErrorProcess(immediate=False)

    result = run_one(
        agent=_ChooseEndTurnAgent(),
        seed="tagged-after-action",
        process=process,
        max_steps=20,
        log_level="full",
    )

    assert result.error == "reward export failed"
    assert result.invalid_states == 1
    assert result.decision == "combat_reward"
    assert result.steps == 1
    assert len(process.sent_commands) == 1
    assert result.extra["last_observed_state"]["error"] == "engine_error"


def test_runner_logs_unsupported_decision_before_stopping(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    process = _UnsupportedDecisionProcess()
    logger = JsonlLogger(trace_path)
    try:
        result = run_one(
            agent=_ChooseEndTurnAgent(),
            seed="unsupported",
            process=process,
            logger=logger,
            log_level="full",
        )
    finally:
        logger.close()

    assert "unsupported decision" in result.error
    assert result.invalid_states == 1
    assert process.sent_commands == []
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    decision = next(row for row in rows if row.get("type") == "decision")
    assert decision["decision"] == "unrecognized_state"
    assert decision["report_required"] is True
    assert "refusing implicit proceed" in decision["decision_error"]
    assert decision["authoritative_state"]["decision"] == "unrecognized_state"


def test_runner_refuses_implicit_proceed_for_recognized_empty_action_state(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    process = _EmptyRecognizedDecisionProcess()
    logger = JsonlLogger(trace_path)
    try:
        result = run_one(
            agent=_ChooseEndTurnAgent(),
            seed="empty-actions",
            process=process,
            logger=logger,
            log_level="full",
        )
    finally:
        logger.close()

    assert result.error == (
        "recognized decision 'event_result' exposed no legal actions; refusing implicit proceed"
    )
    assert result.invalid_states == 1
    assert result.game_action_count == 0
    assert process.sent_commands == []
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert any(row.get("type") == "decision" and row.get("decision") == "event_result" for row in rows)
    error = next(row for row in rows if row.get("type") == "decision_error")
    assert error["report_required"] is True


def test_full_log_preserves_authoritative_terminal_state(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    logger = JsonlLogger(trace_path)
    try:
        result = run_one(
            agent=_ChooseEndTurnAgent(),
            seed="terminal",
            process=_ImmediateTerminalProcess(),
            logger=logger,
            log_level="full",
        )
    finally:
        logger.close()

    terminal = result.extra["authoritative_terminal_state"]
    assert terminal["player"]["deck"][1]["upgrade_level"] == 1
    assert terminal["player"]["relics"][0]["id"] == "burning_blood"
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["result"]["extra"]["authoritative_terminal_state"] == terminal


def test_full_log_preserves_authoritative_post_action_state(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    logger = JsonlLogger(trace_path)
    try:
        run_one(
            agent=_ChooseEndTurnAgent(),
            seed="post-action",
            process=_SafeCustomSequenceValidationProcess(),
            logger=logger,
            log_level="full",
            max_steps=2,
        )
    finally:
        logger.close()

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    action = next(row for row in rows if row.get("type") == "action")
    assert action["state_after"]["decision"] == "game_over"
    assert action["state_after"]["player"] == {"hp": 78, "max_hp": 80}


def test_summary_excludes_truncated_and_error_attempts_from_outcomes():
    completed = RunResult(
        seed="complete",
        character="Ironclad",
        ascension=0,
        decision="game_over",
        victory=True,
        floor=51,
        steps=100,
    )
    truncated = RunResult(
        seed="truncated",
        character="Ironclad",
        ascension=0,
        decision="combat_play",
        floor=99,
        steps=2000,
        truncated=True,
        extra={"truncation_reason": "max_game_actions exceeded (48)"},
    )
    failed = RunResult(
        seed="failed",
        character="Ironclad",
        ascension=0,
        decision="unrecognized_state",
        floor=88,
        error="unsupported decision",
    )

    summary = summarize([completed, truncated, failed])

    assert summary["runs"] == 3
    assert summary["completed"] == 1
    assert summary["victories"] == 1
    assert summary["deaths"] == 0
    assert summary["win_rate"] == 1.0
    assert summary["avg_floor"] == 51.0
    assert summary["truncated"] == 1
    assert summary["errors"] == 1
    assert summary["truncation_reasons"] == ["max_game_actions exceeded (48)"]
    assert "max_steps" not in summary["warning"]


def test_runner_max_game_actions_does_not_count_view_actions():
    result = run_one(
        agent=_ViewThenEndTurnAgent(),
        seed="test",
        process=_NonTerminalEndTurnProcess(),
        max_steps=20,
        max_game_actions=1,
    )

    assert result.steps == 2
    assert result.view_deck_count == 1
    assert result.game_action_count == 1
    assert result.truncated
    assert result.extra["truncation_reason"] == "max_game_actions exceeded (1)"


def test_runner_stops_without_action_and_keeps_state_when_agent_decision_fails(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    logger = JsonlLogger(trace_path)
    result = run_one(
        agent=_RaisingAgent(),
        seed="test",
        process=_NonTerminalEndTurnProcess(),
        max_steps=20,
        logger=logger,
        log_level="full",
    )
    logger.close()

    assert "action_id out of range: 99" in result.error
    assert result.steps == 0
    assert result.game_action_count == 0
    assert result.floor == 1
    assert result.hp == 80
    assert not result.truncated
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    model_error = next(row for row in rows if row.get("type") == "model_error")
    assert model_error["step"] == 0
    assert model_error["decision"] == "combat_play"
    assert model_error["agent_meta"]["model_request_audits"][0]["request_index"] == 0


def test_openai_compat_agent_prompt_includes_short_term_memory():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        short_term_memory_enabled=True,
        short_term_memory_window=2,
    )
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 120, "deck_size": 12},
        "choices": [],
    }
    action = next(
        action
        for action in build_legal_actions(
            {
                **state,
                "choices": [{"col": 1, "row": 3, "type": "Monster"}],
            }
        )
        if action.kind == "map"
    )

    agent.record_transition(
        state,
        action,
        {"parsed": {"reason": "Need one more hallway reward before resting."}},
    )
    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Episode memory:" in prompt
    assert "decision=map_select" in prompt
    assert "floor=3" in prompt
    assert "action=go to Monster at col=1 row=3" in prompt
    assert "Need one more hallway reward before resting." in prompt


def test_openai_compat_agent_prompt_includes_rule_based_run_summary():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        run_summary_enabled=True,
    )
    state = {
        "decision": "combat_play",
        "context": {
            "act": 1,
            "floor": 17,
            "room_type": "Boss",
            "boss": {"name": "Lagavulin Matriarch"},
        },
        "player": {
            "hp": 18,
            "max_hp": 85,
            "gold": 66,
            "deck_size": 22,
            "potion_slots": 4,
            "potions": [],
            "relics": [{"name": "Burning Blood"}, {"name": "Reptile Trinket"}],
        },
        "round": 9,
        "energy": 3,
        "max_energy": 3,
        "player_powers": [{"name": "Dexterity", "amount": -4, "type": "Buff"}],
        "enemies": [
            {
                "index": 0,
                "name": "Lagavulin Matriarch",
                "hp": 101,
                "max_hp": 222,
                "block": 12,
                "intents": [{"type": "Buff"}, {"type": "Debuff"}],
            }
        ],
        "hand": [],
    }

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Run summary:" in prompt
    assert "Write a concise COMBAT_EVALUATION using the current visible combat state and any Combat facts shown." in prompt
    assert "Combat facts:" in prompt
    assert "Combat skills:" in prompt
    assert "order: lethal > survival > reduce actions > damage/setup; one action" in prompt
    assert "no_attack: skip pure block; use damage/draw/scaling" in prompt
    assert "incoming_attack_total=0 current_block=0" in prompt
    assert "hp_if_end_turn_now" not in prompt
    assert "future_risk_check_reasons=boss_or_elite,low_hp,no_current_attack,non_attack_intent" in prompt
    assert "- position: act=1 floor=17 room=Boss boss=Lagavulin Matriarch" in prompt
    assert "- resources: hp=18/85 gold=66 deck_size=22 potions=0/4 relic_count=2" in prompt
    assert "low HP; survival and rest decisions need extra scrutiny" in prompt
    assert "no potions available for this high-risk fight" in prompt
    assert "negative player powers: Dexterity(-4)" in prompt
    assert "visible enemy intents include no attack damage this turn" in prompt


def test_openai_compat_agent_run_summary_can_be_disabled():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        run_summary_enabled=False,
    )
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 120, "deck_size": 12},
        "choices": [],
    }

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Run summary:" not in prompt


def test_openai_compat_agent_prompt_requires_selected_action_label_echo():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
    )
    state = _combat_turn_state()

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "selected_action_label" in prompt
    assert "copy the selected legal action label exactly" in prompt


def test_prompt_contract_is_derived_from_current_decision_and_legal_capabilities():
    combat_state = _combat_turn_state()
    combat_contract = build_prompt_contract(
        combat_state,
        build_legal_actions(combat_state),
        route_plan_enabled=True,
        combat_card_sequences_enabled=True,
        turn_check_enabled=True,
    )

    assert combat_contract == PromptContract(
        decision_type="combat_play",
        route_plan_enabled=False,
        combat_custom_sequence_enabled=False,
        context_patch_fields_enabled=False,
        turn_check_enabled=True,
    )

    custom_state = _custom_gateway_combat_state()
    custom_contract = build_prompt_contract(
        custom_state,
        build_legal_actions(custom_state),
        route_plan_enabled=True,
        combat_card_sequences_enabled=True,
        turn_check_enabled=True,
    )

    assert custom_contract == PromptContract(
        decision_type="combat_play",
        route_plan_enabled=False,
        combat_custom_sequence_enabled=True,
        context_patch_fields_enabled=False,
        turn_check_enabled=True,
    )

    map_state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 120, "deck_size": 12},
        "choices": [],
    }
    map_contract = build_prompt_contract(
        map_state,
        build_legal_actions(map_state),
        route_plan_enabled=True,
        combat_card_sequences_enabled=True,
        turn_check_enabled=True,
    )

    assert map_contract == PromptContract(
        decision_type="map_select",
        route_plan_enabled=True,
        combat_custom_sequence_enabled=False,
        context_patch_fields_enabled=True,
        turn_check_enabled=False,
    )


def test_route_plan_off_prompt_keeps_map_but_omits_plan_contract_and_empty_combat_schema():
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 1, "room_type": "Map"},
        "player": {
            "hp": 70,
            "max_hp": 80,
            "gold": 99,
            "deck_size": 1,
            "deck": [
                {
                    "name": "Defend",
                    "description": "Gain 5 Block.",
                    "hover_tips": [
                        {"name": "Block", "description": "Prevents damage."},
                        {"title": "Block", "description": "Prevents damage."},
                    ],
                }
            ],
        },
        "choices": [{"col": 1, "row": 1, "type": "Monster"}],
        "full_map": {
            "type": "map",
            "current_coord": {"col": 0, "row": 0},
            "rows": [
                [{"col": 0, "row": 0, "type": "Ancient", "children": [{"col": 1, "row": 1}]}],
                [{"col": 1, "row": 1, "type": "Monster", "children": []}],
            ],
        },
    }
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        route_plan_enabled=False,
        run_summary_enabled=False,
    )

    prompt = agent.build_prompt(state, build_legal_actions(state))
    route_map = _json_after(prompt, "Canonical route map JSON:\n")
    route_state = _json_after(prompt, "Route decision state JSON:\n")

    assert "ordered_plan_required" not in route_map["constraints"]
    assert "complete_visible_route_required" not in route_map["constraints"]
    assert len(route_state["player"]["deck"][0]["hover_tips"]) == 1
    assert "Active route plan" not in prompt
    assert "route_plan_event" not in prompt
    assert "combat_notes" not in prompt
    assert "COMBAT_EVALUATION" not in prompt


def test_prompt_context_matrix_keeps_contract_fragments_decision_scoped_and_bounded():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        route_plan_enabled=True,
        combat_card_sequences_enabled=True,
        run_summary_enabled=False,
    )
    map_state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 120, "deck_size": 12},
        "choices": [],
    }
    cases = [
        (
            "plain_combat",
            agent.build_prompt(_combat_turn_state(), build_legal_actions(_combat_turn_state())),
            (),
            ('"cmd":', "route_plan_event", "combat_custom_sequence", '"sequence"'),
            9000,
        ),
        (
            "custom_sequence_combat",
            agent.build_prompt(_custom_gateway_combat_state(), build_legal_actions(_custom_gateway_combat_state())),
            ("combat_custom_sequence", "top-level integer card_index"),
            ('"cmd":', "route_plan_event", "combat_card_sequence"),
            12000,
        ),
        (
            "map",
            agent.build_prompt(map_state, build_legal_actions(map_state)),
            ("route_plan_event", "new_plan"),
            ('"cmd":', "combat_custom_sequence", '"sequence"'),
            12000,
        ),
    ]

    for name, prompt, required, forbidden, max_chars in cases:
        assert len(prompt) <= max_chars, name
        for fragment in required:
            assert fragment in prompt, name
        for fragment in forbidden:
            assert fragment not in prompt, name


def test_openai_compat_agent_prompt_legal_actions_hide_engine_commands_but_full_trace_keeps_them():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        run_summary_enabled=False,
    )
    state = _combat_turn_state()
    legal_actions = build_legal_actions(state)

    prompt = agent.build_prompt(state, legal_actions)
    prompt_actions = _json_after(prompt, "Legal actions:\n")

    assert prompt_actions
    assert all("cmd" not in action for action in prompt_actions)
    assert {"action_id", "kind", "label"} <= set(prompt_actions[0])

    full_entry = _decision_log_entry(
        log_level="full",
        step=0,
        state=state,
        legal_actions=legal_actions,
        prompt=prompt,
    )

    assert full_entry is not None
    assert any("cmd" in action for action in full_entry["legal_actions"])


def test_non_map_prompt_omits_route_plan_contract_even_when_route_plans_are_enabled():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        route_plan_enabled=True,
        run_summary_enabled=False,
    )
    state = _combat_turn_state()

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "route_plan_event" not in prompt
    assert "route_plan_created" not in prompt
    assert "new_plan" not in prompt


def test_combat_prompt_without_custom_gateway_omits_custom_sequence_contract():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        combat_card_sequences_enabled=True,
        run_summary_enabled=False,
    )
    state = _combat_turn_state()
    legal_actions = build_legal_actions(state)

    assert not any(action.kind == "combat_custom_sequence" for action in legal_actions)

    prompt = agent.build_prompt(state, legal_actions)

    assert "combat_custom_sequence" not in prompt
    assert '"sequence"' not in prompt


def test_map_prompt_omits_custom_sequence_contract_even_when_sequence_feature_is_enabled():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        combat_card_sequences_enabled=True,
        route_plan_enabled=True,
        run_summary_enabled=False,
    )
    state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 120, "deck_size": 12},
        "choices": [],
    }

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "route_plan_event" in prompt
    assert "combat_custom_sequence" not in prompt
    assert '"sequence"' not in prompt


def test_openai_compat_agent_prompt_can_disable_model_generated_sequence_actions():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        combat_card_sequences_enabled=False,
    )
    state = _combat_turn_state()

    prompt = agent.build_prompt(state, build_legal_actions(state, allow_card_sequences=False))

    assert "combat_card_sequence" not in prompt
    assert "engine-side queued card sequence" not in prompt
    assert "combat_custom_sequence" not in prompt
    assert "sequence" not in prompt


def test_combat_play_omits_precomputed_sequence_actions_by_default_but_keeps_custom_gateway():
    actions = build_legal_actions(_custom_gateway_combat_state())

    assert not any(action.kind == "combat_card_sequence" for action in actions)
    assert any(action.kind == "combat_custom_sequence" for action in actions)


def test_openai_compat_agent_prompt_requires_custom_sequence_cost_audit_without_listed_sequence_bias():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
    )
    state = _custom_gateway_combat_state()

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Before choosing combat_custom_sequence" in prompt
    assert "sum each step's current cost against current energy" in prompt
    assert "combat_card_sequence" not in prompt
    assert "prefer a listed combat_card_sequence" not in prompt
    assert "If you cannot verify the whole sequence from the visible hand snapshot" in prompt
    assert "If an earlier step can kill or remove a targeted enemy" in prompt
    assert "shown damage is lethal against that enemy's current hp" in prompt
    assert "Track cumulative target-specific shown damage per target" in prompt
    assert "target-specific damage_by_target" in prompt
    assert "calculateddamage_by_target" in prompt
    assert "at most 4 card steps" in prompt
    assert "cost audit" in prompt
    assert "top-level integer card_index" in prompt
    assert "optional top-level integer target_index" in prompt
    assert "Do not wrap sequence steps in cmd, action, or args objects" in prompt
    assert "combat_custom_sequence is an action kind, not a selected_action_label" in prompt


def test_combat_mdp_state_prompt_omits_precomputed_sequence_kind():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_mdp_state",
    )
    state = _custom_gateway_combat_state()

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "combat_custom_sequence" in prompt
    assert "combat_card_sequence" not in prompt
    assert "engine-side queued card sequence" not in prompt


def test_combat_mdp_state_prompt_omits_combat_skill_hints_by_default():
    state = _combat_turn_state()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_mdp_state",
    )

    prompt = agent.build_prompt(state, build_legal_actions(state))
    mdp_state = _json_after(prompt, "Combat MDP state JSON:\n")

    assert "combat_guidance" not in mdp_state


def test_combat_mdp_state_prompt_can_include_structured_combat_skill_hints():
    state = _combat_turn_state()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_mdp_state",
        combat_skill_hints_enabled=True,
    )

    prompt = agent.build_prompt(state, build_legal_actions(state))
    mdp_state = _json_after(prompt, "Combat MDP state JSON:\n")

    assert mdp_state["combat_guidance"]["source"] == "derived_visible_state"
    assert mdp_state["combat_guidance"]["profile"] == "baseline"
    assert mdp_state["combat_guidance"]["mode"] == "structured"
    hints = mdp_state["combat_guidance"]["hints"]
    assert hints[0]["id"] == "priority_order"
    assert "lethal" in hints[0]["text"]
    assert any(hint["id"] == "survival" for hint in hints)


def test_turn_chat_prompt_includes_combat_skill_lines_by_default():
    state = _combat_turn_state()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="turn_chat",
    )

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Combat skills:" in prompt


def test_turn_chat_prompt_can_disable_combat_skill_lines():
    state = _combat_turn_state()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="turn_chat",
        combat_skill_lines_enabled=False,
    )

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "Combat skills:" not in prompt


def test_custom_sequence_feedback_warns_about_targets_removed_by_earlier_steps():
    feedback = _custom_sequence_feedback("custom sequence stopped after 1 step(s): enemy target 0 is not available")

    assert "If an earlier step can kill or remove a targeted enemy" in feedback
    assert "shown damage is lethal against that enemy's current hp" in feedback
    assert "Track cumulative target-specific shown damage per target" in feedback
    assert "target-specific damage_by_target" in feedback
    assert "calculateddamage_by_target" in feedback
    assert "at most 4 card steps" in feedback
    assert "cost audit" in feedback
    assert "top-level integer card_index" in feedback
    assert "Do not wrap sequence steps in cmd, action, or args objects" in feedback


def test_normalize_card_sequence_payload_rejects_engine_command_wrapped_steps_with_shape_hint():
    with pytest.raises(ValueError) as exc_info:
        normalize_card_sequence_payload(
            [
                {
                    "action": "play_card",
                    "args": {"card_index": 0, "target_index": 0},
                }
            ]
        )

    message = str(exc_info.value)
    assert "top-level integer card_index" in message
    assert "Do not wrap sequence steps in cmd, action, or args objects" in message


def test_openai_compat_agent_factual_memory_records_transition_diffs_not_reasons():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        short_term_memory_enabled=True,
        short_term_memory_window=2,
        short_term_memory_mode="factual_diff",
    )
    old_state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {"hp": 80, "max_hp": 80, "gold": 0, "deck_size": 10},
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 5,
        "discard_pile_count": 0,
        "hand": [{"index": 0, "name": "Strike", "cost": 1, "type": "Attack", "target_type": "AnyEnemy", "can_play": True}],
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 20, "max_hp": 40, "block": 0}],
    }
    new_state = {
        **old_state,
        "energy": 2,
        "hand": [],
        "discard_pile_count": 1,
        "enemies": [{"index": 0, "name": "Jaw Worm", "hp": 14, "max_hp": 40, "block": 0}],
    }
    action = next(action for action in build_legal_actions(old_state) if action.label.startswith("play card 0"))

    agent.record_transition(
        old_state,
        action,
        {"parsed": {"reason": "This reason should not be stored."}},
        new_state,
    )
    prompt = agent.build_prompt(new_state, build_legal_actions(new_state))

    assert "Episode memory:" in prompt
    assert "action=play card 0: Strike on enemy 0: Jaw Worm" in prompt
    assert "changes=energy: 3/3 -> 2/3" in prompt
    assert "discard_pile: 0 -> 1" in prompt
    assert "enemy Jaw Worm[0]: hp 20/40 -> 14/40" in prompt
    assert "This reason should not be stored." not in prompt


def test_agent_memory_resets_between_runs():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        short_term_memory_enabled=True,
    )
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 1, "room_type": "Monster"},
        "player": {"hp": 80, "max_hp": 80, "gold": 99, "deck_size": 10},
        "energy": 0,
        "hand": [],
        "enemies": [],
    }
    action = next(action for action in build_legal_actions(state) if action.kind == "combat_end_turn")

    agent.record_transition(state, action, {"parsed": {"reason": "No energy."}})
    assert "Episode memory:" in agent.build_prompt(state, build_legal_actions(state))

    agent.reset_episode()

    assert "Episode memory:" not in agent.build_prompt(state, build_legal_actions(state))


def _combat_turn_state(**overrides):
    state = {
        "decision": "combat_play",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {
            "hp": 70,
            "max_hp": 80,
            "block": 0,
            "gold": 0,
            "deck_size": 10,
            "potion_slots": 3,
            "potions": [],
        },
        "round": 1,
        "energy": 3,
        "max_energy": 3,
        "draw_pile_count": 5,
        "discard_pile_count": 0,
        "exhaust_pile_count": 0,
        "hand": [
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            }
        ],
        "enemies": [
            {
                "index": 0,
                "name": "Jaw Worm",
                "hp": 20,
                "max_hp": 40,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 7}],
            }
        ],
    }
    state.update(overrides)
    return state


def _custom_gateway_combat_state(**overrides):
    state = _combat_turn_state(
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
            {
                "index": 1,
                "name": "Defend",
                "cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"block": 5},
            },
        ],
    )
    state.update(overrides)
    return state


def _turn_chat_agent(responses, **kwargs):
    max_retries = kwargs.pop("max_retries", 0)
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="turn_chat",
        short_term_memory_enabled=False,
        max_retries=max_retries,
        **kwargs,
    )
    sent_messages = []
    response_iter = iter(responses)

    def fake_chat(messages):
        sent_messages.append([dict(message) for message in messages])
        response = next(response_iter)
        if isinstance(response, BaseException):
            raise response
        return {"choices": [{"message": {"content": json.dumps(response)}}]}

    agent._chat = fake_chat
    return agent, sent_messages


def _combat_mdp_agent(responses, **kwargs):
    max_retries = kwargs.pop("max_retries", 0)
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_mdp_state",
        short_term_memory_enabled=False,
        max_retries=max_retries,
        **kwargs,
    )
    sent_messages = []
    response_iter = iter(responses)

    def fake_chat(messages):
        sent_messages.append([dict(message) for message in messages])
        response = next(response_iter)
        if isinstance(response, BaseException):
            raise response
        return {"choices": [{"message": {"content": json.dumps(response)}}]}

    agent._chat = fake_chat
    return agent, sent_messages


def _single_turn_agent(responses, **kwargs):
    max_retries = kwargs.pop("max_retries", 0)
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        max_retries=max_retries,
        **kwargs,
    )
    sent_messages = []
    response_iter = iter(responses)

    def fake_chat(messages):
        sent_messages.append([dict(message) for message in messages])
        response = next(response_iter)
        if isinstance(response, BaseException):
            raise response
        return {"choices": [{"message": {"content": json.dumps(response)}}]}

    agent._chat = fake_chat
    return agent, sent_messages


def test_openai_compat_agent_retries_invalid_action_before_strict_error():
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 99, "reason": "bad id"},
            {"action_id": 5, "reason": "corrected to legal Strike."},
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.kind == "combat_play_card"
    assert meta["attempt"] == 1
    assert meta["previous_errors"] == ["action_id out of range: 99"]
    assert meta["previous_raw_responses"] == [json.dumps({"action_id": 99, "reason": "bad id"})]
    assert len(sent_messages) == 2
    assert [audit["messages"] for audit in meta["model_request_audits"]] == sent_messages
    assert [audit["request_index"] for audit in meta["model_request_audits"]] == [0, 1]
    expected_request_hashes = [
        hashlib.sha256(
            json.dumps(
                messages,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for messages in sent_messages
    ]
    assert [audit["messages_sha256"] for audit in meta["model_request_audits"]] == expected_request_hashes
    retry_text = sent_messages[1][-1]["content"]
    assert "previous response was invalid" in retry_text
    assert "between 0 and" in retry_text
    assert "no combat_custom_sequence gateway is available" in retry_text
    assert "sequence included only when choosing combat_custom_sequence" not in retry_text
    assert "top-level integer card_index" not in retry_text


def test_turn_chat_refuses_to_silently_drop_external_prompt_context():
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent(
        [{"action_id": 5, "reason": "Strike."}],
        fallback_mode="error",
    )

    with pytest.raises(ValueError, match="refusing to discard or flatten"):
        agent.choose(
            state,
            build_legal_actions(state),
            prompt=agent.build_prompt(state, build_legal_actions(state)) + "\nEXTERNAL MEMORY",
        )

    assert sent_messages == []


def test_openai_compat_agent_refuses_final_prompt_without_expected_rule_block():
    state = _combat_turn_state()
    legal_actions = build_legal_actions(state)
    agent, sent_messages = _single_turn_agent(
        [{"action_id": legal_actions[0].action_id, "reason": "unused"}],
        fallback_mode="error",
    )

    with pytest.raises(ValueError, match="expected game-rule block"):
        agent.choose(state, legal_actions, prompt="choose an action")

    assert sent_messages == []


def test_openai_compat_agent_audits_the_actual_final_rule_prompt():
    state = _combat_turn_state()
    legal_actions = build_legal_actions(state)
    selected = legal_actions[0]
    agent, sent_messages = _single_turn_agent(
        [{"action_id": selected.action_id, "reason": "Use the listed action."}],
        fallback_mode="error",
    )
    prompt = agent.build_prompt(state, legal_actions)

    action, meta = agent.choose(state, legal_actions, prompt=prompt)

    assert action == selected
    assert len(sent_messages) == 1
    assert meta["game_rule_exposure_valid"] is True
    assert meta["game_rule_occurrences"] == 1
    assert meta["game_rule_expected_occurrences"] == 1
    assert meta["game_rule_expected_block_sha256"] == meta["game_rule_render_sha256"]
    assert len(meta["game_rule_prompt_sha256"]) == 64


def test_openai_compat_agent_self_consistency_logs_votes_without_changing_by_default():
    state = {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 2, "room_type": "Reward"},
        "player": {"hp": 70, "max_hp": 80, "gold": 0, "deck_size": 10},
        "cards": [{"index": 0, "name": "Anger"}, {"index": 1, "name": "Shrug It Off"}],
        "can_skip": True,
    }
    actions = build_legal_actions(state)
    anger = next(action for action in actions if action.label == "pick card 0: Anger")
    shrug = next(action for action in actions if action.label == "pick card 1: Shrug It Off")
    agent, sent_messages = _single_turn_agent(
        [
            {"action_id": anger.action_id, "selected_action_label": anger.label, "reason": "Anger adds damage."},
            {"action_id": shrug.action_id, "selected_action_label": shrug.label, "reason": "Shrug It Off adds block and draw."},
            {"action_id": shrug.action_id, "selected_action_label": shrug.label, "reason": "Shrug It Off is safer."},
        ],
        self_consistency_enabled=True,
        self_consistency_samples=3,
        self_consistency_decisions=("card_reward",),
    )

    action, meta = agent.choose(state, actions)

    assert action.label == "pick card 0: Anger"
    assert len(sent_messages) == 3
    self_consistency = meta["self_consistency"]
    assert self_consistency["mode"] == "diagnostic"
    assert self_consistency["primary_action_id"] == anger.action_id
    assert self_consistency["selected_action_id"] == anger.action_id
    assert self_consistency["top_action_id"] == shrug.action_id
    assert self_consistency["top_vote_count"] == 2
    assert self_consistency["majority_vote_applied"] is False
    assert [audit["messages"] for audit in meta["model_request_audits"]] == sent_messages
    assert [audit["request_index"] for audit in meta["model_request_audits"]] == [0, 1, 2]


def test_openai_compat_agent_self_consistency_majority_vote_can_change_card_reward():
    state = {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 2, "room_type": "Reward"},
        "player": {"hp": 70, "max_hp": 80, "gold": 0, "deck_size": 10},
        "cards": [{"index": 0, "name": "Anger"}, {"index": 1, "name": "Shrug It Off"}],
        "can_skip": True,
    }
    actions = build_legal_actions(state)
    anger = next(action for action in actions if action.label == "pick card 0: Anger")
    shrug = next(action for action in actions if action.label == "pick card 1: Shrug It Off")
    agent, _sent_messages = _single_turn_agent(
        [
            {"action_id": anger.action_id, "selected_action_label": anger.label, "reason": "Anger adds damage."},
            {"action_id": shrug.action_id, "selected_action_label": shrug.label, "reason": "Shrug It Off adds block and draw."},
            {"action_id": shrug.action_id, "selected_action_label": shrug.label, "reason": "Shrug It Off is safer."},
        ],
        self_consistency_enabled=True,
        self_consistency_samples=3,
        self_consistency_decisions=("card_reward",),
        self_consistency_mode="majority_vote",
    )

    action, meta = agent.choose(state, actions)

    assert action.label == "pick card 1: Shrug It Off"
    assert meta["parsed"]["reason"] == "Shrug It Off adds block and draw."
    assert meta["self_consistency"]["majority_vote_applied"] is True


def test_openai_compat_agent_single_turn_retries_transport_error_without_prompt_correction(monkeypatch):
    monkeypatch.setattr("sts2_bench.agents.time.sleep", lambda _seconds: None)
    state = _combat_turn_state()
    agent, sent_messages = _single_turn_agent(
        [
            urllib.error.URLError("temporary EOF"),
            {"action_id": 5, "reason": "Recovered after transport retry."},
        ],
        max_retries=1,
        fallback_mode="error",
    )

    legal_actions = build_legal_actions(state)
    action, meta = agent.choose(state, legal_actions, prompt=agent.build_prompt(state, legal_actions))

    assert action.action_id == 5
    assert meta["attempt"] == 1
    assert len(sent_messages) == 2
    assert sent_messages[1] == sent_messages[0]
    assert "previous response was invalid" not in "\n".join(message["content"] for message in sent_messages[1])


def test_openai_compat_agent_single_turn_retries_llm_http_500_without_prompt_correction(monkeypatch):
    monkeypatch.setattr("sts2_bench.agents.time.sleep", lambda _seconds: None)
    state = _combat_turn_state()
    agent, sent_messages = _single_turn_agent(
        [
            RuntimeError("LLM HTTP 500: upstream connect failed"),
            {"action_id": 5, "reason": "Recovered after upstream retry."},
        ],
        max_retries=1,
        fallback_mode="error",
    )

    legal_actions = build_legal_actions(state)
    action, meta = agent.choose(state, legal_actions, prompt=agent.build_prompt(state, legal_actions))

    assert action.action_id == 5
    assert meta["attempt"] == 1
    assert len(sent_messages) == 2
    assert sent_messages[1] == sent_messages[0]
    assert "previous response was invalid" not in "\n".join(message["content"] for message in sent_messages[1])


def test_openai_compat_agent_turn_chat_retries_transport_error_without_prompt_correction(monkeypatch):
    monkeypatch.setattr("sts2_bench.agents.time.sleep", lambda _seconds: None)
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent(
        [
            urllib.error.URLError("temporary EOF"),
            {"action_id": 5, "reason": "Recovered after transport retry."},
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.action_id == 5
    assert meta["attempt"] == 1
    assert len(sent_messages) == 2
    assert sent_messages[1] == sent_messages[0]
    assert "previous response was invalid" not in "\n".join(message["content"] for message in sent_messages[1])


def test_openai_compat_agent_turn_chat_retries_llm_http_500_without_prompt_correction(monkeypatch):
    monkeypatch.setattr("sts2_bench.agents.time.sleep", lambda _seconds: None)
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent(
        [
            RuntimeError("LLM HTTP 500: upstream connect failed"),
            {"action_id": 5, "reason": "Recovered after upstream retry."},
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.action_id == 5
    assert meta["attempt"] == 1
    assert len(sent_messages) == 2
    assert sent_messages[1] == sent_messages[0]
    assert "previous response was invalid" not in "\n".join(message["content"] for message in sent_messages[1])


def test_openai_compat_agent_uses_slow_retry_for_auth_unavailable(monkeypatch):
    slept = []
    monkeypatch.setattr("sts2_bench.agents.time.sleep", slept.append)
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent(
        [
            RuntimeError('LLM HTTP 503: {"error":{"message":"auth_unavailable: no auth available"}}'),
            RuntimeError('LLM HTTP 503: {"error":{"message":"auth_unavailable: no auth available"}}'),
            {"action_id": 5, "reason": "Recovered after auth provider retry."},
        ],
        max_retries=2,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.action_id == 5
    assert meta["attempt"] == 2
    assert slept == [10.0, 20.0]
    assert sent_messages[1] == sent_messages[0]
    assert sent_messages[2] == sent_messages[0]
    assert "previous response was invalid" not in "\n".join(message["content"] for message in sent_messages[2])


def test_openai_compat_agent_remaps_when_selected_label_matches_current_action():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": end_turn.action_id,
                "selected_action_label": strike.label,
                "reason": "Play Strike now.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == strike
    assert meta["attempt"] == 0
    assert meta["action_id_remapped_from"] == end_turn.action_id
    assert meta["resolved_action_id"] == strike.action_id
    assert meta["parsed"]["action_id"] == strike.action_id
    assert meta["parsed"]["selected_action_label"] == strike.label
    assert len(sent_messages) == 1


def test_openai_compat_agent_allows_empty_sequence_for_atomic_action():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    agent, _sent_messages = _turn_chat_agent(
        [
            {
                "action_id": strike.action_id,
                "selected_action_label": strike.label,
                "sequence": [],
                "reason": "Play Strike now.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == strike
    assert meta["parsed"]["sequence"] == []


def test_openai_compat_agent_retries_noncustom_action_with_nonempty_sequence():
    state = _custom_gateway_combat_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": strike.action_id,
                "selected_action_label": strike.label,
                "sequence": [{"card_index": 0, "target_index": 0}, {"card_index": 1}],
                "reason": "Play Strike now, then Defend.",
            },
            {
                "action_id": end_turn.action_id,
                "selected_action_label": end_turn.label,
                "sequence": [],
                "reason": "Choose a single legal action after correction.",
            },
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == end_turn
    assert meta["attempt"] == 1
    assert "sequence may only be non-empty when selecting combat_custom_sequence" in "\n".join(
        message["content"] for message in sent_messages[1]
    )


def test_openai_compat_agent_retries_custom_sequence_when_action_id_label_mismatch():
    state = _custom_gateway_combat_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    custom = next(action for action in actions if action.kind == "combat_custom_sequence")
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": strike.action_id,
                "selected_action_label": custom.label,
                "sequence": [{"card_index": 0, "target_index": 0}, {"card_index": 1}],
                "reason": "Use a deterministic custom sequence.",
            },
            {
                "action_id": end_turn.action_id,
                "selected_action_label": end_turn.label,
                "reason": "Use a legal action after correction.",
            },
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == end_turn
    assert meta["attempt"] == 1
    assert len(sent_messages) == 2
    assert "combat_custom_sequence requires action_id and selected_action_label" in "\n".join(
        message["content"] for message in sent_messages[1]
    )


def test_openai_compat_agent_retries_custom_sequence_kind_label_with_clear_feedback():
    state = _custom_gateway_combat_state()
    actions = build_legal_actions(state)
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": -1,
                "selected_action_label": "combat_custom_sequence",
                "sequence": [{"card_index": 0, "target_index": 0}, {"card_index": 1}],
                "reason": "Use the custom sequence gateway.",
            },
            {
                "action_id": end_turn.action_id,
                "selected_action_label": end_turn.label,
                "reason": "Use a legal action after correction.",
            },
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == end_turn
    assert meta["attempt"] == 1
    retry_text = "\n".join(message["content"] for message in sent_messages[1])
    assert "combat_custom_sequence is an action kind, not a selected_action_label" in retry_text
    assert "Use the current legal action_id for the custom sequence gateway" in retry_text


def test_openai_compat_agent_resolves_missing_action_id_from_exact_selected_label():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "selected_action_label": strike.label,
                "reason": "Play Strike now.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == strike
    assert meta["attempt"] == 0
    assert meta["action_id_resolved_from_label"] is True
    assert meta["resolved_action_id"] == strike.action_id
    assert meta["parsed"]["action_id"] == strike.action_id
    assert meta["parsed"]["selected_action_label"] == strike.label
    assert len(sent_messages) == 1


def test_openai_compat_agent_resolves_action_alias_as_selected_label():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action": strike.label,
                "reason": "Play Strike now.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == strike
    assert meta["action_id_resolved_from_label"] is True
    assert meta["resolved_action_id"] == strike.action_id
    assert meta["parsed"]["action_id"] == strike.action_id
    assert meta["parsed"]["selected_action_label"] == strike.label
    assert len(sent_messages) == 1


def test_openai_compat_agent_retries_unopened_card_reward_skip():
    state = {
        "decision": "combat_reward",
        "rewards": [{"index": 0, "kind": "card_reward", "can_claim": True, "can_skip": True}],
        "player": {"hp": 64, "max_hp": 80, "gold": 118, "deck_size": 8},
    }
    actions = build_legal_actions(state)
    claim = next(action for action in actions if action.kind == "combat_reward")
    skip = next(action for action in actions if action.kind == "combat_reward_skip")
    assert claim.label == "open card reward 0: reveal offered cards"
    assert skip.label == "skip unopened card reward 0"
    agent, sent_messages = _single_turn_agent(
        [
            {
                "action_id": skip.action_id,
                "selected_action_label": skip.label,
                "reason": "No card options are visible, so skip the unknown reward.",
            },
            {
                "action_id": claim.action_id,
                "selected_action_label": claim.label,
                "reason": "Open the card reward first so the offered cards are visible.",
            },
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == claim
    assert meta["attempt"] == 1
    assert "open the card reward first" in meta["previous_errors"][0]
    retry_text = sent_messages[1][-1]["content"]
    assert "previous response was invalid" in retry_text
    assert "no combat_custom_sequence gateway is available" in retry_text
    assert "sequence included only when choosing combat_custom_sequence" not in retry_text
    assert "top-level integer card_index" not in retry_text


def test_openai_compat_agent_retries_when_selected_label_matches_no_current_action():
    state = _combat_turn_state()
    actions = build_legal_actions(state)
    strike = next(action for action in actions if action.kind == "combat_play_card")
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": end_turn.action_id,
                "selected_action_label": "play card 99: Ghost Strike",
                "reason": "Play Strike now.",
            },
            {
                "action_id": strike.action_id,
                "selected_action_label": strike.label,
                "reason": "Play Strike now.",
            },
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == strike
    assert meta["attempt"] == 1
    assert "selected_action_label" in sent_messages[1][-1]["content"]


def test_openai_compat_agent_trusts_exact_selected_label_over_discursive_reason():
    state = _combat_turn_state(
        hand=[
            {
                "index": 0,
                "name": "Bloodletting",
                "cost": 0,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
            }
        ],
        energy=0,
    )
    actions = build_legal_actions(state)
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": end_turn.action_id,
                "selected_action_label": end_turn.label,
                "reason": "Play Bloodletting would cost HP here, so skip it.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == end_turn
    assert meta["attempt"] == 0
    assert len(sent_messages) == 1


def test_openai_compat_agent_retries_when_reason_mentions_different_current_action():
    state = _combat_turn_state(
        view_deck=True,
        view_map=True,
        view_draw_pile=True,
        view_discard_pile=True,
        view_exhaust_pile=True,
        hand=[
            {
                "index": 4,
                "name": "Pommel Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 9},
            }
        ],
    )
    actions = build_legal_actions(state)
    pommel = next(action for action in actions if action.kind == "combat_play_card")
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": end_turn.action_id,
                "reason": "Play Pommel Strike first to deal damage before ending.",
            },
            {
                "action_id": pommel.action_id,
                "reason": "Play Pommel Strike first to deal damage.",
            },
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == pommel
    assert meta["attempt"] == 1
    assert len(meta["previous_errors"]) == 1
    assert "does not match selected action" in meta["previous_errors"][0]
    assert "Play Pommel Strike first" in meta["previous_raw_responses"][0]
    assert "does not match selected action" in sent_messages[1][-1]["content"]


def test_combat_play_exposes_atomic_card_sequence_actions():
    state = _combat_turn_state(
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
            {
                "index": 1,
                "name": "Defend",
                "cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"block": 5},
            },
        ],
    )
    actions = build_legal_actions(state, allow_precomputed_card_sequences=True)

    sequence = next(action for action in actions if action.kind == "combat_card_sequence")

    assert sequence.label == "play sequence: 0 Strike on enemy 0 -> 1 Defend"
    assert sequence.command == {
        "cmd": "bench_sequence",
        "sequence": [
            {"card_index": 0, "target_index": 0},
            {"card_index": 1},
        ],
    }


def test_combat_card_sequence_omits_upgrade_or_choice_cards():
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "name": "Armaments",
                "cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "description": "Gain 5 Block. Upgrade a card in your hand for the rest of combat.",
                "stats": {"block": 5},
            },
            {
                "index": 1,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
        ],
    )

    actions = build_legal_actions(state, allow_precomputed_card_sequences=True)

    assert any(action.label == "play card 0: Armaments" for action in actions)
    assert not any(action.kind == "combat_card_sequence" and "Armaments" in action.label for action in actions)


def test_combat_play_exposes_custom_card_sequence_gateway():
    actions = build_legal_actions(_custom_gateway_combat_state())

    custom = next(action for action in actions if action.kind == "combat_custom_sequence")

    assert custom.label == "propose custom deterministic card sequence"
    assert custom.command == {"cmd": "bench_custom_sequence"}


def test_combat_play_hides_custom_sequence_gateway_when_single_card_kills_only_enemy():
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 9, "unblocked_damage": 9},
                    ],
                },
            },
            {
                "index": 1,
                "name": "Defend",
                "cost": 1,
                "type": "Skill",
                "target_type": "Self",
                "can_play": True,
                "stats": {"block": 5},
            },
            {
                "index": 2,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 9, "unblocked_damage": 9},
                    ],
                },
            },
        ],
        enemies=[{"index": 0, "name": "Corpse Slug", "hp": 8, "max_hp": 25, "block": 0}],
    )

    actions = build_legal_actions(state)

    assert any(action.label == "play card 0: Strike on enemy 0: Corpse Slug" for action in actions)
    assert not any(action.kind == "combat_custom_sequence" for action in actions)


def test_combat_play_hides_custom_sequence_gateway_with_fewer_than_two_playable_cards():
    state = _combat_turn_state(
        energy=0,
        hand=[
            {
                "index": 1,
                "name": "Anger",
                "cost": 0,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [{"target_index": 0, "damage": 6, "unblocked_damage": 6}],
                },
            },
            {
                "index": 3,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": False,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [{"target_index": 0, "damage": 6, "unblocked_damage": 6}],
                },
            },
        ],
        enemies=[{"index": 0, "name": "Jaw Worm", "hp": 18, "max_hp": 40, "block": 0}],
    )

    actions = build_legal_actions(state)

    assert any(action.label == "play card 1: Anger on enemy 0: Jaw Worm" for action in actions)
    assert not any(action.kind == "combat_custom_sequence" for action in actions)


def test_combat_play_hides_custom_sequence_gateway_when_playable_card_is_not_sequence_safe():
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
            {
                "index": 1,
                "name": "Sword Boomerang",
                "cost": 1,
                "type": "Attack",
                "target_type": "RandomEnemy",
                "can_play": True,
                "description": "Deal 3 damage to a random enemy 3 times.",
                "stats": {"damage": 3, "repeat": 3},
            },
            {
                "index": 2,
                "name": "Defend",
                "cost": 1,
                "type": "Skill",
                "target_type": "Self",
                "can_play": True,
                "stats": {"block": 5},
            },
        ],
        enemies=[{"index": 0, "name": "Living Fog", "hp": 35, "max_hp": 35, "block": 0}],
    )

    actions = build_legal_actions(state)

    assert any(action.label == "play card 1: Sword Boomerang" for action in actions)
    assert not any(action.kind == "combat_custom_sequence" for action in actions)


def test_combat_play_hides_custom_sequence_gateway_when_target_specific_damage_differs_from_base():
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 9, "unblocked_damage": 9},
                        {"target_index": 1, "damage": 6, "unblocked_damage": 6},
                    ],
                },
            },
            {
                "index": 1,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 9, "unblocked_damage": 9},
                        {"target_index": 1, "damage": 6, "unblocked_damage": 6},
                    ],
                },
            },
        ],
        enemies=[
            {"index": 0, "name": "Corpse Slug", "hp": 17, "max_hp": 25, "block": 0},
            {"index": 1, "name": "Corpse Slug", "hp": 26, "max_hp": 26, "block": 0},
        ],
    )

    actions = build_legal_actions(state)

    assert any(action.label == "play card 0: Strike on enemy 0: Corpse Slug" for action in actions)
    assert not any(action.kind == "combat_custom_sequence" for action in actions)


def test_combat_play_hides_custom_sequence_gateway_for_modifier_changing_cards():
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "name": "Bash",
                "cost": 2,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 8,
                    "vulnerablepower": 2,
                    "damage_by_target": [{"target_index": 0, "damage": 8, "unblocked_damage": 8}],
                },
            },
            {
                "index": 1,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [{"target_index": 0, "damage": 6, "unblocked_damage": 6}],
                },
            },
        ],
        enemies=[{"index": 0, "name": "Jaw Worm", "hp": 40, "max_hp": 40, "block": 0}],
    )

    actions = build_legal_actions(state)

    assert any(action.label == "play card 0: Bash on enemy 0: Jaw Worm" for action in actions)
    assert not any(action.kind == "combat_custom_sequence" for action in actions)


def test_combat_play_hides_custom_sequence_gateway_when_enemy_has_powers():
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 7,
                    "damage_by_target": [{"target_index": 0, "damage": 7, "unblocked_damage": 7}],
                },
            },
            {
                "index": 1,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 7,
                    "damage_by_target": [{"target_index": 0, "damage": 7, "unblocked_damage": 7}],
                },
            },
        ],
        enemies=[
            {
                "index": 0,
                "name": "Phantasmal Gardener",
                "hp": 31,
                "max_hp": 31,
                "block": 0,
                "powers": [{"name": "Skittish", "amount": 6}],
            }
        ],
    )

    actions = build_legal_actions(state)

    assert any(action.label == "play card 0: Strike on enemy 0: Phantasmal Gardener" for action in actions)
    assert not any(action.kind == "combat_custom_sequence" for action in actions)


def test_combat_play_can_expose_custom_sequence_gateway_in_eligible_experiment_mode():
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 7,
                    "damage_by_target": [{"target_index": 0, "damage": 7, "unblocked_damage": 7}],
                },
            },
            {
                "index": 1,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 7,
                    "damage_by_target": [{"target_index": 0, "damage": 7, "unblocked_damage": 7}],
                },
            },
        ],
        enemies=[
            {
                "index": 0,
                "name": "Phantasmal Gardener",
                "hp": 31,
                "max_hp": 31,
                "block": 0,
                "powers": [{"name": "Skittish", "amount": 6}],
            }
        ],
    )

    actions = build_legal_actions(state, custom_sequence_exposure="eligible")

    assert any(action.label == "play card 0: Strike on enemy 0: Phantasmal Gardener" for action in actions)
    assert any(action.kind == "combat_custom_sequence" for action in actions)
    assert not any(action.kind == "combat_card_sequence" for action in actions)


def test_combat_play_hides_custom_sequence_gateway_when_player_has_one_shot_free_attack_power():
    state = _combat_turn_state(
        energy=0,
        hand=[
            {
                "index": 0,
                "instance_id": "strike-a",
                "name": "Strike",
                "cost": 0,
                "energy_cost": 0,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [{"target_index": 0, "damage": 6, "unblocked_damage": 6}],
                },
            },
            {
                "index": 1,
                "instance_id": "strike-b",
                "name": "Strike",
                "cost": 0,
                "energy_cost": 0,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [{"target_index": 0, "damage": 6, "unblocked_damage": 6}],
                },
            },
            {
                "index": 2,
                "instance_id": "slimed",
                "name": "Slimed",
                "cost": 1,
                "energy_cost": 1,
                "type": "Status",
                "target_type": "None",
                "can_play": False,
                "stats": {"cards": 1},
            },
        ],
        player_powers=[
            {
                "id": "FREE_ATTACK_POWER",
                "name": "Free Attack",
                "description": "Your next Attack costs 0 [E].",
                "amount": 1,
            }
        ],
        enemies=[{"index": 0, "name": "Leaf Slime (M)", "hp": 13, "max_hp": 33, "block": 0}],
    )

    actions = build_legal_actions(state)

    assert any(action.label == "play card 0: Strike on enemy 0: Leaf Slime (M)" for action in actions)
    assert any(action.label == "play card 1: Strike on enemy 0: Leaf Slime (M)" for action in actions)
    assert not any(action.kind == "combat_custom_sequence" for action in actions)


def test_combat_play_hides_custom_sequence_gateway_when_attacks_can_cumulatively_kill_target():
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "name": "Anger",
                "cost": 0,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 7,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 7, "unblocked_damage": 7},
                        {"target_index": 1, "damage": 7, "unblocked_damage": 7},
                    ],
                },
            },
            {
                "index": 1,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 7,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 7, "unblocked_damage": 7},
                        {"target_index": 1, "damage": 7, "unblocked_damage": 7},
                    ],
                },
            },
            {
                "index": 2,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 10,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 10, "unblocked_damage": 10},
                        {"target_index": 1, "damage": 10, "unblocked_damage": 10},
                    ],
                },
            },
        ],
        enemies=[
            {"index": 0, "name": "Phantasmal Gardener", "hp": 14, "max_hp": 31, "block": 0},
            {"index": 1, "name": "Phantasmal Gardener", "hp": 29, "max_hp": 29, "block": 0},
        ],
    )

    actions = build_legal_actions(state)

    assert any(action.label == "play card 0: Anger on enemy 0: Phantasmal Gardener" for action in actions)
    assert not any(action.kind == "combat_custom_sequence" for action in actions)


def test_custom_card_sequence_gateway_can_be_disabled_for_env_action_masks():
    actions = build_legal_actions(_combat_turn_state(), allow_custom_sequences=False)

    assert not any(action.kind == "combat_custom_sequence" for action in actions)


class _CapturingRandomAgent(RandomAgent):
    def __init__(self):
        super().__init__(seed=0)
        self.seen_legal_actions = []

    def choose(self, state, legal_actions, *, prompt=None):
        self.seen_legal_actions = list(legal_actions)
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "random"}


class _CapturingEndTurnAgent:
    def __init__(self):
        self.seen_legal_actions = []

    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        self.seen_legal_actions = list(legal_actions)
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "test"}


class _CapturingOpenAIEndTurnAgent(OpenAICompatAgent):
    def __init__(self):
        super().__init__(base_url="http://127.0.0.1:1", model="test", run_summary_enabled=False)
        self.seen_legal_actions = []

    def choose(self, state, legal_actions, *, prompt=None):
        self.seen_legal_actions = list(legal_actions)
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "openai_compat", "parsed": {"reason": "End test turn."}}


class _OneCombatTurnProcess:
    def start(self):
        pass

    def start_run(self, **kwargs):
        return _combat_turn_state()

    def send(self, command):
        return {"decision": "game_over", "victory": False, "player": {"hp": 70, "max_hp": 80}}

    def close(self):
        pass


class _OneCombatSequenceTurnProcess(_OneCombatTurnProcess):
    def start_run(self, **kwargs):
        return _combat_turn_state(
            hand=[
                {
                    "index": 0,
                    "name": "Strike",
                    "cost": 1,
                    "type": "Attack",
                    "target_type": "AnyEnemy",
                    "can_play": True,
                    "stats": {"damage": 6},
                },
                {
                    "index": 1,
                    "name": "Defend",
                    "cost": 1,
                    "type": "Skill",
                    "target_type": "None",
                    "can_play": True,
                    "stats": {"block": 5},
                },
            ],
        )


class _MultiCardRewardProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return {
            "decision": "combat_reward",
            "context": {"act": 1, "floor": 2, "room_type": "Monster"},
            "player": {"hp": 70, "max_hp": 80, "gold": 10, "deck_size": 10},
            "rewards": [
                {"index": 0, "kind": "card_reward", "can_claim": True, "can_skip": True},
                {"index": 1, "kind": "card_reward", "can_claim": True, "can_skip": True},
            ],
        }

    def send(self, command):
        self.sent.append(command)
        if command.get("action") == "claim_reward" and command.get("args", {}).get("reward_index") == 0:
            return {
                "decision": "card_reward",
                "context": {"act": 1, "floor": 2, "room_type": "Monster"},
                "player": {"hp": 70, "max_hp": 80, "gold": 10, "deck_size": 10},
                "cards": [{"index": 0, "name": "Strike"}, {"index": 1, "name": "Shrug It Off"}],
                "can_skip": True,
            }
        if command.get("action") == "skip_card_reward":
            return self.start_run()
        return {"decision": "game_over", "victory": False, "player": {"hp": 70, "max_hp": 80}}

    def close(self):
        pass


class _ScriptedCardRewardAgent:
    def __init__(self):
        self.calls = 0
        self.seen_legal_actions = []

    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        self.seen_legal_actions.append(list(legal_actions))
        if self.calls == 0:
            action = next(action for action in legal_actions if action.label == "open card reward 0: reveal offered cards")
        elif self.calls == 1:
            action = next(action for action in legal_actions if action.kind == "card_reward_skip")
        else:
            action = next(action for action in legal_actions if action.label == "open card reward 1: reveal offered cards")
        self.calls += 1
        return action, {"agent": "test", "parsed": {"reason": action.label}}


def test_runner_does_not_expose_custom_sequence_gateway_to_random_agent():
    agent = _CapturingRandomAgent()

    run_one(agent=agent, seed="test", process=_OneCombatTurnProcess(), max_steps=1)

    assert agent.seen_legal_actions
    assert not any(action.kind == "combat_custom_sequence" for action in agent.seen_legal_actions)


def test_runner_can_disable_custom_sequence_gateway_for_llm_agents():
    agent = _CapturingEndTurnAgent()

    run_one(agent=agent, seed="test", process=_OneCombatTurnProcess(), max_steps=1, allow_custom_sequences=False)

    assert agent.seen_legal_actions
    assert not any(action.kind == "combat_custom_sequence" for action in agent.seen_legal_actions)


def test_runner_hides_precomputed_card_sequences_but_keeps_custom_gateway_by_default():
    agent = _CapturingOpenAIEndTurnAgent()

    run_one(agent=agent, seed="test", process=_OneCombatSequenceTurnProcess(), max_steps=1)

    assert agent.seen_legal_actions
    assert not any(action.kind == "combat_card_sequence" for action in agent.seen_legal_actions)
    assert any(action.kind == "combat_custom_sequence" for action in agent.seen_legal_actions)


def test_runner_can_expose_precomputed_card_sequences_for_diagnostics():
    agent = _CapturingOpenAIEndTurnAgent()

    run_one(
        agent=agent,
        seed="test",
        process=_OneCombatSequenceTurnProcess(),
        max_steps=1,
        precomputed_combat_card_sequences_enabled=True,
    )

    assert agent.seen_legal_actions
    assert any(action.kind == "combat_card_sequence" for action in agent.seen_legal_actions)
    assert any(action.kind == "combat_custom_sequence" for action in agent.seen_legal_actions)


def test_runner_can_disable_combat_sequence_actions_for_llm_agents():
    agent = _CapturingOpenAIEndTurnAgent()

    run_one(
        agent=agent,
        seed="test",
        process=_OneCombatSequenceTurnProcess(),
        max_steps=1,
        combat_card_sequences_enabled=False,
    )

    assert agent.seen_legal_actions
    assert not any(action.kind == "combat_card_sequence" for action in agent.seen_legal_actions)
    assert not any(action.kind == "combat_custom_sequence" for action in agent.seen_legal_actions)


def test_runner_hides_precomputed_card_sequences_from_coordinator_ca_by_default():
    ca_agent = _CapturingOpenAIEndTurnAgent()
    coordinator = CoordinatorAgent(ca_agent=ca_agent, occ_agent=_CapturingEndTurnAgent())

    run_one(agent=coordinator, seed="test", process=_OneCombatSequenceTurnProcess(), max_steps=1)

    assert ca_agent.seen_legal_actions
    assert not any(action.kind == "combat_card_sequence" for action in ca_agent.seen_legal_actions)
    assert any(action.kind == "combat_custom_sequence" for action in ca_agent.seen_legal_actions)


def test_runner_card_reward_reopen_guard_allows_other_unopened_packs():
    agent = _ScriptedCardRewardAgent()
    process = _MultiCardRewardProcess()

    result = run_one(agent=agent, seed="test", process=process, max_steps=4)

    assert result.error is None
    third_decision_labels = {action.label for action in agent.seen_legal_actions[2]}
    assert "open card reward 0: reveal offered cards" not in third_decision_labels
    assert "open card reward 1: reveal offered cards" in third_decision_labels
    reward_commands = [
        command for command in process.sent if command.get("action") in {"claim_reward", "skip_card_reward"}
    ]
    assert reward_commands == [
        {"cmd": "action", "action": "claim_reward", "args": {"reward_index": 0}},
        {"cmd": "action", "action": "skip_card_reward"},
        {"cmd": "action", "action": "claim_reward", "args": {"reward_index": 1}},
    ]


def test_openai_compat_agent_accepts_custom_sequence_payload():
    state = _custom_gateway_combat_state()
    actions = build_legal_actions(state)
    custom = next(action for action in actions if action.kind == "combat_custom_sequence")
    agent, _sent_messages = _turn_chat_agent(
        [
            {
                "action_id": custom.action_id,
                "selected_action_label": custom.label,
                "sequence": [
                    {"card_index": 0, "target_index": 0},
                    {"card_index": 1},
                ],
                "reason": "Use a deterministic custom card sequence.",
            }
        ]
    )

    action, meta = agent.choose(state, actions)

    assert action == custom
    assert meta["parsed"]["sequence"] == [
        {"card_index": 0, "target_index": 0},
        {"card_index": 1},
    ]


def test_openai_compat_agent_combat_mdp_state_accepts_custom_sequence_payload():
    state = _custom_gateway_combat_state()
    actions = build_legal_actions(state)
    custom = next(action for action in actions if action.kind == "combat_custom_sequence")
    agent, sent_messages = _combat_mdp_agent(
        [
            {
                "action_id": custom.action_id,
                "selected_action_label": custom.label,
                "sequence": [
                    {"card_index": 0, "target_index": 0},
                    {"card_index": 1},
                ],
                "reason": "Use a deterministic custom card sequence.",
            }
        ]
    )

    action, meta = agent.choose(state, actions)

    assert action == custom
    assert meta["parsed"]["sequence"] == [
        {"card_index": 0, "target_index": 0},
        {"card_index": 1},
    ]
    assert "Combat MDP state JSON:" in sent_messages[0][1]["content"]
    assert "combat_custom_sequence" in sent_messages[0][1]["content"]


def test_openai_compat_agent_combat_mdp_state_preserves_card_effect_text_and_sequence_safety():
    state = _combat_turn_state(
        hand=[
            {
                "index": 0,
                "id": "CARD.POMMEL_STRIKE",
                "name": "Pommel Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "description": "Deal 9 damage. Draw 1 card.",
                "stats": {"damage": 9, "cards": 1},
            },
            {
                "index": 1,
                "id": "CARD.TRUE_GRIT",
                "name": "True Grit",
                "cost": 1,
                "type": "Skill",
                "target_type": "Self",
                "can_play": True,
                "description": "Gain 7 Block. Exhaust 1 card at random.",
                "stats": {"block": 7, "exhaust": 1},
            },
        ]
    )
    actions = build_legal_actions(state, custom_sequence_exposure="eligible")
    end_turn = next(action for action in actions if action.kind == "combat_end_turn")
    agent, sent_messages = _combat_mdp_agent([{"action_id": end_turn.action_id, "reason": "No safe sequence."}])

    agent.choose(state, actions)

    prompt = sent_messages[0][1]["content"]
    mdp_state = _json_after(prompt, "Combat MDP state JSON:\n")
    by_name = {card["name"]: card for card in mdp_state["hand_snapshot"]}

    pommel = by_name["Pommel Strike"]
    assert pommel["description"] == "Deal 9 damage. Draw 1 card."
    assert pommel["stats"]["cards"] == 1
    assert pommel["sequence_eligible"] is False
    assert any("draw" in reason for reason in pommel["sequence_block_reasons"])
    assert "draw" in pommel["must_recompute_after_play"]

    true_grit = by_name["True Grit"]
    assert true_grit["description"] == "Gain 7 Block. Exhaust 1 card at random."
    assert true_grit["stats"]["exhaust"] == 1
    assert true_grit["sequence_eligible"] is False
    assert any("exhaust" in reason or "random" in reason for reason in true_grit["sequence_block_reasons"])
    assert "random_effect" in true_grit["must_recompute_after_play"]
    assert "hand_change" in true_grit["must_recompute_after_play"]


def test_combat_mdp_omits_disabled_sequence_metadata_dedupes_tips_and_empty_process():
    state = _combat_turn_state(
        hand=[
            {
                "index": 0,
                "id": "CARD.POMMEL_STRIKE",
                "name": "Pommel Strike",
                "cost": 1,
                "type": "Attack",
                "description": "Deal 9 damage. Draw 1 card.",
                "stats": {"damage": 9, "cards": 1},
                "hover_tips": [
                    {"name": "Draw", "description": "Move a card from the draw pile to the hand."},
                    {"title": "Draw", "description": "Move a card from the draw pile to the hand."},
                ],
            }
        ]
    )
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_mdp_state",
        combat_card_sequences_enabled=False,
    )

    prompt = agent.build_prompt(state, build_legal_actions(state, allow_custom_sequences=False))
    card = _json_after(prompt, "Combat MDP state JSON:\n")["hand_snapshot"][0]

    assert "sequence_eligible" not in card
    assert "sequence_block_reasons" not in card
    assert "must_recompute_after_play" not in card
    assert len(card["hover_tips"]) == 1
    assert "Combat turn process JSON:" not in prompt


def test_combat_mdp_state_preserves_visible_engine_facts_for_combat_objects():
    state = _combat_turn_state(
        energy=3,
        player={
            "hp": 70,
            "max_hp": 80,
            "block": 0,
            "gold": 0,
            "deck_size": 10,
            "relics": [
                {
                    "id": "RELIC.AKABEKO",
                    "name": "Akabeko",
                    "description": "Your first Attack each combat deals 8 additional damage.",
                    "counter": 1,
                    "hover_tips": [{"name": "Attack", "description": "A card type."}],
                }
            ],
            "potions": [
                {
                    "index": 0,
                    "instance_id": "potion-1",
                    "id": "POTION.VULNERABLE",
                    "name": "Vulnerable Potion",
                    "description": "Apply 3 Vulnerable.",
                    "target_type": "AnyEnemy",
                    "vars": {"amount": 3},
                    "hover_tips": [
                        {"name": "Vulnerable", "description": "Take more Attack damage."}
                    ],
                }
            ],
        },
        player_powers=[
            {
                "instance_id": "power-1",
                "id": "POWER.RAGE",
                "name": "Rage",
                "amount": 1,
                "type": "Buff",
                "description": "Whenever you play an Attack, gain 3 Block.",
                "vars": {"block": 3},
                "hover_tips": [{"name": "Block", "description": "Prevents damage."}],
            }
        ],
        hand=[
            {
                "index": 0,
                "instance_id": "x-card-1",
                "id": "CARD.WHIRLWIND",
                "name": "Whirlwind",
                "cost": "X",
                "energy_cost": 3,
                "x_value": 3,
                "type": "Attack",
                "rarity": "Uncommon",
                "upgrade_level": 1,
                "target_type": "AllEnemies",
                "can_play": True,
                "description": "Deal 8 damage to ALL enemies X times.",
                "vars": {"damage": 8},
                "hover_tips": [{"name": "X", "description": "Uses all current Energy."}],
                "stats": {"damage": 8},
            }
        ],
        enemies=[
            {
                "index": 0,
                "combat_index": 4,
                "instance_id": "enemy-1",
                "id": "ENEMY.NIBBIT",
                "name": "Nibbit",
                "hp": 24,
                "max_hp": 30,
                "block": 0,
                "alive": True,
                "targetable": True,
                "intends_attack": True,
                "move_id": "BUTT_MOVE",
                "move_name": "Butt",
                "intents": [
                    {
                        "id": "INTENT.BUTT",
                        "type": "Attack",
                        "damage": 7,
                        "hits": 2,
                        "description": "Deal 7 damage twice.",
                        "vars": {"damage": 7, "hits": 2},
                    }
                ],
                "powers": [
                    {
                        "id": "POWER.THORNS",
                        "name": "Thorns",
                        "amount": 3,
                        "type": "Buff",
                        "description": "Deal 3 damage when attacked.",
                        "vars": {"damage": 3},
                    }
                ],
            }
        ],
    )
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_mdp_state",
    )

    prompt = agent.build_prompt(state, build_legal_actions(state, allow_custom_sequences=False))
    mdp_state = _json_after(prompt, "Combat MDP state JSON:\n")

    card = mdp_state["hand_snapshot"][0]
    assert card["instance_id"] == "x-card-1"
    assert card["energy_cost"] == 3
    assert card["x_value"] == 3
    assert card["rarity"] == "Uncommon"
    assert card["upgrade_level"] == 1
    assert card["vars"] == {"damage": 8}
    assert card["hover_tips"][0]["description"] == "Uses all current Energy."

    player = mdp_state["player"]
    assert player["relics"][0]["description"].startswith("Your first Attack")
    assert player["potions"][0]["target_type"] == "AnyEnemy"
    assert player["potions"][0]["vars"] == {"amount": 3}
    assert player["powers"][0]["description"].startswith("Whenever you play an Attack")

    enemy = mdp_state["enemies"][0]
    assert enemy["instance_id"] == "enemy-1"
    assert enemy["combat_index"] == 4
    assert enemy["targetable"] is True
    assert enemy["move_id"] == "BUTT_MOVE"
    assert enemy["move_name"] == "Butt"
    assert enemy["intents"][0]["description"] == "Deal 7 damage twice."
    assert enemy["powers"][0]["description"] == "Deal 3 damage when attacked."


def test_combat_mdp_state_preserves_public_character_mechanics_losslessly():
    state = _combat_turn_state(
        orbs=[
            {
                "index": 0,
                "name": {"en": "Lightning", "zh": "闪电"},
                "type": "Lightning",
                "passive": 3,
                "evoke": 8,
                "position_label": "rightmost",
            }
        ],
        orb_slots=3,
        stars=2,
        osty={
            "name": {"en": "Osty", "zh": "奥斯蒂"},
            "hp": 18,
            "max_hp": 24,
            "block": 4,
            "alive": True,
        },
    )
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_mdp_state",
    )

    prompt = agent.build_prompt(state, build_legal_actions(state, allow_custom_sequences=False))
    mdp_state = _json_after(prompt, "Combat MDP state JSON:\n")

    assert mdp_state["orbs"] == state["orbs"]
    assert mdp_state["orb_slots"] == 3
    assert mdp_state["stars"] == 2
    assert mdp_state["osty"] == state["osty"]


def test_combat_mdp_state_exposes_viewed_card_details_only_after_view_flags():
    deck_card = {
        "index": 8,
        "instance_id": "deck-card",
        "name": "Hidden Deck Card",
        "description": "Deck-only visible effect.",
        "after_upgrade": {"description": "Upgraded deck-only visible effect."},
    }
    draw_card = {
        "index": 9,
        "instance_id": "draw-card",
        "name": "Known Top Card",
        "description": "Draw-pile-only visible effect.",
    }
    base = _combat_turn_state(
        player={
            "hp": 70,
            "max_hp": 80,
            "block": 0,
            "gold": 0,
            "deck_size": 1,
            "deck": [deck_card],
            "potions": [],
            "relics": [],
        },
        draw_pile=[draw_card],
        draw_pile_count=1,
    )
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_mdp_state",
    )

    hidden_prompt = agent.build_prompt(base, build_legal_actions(base, allow_custom_sequences=False))
    hidden = _json_after(hidden_prompt, "Combat MDP state JSON:\n")
    assert "viewed_information" not in hidden
    assert "Hidden Deck Card" not in hidden_prompt
    assert "Known Top Card" not in hidden_prompt

    viewed_state = {**base, "view_deck": True, "view_draw_pile": True}
    viewed_prompt = agent.build_prompt(
        viewed_state,
        build_legal_actions(viewed_state, allow_custom_sequences=False),
    )
    viewed = _json_after(viewed_prompt, "Combat MDP state JSON:\n")["viewed_information"]
    assert viewed["deck"][0]["description"] == "Deck-only visible effect."
    assert viewed["deck"][0]["after_upgrade"]["description"] == "Upgraded deck-only visible effect."
    assert viewed["draw_pile"][0]["description"] == "Draw-pile-only visible effect."
    assert "discard_pile" not in viewed
    assert "exhaust_pile" not in viewed


def test_openai_compat_agent_combat_card_select_preserves_complete_selection_context():
    base_state = _combat_turn_state()
    source_card = {
        "index": 3,
        "id": "CARD.HEADBUTT",
        "name": "Headbutt",
        "cost": 1,
        "type": "Attack",
        "description": "Deal 9 damage. Put a card from your Discard Pile on top of your Draw Pile.",
        "keywords": ["DrawPile", "DiscardPile"],
        "after_upgrade": {"description": "Deal 12 damage. Put a card from your Discard Pile on top of your Draw Pile."},
    }
    spoils_map = {
        "index": 0,
        "id": "CARD.SPOILS_MAP",
        "name": "Spoils Map",
        "cost": "Unplayable",
        "type": "Quest",
        "rarity": "Special",
        "description": "Unplayable. Marks a site on the map and clogs the deck until its condition is met.",
        "keywords": ["Unplayable", "Quest"],
        "after_upgrade": {"description": "Unplayable. The marked site is easier to reach."},
        "hover_tips": [{"title": "Unplayable", "body": "Cannot be played."}],
        "engine_extension": {"must_survive_prompt_serialization": True},
    }
    state = {
        "decision": "card_select",
        "context": {"act": 1, "floor": 9, "room_type": "Monster"},
        "player": base_state["player"],
        "min_select": 1,
        "max_select": 1,
        "can_skip": False,
        "prompt": "Choose a card from your Discard Pile to put on top of your Draw Pile.",
        "source_card": source_card,
        "combat": {
            "round": 2,
            "energy": 1,
            "max_energy": 3,
            "draw_pile_count": 4,
            "discard_pile_count": 5,
            "exhaust_pile_count": 0,
            "enemies": base_state["enemies"],
        },
        "cards": [spoils_map],
    }
    actions = build_legal_actions(state)
    select = next(action for action in actions if action.kind == "card_select")
    agent, sent_messages = _combat_mdp_agent(
        [{"action_id": select.action_id, "reason": "Select the only legal candidate."}]
    )

    agent.choose(state, actions)

    prompt = sent_messages[0][1]["content"]
    mdp_state = _json_after(prompt, "Combat MDP state JSON:\n")
    selection = mdp_state["selection_context"]
    assert selection["prompt"] == state["prompt"]
    assert selection["min_select"] == 1
    assert selection["max_select"] == 1
    assert selection["can_skip"] is False
    assert selection["sources"]["source_card"] == source_card
    assert selection["candidates"] == [spoils_map]


def test_openai_compat_agent_factors_large_card_select_without_losing_candidate_context():
    cards = [
        {
            "index": index,
            "id": f"CARD.TEST_{index}",
            "instance_id": f"test-card-{index}",
            "name": f"Test Card {index}",
            "cost": index % 3,
            "type": "Skill",
            "rarity": "Common",
            "description": f"Complete engine description for candidate {index}.",
            "keywords": ["Test", f"Keyword{index}"],
            "after_upgrade": {
                "description": f"Complete upgraded description for candidate {index}."
            },
            "hover_tips": [{"title": "Test", "body": f"Complete hover tip {index}."}],
            "engine_extension": {"candidate": index, "must_survive_prompt_serialization": True},
        }
        for index in range(16)
    ]
    state = {
        "decision": "card_select",
        "min_select": 4,
        "max_select": 4,
        "can_skip": False,
        "prompt": "Choose exactly four cards to upgrade.",
        "cards": cards,
    }
    actions = build_legal_actions(state)
    assert len(actions) < 10
    assert sum(action.kind == "card_select_space" for action in actions) == 1
    agent, sent_messages = _single_turn_agent(
        [{"selected_card_indices": [0, 3, 7, 11], "reason": "These are the intended four cards."}],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    prompt = sent_messages[0][1]["content"]
    action_space = _json_after(prompt, "Legal actions:\n")
    assert action_space["kind"] == "card_select"
    assert action_space["min_select"] == 4
    assert action_space["max_select"] == 4
    assert action_space["can_skip"] is False
    assert action_space["candidates"] == cards
    assert action.command["args"]["indices"] == "0,3,7,11"
    assert meta["parsed"]["action_id"] == action.action_id
    assert meta["parsed"]["selected_action_label"] == action.label
    assert len(json.dumps(action_space, ensure_ascii=False, separators=(",", ":"))) < 10_000


def test_openai_compat_agent_card_select_empty_indices_resolve_only_to_visible_skip():
    state = {
        "decision": "card_select",
        "min_select": 1,
        "max_select": 2,
        "can_skip": True,
        "prompt": "Choose cards or skip.",
        "cards": [
            {"index": 2, "name": "Strike", "description": "Deal 6 damage."},
            {"index": 5, "name": "Defend", "description": "Gain 5 Block."},
        ],
    }
    actions = build_legal_actions(state)
    agent, _sent_messages = _single_turn_agent(
        [{"selected_card_indices": [], "reason": "Skip this optional selection."}],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action.kind == "card_select_skip"
    assert meta["parsed"]["selected_card_indices"] == []


def test_card_select_min_zero_keeps_nonempty_subset_rule_separate_from_skip():
    state = {
        "decision": "card_select",
        "min_select": 0,
        "max_select": 2,
        "can_skip": True,
        "cards": [
            {"index": 0, "name": "Strike", "description": "Deal 6 damage."},
            {"index": 1, "name": "Defend", "description": "Gain 5 Block."},
        ],
    }
    actions = build_legal_actions(state)
    agent = OpenAICompatAgent(base_url="http://127.0.0.1:1", model="test")

    prompt = agent.build_prompt(state, actions)
    action_space = _json_after(prompt, "Legal actions:\n")

    assert action_space["min_select"] == 0
    assert action_space["nonempty_min_select"] == 1
    assert action_space["can_skip"] is True
    assert "Any non-empty set" in action_space["selection_rule"]
    assert any(action.kind == "card_select_skip" for action in actions)


def test_openai_compat_agent_card_select_indices_cover_every_existing_legal_subset():
    state = {
        "decision": "card_select",
        "min_select": 1,
        "max_select": 3,
        "can_skip": False,
        "cards": [{"index": index, "name": f"Card {index}"} for index in range(5)],
    }
    actions = build_legal_actions(state)
    subsets = [
        combo
        for count in range(1, 4)
        for combo in itertools.combinations(range(5), count)
    ]
    agent, _sent_messages = _single_turn_agent(
        [
            {
                "selected_card_indices": list(reversed(combo)),
                "reason": "Exercise the complete legal subset mapping.",
            }
            for combo in subsets
        ],
        fallback_mode="error",
    )

    resolved = []
    for combo in subsets:
        action, _meta = agent.choose(state, actions)
        resolved.append(tuple(int(value) for value in action.command["args"]["indices"].split(",")))

    assert resolved == subsets


@pytest.mark.parametrize(
    "selected_indices,error",
    [
        ([0, 0], "duplicate"),
        ([True, 1], "not booleans"),
        ([0.5, 1], "must be integers"),
        ([99], "do not identify"),
        ([0], "do not identify"),
        ([0, 1, 2], "do not identify"),
    ],
)
def test_openai_compat_agent_card_select_rejects_nonlegal_index_sets(selected_indices, error):
    state = {
        "decision": "card_select",
        "min_select": 2,
        "max_select": 2,
        "can_skip": False,
        "cards": [{"index": index, "name": f"Card {index}"} for index in range(3)],
    }
    agent, _sent_messages = _single_turn_agent(
        [{"selected_card_indices": selected_indices, "reason": "Invalid test selection."}],
        fallback_mode="error",
    )

    with pytest.raises(ModelActionError, match=error):
        agent.choose(state, build_legal_actions(state))


def test_openai_compat_agent_card_select_uses_indices_as_the_only_action_identity():
    state = {
        "decision": "card_select",
        "min_select": 2,
        "max_select": 2,
        "can_skip": False,
        "cards": [{"index": index, "name": f"Card {index}"} for index in range(3)],
    }
    actions = build_legal_actions(state)
    selected = next(
        action
        for action in actions
        if action.kind == "card_select" and action.command["args"]["indices"] == "0,2"
    )
    conflicting = next(
        action
        for action in actions
        if action.kind == "card_select" and action.command["args"]["indices"] == "0,1"
    )
    agent, _sent_messages = _single_turn_agent(
        [
            {
                "selected_card_indices": [0, 2],
                "action_id": conflicting.action_id,
                "selected_action_label": conflicting.label,
                "reason": "Choose cards 0 and 2.",
            }
        ],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action == selected
    assert meta["parsed"]["selected_card_indices"] == [0, 2]
    assert meta["parsed"]["action_id"] == selected.action_id
    assert meta["parsed"]["selected_action_label"] == selected.label


def test_card_select_format_retry_keeps_the_selected_indices_protocol():
    state = {
        "decision": "card_select",
        "min_select": 2,
        "max_select": 2,
        "can_skip": False,
        "cards": [{"index": index, "name": f"Card {index}"} for index in range(3)],
    }
    agent, sent_messages = _single_turn_agent(
        [
            {"selected_card_indices": [0, 0], "reason": "Invalid duplicate."},
            {"selected_card_indices": [0, 2], "reason": "Corrected distinct selection."},
        ],
        max_retries=1,
        fallback_mode="error",
    )

    action, _meta = agent.choose(state, build_legal_actions(state))

    assert action.command["args"]["indices"] == "0,2"
    correction = sent_messages[1][-1]["content"]
    assert "selected_card_indices" in correction
    assert "distinct current candidate index integers" in correction


def test_card_select_factored_action_space_is_used_in_every_conversation_mode():
    cards = [
        {
            "index": index,
            "name": f"Card {index}",
            "description": f"Complete effect {index}.",
            "engine_extension": {"preserved": index},
        }
        for index in range(8)
    ]
    state = {
        "decision": "card_select",
        "min_select": 3,
        "max_select": 3,
        "can_skip": False,
        "cards": cards,
    }
    actions = build_legal_actions(state)

    for mode in ("single_turn", "turn_chat", "combat_mdp_state"):
        agent = OpenAICompatAgent(
            base_url="http://127.0.0.1:1",
            model="test",
            conversation_mode=mode,
        )
        prompt = agent.build_prompt(state, actions)
        assert "select cards 0,1,2" not in prompt
        assert '"legal_combination_count":56' in prompt
        assert '"engine_extension":{"preserved":7}' in prompt


def test_large_card_select_space_is_not_materialized_but_every_subset_remains_resolvable():
    cards = [
        {
            "index": index,
            "name": f"Card {index}",
            "description": f"Complete effect {index}.",
            "engine_extension": {"preserved": index},
        }
        for index in range(16)
    ]
    state = {
        "decision": "card_select",
        "min_select": 4,
        "max_select": 4,
        "can_skip": False,
        "cards": cards,
    }

    actions = build_legal_actions(state)
    selection_space = [action for action in actions if action.kind == "card_select_space"]
    assert len(selection_space) == 1
    assert not any(action.kind == "card_select" for action in actions)
    assert len(actions) < 10

    selected_indices = [0, 5, 10, 15]
    agent, sent_messages = _single_turn_agent(
        [{"selected_card_indices": selected_indices, "reason": "Choose this legal four-card subset."}],
        fallback_mode="error",
    )

    action, meta = agent.choose(state, actions)

    assert action.kind == "card_select"
    assert action.command == {
        "cmd": "action",
        "action": "select_cards",
        "args": {"indices": "0,5,10,15"},
    }
    assert meta["parsed"]["selected_card_indices"] == selected_indices
    action_space = _json_after(sent_messages[0][1]["content"], "Legal actions:\n")
    assert action_space["legal_combination_count"] == 1820
    assert action_space["candidates"] == cards


def test_card_select_view_action_is_expressible_in_every_conversation_mode():
    state = {
        "decision": "card_select",
        "min_select": 2,
        "max_select": 2,
        "can_skip": False,
        "cards": [{"index": index, "name": f"Card {index}"} for index in range(3)],
    }
    actions = build_legal_actions(state)
    view = next(action for action in actions if action.kind == "view_deck")

    for factory in (_single_turn_agent, _turn_chat_agent, _combat_mdp_agent):
        agent, sent_messages = factory(
            [
                {
                    "action_id": view.action_id,
                    "selected_action_label": view.label,
                    "reason": "Inspect the complete deck before selecting cards.",
                }
            ],
            fallback_mode="error",
        )
        action, _meta = agent.choose(state, actions)
        prompt = "\n".join(message["content"] for message in sent_messages[0])

        assert action == view
        assert '"action_id": <for a listed non-selection action only; otherwise omit>' in prompt
        assert '"selected_action_label": "<for a listed non-selection action only' in prompt


class _ChooseSequenceAgent:
    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        action = next(action for action in legal_actions if action.kind == "combat_card_sequence")
        return action, {"agent": "test"}


class _SequenceProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return _combat_turn_state(
            hand=[
                {
                    "index": 0,
                    "instance_id": "strike-a",
                    "name": "Strike",
                    "cost": 1,
                    "energy_cost": 1,
                    "type": "Attack",
                    "target_type": "AnyEnemy",
                    "can_play": True,
                    "stats": {"damage": 6},
                },
                {
                    "index": 1,
                    "instance_id": "defend-b",
                    "name": "Defend",
                    "cost": 1,
                    "energy_cost": 1,
                    "type": "Skill",
                    "target_type": "None",
                    "can_play": True,
                    "stats": {"block": 5},
                },
            ],
            enemies=[
                {
                    "index": 0,
                    "instance_id": "jaw-worm",
                    "name": "Jaw Worm",
                    "hp": 20,
                    "max_hp": 40,
                    "block": 0,
                    "intents": [{"type": "Attack", "damage": 7}],
                }
            ],
        )

    def send(self, command):
        self.sent.append(command)
        if len(self.sent) == 1:
            return _combat_turn_state(
                energy=2,
                hand=[
                    {
                        "index": 0,
                        "instance_id": "defend-b",
                        "name": "Defend",
                        "cost": 1,
                        "energy_cost": 1,
                        "type": "Skill",
                        "target_type": "None",
                        "can_play": True,
                        "stats": {"block": 5},
                    }
                ],
                enemies=[
                    {
                        "index": 0,
                        "instance_id": "jaw-worm",
                        "name": "Jaw Worm",
                        "hp": 14,
                        "max_hp": 40,
                        "block": 0,
                        "intents": [{"type": "Attack", "damage": 7}],
                    }
                ],
            )
        return _combat_turn_state(energy=1, hand=[], enemies=[])

    def close(self):
        pass


def test_runner_executes_card_sequence_as_one_benchmark_action():
    process = _SequenceProcess()

    result = run_one(
        agent=_ChooseSequenceAgent(),
        seed="test",
        process=process,
        max_steps=1,
        max_game_actions=1,
        precomputed_combat_card_sequences_enabled=True,
    )

    assert [command["args"] for command in process.sent] == [
        {"card_index": 0, "target_index": 0},
        {"card_index": 0},
    ]
    assert result.steps == 1
    assert result.game_action_count == 1
    assert result.extra["last_sequence_steps"] == 2


class _ChooseCustomSequenceAgent:
    def __init__(self, sequence):
        self.sequence = sequence

    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        action = next(action for action in legal_actions if action.kind == "combat_custom_sequence")
        return action, {
            "agent": "test",
            "parsed": {
                "sequence": self.sequence,
                "reason": "Use a custom deterministic sequence.",
            },
        }


def _custom_sequence_state(**overrides):
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "instance_id": "energy-a",
                "name": "Energy Tap",
                "cost": 0,
                "energy_cost": 0,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"energy": 1},
            },
            {
                "index": 1,
                "instance_id": "big-hit",
                "name": "Big Hit",
                "cost": 4,
                "energy_cost": 4,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": False,
                "stats": {"damage": 20},
            },
        ],
        enemies=[
            {
                "index": 0,
                "instance_id": "jaw-worm",
                "name": "Jaw Worm",
                "hp": 30,
                "max_hp": 40,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 7}],
            }
        ],
    )
    state.update(overrides)
    return state


class _CustomSequenceProcess:
    def __init__(self, *, drift=False):
        self.sent = []
        self.drift = drift

    def start(self):
        pass

    def start_run(self, **kwargs):
        state = _custom_sequence_state()
        if self.drift:
            state["enemies"] = [{**state["enemies"][0], "hp": 99}]
        return state

    def send(self, command):
        self.sent.append(command)
        args = command.get("args") or {}
        if args.get("card_index") == 0:
            return _custom_sequence_state(
                energy=4,
                hand=[
                    {
                        "index": 0,
                        "instance_id": "big-hit",
                        "name": "Big Hit",
                        "cost": 4,
                        "energy_cost": 4,
                        "type": "Attack",
                        "target_type": "AnyEnemy",
                        "can_play": True,
                        "stats": {"damage": 20},
                    }
                ],
            )
        if args.get("card_index") == 0 and args.get("target_index") == 0:
            raise AssertionError("unreachable duplicate branch")
        return _custom_sequence_state(
            energy=0,
            hand=[],
            enemies=[
                {
                    "index": 0,
                    "instance_id": "jaw-worm",
                    "name": "Jaw Worm",
                    "hp": 10,
                    "max_hp": 40,
                    "block": 0,
                    "intents": [{"type": "Attack", "damage": 7}],
                }
            ],
        )

    def close(self):
        pass


class _CustomSequenceMainProcess(_CustomSequenceProcess):
    def send(self, command):
        self.sent.append(command)
        if len(self.sent) == 1:
            return _custom_sequence_state(
                energy=4,
                hand=[
                    {
                        "index": 0,
                        "instance_id": "big-hit",
                        "name": "Big Hit",
                        "cost": 4,
                        "energy_cost": 4,
                        "type": "Attack",
                        "target_type": "AnyEnemy",
                        "can_play": True,
                        "stats": {"damage": 20},
                    }
                ],
            )
        return _custom_sequence_state(
            energy=0,
            hand=[],
            enemies=[
                {
                    "index": 0,
                    "instance_id": "jaw-worm",
                    "name": "Jaw Worm",
                    "hp": 10,
                    "max_hp": 40,
                    "block": 0,
                    "intents": [{"type": "Attack", "damage": 7}],
                }
            ],
        )


def test_runner_validates_custom_sequence_with_probe_replay_before_main_execution():
    main = _CustomSequenceMainProcess()
    probe = _CustomSequenceMainProcess()
    sequence = [
        {"card_index": 0},
        {"card_index": 1, "target_index": 0},
    ]

    result = run_one(
        agent=_ChooseCustomSequenceAgent(sequence),
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=1,
        max_game_actions=1,
    )

    assert [command["args"] for command in probe.sent] == [
        {"card_index": 0},
        {"card_index": 0, "target_index": 0},
    ]
    assert [command["args"] for command in main.sent] == [
        {"card_index": 0},
        {"card_index": 0, "target_index": 0},
    ]
    assert result.error is None
    assert result.extra["last_sequence_validation"]["validated"] is True
    assert result.extra["last_sequence_steps"] == 2


def test_runner_ignores_extra_custom_sequence_target_for_non_enemy_cards():
    main = _CustomSequenceMainProcess()
    probe = _CustomSequenceMainProcess()
    sequence = [
        {"card_index": 0, "target_index": 0},
        {"card_index": 1, "target_index": 0},
    ]

    result = run_one(
        agent=_ChooseCustomSequenceAgent(sequence),
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=1,
        max_game_actions=1,
    )

    assert [command["args"] for command in probe.sent] == [
        {"card_index": 0},
        {"card_index": 0, "target_index": 0},
    ]
    assert [command["args"] for command in main.sent] == [
        {"card_index": 0},
        {"card_index": 0, "target_index": 0},
    ]
    assert result.error is None
    assert result.extra["last_sequence_validation"]["validated"] is True
    assert result.extra["last_sequence_steps"] == 2


def test_custom_sequence_validation_rejects_later_target_after_target_specific_lethal_preview():
    state = _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 1,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 9, "unblocked_damage": 9},
                        {"target_index": 1, "damage": 6, "unblocked_damage": 6},
                    ],
                },
            },
            {
                "index": 2,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 9, "unblocked_damage": 9},
                        {"target_index": 1, "damage": 6, "unblocked_damage": 6},
                    ],
                },
            },
            {
                "index": 3,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 9, "unblocked_damage": 9},
                        {"target_index": 1, "damage": 6, "unblocked_damage": 6},
                    ],
                },
            },
            {
                "index": 4,
                "name": "Anger",
                "cost": 0,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 9, "unblocked_damage": 9},
                        {"target_index": 1, "damage": 6, "unblocked_damage": 6},
                    ],
                },
            },
        ],
        enemies=[
            {"index": 0, "name": "Calcified Cultist", "hp": 21, "max_hp": 38, "block": 0},
            {"index": 1, "name": "Damp Cultist", "hp": 52, "max_hp": 52, "block": 0},
        ],
    )

    result = validate_custom_sequence(
        [
            {"card_index": 4, "target_index": 0},
            {"card_index": 1, "target_index": 0},
            {"card_index": 2, "target_index": 0},
            {"card_index": 3, "target_index": 0},
        ],
        current_state=state,
        action_tape=[],
        character="Ironclad",
        seed="test",
        ascension=0,
        lang="en",
        process_factory=lambda: pytest.fail("target-specific lethal preflight should run before probe replay"),
    )

    assert result.validated is False
    assert "target-specific damage_by_target" in result.reason
    assert "kills enemy target 0 after step 3" in result.reason
    assert "later step 4 still targets it" in result.reason


def test_custom_sequence_validation_rejects_later_attack_after_damage_modifier_step():
    state = _combat_turn_state(
        energy=5,
        hand=[
            {
                "index": 5,
                "name": "Perfected Strike",
                "cost": 2,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "calculateddamage": 25,
                    "calculateddamage_by_target": [
                        {"target_index": 0, "calculateddamage": 25},
                        {"target_index": 1, "calculateddamage": 25},
                    ],
                },
            },
            {
                "index": 6,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 10,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 10, "unblocked_damage": 10},
                        {"target_index": 1, "damage": 10, "unblocked_damage": 10},
                    ],
                },
            },
            {
                "index": 7,
                "name": "Bash",
                "cost": 2,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 9,
                    "vulnerablepower": 2,
                    "damage_by_target": [
                        {"target_index": 0, "damage": 9, "unblocked_damage": 9},
                        {"target_index": 1, "damage": 9, "unblocked_damage": 9},
                    ],
                },
            },
        ],
        enemies=[
            {"index": 0, "name": "Punch Construct", "hp": 36, "max_hp": 55, "block": 0},
            {"index": 1, "name": "Punch Construct", "hp": 38, "max_hp": 55, "block": 0},
        ],
    )

    result = validate_custom_sequence(
        [
            {"card_index": 7, "target_index": 0},
            {"card_index": 5, "target_index": 0},
            {"card_index": 6, "target_index": 0},
        ],
        current_state=state,
        action_tape=[],
        character="Ironclad",
        seed="test",
        ascension=0,
        lang="en",
        process_factory=lambda: pytest.fail("modifier-changing preflight should run before probe replay"),
    )

    assert result.validated is False
    assert "step 1 changes enemy target 0 damage modifiers" in result.reason
    assert "later step 2 still attacks that target" in result.reason
    assert "wait for the next state" in result.reason


def _early_terminal_custom_sequence_state():
    return _combat_turn_state(
        energy=3,
        hand=[
            {
                "index": 0,
                "instance_id": "molten-fist",
                "name": "Molten Fist",
                "cost": 1,
                "energy_cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "description": "Deal 13 damage. Exhaust.",
                "stats": {"damage": 13},
            },
            {
                "index": 1,
                "instance_id": "strike-a",
                "name": "Strike",
                "cost": 1,
                "energy_cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
            {
                "index": 2,
                "instance_id": "strike-b",
                "name": "Strike",
                "cost": 1,
                "energy_cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
        ],
        enemies=[
            {
                "index": 0,
                "instance_id": "fuzzy-wurm-crawler",
                "name": "Fuzzy Wurm Crawler",
                "hp": 13,
                "max_hp": 56,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 11}],
            }
        ],
    )


class _EarlyTerminalCustomSequenceProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return _early_terminal_custom_sequence_state()

    def send(self, command):
        self.sent.append(command)
        return {"decision": "game_over", "victory": False, "player": {"hp": 56, "max_hp": 80}}

    def close(self):
        pass


def test_custom_sequence_validation_accepts_prefix_when_combat_ends_early():
    probe = _EarlyTerminalCustomSequenceProcess()
    sequence = [
        {"card_index": 0, "target_index": 0},
        {"card_index": 1, "target_index": 0},
        {"card_index": 2, "target_index": 0},
    ]

    result = validate_custom_sequence(
        sequence,
        current_state=_early_terminal_custom_sequence_state(),
        action_tape=[],
        character="Ironclad",
        seed="test",
        ascension=0,
        lang="en",
        process_factory=lambda: probe,
    )

    assert [command["args"] for command in probe.sent] == [{"card_index": 0, "target_index": 0}]
    assert result.validated is True
    assert result.steps_executed == 1
    assert result.final_state is not None
    assert result.final_state["decision"] == "game_over"


def test_runner_rejects_custom_sequence_when_probe_replay_drifts():
    main = _CustomSequenceMainProcess()
    probe = _CustomSequenceMainProcess(drift=True)
    sequence = [
        {"card_index": 0},
        {"card_index": 1, "target_index": 0},
    ]

    result = run_one(
        agent=_ChooseCustomSequenceAgent(sequence),
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=1,
    )

    assert main.sent == []
    assert "drift" in result.error
    assert result.game_action_count == 0


def test_runner_rejects_custom_sequence_over_validation_limit_without_probe():
    main = _CustomSequenceMainProcess()

    def fail_probe_factory():
        raise AssertionError("probe should not be started for overlong sequence")

    result = run_one(
        agent=_ChooseCustomSequenceAgent([{"card_index": 0}] * 5),
        seed="test",
        process=main,
        sequence_probe_factory=fail_probe_factory,
        max_steps=1,
    )

    assert main.sent == []
    assert "at most 4" in result.error


def _unsafe_custom_sequence_state(unsafe_card=None):
    if unsafe_card is None:
        unsafe_card = {
            "index": 0,
            "instance_id": "chaos-a",
            "name": "Chaos",
            "cost": 0,
            "energy_cost": 0,
            "type": "Skill",
            "target_type": "None",
            "can_play": True,
            "description": "Add a random card to your hand.",
            "stats": {},
        }
    return _custom_sequence_state(
        hand=[
            unsafe_card,
            {
                "index": 1,
                "instance_id": "defend-b",
                "name": "Defend",
                "cost": 1,
                "energy_cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"block": 5},
            },
        ]
    )


class _UnsafeCustomSequenceProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return _unsafe_custom_sequence_state()

    def send(self, command):
        self.sent.append(command)
        if len(self.sent) == 1:
            return _custom_sequence_state(
                energy=3,
                hand=[
                    {
                        "index": 0,
                        "instance_id": "defend-b",
                        "name": "Defend",
                        "cost": 1,
                        "energy_cost": 1,
                        "type": "Skill",
                        "target_type": "None",
                        "can_play": True,
                        "stats": {"block": 5},
                    }
                ],
            )
        return _custom_sequence_state(energy=2, hand=[])

    def close(self):
        pass


def test_custom_sequence_validation_rejects_random_or_choice_card_before_probe_replay():
    probe = _UnsafeCustomSequenceProcess()

    result = validate_custom_sequence(
        [{"card_index": 0}, {"card_index": 1}],
        current_state=_unsafe_custom_sequence_state(),
        action_tape=[],
        character="Ironclad",
        seed="test",
        ascension=0,
        lang="en",
        process_factory=lambda: probe,
    )

    assert probe.sent == []
    assert result.validated is False
    assert "non-deterministic" in result.reason


@pytest.mark.parametrize(
    "unsafe_card, expected_marker",
    [
        (
            {
                "index": 0,
                "instance_id": "armaments-a",
                "name": "Armaments",
                "cost": 1,
                "energy_cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "description": "Gain 5 Block. Upgrade a card in your hand.",
                "stats": {"block": 5},
            },
            "upgrade",
        ),
        (
            {
                "index": 0,
                "instance_id": "burning-pact-a",
                "name": "Burning Pact",
                "cost": 1,
                "energy_cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "description": "Exhaust 1 card. Draw 2 cards.",
                "stats": {"cards": 2},
            },
            "draw",
        ),
        (
            {
                "index": 0,
                "instance_id": "wild-strike-a",
                "name": "Wild Ricochet",
                "cost": 1,
                "energy_cost": 1,
                "type": "Attack",
                "target_type": "None",
                "can_play": True,
                "description": "Deal 8 damage to a random enemy.",
                "stats": {"damage": 8},
            },
            "random",
        ),
    ],
)
def test_custom_sequence_validation_rejects_non_atomic_cards_before_probe_replay(unsafe_card, expected_marker):
    result = validate_custom_sequence(
        [{"card_index": 0}, {"card_index": 1}],
        current_state=_unsafe_custom_sequence_state(unsafe_card),
        action_tape=[],
        character="Ironclad",
        seed="test",
        ascension=0,
        lang="en",
        process_factory=lambda: pytest.fail("non-atomic custom sequence should be rejected before probe replay"),
    )

    assert result.validated is False
    assert expected_marker in result.reason


def _free_attack_custom_sequence_state():
    return _combat_turn_state(
        energy=0,
        hand=[
            {
                "index": 0,
                "instance_id": "strike-a",
                "name": "Strike",
                "cost": 0,
                "energy_cost": 0,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [{"target_index": 0, "damage": 6, "unblocked_damage": 6}],
                },
            },
            {
                "index": 1,
                "instance_id": "strike-b",
                "name": "Strike",
                "cost": 0,
                "energy_cost": 0,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {
                    "damage": 6,
                    "damage_by_target": [{"target_index": 0, "damage": 6, "unblocked_damage": 6}],
                },
            },
        ],
        player_powers=[
            {
                "id": "FREE_ATTACK_POWER",
                "name": "Free Attack",
                "description": "Your next Attack costs 0 [E].",
                "amount": 1,
            }
        ],
        enemies=[{"index": 0, "name": "Leaf Slime (M)", "hp": 13, "max_hp": 33, "block": 0}],
    )


class _FreeAttackCustomSequenceProcess:
    def __init__(self):
        self.sent = []
        self.started = False

    def start(self):
        self.started = True

    def start_run(self, **kwargs):
        return _free_attack_custom_sequence_state()

    def send(self, command):
        self.sent.append(command)
        return _free_attack_custom_sequence_state()

    def close(self):
        pass


def test_custom_sequence_validation_rejects_one_shot_free_attack_power_before_probe_replay():
    probe = _FreeAttackCustomSequenceProcess()

    result = validate_custom_sequence(
        [{"card_index": 0, "target_index": 0}, {"card_index": 1, "target_index": 0}],
        current_state=_free_attack_custom_sequence_state(),
        action_tape=[],
        character="Ironclad",
        seed="test",
        ascension=0,
        lang="en",
        process_factory=lambda: probe,
    )

    assert probe.started is False
    assert probe.sent == []
    assert result.validated is False
    assert "one-shot free attack" in result.reason


def _juggernaut_sequence_state():
    return _combat_turn_state(
        hand=[
            {
                "index": 0,
                "instance_id": "juggernaut",
                "name": "Juggernaut",
                "cost": 0,
                "energy_cost": 0,
                "type": "Power",
                "target_type": "Self",
                "can_play": True,
                "description": "Whenever you gain Block, deal 5 damage to a random enemy.",
                "stats": {"juggernautpower": 5},
            },
            {
                "index": 1,
                "instance_id": "defend-a",
                "name": "Defend",
                "cost": 1,
                "energy_cost": 1,
                "type": "Skill",
                "target_type": "Self",
                "can_play": True,
                "stats": {"block": 5},
            },
        ],
    )


class _InvalidCustomSequenceThenEndTurnAgent:
    def __init__(self, *, invalid_calls=1):
        self.calls = 0
        self.invalid_calls = invalid_calls
        self.feedback = []

    def build_prompt(self, state, legal_actions):
        return ""

    def choose(self, state, legal_actions, *, prompt=None):
        self.calls += 1
        if self.calls <= self.invalid_calls:
            action = next(action for action in legal_actions if action.kind == "combat_custom_sequence")
            return action, {
                "agent": "test",
                "parsed": {
                    "sequence": [{"card_index": 0}] * 5,
                    "reason": "Try an invalid custom sequence.",
                },
            }
        action = next(action for action in legal_actions if action.kind == "combat_end_turn")
        return action, {"agent": "test", "parsed": {"reason": "Use a legal action after feedback."}}

    def receive_external_feedback(self, state, action, meta, feedback, legal_actions):
        self.feedback.append(feedback)


class _SafeCustomSequenceValidationProcess:
    def __init__(self):
        self.sent = []

    def start(self):
        pass

    def start_run(self, **kwargs):
        return _custom_gateway_combat_state()

    def send(self, command):
        self.sent.append(command)
        return {"decision": "game_over", "victory": False, "player": {"hp": 78, "max_hp": 80}}

    def close(self):
        pass


def test_runner_reasks_after_custom_sequence_validation_failure_before_stopping(tmp_path):
    agent = _InvalidCustomSequenceThenEndTurnAgent()
    main = _SafeCustomSequenceValidationProcess()
    probe = _SafeCustomSequenceValidationProcess()
    out = tmp_path / "run.jsonl"
    logger = JsonlLogger(out)

    try:
        result = run_one(
            agent=agent,
            seed="test",
            process=main,
            sequence_probe_factory=lambda: probe,
            logger=logger,
            max_steps=3,
        )
    finally:
        logger.close()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert agent.calls == 2
    assert len(agent.feedback) == 1
    assert "custom sequence validation failed" in agent.feedback[0]
    assert "at most 4" in agent.feedback[0]
    assert probe.sent == []
    assert [command["action"] for command in main.sent] == ["end_turn"]
    assert result.error is None
    assert result.extra["custom_sequence_validation_retry_count"] == 1
    assert any(row.get("type") == "custom_sequence_validation_feedback" for row in rows)
    assert not any(row.get("type") == "custom_sequence_validation_error" for row in rows)
    action_rows = [row for row in rows if row.get("type") == "action"]
    assert len(action_rows) == 1
    assert action_rows[0]["action"]["kind"] == "combat_end_turn"


def test_runner_reasks_twice_after_repeated_custom_sequence_validation_failure():
    agent = _InvalidCustomSequenceThenEndTurnAgent(invalid_calls=2)
    main = _SafeCustomSequenceValidationProcess()
    probe = _SafeCustomSequenceValidationProcess()

    result = run_one(
        agent=agent,
        seed="test",
        process=main,
        sequence_probe_factory=lambda: probe,
        max_steps=4,
    )

    assert agent.calls == 3
    assert len(agent.feedback) == 2
    assert "custom sequence validation failed" in agent.feedback[1]
    assert "at most 4" in agent.feedback[1]
    assert "initial visible hand snapshot" in agent.feedback[1]
    assert "sum step costs against current energy" in agent.feedback[1]
    assert "remain playable after earlier sequence steps resolve" in agent.feedback[1]
    assert [command["action"] for command in main.sent] == ["end_turn"]
    assert result.error is None
    assert result.extra["custom_sequence_validation_retry_count"] == 2


def test_combat_card_sequence_omits_draw_or_random_cards():
    state = _combat_turn_state(
        hand=[
            {
                "index": 0,
                "name": "Strike",
                "cost": 1,
                "type": "Attack",
                "target_type": "AnyEnemy",
                "can_play": True,
                "stats": {"damage": 6},
            },
            {
                "index": 1,
                "name": "Battle Trance",
                "cost": 1,
                "type": "Skill",
                "target_type": "None",
                "can_play": True,
                "stats": {"cards": 3},
                "description": "Draw 3 cards.",
            },
        ],
    )
    actions = build_legal_actions(state, allow_precomputed_card_sequences=True)

    assert not any(action.kind == "combat_card_sequence" for action in actions)


def test_openai_compat_agent_strict_mode_raises_after_invalid_action_without_fallback():
    state = _combat_turn_state()
    agent, _sent_messages = _turn_chat_agent(
        [{"action_id": 99, "reason": "bad id"}],
        max_retries=0,
        fallback_mode="error",
    )

    with pytest.raises(ModelActionError, match="action_id out of range: 99") as exc_info:
        agent.choose(state, build_legal_actions(state))

    assert len(exc_info.value.audit_meta["model_request_audits"]) == 1
    assert exc_info.value.audit_meta["model_request_audits"][0]["request_index"] == 0


def test_compact_assistant_history_omits_state_local_action_ids():
    state = _combat_turn_state()
    action = next(action for action in build_legal_actions(state) if action.kind == "combat_play_card")

    payload = json.loads(build_compact_assistant_message(action, {"reason": "Play Strike."}))

    assert "action_id" not in payload
    assert payload["action"] == action.label
    assert payload["action_kind"] == action.kind


def test_openai_compat_agent_turn_chat_keeps_history_within_combat_turn():
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        energy=2,
        hand=[],
        discard_pile_count=1,
        enemies=[
            {
                "index": 0,
                "name": "Jaw Worm",
                "hp": 14,
                "max_hp": 40,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 7}],
            }
        ],
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "combat_play",
            "changes": ["energy: 3/3 -> 2/3"],
        },
    )
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "action_id": 5,
                "reason": "Strike now.",
                "calculations": ["full private-ish calculation should not persist"],
                "candidates": [{"action_id": 5, "label": "play card"}],
            },
            {"action_id": 5, "reason": "No useful cards remain."},
        ]
    )

    first_action, _first_meta = agent.choose(state, build_legal_actions(state))
    second_action, second_meta = agent.choose(next_state, build_legal_actions(next_state))

    assert first_action.kind == "combat_play_card"
    assert second_action.kind == "combat_end_turn"
    assert len(sent_messages[1]) == 4
    assistant_history = sent_messages[1][2]["content"]
    assert "Strike now." in assistant_history
    assert "calculations" not in assistant_history
    assert "candidates" not in assistant_history
    assert "Turn update." in sent_messages[1][3]["content"]
    assert "Only the latest Current turn state and Legal actions are valid" in sent_messages[1][3]["content"]
    assert "Current turn state:" in sent_messages[1][3]["content"]
    assert second_meta["conversation_mode"] == "turn_chat"
    assert second_meta["turn_chat_history_turns"] == 1


def test_openai_compat_agent_combat_mdp_state_keeps_structured_turn_history_without_chat_history():
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        energy=2,
        hand=[],
        discard_pile_count=1,
        enemies=[
            {
                "index": 0,
                "name": "Jaw Worm",
                "hp": 14,
                "max_hp": 40,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 7}],
            }
        ],
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "combat_play",
            "changes": [
                "energy: 3/3 -> 2/3",
                "discard_pile: 0 -> 1",
                "enemy Jaw Worm[0]: hp 20/40 -> 14/40",
            ],
        },
    )
    agent, sent_messages = _combat_mdp_agent(
        [
            {
                "action_id": 5,
                "reason": "Strike now.",
                "calculations": ["must not persist as chat history"],
            },
            {"action_id": 5, "reason": "End after Strike."},
        ]
    )

    first_action, first_meta = agent.choose(state, build_legal_actions(state))
    agent.record_transition(state, first_action, first_meta, next_state)
    second_action, second_meta = agent.choose(next_state, build_legal_actions(next_state))

    assert first_action.kind == "combat_play_card"
    assert second_action.kind == "combat_end_turn"
    assert len(sent_messages[1]) == 2
    user_prompt = sent_messages[1][1]["content"]
    assert "Combat MDP state JSON:" in user_prompt
    assert "Combat turn process JSON:" in user_prompt
    assert "actions_this_turn" in user_prompt
    assert '"card_name":"Strike"' in user_prompt
    assert '"target":{"index":0,"name":"Jaw Worm"}' in user_prompt
    assert "enemy Jaw Worm[0]: hp 20/40 -> 14/40" in user_prompt
    assert "Strike now." not in user_prompt
    assert "do not persist verbose scratchpad" not in user_prompt
    process_json = user_prompt.split("Combat turn process JSON:\n", 1)[1].split("\n\nLegal actions:", 1)[0]
    assert '"action_id":5' not in process_json
    assert second_meta["conversation_mode"] == "combat_mdp_state"
    assert second_meta["combat_mdp_turn_actions"] == 1


def test_openai_compat_agent_combat_mdp_state_keeps_only_code_owned_turn_intent_checkpoint():
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        energy=2,
        hand=[],
        discard_pile_count=1,
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "combat_play",
            "changes": ["energy: 3/3 -> 2/3"],
        },
    )
    agent, sent_messages = _combat_mdp_agent(
        [
            {
                "turn_intent": "Use potion for 1 energy, then play Strike for lethal.",
                "turn_check": {
                    "observed_numbers": {},
                    "incoming_damage_calculation": {},
                    "selected_action_calculation": {},
                    "end_turn_audit": None,
                },
                "action_id": 5,
                "reason": "Strike now to push lethal setup.",
                "combat_notes": ["6 damage, 2 energy remains"],
                "calculations": ["do not persist verbose scratchpad"],
                "candidates": [{"action_id": 5, "label": "play card"}],
            },
            {
                "turn_intent": "End only after checking the fresh state.",
                "turn_check": {
                    "observed_numbers": {},
                    "incoming_damage_calculation": {},
                    "selected_action_calculation": None,
                    "end_turn_audit": {},
                },
                "action_id": 5,
                "reason": "End after Strike.",
            },
        ],
        combat_reasoning_pool_enabled=True,
        turn_check_enabled=True,
    )

    first_action, first_meta = agent.choose(state, build_legal_actions(state))
    agent.record_transition(state, first_action, first_meta, next_state)
    agent.choose(next_state, build_legal_actions(next_state))

    user_prompt = sent_messages[1][1]["content"]
    assert '"current_turn_intent"' in user_prompt
    assert "Re-evaluate the provisional action order or resource objective against the current state." in user_prompt
    assert "Use potion for 1 energy, then play Strike for lethal." not in user_prompt
    assert "Strike now to push lethal setup." not in user_prompt
    assert "6 damage, 2 energy remains" not in user_prompt
    assert '"primary_goal"' not in user_prompt
    assert '"reasoning_pool"' not in user_prompt
    assert "do not persist verbose scratchpad" not in user_prompt
    assert '"label":"play card"' not in user_prompt
    process_json = user_prompt.split("Combat turn process JSON:\n", 1)[1].split("\n\nLegal actions:", 1)[0]
    assert '"action_id":5' not in process_json
    assert user_prompt.index("Game rules for the current decision:") > user_prompt.index(
        "Combat turn process JSON:"
    )
    assert user_prompt.index("Game rules for the current decision:") < user_prompt.index(
        "Legal actions:"
    )
    assert '"risk_features"' not in user_prompt


def test_openai_compat_agent_combat_mdp_state_preserves_turn_intent_across_card_select_modal():
    state = _combat_turn_state()
    modal_state = {
        "decision": "card_select",
        "context": state["context"],
        "player": state["player"],
        "min_select": 1,
        "max_select": 1,
        "prompt": "Choose a card.",
        "source_card": {"name": "Armaments"},
        "combat": {
            "round": state["round"],
            "energy": 2,
            "max_energy": 3,
            "draw_pile_count": 4,
            "discard_pile_count": 1,
            "exhaust_pile_count": 0,
            "enemies": state["enemies"],
        },
        "cards": [{"index": 0, "name": "Defend", "cost": 1, "type": "Skill"}],
        "last_action_result": {
            "action_label": "play card 0: Armaments",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "card_select",
            "changes": ["energy: 3/3 -> 2/3"],
        },
    }
    next_state = _combat_turn_state(
        energy=2,
        hand=[],
        discard_pile_count=1,
        last_action_result={
            "action_label": "select cards 0",
            "action_kind": "card_select",
            "decision_before": "card_select",
            "decision_after": "combat_play",
            "changes": [],
        },
    )
    non_end_contract_fields = {
        "observed_numbers": {},
        "incoming_damage_calculation": {},
        "selected_action_calculation": {},
        "end_turn_audit": None,
    }
    end_turn_contract_fields = {
        "observed_numbers": {},
        "incoming_damage_calculation": {},
        "selected_action_calculation": None,
        "end_turn_audit": {},
    }
    agent, sent_messages = _combat_mdp_agent(
        [
            {
                "turn_intent": "Preserve the same turn plan through the modal choice.",
                "turn_check": non_end_contract_fields,
                "action_id": 5,
                "reason": "Open the card selection.",
            },
            {
                "selected_card_indices": [0],
                "reason": "Choose Defend.",
            },
            {
                "turn_intent": "Finish after rechecking the returned combat state.",
                "turn_check": end_turn_contract_fields,
                "action_id": 5,
                "reason": "End the turn.",
            },
        ],
        combat_reasoning_pool_enabled=True,
        turn_check_enabled=True,
    )

    first_action, first_meta = agent.choose(state, build_legal_actions(state))
    agent.record_transition(state, first_action, first_meta, modal_state)
    modal_action, modal_meta = agent.choose(modal_state, build_legal_actions(modal_state))
    agent.record_transition(modal_state, modal_action, modal_meta, next_state)
    agent.choose(next_state, build_legal_actions(next_state))

    returned_prompt = sent_messages[2][1]["content"]
    assert '"current_turn_intent"' in returned_prompt
    assert "Re-evaluate the provisional action order or resource objective against the current state." in returned_prompt
    assert "Preserve the same turn plan through the modal choice." not in returned_prompt
    assert "Choose Defend." not in returned_prompt


def test_openai_compat_agent_combat_mdp_turn_check_contract_is_retried_without_dropping_fields():
    state = _combat_turn_state()
    agent, sent_messages = _combat_mdp_agent(
        [
            {"action_id": 5, "reason": "Missing required treatment fields."},
            {
                "turn_intent": "Deal damage and reassess.",
                "turn_check": {
                    "observed_numbers": {},
                    "incoming_damage_calculation": {},
                    "selected_action_calculation": {},
                    "end_turn_audit": None,
                },
                "action_id": 5,
                "reason": "Strike.",
            },
        ],
        combat_reasoning_pool_enabled=True,
        turn_check_enabled=True,
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.kind == "combat_play_card"
    assert meta["attempt"] == 1
    assert meta["previous_errors"] == [
        "turn_intent is required by the current combat response contract"
    ]
    correction = sent_messages[1][-1]["content"]
    assert "Preserve the required top-level turn_intent and turn_check fields" in correction


def test_openai_compat_agent_turn_check_requires_only_non_end_calculation_for_card_action():
    state = _combat_turn_state()
    base_turn_check = {
        "observed_numbers": {},
        "incoming_damage_calculation": {},
        "selected_action_calculation": {},
    }
    agent, _ = _combat_mdp_agent(
        [
            {
                "turn_intent": "Strike and reassess.",
                "turn_check": {**base_turn_check, "end_turn_audit": {}},
                "action_id": 5,
                "reason": "Strike.",
            },
            {
                "turn_intent": "Strike and reassess.",
                "turn_check": base_turn_check,
                "action_id": 5,
                "reason": "Strike.",
            },
        ],
        combat_reasoning_pool_enabled=True,
        turn_check_enabled=True,
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.kind == "combat_play_card"
    assert meta["attempt"] == 1
    assert meta["previous_errors"] == [
        "turn_check.end_turn_audit must be null for a non-end-turn action"
    ]


def test_openai_compat_agent_turn_check_requires_only_end_audit_for_end_turn_action():
    state = _combat_turn_state(hand=[], energy=0)
    agent, _ = _combat_mdp_agent(
        [
            {
                "turn_intent": "End the spent turn.",
                "turn_check": {
                    "observed_numbers": {},
                    "incoming_damage_calculation": {},
                    "selected_action_calculation": {},
                    "end_turn_audit": None,
                },
                "action_id": 5,
                "reason": "End.",
            },
            {
                "turn_intent": "End the spent turn.",
                "turn_check": {
                    "observed_numbers": {},
                    "incoming_damage_calculation": {},
                    "end_turn_audit": {},
                },
                "action_id": 5,
                "reason": "End.",
            },
        ],
        combat_reasoning_pool_enabled=True,
        turn_check_enabled=True,
        max_retries=1,
        fallback_mode="error",
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.kind == "combat_end_turn"
    assert meta["attempt"] == 1
    assert meta["previous_errors"] == [
        "turn_check.selected_action_calculation must be null when selecting end turn"
    ]


def test_openai_compat_agent_combat_mdp_state_keeps_compact_recent_turn_delta():
    state = _combat_turn_state(
        energy=0,
        hand=[],
        player={"hp": 70, "max_hp": 80, "block": 5, "gold": 0, "deck_size": 10, "potions": []},
        player_powers=[{"name": "Vulnerable", "amount": 2, "type": "Debuff"}],
    )
    next_state = _combat_turn_state(
        round=2,
        energy=3,
        hp=68,
        hand=[{"index": 0, "name": "Defend", "cost": 1, "type": "Skill", "can_play": True}],
        player={"hp": 68, "max_hp": 80, "block": 0, "gold": 0, "deck_size": 10, "potions": []},
        player_powers=[{"name": "Vulnerable", "amount": 1, "type": "Debuff"}],
        enemies=[
            {
                "index": 0,
                "name": "Jaw Worm",
                "hp": 14,
                "max_hp": 40,
                "block": 0,
                "intents": [{"type": "Buff"}],
                "powers": [{"name": "Poison", "amount": 2, "type": "Debuff"}],
            }
        ],
        last_action_result={
            "action_label": "end turn",
            "action_kind": "combat_end_turn",
            "decision_before": "combat_play",
            "decision_after": "combat_play",
            "changes": [
                "hp: 70/80 -> 68/80",
                "block: 5 -> 0",
                "round: 1 -> 2",
            ],
        },
    )
    agent, sent_messages = _combat_mdp_agent(
        [
            {
                "action_id": 5,
                "reason": "End the spent turn.",
                "candidates": [{"action_id": 5}],
            },
            {"action_id": 5, "reason": "Fresh turn."},
        ],
        combat_turn_board_enabled=True,
    )

    action, meta = agent.choose(state, build_legal_actions(state))
    agent.record_transition(state, action, meta, next_state)
    agent.choose(next_state, build_legal_actions(next_state))

    fresh_prompt = sent_messages[1][1]["content"]
    process = _json_after(fresh_prompt, "Combat turn process JSON:\n")
    assert "cross_turn_board" not in process
    assert "combat_summary" not in process
    assert process["recent_window_start"]["player"]["hp"] == 70
    assert "hand" not in process["recent_window_start"]
    assert "piles" not in process["recent_window_start"]
    assert "energy" not in process["recent_window_start"]
    assert len(process["recent_turns"]) == 1
    completed_turn = process["recent_turns"][0]
    assert completed_turn["round"] == 1
    assert "turn_start" not in completed_turn
    assert "turn_boundary" not in completed_turn
    transition = completed_turn["end_turn_transition"]
    assert transition["source"] == "snapshot_diff"
    assert transition["ordering"] == "not_observed"
    assert transition["announced_enemy_intents"] == [
        {
            "index": 0,
            "name": "Jaw Worm",
            "intents": [{"type": "Attack", "damage": 7, "hits": 1}],
        }
    ]

    facts = {(fact["entity"], fact["field"]): fact for fact in transition["observed_facts"]}
    assert facts[("player", "hp")]["delta"] == -2
    assert facts[("player", "power:Vulnerable")]["change"] == "changed"
    assert facts[("enemy:0:Jaw Worm", "hp")]["delta"] == -6
    assert facts[("enemy:0:Jaw Worm", "power:Poison")]["change"] == "added"
    assert ("player", "block") not in facts
    assert ("enemy:0:Jaw Worm", "intents") not in facts
    assert all(entity != "combat" for entity, _field in facts)
    assert "End the spent turn." not in fresh_prompt
    process_json = fresh_prompt.split("Combat turn process JSON:\n", 1)[1].split("\n\nLegal actions:", 1)[0]
    assert '"action_id":5' not in process_json
    assert "candidates" not in process_json


def test_openai_compat_agent_combat_mdp_state_rolling_memory_resets_after_combat_exit():
    state = _combat_turn_state(energy=0, hand=[])
    reward_state = {
        "decision": "combat_reward",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {"hp": 70, "max_hp": 80},
        "last_action_result": {"changes": ["combat ended"]},
    }
    later_combat = _combat_turn_state(context={"act": 1, "floor": 3, "room_type": "Monster"})
    agent, sent_messages = _combat_mdp_agent(
        [
            {"action_id": 5, "reason": "End combat."},
            {"action_id": 5, "reason": "New fight."},
        ],
        combat_turn_board_enabled=True,
    )

    action, meta = agent.choose(state, build_legal_actions(state))
    agent.record_transition(state, action, meta, reward_state)
    agent.choose(later_combat, build_legal_actions(later_combat))

    later_prompt = sent_messages[1][1]["content"]
    assert '"recent_turns"' not in later_prompt
    assert '"combat_summary"' not in later_prompt
    assert "End combat." not in later_prompt


def test_openai_compat_agent_combat_mdp_state_rolls_old_turn_into_summary():
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        conversation_mode="combat_mdp_state",
        short_term_memory_enabled=False,
        combat_turn_board_enabled=True,
    )

    for round_number in range(1, 6):
        start_hp = 71 - round_number
        start_enemy_hp = 22 - (2 * round_number)
        start_state = _combat_turn_state(
            round=round_number,
            energy=3,
            hand=[],
            player={
                "hp": start_hp,
                "max_hp": 80,
                "block": 0,
                "gold": 0,
                "deck_size": 10,
                "potions": [],
            },
            enemies=[
                {
                    "index": 0,
                    "name": "Jaw Worm",
                    "hp": start_enemy_hp,
                    "max_hp": 40,
                    "block": 0,
                    "intents": [{"type": "Attack", "damage": 1}],
                }
            ],
        )
        agent.build_prompt(start_state, build_legal_actions(start_state))

        before_end_state = _combat_turn_state(
            round=round_number,
            energy=0,
            hand=[],
            player={
                "hp": start_hp,
                "max_hp": 80,
                "block": 0,
                "gold": 0,
                "deck_size": 10,
                "potions": [],
            },
            enemies=[
                {
                    "index": 0,
                    "name": "Jaw Worm",
                    "hp": start_enemy_hp - 2,
                    "max_hp": 40,
                    "block": 0,
                    "intents": [{"type": "Attack", "damage": 1}],
                }
            ],
        )
        next_state = _combat_turn_state(
            round=round_number + 1,
            energy=3,
            hand=[],
            player={
                "hp": start_hp - 1,
                "max_hp": 80,
                "block": 0,
                "gold": 0,
                "deck_size": 10,
                "potions": [],
            },
            enemies=[
                {
                    "index": 0,
                    "name": "Jaw Worm",
                    "hp": start_enemy_hp - 2,
                    "max_hp": 40,
                    "block": 0,
                    "intents": [{"type": "Attack", "damage": 1}],
                }
            ],
        )
        end_turn = next(
            action for action in build_legal_actions(before_end_state) if action.kind == "combat_end_turn"
        )
        agent.record_transition(
            before_end_state,
            end_turn,
            {"parsed": {"reason": f"finish round {round_number}"}},
            next_state,
        )

    prompt = agent.build_prompt(next_state, build_legal_actions(next_state))
    process = _json_after(prompt, "Combat turn process JSON:\n")

    assert [turn["round"] for turn in process["recent_turns"]] == [3, 4, 5]
    summary = process["combat_summary"]
    assert summary["source"] == "deterministic_fold"
    assert summary["covered_rounds"] == {"first": 1, "last": 2, "count": 2}
    assert summary["completed_turns"] == 2
    assert summary["player_hp_delta"] == -2
    assert summary["boundary_player_hp_delta"] == -2
    assert summary["enemy_hp_delta_by_enemy"] == {"0:Jaw Worm": -4}

    reward_state = {
        "decision": "combat_reward",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {"hp": 65, "max_hp": 80},
    }
    end_turn = next(action for action in build_legal_actions(next_state) if action.kind == "combat_end_turn")
    agent.record_transition(next_state, end_turn, {"parsed": {"reason": "combat ended"}}, reward_state)
    later_combat = _combat_turn_state(context={"act": 1, "floor": 3, "room_type": "Monster"})
    later_prompt = agent.build_prompt(later_combat, build_legal_actions(later_combat))
    assert "Combat turn process JSON:" not in later_prompt


def test_openai_compat_agent_combat_mdp_state_resets_after_end_turn_transition():
    state = _combat_turn_state(energy=0, hand=[])
    next_state = _combat_turn_state(round=2, energy=3, hand=[])
    agent, sent_messages = _combat_mdp_agent(
        [
            {"action_id": 5, "reason": "End the spent turn."},
            {"action_id": 5, "reason": "Fresh turn."},
        ]
    )

    action, meta = agent.choose(state, build_legal_actions(state))
    agent.record_transition(state, action, meta, next_state)
    agent.choose(next_state, build_legal_actions(next_state))

    assert action.kind == "combat_end_turn"
    fresh_prompt = sent_messages[1][1]["content"]
    assert "Combat MDP state JSON:" in fresh_prompt
    assert "End the spent turn." not in fresh_prompt
    assert "Combat turn process JSON:" not in fresh_prompt


def test_openai_compat_agent_turn_check_is_added_to_combat_prompt_and_kept_compact():
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        energy=2,
        hand=[],
        discard_pile_count=1,
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "combat_play",
            "changes": ["energy: 3/3 -> 2/3"],
        },
    )
    agent, sent_messages = _turn_chat_agent(
        [
            {
                "turn_intent": "Apply pressure and reassess after the selected action.",
                "turn_check": {
                    "observed_numbers": {
                        "current_hp": 70,
                        "current_block": 0,
                        "current_energy": 3,
                    },
                    "incoming_damage_calculation": {
                        "parts": ["enemy 0: 7 x 1 = 7"],
                        "total": 7,
                    },
                    "selected_action_calculation": {
                        "cost": 1,
                        "displayed_damage": 6,
                        "displayed_block": None,
                        "expected_energy_after": 2,
                        "expected_block_after": 0,
                        "expected_target_hp_after": 34,
                        "effect_summary": "Deal 6 damage.",
                    },
                    "end_turn_audit": None,
                },
                "action_id": 5,
                "reason": "Start with Strike.",
                "calculations": ["do not keep this verbose field"],
            },
            {
                "turn_intent": "End after checking the refreshed state.",
                "turn_check": {
                    "observed_numbers": {},
                    "incoming_damage_calculation": {},
                    "selected_action_calculation": None,
                    "end_turn_audit": {},
                },
                "action_id": 5,
                "reason": "End after the plan.",
            },
        ],
        turn_check_enabled=True,
    )

    agent.choose(state, build_legal_actions(state))
    agent.choose(next_state, build_legal_actions(next_state))

    assert "turn_check" in sent_messages[0][0]["content"]
    assert "your own current-state arithmetic scratchpad" in sent_messages[0][0]["content"]
    assert "Do not enumerate or calculate every unselected alternative" in sent_messages[0][0]["content"]
    assert "expected_hp_after_enemy_attacks" in sent_messages[0][0]["content"]
    assert "omit selected_action_calculation" in sent_messages[0][0]["content"]
    assert "recompute_after_action" not in sent_messages[0][0]["content"]
    assert "Turn planning instruction:" not in sent_messages[0][1]["content"]
    assistant_history = sent_messages[1][2]["content"]
    assert "turn_check" not in assistant_history
    assert "incoming_damage_calculation" not in assistant_history
    assert "calculations" not in assistant_history


def test_openai_compat_agent_turn_chat_omits_turn_check_prompt_by_default():
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent([{"action_id": 5, "reason": "Strike."}])

    agent.choose(state, build_legal_actions(state))

    assert "turn_check" not in sent_messages[0][0]["content"]
    assert "Turn planning instruction:" not in sent_messages[0][1]["content"]
    assert "turn_check" not in sent_messages[0][1]["content"]


def test_openai_compat_agent_combat_mdp_state_can_require_turn_check():
    state = _combat_turn_state()
    agent, sent_messages = _combat_mdp_agent(
        [
            {
                "action_id": 5,
                "turn_intent": "Deal damage and reassess.",
                "turn_check": {
                    "observed_numbers": {
                        "current_hp": 70,
                        "current_block": 0,
                        "current_energy": 3,
                    },
                    "incoming_damage_calculation": {
                        "parts": ["enemy 0: 7 x 1 = 7"],
                        "total": 7,
                    },
                    "selected_action_calculation": {
                        "cost": 1,
                        "displayed_damage": 6,
                        "displayed_block": None,
                        "expected_energy_after": 2,
                        "expected_block_after": 0,
                        "expected_target_hp_after": 34,
                        "effect_summary": "Deal 6 damage.",
                    },
                    "end_turn_audit": None,
                },
                "reason": "Strike.",
            }
        ],
        turn_check_enabled=True,
    )

    action, meta = agent.choose(state, build_legal_actions(state))

    assert action.kind == "combat_play_card"
    assert meta["parsed"]["turn_check"]["incoming_damage_calculation"]["total"] == 7
    assert meta["parsed"]["turn_intent"] == "Deal damage and reassess."
    assert "turn_check" in sent_messages[0][1]["content"]
    assert "your own current-state arithmetic scratchpad" in sent_messages[0][1]["content"]
    assert '"turn_intent"' not in sent_messages[0][1]["content"]
    assert '"candidates"' not in sent_messages[0][1]["content"]
    assert '"calculations"' not in sent_messages[0][1]["content"]
    assert '"situation"' not in sent_messages[0][1]["content"]
    assert '"why_selected_action_is_safe"' not in sent_messages[0][1]["content"]
    assert "expected_energy_next_player_turn" in sent_messages[0][1]["content"]


def test_openai_compat_agent_combat_reasoning_pool_requires_intent_without_turn_check():
    state = _combat_turn_state()
    next_state = _combat_turn_state(
        energy=2,
        hand=[],
        discard_pile_count=1,
        last_action_result={
            "action_label": "play card 0: Strike on enemy 0: Jaw Worm",
            "action_kind": "combat_play_card",
            "decision_before": "combat_play",
            "decision_after": "combat_play",
            "changes": ["energy: 3/3 -> 2/3"],
        },
    )
    agent, sent_messages = _combat_mdp_agent(
        [
            {
                "turn_intent": "Apply pressure, then reassess the observed state.",
                "action_id": 5,
                "reason": "Strike.",
            },
            {
                "turn_intent": "End only after rechecking the current state.",
                "action_id": 5,
                "reason": "End.",
            },
        ],
        combat_reasoning_pool_enabled=True,
        turn_check_enabled=False,
    )

    first_action, first_meta = agent.choose(state, build_legal_actions(state))
    agent.record_transition(state, first_action, first_meta, next_state)
    agent.choose(next_state, build_legal_actions(next_state))

    first_prompt = sent_messages[0][1]["content"]
    second_prompt = sent_messages[1][1]["content"]
    assert '"turn_intent"' in first_prompt
    assert '"turn_check"' not in first_prompt
    assert "Do not put arithmetic, lethal claims, future enemy intents, or unobserved consequences" in first_prompt
    assert '"current_turn_intent"' in second_prompt
    assert "Re-evaluate the provisional action order or resource objective against the current state." in second_prompt
    assert "Apply pressure, then reassess the observed state." not in second_prompt


def test_openai_compat_agent_turn_chat_clears_after_end_turn_transition():
    state = _combat_turn_state(energy=0, hand=[])
    next_state = _combat_turn_state(round=2, energy=3, hand=[])
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 5, "reason": "End the spent turn."},
            {"action_id": 5, "reason": "Fresh turn prompt."},
        ]
    )

    action, meta = agent.choose(state, build_legal_actions(state))
    agent.record_transition(state, action, meta, next_state)
    agent.choose(next_state, build_legal_actions(next_state))

    assert action.kind == "combat_end_turn"
    assert len(sent_messages[1]) == 2
    assert "End the spent turn." not in sent_messages[1][-1]["content"]
    assert "New player turn." in sent_messages[1][-1]["content"]


def test_openai_compat_agent_turn_chat_does_not_keep_noncombat_history():
    combat_state = _combat_turn_state()
    map_state = {
        "decision": "map_select",
        "context": {"act": 1, "floor": 3, "room_type": "Map"},
        "player": {"hp": 70, "max_hp": 80, "gold": 0, "deck_size": 10},
        "choices": [],
    }
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 5, "reason": "Strike before leaving combat context."},
            {"action_id": 0, "reason": "View deck on map."},
        ]
    )

    agent.choose(combat_state, build_legal_actions(combat_state))
    agent.choose(map_state, build_legal_actions(map_state))

    assert len(sent_messages[1]) == 2
    assert "Strike before leaving combat context." not in sent_messages[1][-1]["content"]
    assert "Route decision state JSON:" in sent_messages[1][-1]["content"]
    assert "Game state:" not in sent_messages[1][-1]["content"]


def test_openai_compat_agent_turn_chat_window_trims_old_pairs():
    state = _combat_turn_state()
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 5, "reason": "first action"},
            {"action_id": 5, "reason": "second action"},
            {"action_id": 5, "reason": "third action"},
        ],
        turn_chat_window=1,
    )

    actions = build_legal_actions(state)
    agent.choose(state, actions)
    agent.choose(state, actions)
    agent.choose(state, actions)

    third_transcript = "\n".join(message["content"] for message in sent_messages[2])
    assert "first action" not in third_transcript
    assert "second action" in third_transcript


def test_openai_compat_agent_turn_chat_view_update_keeps_current_turn_reminder():
    state = _combat_turn_state()
    viewed_state = _combat_turn_state(
        view_draw_pile=True,
        draw_pile=[{"index": 0, "name": "Defend", "cost": 1, "type": "Skill", "stats": {"block": 5}}],
    )
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 2, "reason": "Check draw pile."},
            {"action_id": 5, "reason": "End after checking."},
        ]
    )

    agent.choose(state, build_legal_actions(state))
    agent.choose(viewed_state, build_legal_actions(viewed_state))

    update = sent_messages[1][-1]["content"]
    assert "Requested information: draw pile." in update
    assert "Viewed draw pile:" in update
    assert "Current turn reminder:" in update
    assert "Legal actions:" in update


def test_map_prompt_preserves_complete_deck_relic_and_potion_mechanics():
    state = {
        "decision": "map_select",
        "context": {
            "act": 1,
            "floor": 4,
            "room_type": "Map",
            "boss": {"id": "BOSS.TEST", "name": "Test Boss", "description": "Boss mechanic."},
        },
        "player": {
            "hp": 61,
            "max_hp": 80,
            "gold": 99,
            "deck_size": 1,
            "deck": [
                {
                    "id": "CARD.TEST",
                    "instance_id": "card-1",
                    "name": "Route Card",
                    "upgraded": False,
                    "description": "Current card mechanic.",
                    "after_upgrade": {"description": "Upgraded card mechanic."},
                }
            ],
            "relics": [
                {
                    "id": "RELIC.TEST",
                    "name": "Route Relic",
                    "description": "Relic route mechanic.",
                    "vars": {"amount": 2},
                }
            ],
            "potions": [
                {
                    "index": 0,
                    "id": "POTION.TEST",
                    "name": "Route Potion",
                    "description": "Potion route mechanic.",
                    "target_type": "AnyEnemy",
                }
            ],
        },
        "choices": [{"col": 2, "row": 4, "type": "Unknown"}],
    }
    actions = build_legal_actions(state)
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        run_summary_enabled=False,
    )

    prompt = agent.build_prompt(state, actions)
    route_state = _json_after(prompt, "Route decision state JSON:\n")

    assert route_state["player"]["deck"][0]["description"] == "Current card mechanic."
    assert route_state["player"]["deck"][0]["after_upgrade"]["description"] == "Upgraded card mechanic."
    assert route_state["player"]["relics"][0]["description"] == "Relic route mechanic."
    assert route_state["player"]["potions"][0]["target_type"] == "AnyEnemy"
    assert route_state["choices"] == state["choices"]
    assert "go to Unknown at col=2 row=4" in prompt


def test_turn_chat_update_restates_complete_current_combat_information():
    state = _combat_turn_state(
        player={
            "hp": 70,
            "max_hp": 80,
            "block": 3,
            "gold": 0,
            "deck_size": 10,
            "relics": [],
            "potion_slots": 3,
            "potion_empty_slots": 2,
            "potions": [
                {
                    "index": 0,
                    "name": "Fire Potion",
                    "description": "Deal 20 damage.",
                    "target_type": "AnyEnemy",
                }
            ],
        },
        player_powers=[
            {"name": "Strength", "amount": 2, "type": "Buff", "description": "Attacks deal more damage."}
        ],
        hand=[
            {
                "index": 0,
                "id": "CARD.TEST",
                "name": "Fresh Hand Card",
                "cost": 1,
                "type": "Attack",
                "can_play": True,
                "description": "Deal 8 damage.",
                "after_upgrade": {
                    "description": "Deal 11 damage.",
                    "added_keywords": ["Retain"],
                    "removed_keywords": ["Exhaust"],
                },
            }
        ],
        orbs=[],
        orb_slots=3,
        stars=2,
        osty={
            "name": "Osty",
            "hp": 18,
            "max_hp": 24,
            "block": 4,
            "alive": True,
            "powers": [{"name": "Osty Power", "amount": 1, "description": "Osty mechanic."}],
        },
    )

    update = agents_module.build_turn_update_user_message(state, build_legal_actions(state))

    assert "description: Deal 8 damage." in update
    assert "upgrade_description: Deal 11 damage." in update
    assert "upgrade_added_keywords: Retain" in update
    assert "upgrade_removed_keywords: Exhaust" in update
    assert "description: Deal 20 damage." in update
    assert "Buff Strength(2): Attacks deal more damage." in update
    assert "Orbs: none" in update
    assert "Orb slots: 3" in update
    assert "Stars: 2" in update
    assert "Osty powers (1):" in update
    assert "Osty Power(1): Osty mechanic." in update


def test_card_reward_prompt_exposes_upgrade_keyword_delta():
    state = {
        "decision": "card_reward",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": {"hp": 55, "max_hp": 80, "gold": 0, "deck_size": 0, "deck": []},
        "cards": [
            {
                "index": 0,
                "name": "Wisp",
                "type": "Skill",
                "cost": 1,
                "keywords": ["Exhaust"],
                "description": "Gain [E].\nExhaust.",
                "after_upgrade": {
                    "description": "Gain [E].\nExhaust.\nRetain.",
                    "added_keywords": ["Retain"],
                },
            }
        ],
        "can_skip": True,
    }
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:1",
        model="test",
        run_summary_enabled=False,
    )

    prompt = agent.build_prompt(state, build_legal_actions(state))

    assert "upgrade_description: Gain [E]. Exhaust. Retain." in prompt
    assert "upgrade_added_keywords: Retain" in prompt


def test_openai_compat_agent_turn_chat_treats_combat_card_select_as_same_turn_modal():
    state = _combat_turn_state()
    modal_state = {
        "decision": "card_select",
        "context": {"act": 1, "floor": 2, "room_type": "Monster"},
        "player": state["player"],
        "min_select": 1,
        "max_select": 1,
        "prompt": "Exhaust 1 card.",
        "source_card": {"name": "Burning Pact"},
        "combat": {
            "round": 1,
            "energy": 2,
            "max_energy": 3,
            "draw_pile_count": 4,
            "discard_pile_count": 1,
            "exhaust_pile_count": 0,
            "enemies": state["enemies"],
        },
        "cards": [{"index": 0, "name": "Injury", "cost": 0, "type": "Curse"}],
    }
    agent, sent_messages = _turn_chat_agent(
        [
            {"action_id": 5, "reason": "Play Burning Pact."},
            {"action_id": 5, "reason": "Exhaust Injury."},
        ]
    )

    agent.choose(state, build_legal_actions(state))
    agent.choose(modal_state, build_legal_actions(modal_state))

    assert len(sent_messages[1]) == 4
    assert "Play Burning Pact." in sent_messages[1][2]["content"]
    assert "Turn modal." in sent_messages[1][3]["content"]
    assert "Decision: card_select" in sent_messages[1][3]["content"]


def test_openai_compat_agent_rejects_turn_chat_with_episode_memory():
    with pytest.raises(ValueError, match="parallel"):
        OpenAICompatAgent(
            base_url="http://127.0.0.1:1",
            model="test",
            conversation_mode="turn_chat",
            short_term_memory_enabled=True,
        )


def test_run_config_redacts_api_key_cli_arg():
    assert _redact_argv(["--api-key", "secret", "--model", "m"]) == [
        "--api-key",
        "<redacted>",
        "--model",
        "m",
    ]
    assert _redact_argv(["--api-key=secret"]) == ["--api-key=<redacted>"]


def test_agent_from_args_passes_llm_temperature():
    agent = agent_from_args(
        "llm",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
        llm_temperature=0.25,
    )

    assert isinstance(agent, OutOfCombatContextAgent)
    assert isinstance(agent.inner, OpenAICompatAgent)
    assert agent.reward_only is True
    assert agent.inner.temperature == 0.25


def test_agent_from_args_passes_strict_fallback_mode_by_default():
    agent = agent_from_args(
        "llm",
        base_url="http://127.0.0.1:1",
        model="test",
        api_key="local",
    )

    assert isinstance(agent, OutOfCombatContextAgent)
    assert isinstance(agent.inner, OpenAICompatAgent)
    assert agent.inner.fallback_mode == "error"


def test_load_benchmark_config_reads_nested_context_management(tmp_path, monkeypatch):
    pytest.importorskip("omegaconf")
    for key in (
        "STS2_BENCH_LLM_TEMPERATURE",
        "STS2_BENCH_CONTEXT_MODE",
        "STS2_BENCH_TURN_CHAT_WINDOW",
        "STS2_BENCH_TURN_CHAT_UPDATE_MODE",
        "STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY",
        "STS2_BENCH_TURN_CHAT_PLAN_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        """
llm_temperature: 0.2
context_management:
  mode: turn_chat
  turn_chat:
    window: 3
    update_mode: delta
    assistant_history: compact
    plan:
      enabled: true
""",
        encoding="utf-8",
    )

    load_benchmark_config(config)

    assert os.environ["STS2_BENCH_LLM_TEMPERATURE"] == "0.2"
    assert os.environ["STS2_BENCH_CONTEXT_MODE"] == "turn_chat"
    assert os.environ["STS2_BENCH_TURN_CHAT_WINDOW"] == "3"
    assert os.environ["STS2_BENCH_TURN_CHAT_UPDATE_MODE"] == "delta"
    assert os.environ["STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY"] == "compact"
    assert os.environ["STS2_BENCH_TURN_CHAT_PLAN_ENABLED"] == "true"
