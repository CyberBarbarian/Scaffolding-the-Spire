import base64
import hashlib
import http.client
import io
import json
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from sts2_bench.direct_cli_baseline import (
    GAME_RULES_IDENTITY,
    OUTPUT_RETRY_MESSAGE,
    SYSTEM_PROMPT,
    DirectCliInput,
    JsonlLogger,
    ModelReply,
    ModelProtocolError,
    OpenAICompatTerminalModel,
    ProviderConfigurationError,
    ProviderTransportError,
    TerminalRecorder,
    build_parser,
    extract_chat_text,
    load_human_cli,
    run_direct_cli,
    summarize_terminal_run,
    terminal_command_from_response,
    terminal_output_format_valid,
)
from sts2_bench.game_rules import GAME_RULE_MODULES


class FakeModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return ModelReply(
            text=self.replies.pop(0),
            usage={"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
            latency_sec=0.25,
        )


def test_direct_system_prompt_exposes_generic_card_type_timing_once():
    timing_rule = '"At the start...", "At the end...", and "Whenever..." clauses are delayed triggers.'

    assert SYSTEM_PROMPT.count(timing_rule) == 1
    assert "Inferno" not in SYSTEM_PROMPT


def test_mini_real_cli_input_contract_round_trip_without_model_api(tmp_path, monkeypatch):
    cli_root = Path(__file__).resolve().parents[2] / "sts2-cli"
    play = load_human_cli(cli_root)
    play.LANG = "en"
    recorder = TerminalRecorder(tmp_path / "terminal.txt", history_chars=24000)
    events = JsonlLogger(tmp_path / "events.jsonl")
    model = FakeModel(["0 1@0", "seq 0 1@0"])
    direct_input = DirectCliInput(
        model=model,
        recorder=recorder,
        events=events,
        max_decisions=3,
    )
    recorder.write(
        "Energy: 3\n"
        "Hand:\n"
        "  [0] Defend (1)\n"
        "  [1] Strike (1) -> target enemy\n"
        "Enemies:\n"
        "  [0] Test Enemy\n"
    )
    monkeypatch.setattr("builtins.input", direct_input)

    with redirect_stdout(recorder):
        choice = play.get_input(
            play.combat_input_prompt(
                {"decision": "combat_play", "player": {"potions": []}}
            ),
            {"0", "1", "e", "help"},
            state={"decision": "combat_play"},
            additional_accepted_syntax=("combat_card_or_sequence",),
        )

    events.close()
    recorder.close()

    assert choice == "seq 0 1@0"
    assert len(model.messages) == 2
    assert "seq CARD_INDEX CARD_INDEX@ENEMY_INDEX ..." in model.messages[0][0]["content"]
    assert "not a list of actions that are legal in the current state" in model.messages[0][0]["content"]
    assert "canonical_option_tokens" not in json.dumps(model.messages, ensure_ascii=False)
    second_window = model.messages[1][1]["content"]
    assert "Invalid input. Follow the current prompt; type help for input syntax." in second_window
    assert "Invalid. Options:" not in second_window
    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decisions = [row for row in rows if row["type"] == "terminal_decision"]
    assert [row["command"] for row in decisions] == ["0 1@0", "seq 0 1@0"]
    assert all(
        row["prompt_contract"]["canonical_option_tokens"] == ["0", "1", "e", "help"]
        for row in decisions
    )


def test_terminal_command_only_converts_documented_enter_marker():
    assert terminal_command_from_response("  0@1\nI attack the enemy") == "0@1\nI attack the enemy"
    assert terminal_command_from_response("\n\n") == ""
    assert terminal_command_from_response("```0```") == "```0```"
    assert terminal_command_from_response("<ENTER>") == ""
    assert terminal_output_format_valid("0@1") is True
    assert terminal_output_format_valid("<ENTER>") is True
    assert terminal_output_format_valid("0@1\nI attack") is False
    assert terminal_output_format_valid("") is False


def test_direct_cli_input_records_human_cli_prompt_contract(tmp_path):
    recorder = TerminalRecorder(tmp_path / "terminal.txt", history_chars=1000)
    events = JsonlLogger(tmp_path / "events.jsonl")
    direct_input = DirectCliInput(
        model=FakeModel(["c2"]),
        recorder=recorder,
        events=events,
        max_decisions=3,
    )
    direct_input.set_prompt_contract(
        decision="shop",
        canonical_option_tokens=["c2", "leave"],
        accepted_syntax=["canonical_token"],
        meta_commands=["deck", "help", "saves"],
        multi_select=False,
        multi_min=1,
        multi_max=1,
    )

    assert direct_input("Buy [c2/leave]: ") == "c2"
    events.close()
    recorder.close()

    row = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["type"] == "terminal_decision"
    assert row["prompt_contract"] == {
        "decision": "shop",
        "canonical_option_tokens": ["c2", "leave"],
        "accepted_syntax": ["canonical_token"],
        "meta_commands": ["deck", "help"],
        "forbidden_meta_commands": ["save", "saves"],
        "multi_select": False,
        "multi_min": 1,
        "multi_max": 1,
    }


def test_extract_chat_text_accepts_string_and_text_blocks():
    assert extract_chat_text({"choices": [{"message": {"content": "deck"}}]}) == "deck"
    assert extract_chat_text(
        {"choices": [{"message": {"content": [{"type": "text", "text": "m"}, {"type": "text", "text": "ap"}]}}]}
    ) == "map"


def test_openai_compat_terminal_model_sends_plain_chat_request(monkeypatch):
    captured = {}
    observed = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {
                    "id": "req-123",
                    "model": "resolved-model",
                    "system_fingerprint": "fp-1",
                    "choices": [{"message": {"content": "e"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        temperature=0.0,
        timeout=15.0,
        max_tokens=32,
        request_retries=0,
        attempt_observer=observed.append,
    )
    reply = model.complete([{"role": "user", "content": "screen"}])

    assert reply.text == "e"
    assert captured["headers"]["user-agent"] == "Mozilla/5.0 (compatible; sts2-agent-bench/1.0)"
    assert reply.usage["total_tokens"] == 10
    assert reply.provider_metadata == {
        "model": "resolved-model",
        "id": "req-123",
        "system_fingerprint": "fp-1",
        "finish_reason": "stop",
    }
    assert captured["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert captured["payload"] == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "screen"}],
        "temperature": 0.0,
        "max_tokens": 32,
    }
    assert captured["timeout"] == 15.0
    assert observed[0]["status"] == "succeeded"
    assert observed[0]["provider_metadata"]["finish_reason"] == "stop"
    assert len(observed[0]["provider_response_sha256"]) == 64
    assert json.loads(observed[0]["provider_response_body"])["choices"][0]["message"]["content"] == "e"


def test_openai_compat_terminal_model_rejects_empty_content(monkeypatch):
    calls = 0
    observed = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {
                    "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 64, "completion_tokens_details": {"reasoning_tokens": 64}},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="reasoning-only-response",
        request_retries=0,
        attempt_observer=observed.append,
    )

    with pytest.raises(ModelProtocolError, match="LLM response content is empty") as captured_error:
        model.complete([{"role": "user", "content": "screen"}])
    assert captured_error.value.usage == {
        "completion_tokens": 64,
        "completion_tokens_details": {"reasoning_tokens": 64},
    }
    assert captured_error.value.latency_sec >= 0
    assert calls == 1
    assert observed[0]["failure_class"] == "model_protocol_error"


def test_openai_compat_terminal_model_rejects_non_object_json_with_audited_protocol_error(monkeypatch):
    observed = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return b"[]"

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="non-object-response",
        request_retries=0,
        attempt_observer=observed.append,
    )

    with pytest.raises(ModelProtocolError, match="not a JSON object"):
        model.complete([{"role": "user", "content": "screen"}])

    assert len(observed) == 1
    assert observed[0]["failure_class"] == "model_protocol_error"
    assert observed[0]["provider_response_body"] == "[]"
    assert observed[0]["provider_response_sha256"] == hashlib.sha256(b"[]").hexdigest()
    assert observed[0]["provider_response_body"]
    assert len(observed[0]["provider_response_sha256"]) == 64


def test_direct_cli_input_accounts_for_protocol_error_usage_and_latency(tmp_path):
    class ProtocolFailureModel:
        @staticmethod
        def complete(_messages):
            raise ModelProtocolError(
                "truncated",
                usage={"prompt_tokens": 20, "completion_tokens": 64, "total_tokens": 84},
                latency_sec=0.75,
                provider_metadata={"finish_reason": "length"},
            )

    recorder = TerminalRecorder(tmp_path / "terminal.txt", history_chars=100)
    events = JsonlLogger(tmp_path / "events.jsonl")
    direct_input = DirectCliInput(
        model=ProtocolFailureModel(),
        recorder=recorder,
        events=events,
        max_decisions=3,
    )

    with pytest.raises(ModelProtocolError, match="truncated"):
        direct_input("Choose: ")
    events.close()
    recorder.close()

    assert direct_input.usage_totals == {
        "prompt_tokens": 20,
        "completion_tokens": 64,
        "total_tokens": 84,
    }
    assert direct_input.model_latency_sec == 0.75
    row = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["type"] == "model_protocol_error"
    assert row["usage"]["total_tokens"] == 84
    assert row["latency_sec"] == 0.75
    assert row["provider_attempt_count"] == 1
    assert row["provider_metadata"]["finish_reason"] == "length"


def test_openai_compat_terminal_model_rejects_length_truncated_content(monkeypatch):
    observed = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {"content": "3 " * 64},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"completion_tokens": 64},
                }
            ).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="truncated-response",
        request_retries=0,
        attempt_observer=observed.append,
    )

    with pytest.raises(ModelProtocolError, match="finish_reason=length"):
        model.complete([{"role": "user", "content": "screen"}])
    assert observed[0]["failure_class"] == "provider_output_truncated"
    assert observed[0]["provider_metadata"]["finish_reason"] == "length"


def test_openai_compat_terminal_model_omits_provider_default_max_tokens(monkeypatch):
    captured_payloads = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {
                    "choices": [{"message": {"content": "e"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured_payloads.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="provider-default-output-budget",
        max_tokens=None,
        request_retries=0,
    )

    reply = model.complete([{"role": "user", "content": "screen"}])

    assert reply.provider_attempt_count == 1
    assert captured_payloads == [
        {
            "model": "provider-default-output-budget",
            "messages": [{"role": "user", "content": "screen"}],
            "temperature": 0.0,
        }
    ]
    assert "max_tokens" not in captured_payloads[0]


def test_openai_compat_terminal_model_retries_length_with_correction_and_accumulates_usage(monkeypatch):
    requests = []
    observed = []
    bodies = [
        {
            "id": "truncated-1",
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4097,
                "total_tokens": 4107,
                "completion_tokens_details": {"reasoning_tokens": 4097},
            },
        },
        {
            "id": "completed-2",
            "choices": [{"message": {"content": "e"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 1,
                "total_tokens": 21,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        },
    ]

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self.body).encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append(request.data)
        return FakeResponse(bodies[len(requests) - 1])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="provider-default-output-budget",
        max_tokens=None,
        request_retries=1,
        attempt_observer=observed.append,
    )
    original_messages = [{"role": "user", "content": "screen"}]

    reply = model.complete(original_messages)

    payloads = [json.loads(value.decode("utf-8")) for value in requests]
    assert reply.text == "e"
    assert reply.provider_attempt_count == 2
    assert reply.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 4098,
        "total_tokens": 4128,
        "completion_tokens_details": {"reasoning_tokens": 4097},
    }
    assert original_messages == [{"role": "user", "content": "screen"}]
    assert payloads[0]["messages"] == original_messages
    assert payloads[1]["messages"] == [
        *original_messages,
        {"role": "user", "content": OUTPUT_RETRY_MESSAGE},
    ]
    assert all("max_tokens" not in payload for payload in payloads)
    assert [item["provider_attempt_index"] for item in observed] == [0, 1]
    assert [item["status"] for item in observed] == ["failed", "succeeded"]
    assert [item["failure_class"] for item in observed] == ["provider_output_truncated", None]
    assert len({item["model_messages_sha256"] for item in observed}) == 2
    assert len({item["provider_request_sha256"] for item in observed}) == 2
    for request_bytes, attempt in zip(requests, observed, strict=True):
        payload = json.loads(request_bytes.decode("utf-8"))
        expected_messages_hash = hashlib.sha256(
            json.dumps(
                payload["messages"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        assert attempt["provider_request_sha256"] == hashlib.sha256(request_bytes).hexdigest()
        assert attempt["model_messages_sha256"] == expected_messages_hash
    assert [json.loads(item["provider_response_body"])["usage"] for item in observed] == [
        bodies[0]["usage"],
        bodies[1]["usage"],
    ]


def test_direct_cli_input_excludes_transport_retry_backoff_from_provider_latency(monkeypatch, tmp_path):
    clock = [0.0]
    calls = 0
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {
                    "choices": [{"message": {"content": "e"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            clock[0] += 0.25
            raise ConnectionResetError("retry me")
        clock[0] += 0.5
        return FakeResponse()

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.sleep", fake_sleep)
    recorder = TerminalRecorder(tmp_path / "terminal.txt", history_chars=100)
    events = JsonlLogger(tmp_path / "events.jsonl")
    direct_input = DirectCliInput(
        model=OpenAICompatTerminalModel(
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            request_retries=1,
        ),
        recorder=recorder,
        events=events,
        max_decisions=3,
    )

    assert direct_input("Choose: ") == "e"
    events.close()
    recorder.close()

    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    attempts = [row for row in rows if row["type"] == "provider_request_attempt"]
    decision = next(row for row in rows if row["type"] == "terminal_decision")
    assert sleeps == [1.0]
    assert [row["latency_sec"] for row in attempts] == pytest.approx([0.25, 0.5])
    assert decision["latency_sec"] == pytest.approx(1.75)
    assert decision["provider_latency_sec"] == pytest.approx(0.75)
    assert direct_input.model_latency_sec == pytest.approx(0.75)


def test_direct_cli_input_preserves_nested_usage_across_provider_attempts(monkeypatch, tmp_path):
    bodies = [
        {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
        },
        {
            "choices": [{"message": {"content": "e"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 2,
                "total_tokens": 22,
                "prompt_tokens_details": {"cached_tokens": 7},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
    ]
    calls = 0

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self.body).encode("utf-8")

    def fake_urlopen(request, timeout):
        nonlocal calls
        response = FakeResponse(bodies[calls])
        calls += 1
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    recorder = TerminalRecorder(tmp_path / "terminal.txt", history_chars=100)
    events = JsonlLogger(tmp_path / "events.jsonl")
    direct_input = DirectCliInput(
        model=OpenAICompatTerminalModel(
            base_url="http://127.0.0.1:8000/v1",
            model="test-model",
            request_retries=1,
        ),
        recorder=recorder,
        events=events,
        max_decisions=3,
    )

    assert direct_input("Choose: ") == "e"
    events.close()
    recorder.close()

    expected_usage = {
        "prompt_tokens": 30,
        "completion_tokens": 6,
        "total_tokens": 36,
        "prompt_tokens_details": {"cached_tokens": 10},
        "completion_tokens_details": {"reasoning_tokens": 6},
    }
    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decision = next(row for row in rows if row["type"] == "terminal_decision")
    assert direct_input.usage_totals == expected_usage
    assert decision["usage"] == expected_usage


def test_openai_compat_terminal_model_repeated_length_exhausts_as_protocol_error(monkeypatch):
    requests = []
    observed = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {
                    "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                    "usage": {"completion_tokens": 4097, "total_tokens": 4107},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="provider-default-output-budget",
        max_tokens=None,
        request_retries=1,
        attempt_observer=observed.append,
    )

    with pytest.raises(ModelProtocolError, match="finish_reason=length") as captured_error:
        model.complete([{"role": "user", "content": "screen"}])

    assert captured_error.value.usage == {"completion_tokens": 8194, "total_tokens": 8214}
    assert captured_error.value.provider_attempt_count == 2
    assert len(requests) == 2
    assert requests[1]["messages"][-1] == {"role": "user", "content": OUTPUT_RETRY_MESSAGE}
    assert all("max_tokens" not in payload for payload in requests)
    assert [item["provider_attempt_index"] for item in observed] == [0, 1]
    assert all(item["status"] == "failed" for item in observed)
    assert all(item["failure_class"] == "provider_output_truncated" for item in observed)
    assert len({item["model_messages_sha256"] for item in observed}) == 2
    assert len({item["provider_request_sha256"] for item in observed}) == 2


@pytest.mark.parametrize("finish_reason", [None, "content_filter", "tool_calls"])
def test_openai_compat_terminal_model_requires_normal_stop(monkeypatch, finish_reason):
    choice = {"message": {"content": "0"}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps({"choices": [choice]}).encode("utf-8")

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="non-stop-response",
        request_retries=0,
    )

    expected = "missing" if finish_reason is None else finish_reason
    with pytest.raises(ModelProtocolError, match=f"finish_reason={expected}"):
        model.complete([{"role": "user", "content": "screen"}])


def test_openai_compat_terminal_model_retries_dropped_connection(monkeypatch):
    attempts = []
    observed = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {"choices": [{"message": {"content": "e"}, "finish_reason": "stop"}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        attempts.append((request, timeout))
        if len(attempts) == 1:
            raise ConnectionResetError("remote end closed connection")
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.sleep", lambda _seconds: None)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=1,
        attempt_observer=observed.append,
    )

    reply = model.complete([{"role": "user", "content": "screen"}])
    assert reply.text == "e"
    assert reply.provider_attempt_count == 2
    assert len(attempts) == 2
    assert [item["provider_attempt_index"] for item in observed] == [0, 1]
    assert [item["status"] for item in observed] == ["failed", "succeeded"]
    assert [item["failure_class"] for item in observed] == ["provider_transport_error", None]
    assert len({item["model_messages_sha256"] for item in observed}) == 1


def test_openai_compat_terminal_model_retries_incomplete_read_with_exact_partial_evidence(monkeypatch):
    requests = []
    observed = []
    partial_usage = {"prompt_tokens": 7, "total_tokens": 7}
    partial_body = json.dumps({"usage": partial_usage}, separators=(",", ":")).encode("utf-8")
    success_usage = {"completion_tokens": 1, "total_tokens": 10}
    success_body = json.dumps(
        {
            "choices": [{"message": {"content": "e"}, "finish_reason": "stop"}],
            "usage": success_usage,
        },
        separators=(",", ":"),
    ).encode("utf-8")

    class IncompleteResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            raise http.client.IncompleteRead(partial_body, 19)

    class SuccessResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return success_body

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        return IncompleteResponse() if len(requests) == 1 else SuccessResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.sleep", lambda _seconds: None)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=1,
        attempt_observer=observed.append,
    )

    reply = model.complete([{"role": "user", "content": "screen"}])

    assert reply.text == "e"
    assert reply.provider_attempt_count == 2
    assert reply.usage == {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 17}
    assert requests[0]["messages"] == requests[1]["messages"]
    assert [item["status"] for item in observed] == ["failed", "succeeded"]
    assert [item["failure_class"] for item in observed] == ["provider_transport_error", None]
    partial = observed[0]
    assert partial["partial_response"] is True
    assert partial["partial_response_received_bytes"] == len(partial_body)
    assert partial["partial_response_expected_remaining_bytes"] == 19
    assert partial["provider_response_body"] == partial_body.decode("utf-8")
    assert partial["provider_response_sha256"] == hashlib.sha256(partial_body).hexdigest()
    assert len({item["model_messages_sha256"] for item in observed}) == 1


def test_openai_compat_terminal_model_exhausted_incomplete_reads_preserve_evidence_usage_and_latency(
    monkeypatch,
):
    observed = []
    requests = []
    partial_bodies = [
        json.dumps(
            {"usage": {"prompt_tokens": 3, "total_tokens": 3}},
            separators=(",", ":"),
        ).encode("utf-8"),
        json.dumps(
            {"usage": {"completion_tokens": 5, "total_tokens": 5}},
            separators=(",", ":"),
        ).encode("utf-8"),
    ]
    expected_remaining = [11, 13]

    class IncompleteResponse:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            raise http.client.IncompleteRead(
                partial_bodies[self.index],
                expected_remaining[self.index],
            )

    def fake_urlopen(request, timeout):
        index = len(requests)
        requests.append(json.loads(request.data.decode("utf-8")))
        return IncompleteResponse(index)

    ticks = iter([0.0, 0.1, 0.4, 0.5, 0.9, 1.0])
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "sts2_bench.direct_cli_baseline.time.perf_counter",
        lambda: next(ticks),
    )
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=1,
        attempt_observer=observed.append,
    )

    with pytest.raises(ProviderTransportError, match="failed after 2 attempt") as captured:
        model.complete([{"role": "user", "content": "screen"}])

    assert captured.value.provider_attempt_count == 2
    assert captured.value.usage == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
    }
    assert captured.value.latency_sec == pytest.approx(1.0)
    assert requests[0]["messages"] == requests[1]["messages"]
    assert [item["latency_sec"] for item in observed] == pytest.approx([0.3, 0.4])
    assert [item["failure_class"] for item in observed] == [
        "provider_transport_error",
        "provider_transport_error",
    ]
    for index, item in enumerate(observed):
        partial_body = partial_bodies[index]
        assert item["partial_response"] is True
        assert item["partial_response_received_bytes"] == len(partial_body)
        assert item["partial_response_expected_remaining_bytes"] == expected_remaining[index]
        assert item["provider_response_body"] == partial_body.decode("utf-8")
        assert item["provider_response_sha256"] == hashlib.sha256(partial_body).hexdigest()


def test_openai_compat_terminal_model_retries_general_http_exception_as_no_response_transport(
    monkeypatch,
):
    requests = []
    observed = []
    success_body = b'{"choices":[{"message":{"content":"e"},"finish_reason":"stop"}]}'

    class HttpExceptionResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            raise http.client.HTTPException("response framing failed")

    class SuccessResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return success_body

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        return HttpExceptionResponse() if len(requests) == 1 else SuccessResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.sleep", lambda _seconds: None)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=1,
        attempt_observer=observed.append,
    )

    reply = model.complete([{"role": "user", "content": "screen"}])

    assert reply.text == "e"
    assert reply.provider_attempt_count == 2
    assert requests[0]["messages"] == requests[1]["messages"]
    assert [item["failure_class"] for item in observed] == ["provider_transport_error", None]
    assert "provider_response_body" not in observed[0]
    assert "provider_response_sha256" not in observed[0]
    assert "partial_response" not in observed[0]
    assert "http_status" not in observed[0]


def test_openai_compat_terminal_model_preserves_complete_http_error_response(monkeypatch):
    observed = []
    attempts = []
    error_body = json.dumps({"error": {"message": "rate limited", "details": "x" * 800}}).encode("utf-8")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(
                {"choices": [{"message": {"content": "e"}, "finish_reason": "stop"}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        attempts.append((request, timeout))
        if len(attempts) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                hdrs=None,
                fp=io.BytesIO(error_body),
            )
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.sleep", lambda _seconds: None)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=1,
        attempt_observer=observed.append,
    )

    reply = model.complete([{"role": "user", "content": "screen"}])

    assert reply.text == "e"
    assert observed[0]["failure_class"] == "provider_transport_error"
    assert observed[0]["http_status"] == 429
    assert observed[0]["provider_response_body"] == error_body.decode("utf-8")
    assert observed[0]["provider_response_sha256"] == hashlib.sha256(error_body).hexdigest()


def test_openai_compat_terminal_model_retries_auth_unavailable_without_changing_messages(monkeypatch):
    requests = []
    observed = []
    error_body = b'{"error":{"message":"auth_unavailable"}}'

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return b'{"choices":[{"message":{"content":"e"},"finish_reason":"stop"}]}'

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                hdrs=None,
                fp=io.BytesIO(error_body),
            )
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.sleep", lambda _seconds: None)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=1,
        attempt_observer=observed.append,
    )

    assert model.complete([{"role": "user", "content": "screen"}]).text == "e"
    assert requests[0]["messages"] == requests[1]["messages"]
    assert [item["failure_class"] for item in observed] == ["provider_transport_error", None]


@pytest.mark.parametrize("http_status", [400, 401, 403, 404])
def test_openai_compat_terminal_model_fails_closed_on_nonretryable_http_4xx(monkeypatch, http_status):
    requests = []
    observed = []
    error_body = json.dumps({"error": {"message": "bad request"}}).encode("utf-8")

    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data.decode("utf-8")))
        raise urllib.error.HTTPError(
            request.full_url,
            http_status,
            "Bad Request",
            hdrs=None,
            fp=io.BytesIO(error_body),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=1,
        attempt_observer=observed.append,
    )

    with pytest.raises(ProviderConfigurationError) as captured:
        model.complete([{"role": "user", "content": "screen"}])

    assert captured.value.provider_attempt_count == 1
    assert len(requests) == 1
    assert [item["failure_class"] for item in observed] == ["provider_configuration_error"]
    assert observed[0]["http_status"] == http_status


def test_openai_compat_terminal_model_preserves_invalid_utf8_response_bytes(monkeypatch):
    observed = []
    response_bytes = b"\xff"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return response_bytes

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: FakeResponse())
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=0,
        attempt_observer=observed.append,
    )

    with pytest.raises(ModelProtocolError):
        model.complete([{"role": "user", "content": "screen"}])

    assert observed[0]["provider_response_sha256"] == hashlib.sha256(response_bytes).hexdigest()
    assert observed[0]["provider_response_encoding"] == "invalid_utf8_with_base64"
    assert base64.b64decode(observed[0]["provider_response_body_base64"], validate=True) == response_bytes


def test_protocol_then_transport_exhaustion_preserves_billed_usage_and_latency(monkeypatch):
    calls = 0
    protocol_body = {
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"completion_tokens": 17, "total_tokens": 31},
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @staticmethod
        def read():
            return json.dumps(protocol_body).encode("utf-8")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse()
        raise ConnectionResetError("dropped after billed response")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.sleep", lambda _seconds: None)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=1,
    )

    with pytest.raises(ProviderTransportError) as captured:
        model.complete([{"role": "user", "content": "screen"}])

    assert captured.value.usage == protocol_body["usage"]
    assert captured.value.latency_sec >= 0
    assert captured.value.provider_attempt_count == 2


def test_direct_cli_input_accounts_for_usage_before_transport_exhaustion(tmp_path):
    class MixedFailureModel:
        def complete(self, messages):
            raise ProviderTransportError(
                "transport exhausted",
                usage={"completion_tokens": 17, "total_tokens": 31},
                latency_sec=0.5,
                provider_attempt_count=2,
            )

    recorder = TerminalRecorder(tmp_path / "terminal.txt", history_chars=100)
    events = JsonlLogger(tmp_path / "events.jsonl")
    direct_input = DirectCliInput(
        model=MixedFailureModel(),
        recorder=recorder,
        events=events,
        max_decisions=3,
    )

    with pytest.raises(ProviderTransportError):
        direct_input("Choose: ")
    events.close()
    recorder.close()

    assert direct_input.usage_totals == {"completion_tokens": 17, "total_tokens": 31}
    assert direct_input.model_latency_sec == 0.5


def test_openai_compat_terminal_model_exhausted_transport_is_retryable_class(monkeypatch):
    observed = []
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(ConnectionResetError("dropped")),
    )
    monkeypatch.setattr("sts2_bench.direct_cli_baseline.time.sleep", lambda _seconds: None)
    model = OpenAICompatTerminalModel(
        base_url="http://127.0.0.1:8000/v1",
        model="test-model",
        request_retries=2,
        attempt_observer=observed.append,
    )

    with pytest.raises(ProviderTransportError, match="failed after 3 attempt"):
        model.complete([{"role": "user", "content": "screen"}])
    assert len(observed) == 3
    assert all(item["failure_class"] == "provider_transport_error" for item in observed)


def test_direct_cli_api_key_prefers_qwen_then_deepseek_before_openai(monkeypatch):
    monkeypatch.delenv("STS2_BENCH_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_API_KEY", "qwen-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert build_parser().parse_args(["--omit-max-tokens"]).api_key == "qwen-key"


@pytest.mark.parametrize(
    "argv",
    [[], ["--max-tokens", "4096", "--omit-max-tokens"]],
)
def test_direct_cli_output_budget_control_is_explicit_and_exclusive(argv):
    with pytest.raises(SystemExit) as captured:
        build_parser().parse_args(argv)
    assert captured.value.code == 2


def test_direct_cli_input_exposes_only_fifo_terminal_transcript(tmp_path):
    recorder = TerminalRecorder(tmp_path / "terminal.txt", history_chars=28)
    events = JsonlLogger(tmp_path / "events.jsonl")
    model = FakeModel(["deck"])
    direct_input = DirectCliInput(model=model, recorder=recorder, events=events, max_decisions=3)

    recorder.write("OLD-STATE-THAT-FALLS-OUT\nCurrent screen\n")
    assert direct_input("> Choose: ") == "deck"

    visible = model.messages[0][1]["content"]
    assert "Current screen" in visible
    assert "OLD-STATE" not in visible
    assert "legal_actions" not in visible
    assert "Compact state JSON" not in visible
    assert model.messages[0][0]["content"] == SYSTEM_PROMPT
    assert "CARD_INDEX@ENEMY_INDEX" in model.messages[0][0]["content"]
    assert "seq CARD_INDEX CARD_INDEX@ENEMY_INDEX ..." in model.messages[0][0]["content"]
    assert 'At a separate "Target enemy [index]" prompt, output only the enemy index.' in model.messages[0][0]["content"]
    assert "space-separated indices unless the line begins with seq" in model.messages[0][0]["content"]
    assert "output the requested displayed indices separated by commas" in model.messages[0][0]["content"]
    assert "not a list of actions that are legal in the current state" in model.messages[0][0]["content"]
    assert "canonical_option_tokens" not in model.messages[0][0]["content"]
    assert GAME_RULES_IDENTITY["ruleset_id"] in SYSTEM_PROMPT
    assert all(SYSTEM_PROMPT.count(f"[{module_id}]") == 1 for module_id in GAME_RULE_MODULES)
    assert direct_input.usage_totals["total_tokens"] == 12
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert event["game_ruleset_sha256"] == GAME_RULES_IDENTITY["ruleset_sha256"]
    assert event["game_rule_module_ids"] == list(GAME_RULE_MODULES)
    assert event["game_rule_delivery"] == "static_full_system_prompt"
    events.close()
    recorder.close()


@pytest.mark.parametrize("command", ["save", "saves"])
def test_direct_cli_input_blocks_persistent_meta_commands(command, tmp_path):
    recorder = TerminalRecorder(tmp_path / "terminal.txt", history_chars=100)
    events = JsonlLogger(tmp_path / "events.jsonl")
    direct_input = DirectCliInput(
        model=FakeModel([command]),
        recorder=recorder,
        events=events,
        max_decisions=3,
    )

    with pytest.raises(ModelProtocolError, match="meta-command"):
        direct_input("> ")
    assert direct_input.termination_class == "model_protocol_error"
    events.close()
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["type"] == "forbidden_meta_command"


def test_direct_cli_input_classifies_quit_and_confirmed_abandon_as_forfeit(tmp_path):
    recorder = TerminalRecorder(tmp_path / "terminal.txt", history_chars=1000)
    events = JsonlLogger(tmp_path / "events.jsonl")
    quit_input = DirectCliInput(
        model=FakeModel(["quit"]), recorder=recorder, events=events, max_decisions=3
    )
    assert quit_input("> ") == "quit"
    assert quit_input.termination_class == "model_forfeit"
    assert quit_input("Save before quitting? (y/n): ") == "n"
    assert quit_input.decision_count == 1
    events.close()
    recorder.close()

    recorder = TerminalRecorder(tmp_path / "abandon-terminal.txt", history_chars=1000)
    events = JsonlLogger(tmp_path / "abandon-events.jsonl")
    abandon_input = DirectCliInput(
        model=FakeModel(["abandon", "y"]), recorder=recorder, events=events, max_decisions=3
    )
    assert abandon_input("> ") == "abandon"
    assert abandon_input("Confirm abandon? (y/n): ") == "y"
    assert abandon_input.termination_class == "model_forfeit"
    events.close()
    recorder.close()


def test_run_direct_cli_uses_human_input_and_auto_exits_finished_run(tmp_path):
    model = FakeModel(["0"])

    class FakeHumanCli:
        LANG = ""
        setup_called = False

        @classmethod
        def ensure_setup(cls):
            cls.setup_called = True

        @staticmethod
        def play(**kwargs):
            assert kwargs["auto"] is False
            assert kwargs["log"] is True
            print("Slay the Spire 2 — Headless CLI")
            # Deliberately stale prose: paper metrics must come from the
            # authoritative engine trace below, not from terminal wording.
            print("Floor: 6")
            command = input("> Choose card: ")
            print(f"Played {command}")
            print("DEFEAT")
            print("q Quit    n New run")
            assert input("  > ") == "q"
            engine_rows = [
                {"step": 1, "type": "state", "data": {"type": "ready"}},
                {
                    "step": 1,
                    "type": "action",
                    "data": {
                        "cmd": "start_run",
                        "seed": kwargs["seed"],
                        "character": kwargs["character"],
                        "ascension": kwargs["ascension"],
                        "lang": "en",
                    },
                },
                {
                    "step": 2,
                    "type": "state",
                    "data": {
                        "type": "decision",
                        "decision": "game_over",
                        "context": {"act": 1, "floor": 7, "room_type": "Boss"},
                        "victory": False,
                        "player": {
                            "hp": 0,
                            "max_hp": 80,
                            "gold": 50,
                            "deck_size": 1,
                            "deck": [{"id": "CARD.STRIKE_IRONCLAD", "name": "Strike"}],
                            "relics": [],
                            "potions": [],
                        },
                    },
                },
                {"step": 2, "type": "action", "data": {"cmd": "quit"}},
            ]
            Path(kwargs["log_path"]).write_text(
                "".join(json.dumps(row) + "\n" for row in engine_rows),
                encoding="utf-8",
            )
            return False

    summary = run_direct_cli(
        human_cli=FakeHumanCli,
        model=model,
        out_dir=tmp_path,
        seed="fixed_seed",
        character="Ironclad",
        ascension=0,
        lang="en",
        history_chars=1000,
        max_decisions=10,
        echo_terminal=False,
    )

    assert FakeHumanCli.setup_called
    assert summary["outcome"] == "defeat"
    assert summary["completed"] is True
    assert summary["max_visible_floor"] == 7
    assert summary["decisions"] == 1
    assert summary["usage"]["total_tokens"] == 12
    assert summary["analysis_error"] is None
    assert summary["authoritative_terminal_state"]["decision"] == "game_over"
    assert summary["authoritative_terminal_deck"][0]["name"] == "Strike"
    assert "Played 0" in (tmp_path / "terminal.txt").read_text(encoding="utf-8")
    assert (tmp_path / "engine_trace.jsonl").is_file()
    assert (tmp_path / "benchmark_trace.jsonl").is_file()
    assert (tmp_path / "trace_analysis" / "replay.md").is_file()
    assert (tmp_path / "trace_analysis" / "deck_and_card_rewards_zh.md").is_file()
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["game_rules"] == {
        **GAME_RULES_IDENTITY,
        "delivery": "static_full_system_prompt",
    }
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows].count("terminal_decision") == 1
    decision = next(row for row in rows if row["type"] == "terminal_decision")
    transcript = (tmp_path / "terminal.txt").read_text(encoding="utf-8")
    reconstructed = transcript[decision["terminal_window_start"] : decision["terminal_window_end"]]
    assert reconstructed.endswith("> Choose card: ")
    assert any(row.get("reason") == "finished_run_exit" for row in rows)


def test_natural_terminal_is_retained_when_analysis_needs_offline_rebuild(tmp_path, monkeypatch):
    class FakeHumanCli:
        LANG = ""

        @staticmethod
        def ensure_setup():
            return None

        @staticmethod
        def play(**kwargs):
            print("Floor: 1")
            input("Choose: ")
            print("DEFEAT")
            print("q Quit    n New run")
            assert input("  > ") == "q"
            engine_rows = [
                {
                    "step": 1,
                    "type": "action",
                    "data": {
                        "cmd": "start_run",
                        "seed": kwargs["seed"],
                        "character": kwargs["character"],
                        "ascension": kwargs["ascension"],
                        "lang": "en",
                    },
                },
                {
                    "step": 2,
                    "type": "state",
                    "data": {
                        "type": "decision",
                        "decision": "game_over",
                        "victory": False,
                        "context": {"act": 1, "floor": 2, "room_type": "Boss"},
                        "player": {
                            "hp": 0,
                            "max_hp": 80,
                            "deck_size": 1,
                            "deck": [{"id": "CARD.STRIKE_IRONCLAD", "name": "Strike"}],
                        },
                    },
                },
                {"step": 2, "type": "action", "data": {"cmd": "quit"}},
            ]
            Path(kwargs["log_path"]).write_text(
                "".join(json.dumps(row) + "\n" for row in engine_rows),
                encoding="utf-8",
            )
            return False

    monkeypatch.setattr(
        "sts2_bench.direct_cli_baseline.analyze_direct_cli_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("report renderer failed")),
    )
    summary = run_direct_cli(
        human_cli=FakeHumanCli,
        model=FakeModel(["0"]),
        out_dir=tmp_path,
        seed="fixed",
        character="Ironclad",
        ascension=0,
        lang="en",
        history_chars=1000,
        max_decisions=10,
        echo_terminal=False,
    )

    assert summary["termination_class"] == "natural_game_over"
    assert summary["formal_behavior_sample"] is True
    assert summary["completed"] is True
    assert summary["analysis_pending"] is True
    assert summary["needs_attention"] is True
    assert "report renderer failed" in summary["analysis_error"]
    assert summary["max_visible_floor"] == 2
    assert summary["authoritative_terminal_state"]["decision"] == "game_over"


def test_load_human_cli_uses_the_real_play_entrypoint():
    cli_root = Path(__file__).resolve().parents[2] / "sts2-cli"
    module = load_human_cli(cli_root)

    assert Path(module.__file__).resolve() == (cli_root / "python" / "play.py").resolve()
    assert "log_path" in module.play.__code__.co_varnames


def test_summary_counts_visible_invalid_inputs_and_incomplete_limit():
    summary = summarize_terminal_run(
        "Floor: 3\nInvalid input. Follow the current prompt; type help for input syntax.\n"
        "输入无效。请按当前提示输入；可输入 help 查看语法。\nFloor: 4\n",
        decisions=5,
        format_violations=2,
        usage={},
        model_latency_sec=1.0,
        elapsed_sec=2.0,
        decision_limit_reached=True,
        error=None,
    )
    assert summary["outcome"] == "incomplete"
    assert summary["max_visible_floor"] == 4
    assert summary["invalid_cli_inputs"] == 2
    assert summary["model_output_format_violations"] == 2
    assert summary["decision_limit_reached"] is True


@pytest.mark.parametrize(
    ("termination_class", "error"),
    [
        ("natural_game_over", None),
        ("model_forfeit", None),
        ("decision_budget_exhausted", "EOFError: direct CLI decision limit reached"),
        ("model_protocol_error", "ModelProtocolError: response has no choices"),
    ],
)
def test_summary_counts_attributable_agent_terminations_as_paper_behavior_samples(
    termination_class,
    error,
):
    transcript = "DEFEAT\n" if termination_class == "natural_game_over" else "Floor: 3\nChoose: "
    summary = summarize_terminal_run(
        transcript,
        decisions=5,
        format_violations=0,
        usage={},
        model_latency_sec=1.0,
        elapsed_sec=2.0,
        decision_limit_reached=termination_class == "decision_budget_exhausted",
        error=error,
        termination_class=termination_class,
    )

    assert summary["formal_behavior_sample"] is True
    assert summary["termination_class"] == termination_class


def test_run_refuses_to_overwrite_existing_artifacts(tmp_path):
    (tmp_path / "existing.txt").write_text("owned", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_direct_cli(
            human_cli=object(),
            model=FakeModel(["0"]),
            out_dir=tmp_path,
            seed="fixed_seed",
            character="Ironclad",
            ascension=0,
            lang="en",
            history_chars=1000,
            max_decisions=10,
            echo_terminal=False,
        )
