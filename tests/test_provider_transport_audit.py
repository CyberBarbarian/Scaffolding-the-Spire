from __future__ import annotations

import hashlib
import json

import pytest

from sts2_bench.agents import FORMAL_PROVIDER_AUDIT_ENV, OpenAICompatAgent


class _Response:
    def __init__(self, body: bytes, status_code: int = 200) -> None:
        self.content = body
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response

    def post(self, *args, **kwargs):
        return self.response


class _WorkerConnection:
    def __init__(self, result: dict) -> None:
        self.result = result

    def send(self, value: dict) -> None:
        assert value["type"] == "request"

    def poll(self, timeout: float) -> bool:
        return True

    def recv(self) -> dict:
        return self.result


class _WorkerProcess:
    pass


def _length_response() -> tuple[dict, bytes]:
    response = {
        "id": "truncated",
        "model": "test-model",
        "choices": [{"message": {"content": '{"action_id":0}'}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    }
    return response, json.dumps(response, separators=(",", ":")).encode("utf-8")


def _exercise_protocol_failure(agent: OpenAICompatAgent, body: bytes) -> dict:
    messages = [{"role": "user", "content": "choose"}]
    audits: list[dict] = []
    agent._begin_model_request(audits, messages)
    response = agent._chat(messages)
    with pytest.raises(ValueError, match="finish_reason=length") as captured:
        agent._complete_model_request(audits, response)
    agent._fail_model_request(audits, captured.value)
    assert audits[0]["provider_response_status"] == "model_protocol_error"
    assert audits[0]["provider_response_body"] == body.decode("utf-8")
    assert audits[0]["provider_response_sha256"] == hashlib.sha256(body).hexdigest()
    assert audits[0]["provider_http_status"] == 200
    assert audits[0]["provider_finish_reason"] == "length"
    assert audits[0]["provider_usage"] == response["usage"]
    assert isinstance(audits[0]["provider_latency_sec"], float)
    assert audits[0]["provider_latency_sec"] >= 0.0
    return audits[0]


def test_requests_session_preserves_raw_body_and_rejects_length(monkeypatch):
    monkeypatch.setenv(FORMAL_PROVIDER_AUDIT_ENV, "1")
    response, body = _length_response()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:8000/v1",
        model="test",
        max_retries=0,
        http_backend="requests_session",
    )
    agent._http_session = _Session(_Response(body))
    audit = _exercise_protocol_failure(agent, body)
    assert json.loads(audit["provider_response_body"]) == response


def test_requests_session_worker_preserves_raw_body_and_rejects_length(monkeypatch):
    monkeypatch.setenv(FORMAL_PROVIDER_AUDIT_ENV, "1")
    response, body = _length_response()
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:8000/v1",
        model="test",
        max_retries=0,
        http_backend="requests_session_worker",
    )
    connection = _WorkerConnection(
        {"status": "ok", "code": 200, "response": response, "response_body": body}
    )
    monkeypatch.setattr(agent, "_ensure_http_worker", lambda: (_WorkerProcess(), connection, 0))
    audit = _exercise_protocol_failure(agent, body)
    assert json.loads(audit["provider_response_body"]) == response


@pytest.mark.parametrize("backend", ["requests_session", "requests_session_worker"])
def test_requests_backends_preserve_complete_http_error_body(monkeypatch, backend):
    monkeypatch.setenv(FORMAL_PROVIDER_AUDIT_ENV, "1")
    marker = "tail-marker-after-500-characters"
    body = json.dumps(
        {"error": {"message": ("x" * 700) + marker}}, separators=(",", ":")
    ).encode("utf-8")
    agent = OpenAICompatAgent(
        base_url="http://127.0.0.1:8000/v1",
        model="test",
        max_retries=0,
        http_backend=backend,
    )
    if backend == "requests_session":
        agent._http_session = _Session(_Response(body, status_code=429))
    else:
        connection = _WorkerConnection(
            {"status": "http_error", "code": 429, "response_body": body}
        )
        monkeypatch.setattr(agent, "_ensure_http_worker", lambda: (_WorkerProcess(), connection, 0))

    messages = [{"role": "user", "content": "choose"}]
    audits: list[dict] = []
    agent._begin_model_request(audits, messages)
    with pytest.raises(RuntimeError, match="LLM HTTP 429") as captured:
        agent._chat(messages)
    agent._fail_model_request(audits, captured.value)

    assert audits[0]["provider_response_status"] == "transport_error"
    assert audits[0]["provider_http_status"] == 429
    assert audits[0]["provider_response_body"].endswith(marker + '"}}')
    assert len(audits[0]["provider_response_body"]) > 500
    assert audits[0]["provider_response_sha256"] == hashlib.sha256(body).hexdigest()
    assert isinstance(audits[0]["provider_latency_sec"], float)
    assert audits[0]["provider_latency_sec"] >= 0.0
