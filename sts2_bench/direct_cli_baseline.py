"""Zero-scaffold LLM baseline over the human-facing ``sts2-cli`` terminal.

The acting model never receives engine JSON, benchmark ``LegalAction`` objects,
or any of the structured benchmark prompts.  It sees the same rendered terminal
transcript as a human player and returns one line for the current ``input()``
prompt.  History management is deliberately limited to raw FIFO truncation.
"""

from __future__ import annotations

import argparse
import base64
import builtins
import hashlib
import http.client
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Protocol, TextIO

from .direct_cli_report import analyze_direct_cli_run, audit_engine_trace
from .game_rules import GAME_RULE_MODULES, game_rules_identity, render_full_game_rules
from .seeds import DEFAULT_TEST_SEED


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STS2_CLI_ROOT = ROOT.parent / "sts2-cli"

_BASE_SYSTEM_PROMPT = """You are playing Slay the Spire 2 directly through a human-facing terminal.
You receive only the raw terminal transcript that a human player could read.
Choose exactly what you would type at the current terminal prompt to maximize the chance of winning the complete run.
Use only commands and choices exposed by the terminal.

Output rules:
1. Output exactly one line containing only the text to enter at the current prompt.
2. Do not include analysis, explanations, Markdown, quotation marks, or prefixes such as "command:".
3. To press Enter without typing anything, output exactly <ENTER>.

Terminal input syntax:
1. Follow the current prompt exactly. For ordinary choices, output one option token shown by the terminal.
2. In combat, output CARD_INDEX to play one card, CARD_INDEX@ENEMY_INDEX to target an enemy, or
   seq CARD_INDEX CARD_INDEX@ENEMY_INDEX ... to queue multiple card plays. Output e to end the turn.
3. At a separate "Target enemy [index]" prompt, output only the enemy index.
4. Never put multiple card plays on one line as space-separated indices unless the line begins with seq.
5. When a prompt explicitly requests multiple cards, output the requested displayed indices separated by commas.
6. Syntax examples are not a list of actions that are legal in the current state. Infer legality from the visible game state.
7. If the input syntax is unclear and the current terminal accepts it, output help."""

GAME_RULES_IDENTITY = game_rules_identity()
SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + "\n\n" + render_full_game_rules()
OUTPUT_RETRY_MESSAGE = (
    "The previous provider response did not yield a complete terminal input. "
    "Retry the same unchanged game state and output exactly one terminal input line with no analysis."
)
FORMAL_FORBIDDEN_META_COMMANDS = ("save", "saves")
PAPER_BEHAVIOR_TERMINATION_CLASSES = frozenset(
    {
        "natural_game_over",
        "model_forfeit",
        "decision_budget_exhausted",
        "model_protocol_error",
    }
)
_INVALID_CLI_INPUT_PATTERN = re.compile(
    r"Invalid input\.|输入无效。|Invalid\. Options:|无效。选项:"
)


def count_invalid_cli_inputs(transcript: str) -> int:
    """Count canonical and historical Human CLI invalid-input feedback."""
    return len(_INVALID_CLI_INPUT_PATTERN.findall(transcript))


def _merge_provider_usage(*values: dict[str, Any]) -> dict[str, Any]:
    """Merge usage objects without discarding billed retry tokens."""

    def merge(left: Any, right: Any) -> Any:
        if isinstance(left, bool) or isinstance(right, bool):
            return right
        if isinstance(left, int) and isinstance(right, int):
            return left + right
        if isinstance(left, dict) and isinstance(right, dict):
            combined = dict(left)
            for key, value in right.items():
                combined[key] = merge(combined[key], value) if key in combined else value
            return combined
        return right if left != right else left

    result: dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            result = merge(result, value)
    return result


class TerminalModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> "ModelReply": ...


@dataclass(frozen=True)
class ModelReply:
    text: str
    usage: dict[str, Any]
    latency_sec: float
    provider_attempt_count: int = 1
    provider_metadata: dict[str, Any] | None = None


class ProviderTransportError(RuntimeError):
    """The provider request did not yield a successful HTTP response."""

    def __init__(
        self,
        message: str,
        *,
        usage: dict[str, Any] | None = None,
        latency_sec: float = 0.0,
        provider_attempt_count: int = 1,
    ) -> None:
        super().__init__(message)
        self.usage = dict(usage or {})
        self.latency_sec = float(latency_sec)
        self.provider_attempt_count = int(provider_attempt_count)


class ProviderConfigurationError(RuntimeError):
    """The provider rejected the request for a non-behavioral configuration reason."""

    def __init__(
        self,
        message: str,
        *,
        usage: dict[str, Any] | None = None,
        latency_sec: float = 0.0,
        provider_attempt_count: int = 1,
        provider_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.usage = dict(usage or {})
        self.latency_sec = float(latency_sec)
        self.provider_attempt_count = int(provider_attempt_count)
        self.provider_metadata = dict(provider_metadata or {})


class ModelProtocolError(RuntimeError):
    """The provider answered, but the response could not produce a CLI input."""

    def __init__(
        self,
        message: str,
        *,
        usage: dict[str, Any] | None = None,
        latency_sec: float = 0.0,
        provider_attempt_count: int = 1,
        provider_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.usage = dict(usage or {})
        self.latency_sec = float(latency_sec)
        self.provider_attempt_count = int(provider_attempt_count)
        self.provider_metadata = dict(provider_metadata or {})


def _provider_response_evidence(response_bytes: bytes) -> tuple[str, dict[str, Any]]:
    """Preserve exact response bytes while retaining readable UTF-8 when possible."""

    evidence: dict[str, Any] = {
        "provider_response_sha256": hashlib.sha256(response_bytes).hexdigest(),
    }
    try:
        raw_body = response_bytes.decode("utf-8")
        evidence["provider_response_encoding"] = "utf-8"
    except UnicodeDecodeError:
        raw_body = response_bytes.decode("utf-8", errors="replace")
        evidence.update(
            {
                "provider_response_encoding": "invalid_utf8_with_base64",
                "provider_response_body_base64": base64.b64encode(response_bytes).decode("ascii"),
            }
        )
    evidence["provider_response_body"] = raw_body
    return raw_body, evidence


def _http_failure_is_transport(status: int, raw_body: str) -> bool:
    lowered = raw_body.lower()
    return (
        status in {408, 429}
        or 500 <= status <= 599
        or "auth_unavailable" in lowered
        or "no auth available" in lowered
    )


@dataclass
class OpenAICompatTerminalModel:
    base_url: str
    model: str
    api_key: str = "local"
    temperature: float = 0.0
    timeout: float = 120.0
    max_tokens: int | None = None
    request_retries: int = 1
    attempt_observer: Callable[[dict[str, Any]], None] | None = None

    def complete(self, messages: list[dict[str, str]]) -> ModelReply:
        base = self.base_url.rstrip("/")
        url = base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"
        started = time.perf_counter()
        last_error: BaseException | None = None
        request_messages = [dict(message) for message in messages]
        accumulated_usage: dict[str, Any] = {}
        for attempt in range(self.request_retries + 1):
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": request_messages,
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                payload["max_tokens"] = self.max_tokens
            request_bytes = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            request_sha256 = hashlib.sha256(request_bytes).hexdigest()
            messages_sha256 = hashlib.sha256(
                json.dumps(
                    request_messages,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            request = urllib.request.Request(
                url,
                data=request_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "Mozilla/5.0 (compatible; sts2-agent-bench/1.0)",
                },
                method="POST",
            )
            attempt_started = time.perf_counter()
            body: dict[str, Any] = {}
            transport_failure_evidence: dict[str, Any] = {}
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_bytes = response.read()
                raw_body, response_evidence = _provider_response_evidence(response_bytes)
                try:
                    body = json.loads(raw_body)
                    if not isinstance(body, dict):
                        raise ValueError("LLM response body is not a JSON object")
                    finish_reason = _provider_finish_reason(body)
                    if finish_reason != "stop":
                        detail = "missing" if finish_reason is None else finish_reason
                        raise ValueError(f"LLM response did not complete normally (finish_reason={detail})")
                    text = extract_chat_text(body)
                    if not text.strip():
                        raise ValueError("LLM response content is empty")
                except (json.JSONDecodeError, UnicodeError, KeyError, TypeError, ValueError, AttributeError) as exc:
                    metadata = _provider_response_metadata(body if isinstance(body, dict) else {})
                    usage = (
                        body.get("usage")
                        if isinstance(body, dict) and isinstance(body.get("usage"), dict)
                        else {}
                    )
                    accumulated_usage = _merge_provider_usage(accumulated_usage, usage)
                    protocol_error = ModelProtocolError(
                        f"{type(exc).__name__}: {exc}",
                        usage=accumulated_usage,
                        latency_sec=time.perf_counter() - started,
                        provider_attempt_count=attempt + 1,
                        provider_metadata=metadata,
                    )
                    failure_class = (
                        "provider_output_truncated"
                        if metadata.get("finish_reason") == "length"
                        else "model_protocol_error"
                    )
                    if self.attempt_observer is not None:
                        self.attempt_observer(
                            {
                                "provider_attempt_index": attempt,
                                "model_messages_sha256": messages_sha256,
                                "provider_request_sha256": request_sha256,
                                "status": "failed",
                                "failure_class": failure_class,
                                "error": str(protocol_error),
                                "latency_sec": time.perf_counter() - attempt_started,
                                "provider_metadata": metadata,
                                **response_evidence,
                            }
                        )
                    if attempt < self.request_retries:
                        request_messages = [
                            *request_messages,
                            {"role": "user", "content": OUTPUT_RETRY_MESSAGE},
                        ]
                        continue
                    raise protocol_error from exc
                usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
                accumulated_usage = _merge_provider_usage(accumulated_usage, usage)
                metadata = _provider_response_metadata(body)
                if self.attempt_observer is not None:
                    self.attempt_observer(
                        {
                            "provider_attempt_index": attempt,
                            "model_messages_sha256": messages_sha256,
                            "provider_request_sha256": request_sha256,
                            "status": "succeeded",
                            "failure_class": None,
                            "error": None,
                            "latency_sec": time.perf_counter() - attempt_started,
                            "provider_metadata": metadata,
                            **response_evidence,
                        }
                    )
                return ModelReply(
                    text=text,
                    usage=dict(accumulated_usage),
                    latency_sec=time.perf_counter() - started,
                    provider_attempt_count=attempt + 1,
                    provider_metadata=metadata,
                )
            except urllib.error.HTTPError as exc:
                error_bytes = exc.read()
                error_body, response_evidence = _provider_response_evidence(error_bytes)
                error_message = f"LLM HTTP {exc.code}: {error_body[:500]}"
                if not _http_failure_is_transport(exc.code, error_body):
                    try:
                        candidate = json.loads(error_body)
                        error_body_json = candidate if isinstance(candidate, dict) else {}
                    except (json.JSONDecodeError, UnicodeError):
                        error_body_json = {}
                    metadata = _provider_response_metadata(error_body_json)
                    usage = (
                        error_body_json.get("usage")
                        if isinstance(error_body_json.get("usage"), dict)
                        else {}
                    )
                    accumulated_usage = _merge_provider_usage(accumulated_usage, usage)
                    configuration_error = ProviderConfigurationError(
                        error_message,
                        usage=accumulated_usage,
                        latency_sec=time.perf_counter() - started,
                        provider_attempt_count=attempt + 1,
                        provider_metadata=metadata,
                    )
                    if self.attempt_observer is not None:
                        self.attempt_observer(
                            {
                                "provider_attempt_index": attempt,
                                "model_messages_sha256": messages_sha256,
                                "provider_request_sha256": request_sha256,
                                "status": "failed",
                                "failure_class": "provider_configuration_error",
                                "error": str(configuration_error),
                                "latency_sec": time.perf_counter() - attempt_started,
                                "http_status": exc.code,
                                "provider_metadata": metadata,
                                **response_evidence,
                            }
                        )
                    raise configuration_error from exc
                last_error = RuntimeError(error_message)
                transport_failure_evidence = {"http_status": exc.code, **response_evidence}
            except http.client.IncompleteRead as exc:
                partial_bytes = bytes(exc.partial or b"")
                partial_body, response_evidence = _provider_response_evidence(partial_bytes)
                try:
                    partial_json = json.loads(partial_body)
                except (json.JSONDecodeError, UnicodeError):
                    partial_json = {}
                partial_usage = (
                    partial_json.get("usage")
                    if isinstance(partial_json, dict) and isinstance(partial_json.get("usage"), dict)
                    else {}
                )
                accumulated_usage = _merge_provider_usage(accumulated_usage, partial_usage)
                last_error = RuntimeError(
                    "LLM response body was incomplete "
                    f"(received={len(partial_bytes)} bytes, expected_remaining={exc.expected})"
                )
                transport_failure_evidence = {
                    "partial_response": True,
                    "partial_response_received_bytes": len(partial_bytes),
                    "partial_response_expected_remaining_bytes": exc.expected,
                    **response_evidence,
                }
            except http.client.HTTPException as exc:
                last_error = exc
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                OSError,
            ) as exc:
                last_error = exc
            if self.attempt_observer is not None:
                self.attempt_observer(
                    {
                        "provider_attempt_index": attempt,
                        "model_messages_sha256": messages_sha256,
                        "provider_request_sha256": request_sha256,
                        "status": "failed",
                        "failure_class": "provider_transport_error",
                        "error": f"{type(last_error).__name__}: {last_error}",
                        "latency_sec": time.perf_counter() - attempt_started,
                        "provider_metadata": {},
                        **transport_failure_evidence,
                    }
                )
            if attempt < self.request_retries:
                time.sleep(min(4.0, 2.0**attempt))
                continue
        assert last_error is not None
        raise ProviderTransportError(
            f"LLM request failed after {self.request_retries + 1} attempt(s): {last_error}",
            usage=accumulated_usage,
            latency_sec=time.perf_counter() - started,
            provider_attempt_count=self.request_retries + 1,
        ) from last_error


def _provider_response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    """Keep only non-secret provider provenance exposed by compatible APIs."""

    keys = ("model", "id", "created", "system_fingerprint", "service_tier")
    metadata = {key: response[key] for key in keys if response.get(key) is not None}
    finish_reason = _provider_finish_reason(response)
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason
    return metadata


def _provider_finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    value = choices[0].get("finish_reason")
    return str(value) if value is not None else None


def extract_chat_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response choice has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "".join(str(part) for part in parts)
    raise ValueError("LLM response message content is not text")


def terminal_command_from_response(text: str) -> str:
    """Convert the explicitly documented Enter marker without repairing output."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if normalized == "<ENTER>":
        return ""
    return normalized


def terminal_output_format_valid(text: str) -> bool:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return normalized == "<ENTER>" or bool(normalized) and "\n" not in normalized


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._handle = path.open("w", encoding="utf-8", newline="\n")

    def write(self, row: dict[str, Any]) -> None:
        self._handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class TerminalRecorder(io.TextIOBase):
    """Tee terminal output while retaining only a raw FIFO model window."""

    def __init__(self, path: Path, *, history_chars: int, echo: TextIO | None = None) -> None:
        if history_chars < 1:
            raise ValueError("history_chars must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.history_chars = history_chars
        self.echo = echo
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._window = ""
        self.total_chars = 0

    @property
    def encoding(self) -> str:  # pragma: no cover - compatibility surface
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        self._handle.write(text)
        self._handle.flush()
        if self.echo is not None:
            self.echo.write(text)
            self.echo.flush()
        self.total_chars += len(text)
        self._window = (self._window + text)[-self.history_chars :]
        return len(text)

    def flush(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
        if self.echo is not None:
            self.echo.flush()

    def window(self) -> str:
        return self._window

    def close(self) -> None:
        if self.closed:
            return
        super().close()
        self._handle.close()


class DirectCliInput:
    """Replacement for ``input`` that delegates only gameplay text to an LLM."""

    def __init__(
        self,
        *,
        model: TerminalModel,
        recorder: TerminalRecorder,
        events: JsonlLogger,
        max_decisions: int,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        if max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        self.model = model
        self.recorder = recorder
        self.events = events
        self.max_decisions = max_decisions
        self.system_prompt = system_prompt
        self.decision_count = 0
        self.stop_requested = False
        self.format_violation_count = 0
        self.usage_totals: dict[str, Any] = {}
        self.model_latency_sec = 0.0
        self.termination_class: str | None = None
        self._abandon_confirmation_pending = False
        self._pending_prompt_contract: dict[str, Any] | None = None

    def set_prompt_contract(
        self,
        *,
        decision: str | None,
        canonical_option_tokens: list[str],
        accepted_syntax: list[str],
        meta_commands: list[str],
        multi_select: bool,
        multi_min: int,
        multi_max: int,
    ) -> None:
        """Receive the exact validation contract from the human CLI before input."""
        allowed_meta_commands = sorted(
            command
            for command in meta_commands
            if command not in FORMAL_FORBIDDEN_META_COMMANDS
        )
        self._pending_prompt_contract = {
            "decision": decision,
            "canonical_option_tokens": list(canonical_option_tokens),
            "accepted_syntax": list(accepted_syntax),
            "meta_commands": allowed_meta_commands,
            "forbidden_meta_commands": list(FORMAL_FORBIDDEN_META_COMMANDS),
            "multi_select": bool(multi_select),
            "multi_min": int(multi_min),
            "multi_max": int(multi_max),
        }

    def __call__(self, prompt: str = "") -> str:
        prompt_contract = self._pending_prompt_contract
        self._pending_prompt_contract = None
        self.recorder.write(prompt)
        visible = self.recorder.window()

        if self._is_finished_run_prompt(visible):
            return self._harness_input("q", "finished_run_exit", prompt)
        if self.termination_class == "model_forfeit" and self._is_quit_save_prompt(prompt):
            return self._harness_input("n", "persistent_save_disabled", prompt)
        if self.stop_requested and self._is_confirmation_prompt(prompt):
            return self._harness_input("n", "decision_limit_shutdown", prompt)
        if self.decision_count >= self.max_decisions:
            self.stop_requested = True
            self.termination_class = "decision_budget_exhausted"
            self.events.write(
                {
                    "type": "harness_stop",
                    "reason": "decision_limit",
                    "decision_count": self.decision_count,
                    "timestamp": _utc_now(),
                }
            )
            raise EOFError("direct CLI decision limit reached")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Raw terminal transcript follows. Older text may have been removed by FIFO truncation.\n\n"
                    + visible
                    + "\n\nReturn only the next terminal input line."
                ),
            },
        ]
        terminal_window_end = self.recorder.total_chars
        terminal_window_start = terminal_window_end - len(visible)
        messages_sha256 = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        previous_observer = None
        provider_attempt_latency_before = self.model_latency_sec
        provider_attempts_observed = False
        if isinstance(self.model, OpenAICompatTerminalModel):
            previous_observer = self.model.attempt_observer

            def record_provider_attempt(value: dict[str, Any]) -> None:
                nonlocal provider_attempts_observed
                latency = value.get("latency_sec")
                if isinstance(latency, (int, float)) and not isinstance(latency, bool) and latency >= 0:
                    self.model_latency_sec += float(latency)
                    provider_attempts_observed = True
                self.events.write(
                    {
                        "type": "provider_request_attempt",
                        "decision_index": self.decision_count,
                        **value,
                        "timestamp": _utc_now(),
                    }
                )

            self.model.attempt_observer = record_provider_attempt
        try:
            reply = self.model.complete(messages)
        except ModelProtocolError as exc:
            self.decision_count += 1
            self.termination_class = "model_protocol_error"
            if not provider_attempts_observed:
                self.model_latency_sec += exc.latency_sec
            _accumulate_usage(self.usage_totals, exc.usage)
            self.events.write(
                {
                    "type": "model_protocol_error",
                    "decision_index": self.decision_count - 1,
                    "prompt": prompt,
                    "prompt_contract": prompt_contract,
                    "history_chars": len(visible),
                    "terminal_window_start": terminal_window_start,
                    "terminal_window_end": terminal_window_end,
                    "terminal_window_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
                    "model_messages_sha256": messages_sha256,
                    "provider_attempt_count": exc.provider_attempt_count,
                    "game_ruleset_id": GAME_RULES_IDENTITY["ruleset_id"],
                    "game_ruleset_sha256": GAME_RULES_IDENTITY["ruleset_sha256"],
                    "game_rule_module_ids": list(GAME_RULE_MODULES),
                    "game_rule_delivery": "static_full_system_prompt",
                    "usage": exc.usage,
                    "latency_sec": exc.latency_sec,
                    "provider_latency_sec": self.model_latency_sec - provider_attempt_latency_before,
                    "provider_metadata": exc.provider_metadata,
                    "error": str(exc),
                    "timestamp": _utc_now(),
                }
            )
            raise
        except ProviderTransportError as exc:
            if not provider_attempts_observed:
                self.model_latency_sec += exc.latency_sec
            _accumulate_usage(self.usage_totals, exc.usage)
            raise
        except ProviderConfigurationError as exc:
            if not provider_attempts_observed:
                self.model_latency_sec += exc.latency_sec
            _accumulate_usage(self.usage_totals, exc.usage)
            raise
        finally:
            if isinstance(self.model, OpenAICompatTerminalModel):
                self.model.attempt_observer = previous_observer
        command = terminal_command_from_response(reply.text)
        format_valid = terminal_output_format_valid(reply.text)
        if not format_valid:
            self.format_violation_count += 1
        self.decision_count += 1
        if not provider_attempts_observed:
            self.model_latency_sec += reply.latency_sec
        _accumulate_usage(self.usage_totals, reply.usage)
        self.events.write(
            {
                "type": "terminal_decision",
                "decision_index": self.decision_count - 1,
                "prompt": prompt,
                "prompt_contract": prompt_contract,
                "history_chars": len(visible),
                "terminal_window_start": terminal_window_start,
                "terminal_window_end": terminal_window_end,
                "terminal_window_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
                "model_messages_sha256": messages_sha256,
                "provider_attempt_count": reply.provider_attempt_count,
                "game_ruleset_id": GAME_RULES_IDENTITY["ruleset_id"],
                "game_ruleset_sha256": GAME_RULES_IDENTITY["ruleset_sha256"],
                "game_rule_module_ids": list(GAME_RULE_MODULES),
                "game_rule_delivery": "static_full_system_prompt",
                "model_raw": reply.text,
                "command": command,
                "output_format_valid": format_valid,
                "usage": reply.usage,
                "latency_sec": reply.latency_sec,
                "provider_latency_sec": self.model_latency_sec - provider_attempt_latency_before,
                "provider_metadata": reply.provider_metadata or {},
                "timestamp": _utc_now(),
            }
        )
        self.recorder.write(command + "\n")
        lowered = command.strip().lower()
        if lowered in {"save", "saves"}:
            self.termination_class = "model_protocol_error"
            self.events.write(
                {
                    "type": "forbidden_meta_command",
                    "decision_index": self.decision_count - 1,
                    "command": command,
                    "reason": "persistent CLI commands are disabled in formal Direct runs",
                    "timestamp": _utc_now(),
                }
            )
            raise ModelProtocolError(f"forbidden formal Direct CLI meta-command: {lowered}")
        if lowered == "quit":
            self.termination_class = "model_forfeit"
        elif lowered == "abandon":
            self._abandon_confirmation_pending = True
        elif self._abandon_confirmation_pending and self._is_confirmation_prompt(prompt):
            if lowered in {"y", "yes", "是"}:
                self.termination_class = "model_forfeit"
            self._abandon_confirmation_pending = False
        return command

    def _harness_input(self, command: str, reason: str, prompt: str) -> str:
        self.events.write(
            {
                "type": "harness_input",
                "reason": reason,
                "prompt": prompt,
                "command": command,
                "timestamp": _utc_now(),
            }
        )
        self.recorder.write(command + "\n")
        return command

    @staticmethod
    def _is_confirmation_prompt(prompt: str) -> bool:
        lowered = prompt.lower()
        return "(y/n)" in lowered or "（是/否）" in prompt

    @staticmethod
    def _is_quit_save_prompt(prompt: str) -> bool:
        lowered = prompt.lower()
        return "save before quitting" in lowered or "退出前是否存档" in prompt

    @staticmethod
    def _is_finished_run_prompt(visible: str) -> bool:
        ended = "VICTORY!" in visible or "DEFEAT" in visible or "胜利!" in visible or "战败" in visible
        choice = ("New run" in visible and "Quit" in visible) or ("开始新一局" in visible and "退出" in visible)
        return ended and choice


def _accumulate_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    merged = _merge_provider_usage(total, usage)
    total.clear()
    total.update(merged)


def load_human_cli(sts2_cli_root: Path) -> ModuleType:
    play_path = sts2_cli_root / "python" / "play.py"
    if not play_path.is_file():
        raise FileNotFoundError(f"Human CLI entrypoint not found: {play_path}")
    module_name = "sts2_direct_human_cli"
    spec = importlib.util.spec_from_file_location(module_name, play_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load human CLI from {play_path}")
    module = importlib.util.module_from_spec(spec)
    python_dir = str(play_path.parent)
    inserted = python_dir not in sys.path
    if inserted:
        sys.path.insert(0, python_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(python_dir)
    return module


def run_direct_cli(
    *,
    human_cli: ModuleType,
    model: TerminalModel,
    out_dir: Path,
    seed: str,
    character: str,
    ascension: int,
    lang: str,
    history_chars: int,
    max_decisions: int,
    echo_terminal: bool,
    source_lock_sha256: str | None = None,
    source_tree_identity_sha256: str | None = None,
) -> dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    events = JsonlLogger(out_dir / "events.jsonl")
    original_stdout = sys.stdout
    recorder = TerminalRecorder(
        out_dir / "terminal.txt",
        history_chars=history_chars,
        echo=original_stdout if echo_terminal else None,
    )
    direct_input = DirectCliInput(
        model=model,
        recorder=recorder,
        events=events,
        max_decisions=max_decisions,
    )
    original_input = builtins.input
    started = time.perf_counter()
    error: str | None = None
    caught_exception: BaseException | None = None
    engine_trace_path = out_dir / "engine_trace.jsonl"
    benchmark_trace_path = out_dir / "benchmark_trace.jsonl"
    analysis_dir = out_dir / "trace_analysis"
    dotnet_value = getattr(human_cli, "DOTNET", None)
    dotnet_path = Path(dotnet_value).resolve() if dotnet_value and Path(dotnet_value).is_file() else None

    config = {
        "protocol": "human_cli_raw_fifo_v1",
        "model": getattr(model, "model", type(model).__name__),
        "base_url": getattr(model, "base_url", None),
        "temperature": getattr(model, "temperature", None),
        "timeout": getattr(model, "timeout", None),
        "max_tokens": getattr(model, "max_tokens", None),
        "request_retries": getattr(model, "request_retries", None),
        "thinking_control": "provider_default_omitted",
        "response_format_control": "provider_default_omitted",
        "controller_path": str(Path(__file__).resolve()),
        "controller_sha256": _file_sha256(Path(__file__).resolve()),
        "human_cli_path": str(Path(human_cli.__file__).resolve()) if getattr(human_cli, "__file__", None) else None,
        "human_cli_sha256": _file_sha256(Path(human_cli.__file__).resolve()) if getattr(human_cli, "__file__", None) else None,
        "dotnet_path": str(dotnet_path) if dotnet_path is not None else None,
        "dotnet_sha256": _file_sha256(dotnet_path) if dotnet_path is not None else None,
        "seed": seed,
        "character": character,
        "ascension": ascension,
        "lang": lang,
        "history_chars": history_chars,
        "max_decisions": max_decisions,
        "python_runtime": _python_runtime_identity(),
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "game_rules": {
            **GAME_RULES_IDENTITY,
            "delivery": "static_full_system_prompt",
        },
        "engine_trace_path": str(engine_trace_path),
        "benchmark_trace_path": str(benchmark_trace_path),
        "trace_analysis_dir": str(analysis_dir),
        "engine_telemetry_visible_to_model": False,
        "source_locked_runtime_mode": bool(
            source_lock_sha256 and source_tree_identity_sha256
        ),
        "source_lock_sha256": source_lock_sha256,
        "source_tree_identity_sha256": source_tree_identity_sha256,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        human_cli.LANG = lang
        human_cli.ensure_setup()
        builtins.input = direct_input
        sys.stdout = recorder
        human_cli.play(
            character=character,
            seed=seed,
            auto=False,
            ascension=ascension,
            log=True,
            log_path=str(engine_trace_path),
        )
    except BaseException as exc:
        caught_exception = exc
        error = f"{type(exc).__name__}: {exc}"
        events.write({"type": "runner_error", "error": error, "timestamp": _utc_now()})
    finally:
        sys.stdout = original_stdout
        builtins.input = original_input
        recorder.flush()
        transcript = (out_dir / "terminal.txt").read_text(encoding="utf-8")
        natural_outcome = _terminal_outcome(transcript)
        if natural_outcome in {"victory", "defeat"} and error is None:
            termination_class = "natural_game_over"
        elif direct_input.termination_class is not None:
            termination_class = direct_input.termination_class
        elif isinstance(caught_exception, ProviderTransportError):
            termination_class = "provider_transport_error"
        elif isinstance(caught_exception, ProviderConfigurationError):
            termination_class = "provider_configuration_error"
        elif isinstance(caught_exception, ModelProtocolError):
            termination_class = "model_protocol_error"
        else:
            termination_class = "process_infrastructure_error"
        summary = summarize_terminal_run(
            transcript,
            decisions=direct_input.decision_count,
            format_violations=direct_input.format_violation_count,
            usage=direct_input.usage_totals,
            model_latency_sec=direct_input.model_latency_sec,
            elapsed_sec=time.perf_counter() - started,
            decision_limit_reached=direct_input.stop_requested,
            error=error,
            termination_class=termination_class,
        )
        analysis_error: str | None = None
        analysis_artifacts: dict[str, Any] | None = None
        if engine_trace_path.is_file():
            try:
                analysis_artifacts = analyze_direct_cli_run(
                    engine_trace_path,
                    benchmark_trace_path,
                    analysis_dir,
                    config=config,
                    cli_summary=summary,
                )
                engine_audit = analysis_artifacts.get("engine_audit") or {}
                benchmark_audit = analysis_artifacts.get("benchmark_audit") or {}
                # The engine trace is the authoritative source for paper
                # progress metrics.  Terminal prose is retained for the raw
                # human-CLI transcript, but its wording is not a stable metric
                # interface and can lag the state that triggered a model stop.
                summary["max_visible_floor"] = engine_audit.get("cumulative_observed_floor")
                summary["max_within_act_visible_floor"] = engine_audit.get("max_observed_floor")
                summary["max_visible_floor_by_act"] = engine_audit.get("max_observed_floor_by_act")
                summary["authoritative_terminal_state"] = engine_audit.get("authoritative_terminal_state")
                summary["authoritative_terminal_state_sha256"] = engine_audit.get(
                    "authoritative_terminal_state_sha256"
                )
                summary["authoritative_terminal_deck"] = engine_audit.get("authoritative_terminal_deck")
                summary["authoritative_terminal_deck_sha256"] = engine_audit.get(
                    "authoritative_terminal_deck_sha256"
                )
                if engine_audit.get("valid") is not True or benchmark_audit.get("valid") is not True:
                    failed = [
                        name
                        for audit_name, audit in (("engine", engine_audit), ("benchmark", benchmark_audit))
                        for name, passed in (audit.get("checks") or {}).items()
                        if not passed
                    ]
                    analysis_error = "trace closure validation failed: " + ", ".join(failed)
            except Exception as exc:
                analysis_error = f"{type(exc).__name__}: {exc}"
                try:
                    engine_audit = audit_engine_trace(engine_trace_path, config=config, cli_summary=summary)
                    summary["max_visible_floor"] = engine_audit.get("cumulative_observed_floor")
                    summary["max_within_act_visible_floor"] = engine_audit.get("max_observed_floor")
                    summary["max_visible_floor_by_act"] = engine_audit.get("max_observed_floor_by_act")
                    summary["authoritative_terminal_state"] = engine_audit.get("authoritative_terminal_state")
                    summary["authoritative_terminal_state_sha256"] = engine_audit.get(
                        "authoritative_terminal_state_sha256"
                    )
                    summary["authoritative_terminal_deck"] = engine_audit.get("authoritative_terminal_deck")
                    summary["authoritative_terminal_deck_sha256"] = engine_audit.get(
                        "authoritative_terminal_deck_sha256"
                    )
                except Exception as audit_exc:
                    analysis_error += f"; engine audit fallback failed: {type(audit_exc).__name__}: {audit_exc}"
        else:
            analysis_error = "engine_trace.jsonl was not created"
        summary["analysis"] = analysis_artifacts
        summary["analysis_error"] = analysis_error
        summary["analysis_pending"] = analysis_error is not None
        summary["needs_attention"] = analysis_error is not None
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        events.write(
            {
                "type": "analysis",
                "artifacts": analysis_artifacts,
                "error": analysis_error,
                "timestamp": _utc_now(),
            }
        )
        events.write({"type": "summary", **summary})
        events.close()
        recorder.close()
    return summary


def summarize_terminal_run(
    transcript: str,
    *,
    decisions: int,
    format_violations: int,
    usage: dict[str, Any],
    model_latency_sec: float,
    elapsed_sec: float,
    decision_limit_reached: bool,
    error: str | None,
    termination_class: str | None = None,
) -> dict[str, Any]:
    natural_outcome = _terminal_outcome(transcript)
    effective_class = termination_class or (
        "natural_game_over" if natural_outcome in {"victory", "defeat"} and error is None else "process_infrastructure_error"
    )
    outcome = {
        "model_forfeit": "forfeit",
        "decision_budget_exhausted": "decision_budget_exhausted",
        "model_protocol_error": "model_protocol_error",
    }.get(effective_class, natural_outcome)
    floors = [int(value) for value in re.findall(r"(?:Floor|层)\s*[:：]?\s*(\d+)", transcript, flags=re.IGNORECASE)]
    invalid_count = count_invalid_cli_inputs(transcript)
    return {
        "outcome": outcome,
        "max_visible_floor": max(floors) if floors else None,
        "decisions": decisions,
        "model_output_format_violations": format_violations,
        "invalid_cli_inputs": invalid_count,
        "usage": usage,
        "model_latency_sec": round(model_latency_sec, 6),
        "elapsed_sec": round(elapsed_sec, 6),
        "decision_limit_reached": decision_limit_reached,
        "error": error,
        "termination_class": effective_class,
        "formal_behavior_sample": effective_class in PAPER_BEHAVIOR_TERMINATION_CLASSES,
        "completed": effective_class == "natural_game_over" and outcome in {"victory", "defeat"} and error is None,
    }


def _terminal_outcome(transcript: str) -> str:
    if "VICTORY!" in transcript or "胜利!" in transcript:
        return "victory"
    if "DEFEAT" in transcript or "战败" in transcript:
        return "defeat"
    return "incomplete"


def _load_env_file(path: Path, *, override: bool = False) -> set[str]:
    if not path.is_file():
        return set()
    loaded: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        normalized_key = key.strip()
        if override:
            os.environ[normalized_key] = value
        else:
            os.environ.setdefault(normalized_key, value)
        loaded.add(normalized_key)
    return loaded


def _default_out_dir(model: str, seed: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"
    safe_seed = re.sub(r"[^A-Za-z0-9_.-]+", "_", seed).strip("_") or "seed"
    return ROOT / "results" / "direct_cli" / f"{stamp}_{safe_model}_{safe_seed}"


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _python_runtime_identity() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    distributions = sorted(
        {
            (
                str(distribution.metadata.get("Name") or "").strip().lower(),
                str(distribution.version),
            )
            for distribution in importlib.metadata.distributions()
            if str(distribution.metadata.get("Name") or "").strip()
        }
    )
    customizations: list[dict[str, Any]] = []
    for name in ("sitecustomize", "usercustomize"):
        module = sys.modules.get(name)
        path_value = getattr(module, "__file__", None) if module is not None else None
        if path_value:
            path = Path(path_value).resolve()
            customizations.append(
                {"module": name, "path": str(path), "sha256": _file_sha256(path)}
            )
    return {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "cache_tag": str(sys.implementation.cache_tag or ""),
        "executable": str(executable),
        "executable_sha256": _file_sha256(executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "installed_distributions": [
            {"name": name, "version": version} for name, version in distributions
        ],
        "site_customizations": customizations,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a zero-scaffold LLM through the human-facing sts2-cli terminal.",
    )
    parser.add_argument("--sts2-cli-root", type=Path, default=DEFAULT_STS2_CLI_ROOT)
    parser.add_argument("--base-url", default=os.environ.get("STS2_BENCH_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--model", default=os.environ.get("STS2_BENCH_MODEL"))
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("STS2_BENCH_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "local"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    output_limit = parser.add_mutually_exclusive_group(required=True)
    output_limit.add_argument("--max-tokens", type=int)
    output_limit.add_argument("--omit-max-tokens", action="store_true")
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--history-chars", type=int, default=24000)
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--seed", default=DEFAULT_TEST_SEED)
    parser.add_argument("--character", default="Ironclad")
    parser.add_argument("--ascension", type=int, default=0)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--no-echo-terminal", action="store_true")
    parser.add_argument("--source-lock-sha256")
    parser.add_argument("--source-tree-identity-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_env_file(ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model is required (or set STS2_BENCH_MODEL)")
    if args.history_chars < 1 or args.max_decisions < 1:
        parser.error("history and decision limits must be positive")
    if not args.omit_max_tokens and args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.request_retries < 0:
        parser.error("--request-retries must be non-negative")
    if not 0 <= args.ascension <= 10:
        parser.error("--ascension must be between 0 and 10")

    out_dir = args.out_dir or _default_out_dir(args.model, args.seed)
    model = OpenAICompatTerminalModel(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        timeout=args.timeout,
        max_tokens=None if args.omit_max_tokens else args.max_tokens,
        request_retries=args.request_retries,
    )
    human_cli = load_human_cli(args.sts2_cli_root.resolve())
    summary = run_direct_cli(
        human_cli=human_cli,
        model=model,
        out_dir=out_dir.resolve(),
        seed=args.seed,
        character=args.character,
        ascension=args.ascension,
        lang=args.lang,
        history_chars=args.history_chars,
        max_decisions=args.max_decisions,
        echo_terminal=not args.no_echo_terminal,
        source_lock_sha256=args.source_lock_sha256,
        source_tree_identity_sha256=args.source_tree_identity_sha256,
    )
    print(json.dumps({"out_dir": str(out_dir.resolve()), "summary": summary}, ensure_ascii=False, indent=2))
    return 0 if summary.get("formal_behavior_sample") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
