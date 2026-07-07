import os
import sys
from pathlib import Path

from sts2_bench import run_benchmark as run_benchmark_module
from sts2_bench.run_benchmark import (
    _default_trace_analysis_dir,
    _load_simple_yaml_mapping,
    _resolve_trace_analysis_dir,
    DEFAULT_SEEDS_FILE,
    load_benchmark_config,
    load_env_file,
    load_seeds,
)
from sts2_bench.runner import RunResult


def test_load_benchmark_config_falls_back_to_simple_scalar_parser(tmp_path, monkeypatch):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "\n".join(
            [
                "# Top-level scalar config only.",
                "agent: random",
                "base_url: \"\"",
                "model: ''",
                "include_json_state: false",
                "allow_custom_sequences: false",
                "memory_window: 8",
                "max_steps: 300",
                "out: results/random_ironclad_a0.jsonl",
                "trace_analysis: true",
            ]
        ),
        encoding="utf-8",
    )
    for key in (
        "STS2_BENCH_AGENT",
        "STS2_BENCH_BASE_URL",
        "STS2_BENCH_MODEL",
        "STS2_BENCH_INCLUDE_JSON_STATE",
        "STS2_BENCH_ALLOW_CUSTOM_SEQUENCES",
        "STS2_BENCH_MEMORY_WINDOW",
        "STS2_BENCH_MAX_STEPS",
        "STS2_BENCH_OUT",
        "STS2_BENCH_TRACE_ANALYSIS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, "omegaconf", None)

    load_benchmark_config(config)

    assert os.environ["STS2_BENCH_BASE_URL"] == ""
    assert os.environ["STS2_BENCH_MODEL"] == ""
    assert os.environ["STS2_BENCH_AGENT"] == "random"
    assert os.environ["STS2_BENCH_INCLUDE_JSON_STATE"] == "false"
    assert os.environ["STS2_BENCH_ALLOW_CUSTOM_SEQUENCES"] == "false"
    assert os.environ["STS2_BENCH_MEMORY_WINDOW"] == "8"
    assert os.environ["STS2_BENCH_MAX_STEPS"] == "300"
    assert os.environ["STS2_BENCH_OUT"] == "results/random_ironclad_a0.jsonl"
    assert os.environ["STS2_BENCH_TRACE_ANALYSIS"] == "true"


def test_load_benchmark_config_falls_back_to_simple_nested_parser(tmp_path, monkeypatch):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        """
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
    for key in (
        "STS2_BENCH_CONTEXT_MODE",
        "STS2_BENCH_TURN_CHAT_WINDOW",
        "STS2_BENCH_TURN_CHAT_UPDATE_MODE",
        "STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY",
        "STS2_BENCH_TURN_CHAT_PLAN_ENABLED",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, "omegaconf", None)

    load_benchmark_config(config)

    assert os.environ["STS2_BENCH_CONTEXT_MODE"] == "turn_chat"
    assert os.environ["STS2_BENCH_TURN_CHAT_WINDOW"] == "3"
    assert os.environ["STS2_BENCH_TURN_CHAT_UPDATE_MODE"] == "delta"
    assert os.environ["STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY"] == "compact"
    assert os.environ["STS2_BENCH_TURN_CHAT_PLAN_ENABLED"] == "true"


def test_tracked_benchmark_config_allows_reverse_proxy_llm_latency(monkeypatch):
    config = Path(__file__).resolve().parents[1] / "sts2_bench" / "benchmark.yaml"
    monkeypatch.delenv("STS2_BENCH_LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("STS2_BENCH_LLM_MAX_RETRIES", raising=False)

    load_benchmark_config(config)

    assert float(os.environ["STS2_BENCH_LLM_TIMEOUT"]) >= 90.0
    assert int(os.environ["STS2_BENCH_LLM_MAX_RETRIES"]) >= 1


def test_tracked_benchmark_config_defaults_to_turn_chat_route_baseline(monkeypatch):
    config = Path(__file__).resolve().parents[1] / "sts2_bench" / "benchmark.yaml"
    monkeypatch.delenv("STS2_BENCH_CONTEXT_MODE", raising=False)
    monkeypatch.delenv("STS2_BENCH_AGENT", raising=False)
    monkeypatch.delenv("STS2_BENCH_ROUTE_CANDIDATE_OBJECTIVE", raising=False)
    monkeypatch.delenv("STS2_BENCH_OCC_CONTEXT_MODE", raising=False)

    load_benchmark_config(config)

    assert os.environ["STS2_BENCH_AGENT"] == "coordinator"
    assert os.environ["STS2_BENCH_CONTEXT_MODE"] == "turn_chat"
    assert os.environ["STS2_BENCH_ROUTE_CANDIDATE_OBJECTIVE"] == "max_restsites_low_risk"
    assert os.environ["STS2_BENCH_OCC_CONTEXT_MODE"] == "reference_turn"


def test_tracked_benchmark_config_enables_trace_analysis(monkeypatch):
    config = Path(__file__).resolve().parents[1] / "sts2_bench" / "benchmark.yaml"
    monkeypatch.delenv("STS2_BENCH_TRACE_ANALYSIS", raising=False)

    load_benchmark_config(config)

    assert os.environ["STS2_BENCH_TRACE_ANALYSIS"] == "true"


def test_tracked_benchmark_config_defaults_to_deepseek_v4_flash(monkeypatch):
    config = Path(__file__).resolve().parents[1] / "sts2_bench" / "benchmark.yaml"
    monkeypatch.delenv("STS2_BENCH_MODEL", raising=False)

    load_benchmark_config(config)

    assert os.environ["STS2_BENCH_MODEL"] == "deepseek-v4-flash"
    assert _load_simple_yaml_mapping(config)["model"] == "deepseek-v4-flash"


def test_tracked_benchmark_config_documents_apihy_model_catalog():
    config = Path(__file__).resolve().parents[1] / "sts2_bench" / "benchmark.yaml"

    data = _load_simple_yaml_mapping(config)
    catalog = data["model_catalog"]
    deepseek = catalog["apihy_deepseek"]
    openai = catalog["apihy_openai"]

    assert deepseek["base_url"] == "https://zgc.apihy.com/v1"
    assert deepseek["default_model"] == "deepseek-v4-pro"
    assert "deepseek-v4-pro" in deepseek["chat_models"].split(", ")
    assert "deepseek-v4-flash" in deepseek["chat_models"].split(", ")
    assert openai["base_url"] == "https://zgc.apihy.com/v1"
    assert openai["default_model"] == "gpt-5.5"
    assert "gpt-5.5" in openai["chat_models"].split(", ")
    assert "gpt-image-2" in openai["non_chat_models"].split(", ")


def test_trace_analysis_dir_defaults_to_artifact_bundle_for_trace_out():
    assert _default_trace_analysis_dir("results/sts2bench_run.jsonl") == Path(
        "artifacts/trace_analysis/sts2bench_run"
    )
    assert _resolve_trace_analysis_dir(
        out="results/sts2bench_run.jsonl",
        trace_analysis_enabled=True,
        trace_analysis_dir=None,
    ) == Path("artifacts/trace_analysis/sts2bench_run")


def test_trace_analysis_dir_respects_explicit_dir_and_disable_switch():
    assert _resolve_trace_analysis_dir(
        out="results/sts2bench_run.jsonl",
        trace_analysis_enabled=True,
        trace_analysis_dir="artifacts/reports/custom",
    ) == Path("artifacts/reports/custom")
    assert (
        _resolve_trace_analysis_dir(
            out="results/sts2bench_run.jsonl",
            trace_analysis_enabled=False,
            trace_analysis_dir="artifacts/reports/custom",
        )
        is None
    )


def test_run_benchmark_main_runs_trace_analysis_by_default(tmp_path, monkeypatch):
    out = tmp_path / "results" / "sts2bench_run.jsonl"
    calls = []

    monkeypatch.setattr(run_benchmark_module, "load_benchmark_env", lambda: None)
    monkeypatch.setattr(run_benchmark_module, "agent_from_args", lambda *args, **kwargs: object())

    def fake_run_one(*, seed, character, ascension, **kwargs):
        return RunResult(
            seed=seed,
            character=character,
            ascension=ascension,
            decision="game_over",
            act=1,
            floor=1,
            room_type="Monster",
        )

    def fake_analyze_benchmark_trace(trace_path, out_dir):
        calls.append((Path(trace_path), Path(out_dir)))
        return {"source": str(trace_path), "out_dir": str(out_dir), "run_count": 1, "runs": []}

    monkeypatch.setattr(run_benchmark_module, "run_one", fake_run_one)
    monkeypatch.setattr(run_benchmark_module, "analyze_benchmark_trace", fake_analyze_benchmark_trace)

    exit_code = run_benchmark_module.main(
        [
            "--agent",
            "random",
            "--context-mode",
            "single_turn",
            "--count",
            "1",
            "--out",
            str(out),
            "--max-steps",
            "1",
        ]
    )

    assert exit_code == 0
    assert calls == [(out, Path("artifacts/trace_analysis/sts2bench_run"))]


def test_load_env_file_handles_utf8_bom(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\ufeffDEEPSEEK_API_KEY=secret\nSTS2_BENCH_MODEL=deepseek-v4-flash\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("STS2_BENCH_MODEL", raising=False)

    load_env_file(env_file)

    assert os.environ["DEEPSEEK_API_KEY"] == "secret"
    assert os.environ["STS2_BENCH_MODEL"] == "deepseek-v4-flash"


def test_load_seeds_handles_utf8_bom(tmp_path):
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text('\ufeff["ironclad_a0_3407"]\n', encoding="utf-8")

    assert load_seeds(str(seed_file), count=1) == ["ironclad_a0_3407"]


def test_load_seeds_defaults_to_preconfigured_fixed_seed_file():
    assert load_seeds(None, count=2) == ["ironclad_a0_3407", "ironclad_a0_1007"]
    assert DEFAULT_SEEDS_FILE.name == "seeds_ironclad_a0.json"


def test_load_seeds_requires_explicit_file_when_count_exceeds_preset_pool():
    try:
        load_seeds(None, count=3)
    except ValueError as exc:
        assert "only 2 preconfigured fixed seeds" in str(exc)
        assert "--seeds" in str(exc)
    else:
        raise AssertionError("expected load_seeds to reject generated default seeds")
