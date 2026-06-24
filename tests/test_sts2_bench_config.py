import os
import sys

from sts2_bench.run_benchmark import load_benchmark_config, load_env_file, load_seeds


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
                "memory_window: 8",
                "max_steps: 300",
                "out: results/random_ironclad_a0.jsonl",
            ]
        ),
        encoding="utf-8",
    )
    for key in (
        "STS2_BENCH_AGENT",
        "STS2_BENCH_BASE_URL",
        "STS2_BENCH_MODEL",
        "STS2_BENCH_INCLUDE_JSON_STATE",
        "STS2_BENCH_MEMORY_WINDOW",
        "STS2_BENCH_MAX_STEPS",
        "STS2_BENCH_OUT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, "omegaconf", None)

    load_benchmark_config(config)

    assert os.environ["STS2_BENCH_BASE_URL"] == ""
    assert os.environ["STS2_BENCH_MODEL"] == ""
    assert os.environ["STS2_BENCH_AGENT"] == "random"
    assert os.environ["STS2_BENCH_INCLUDE_JSON_STATE"] == "false"
    assert os.environ["STS2_BENCH_MEMORY_WINDOW"] == "8"
    assert os.environ["STS2_BENCH_MAX_STEPS"] == "300"
    assert os.environ["STS2_BENCH_OUT"] == "results/random_ironclad_a0.jsonl"


def test_load_benchmark_config_falls_back_to_simple_nested_parser(tmp_path, monkeypatch):
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        """
context_management:
  mode: turn_chat
  route_plan:
    enabled: false
    enforcement: soft
  run_plan:
    route_plan:
      enabled: false
  turn_chat:
    window: 3
    update_mode: delta
    assistant_history: compact
    turn_check:
      enabled: true
  combat:
    card_sequences:
      enabled: false
diagnostics:
  self_consistency:
    enabled: true
    samples: 5
    decisions: map_select,card_reward
    mode: majority_vote
  reflection:
    enabled: true
    decisions: noncombat_non_map
    mode: verifier_vote
    max_options: 6
  action_prompt_order:
    mode: shuffle_map
    seed: 7
""",
        encoding="utf-8",
    )
    for key in (
        "STS2_BENCH_CONTEXT_MODE",
        "STS2_BENCH_ROUTE_PLAN_ENABLED",
        "STS2_BENCH_ROUTE_PLAN_ENFORCEMENT",
        "STS2_BENCH_RUN_PLAN_ROUTE_PLAN_ENABLED",
        "STS2_BENCH_TURN_CHAT_WINDOW",
        "STS2_BENCH_TURN_CHAT_UPDATE_MODE",
        "STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY",
        "STS2_BENCH_TURN_CHECK_ENABLED",
        "STS2_BENCH_TURN_CHAT_PLAN_ENABLED",
        "STS2_BENCH_COMBAT_CARD_SEQUENCES_ENABLED",
        "STS2_BENCH_SELF_CONSISTENCY_ENABLED",
        "STS2_BENCH_SELF_CONSISTENCY_SAMPLES",
        "STS2_BENCH_SELF_CONSISTENCY_DECISIONS",
        "STS2_BENCH_SELF_CONSISTENCY_MODE",
        "STS2_BENCH_REFLECTION_ENABLED",
        "STS2_BENCH_REFLECTION_DECISIONS",
        "STS2_BENCH_REFLECTION_MODE",
        "STS2_BENCH_REFLECTION_MAX_OPTIONS",
        "STS2_BENCH_ACTION_PROMPT_ORDER",
        "STS2_BENCH_ACTION_PROMPT_ORDER_SEED",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, "omegaconf", None)

    load_benchmark_config(config)

    assert os.environ["STS2_BENCH_CONTEXT_MODE"] == "turn_chat"
    assert os.environ["STS2_BENCH_ROUTE_PLAN_ENABLED"] == "false"
    assert os.environ["STS2_BENCH_ROUTE_PLAN_ENFORCEMENT"] == "soft"
    assert os.environ["STS2_BENCH_RUN_PLAN_ROUTE_PLAN_ENABLED"] == "false"
    assert os.environ["STS2_BENCH_TURN_CHAT_WINDOW"] == "3"
    assert os.environ["STS2_BENCH_TURN_CHAT_UPDATE_MODE"] == "delta"
    assert os.environ["STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY"] == "compact"
    assert os.environ["STS2_BENCH_TURN_CHECK_ENABLED"] == "true"
    assert os.environ["STS2_BENCH_COMBAT_CARD_SEQUENCES_ENABLED"] == "false"
    assert os.environ["STS2_BENCH_SELF_CONSISTENCY_ENABLED"] == "true"
    assert os.environ["STS2_BENCH_SELF_CONSISTENCY_SAMPLES"] == "5"
    assert os.environ["STS2_BENCH_SELF_CONSISTENCY_DECISIONS"] == "map_select,card_reward"
    assert os.environ["STS2_BENCH_SELF_CONSISTENCY_MODE"] == "majority_vote"
    assert os.environ["STS2_BENCH_REFLECTION_ENABLED"] == "true"
    assert os.environ["STS2_BENCH_REFLECTION_DECISIONS"] == "noncombat_non_map"
    assert os.environ["STS2_BENCH_REFLECTION_MODE"] == "verifier_vote"
    assert os.environ["STS2_BENCH_REFLECTION_MAX_OPTIONS"] == "6"
    assert os.environ["STS2_BENCH_ACTION_PROMPT_ORDER"] == "shuffle_map"
    assert os.environ["STS2_BENCH_ACTION_PROMPT_ORDER_SEED"] == "7"


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
