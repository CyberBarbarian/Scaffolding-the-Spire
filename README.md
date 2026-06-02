# STS2 Agent Bench

[Chinese version](README.zh-CN.md)

`sts2-agent-bench` is the private BG-Agent research repository for comparing
LLM and hybrid-agent methods on Slay the Spire 2. The repository includes the
headless `sts2-cli` runtime, the `sts2_bench` benchmark scaffold, experiment
configuration, traces, metrics, and failure-analysis material.

The public CLI-only runtime is maintained separately at
[CyberBarbarian/sts2-cli](https://github.com/CyberBarbarian/sts2-cli). Keep
CLI/headless adapter fixes there. Keep LLM policy, benchmark, trace, prompt,
memory, planning, search, and research-method work here.

## Scope

This repository is for:

- comparing LLM-based STS2 playing methods;
- running reproducible random, LLM, memory, chat-history, planning, and hybrid-agent baselines;
- logging traces, run configurations, metrics, and failure labels;
- turning failures into regression spots and benchmark cases;
- maintaining the private research workflow around STS2 agent evaluation.

This repository is not for:

- redistributing Slay the Spire 2 game DLLs;
- changing game card, relic, enemy, event, combat, or reward semantics;
- keeping the public CLI runtime boundary clean. That belongs in `sts2-cli`.

## Runtime Lineage

The runtime started from the
[CyberBarbarian/sts2-cli](https://github.com/CyberBarbarian/sts2-cli) fork of
[wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli). It runs the real Slay
the Spire 2 engine headlessly and exposes:

- an interactive terminal UI through `python/play.py`;
- a stdin/stdout JSON protocol through `src/Sts2Headless`;
- reusable state export, localization, and action surfaces for benchmarks.

You must own and install Slay the Spire 2 through Steam. This repository does
not contain or redistribute game DLLs.

## Layout

- `sts2_bench/` - benchmark loop, action generation, state rendering, agents,
  configuration loading, and RL-style environment scaffolding.
- `src/Sts2Headless/` - C# headless adapter around the real STS2 engine.
- `src/GodotStubs/` - narrow Godot presentation stubs required by the headless
  adapter.
- `python/` - interactive CLI player, log helpers, and full-run smoke tools.
- `localization_eng/` and `localization_zhs/` - engine-derived localization
  data used by the runtime and prompts.
- `tests/` - CLI/runtime and benchmark regression coverage.
- `docs/` - runtime notes, benchmark plans, and research notes.
- `logs/`, `results/`, `artifacts/`, `lib/`, `.tools/`, and `saves/` - local
  runtime or experiment artifacts. They should stay out of commits unless a
  specific artifact is intentionally promoted.

## Quick Start

Requirements:

- Slay the Spire 2 installed through Steam.
- Python 3.9 or newer.
- .NET 9 SDK or the repo-local `.tools/dotnet` runtime.
- On Unix-like systems, `bash` for `setup.sh` and `copy_dlls.sh`.

Clone the private repository:

```bash
git clone https://github.com/CyberBarbarian/sts2-agent-bench.git
cd sts2-agent-bench
```

On Windows:

```cmd
.\setup-windows.bat
.\sts2-cli.bat
```

On macOS, Linux, or Git Bash:

```bash
./setup.sh
python3 python/play.py
```

The first setup locates the Steam install, copies required local DLLs into
`lib/`, and builds the headless adapter. Do not commit or redistribute `lib/`.

## Running Benchmarks

Install benchmark dependencies when needed:

```bash
python -m pip install -r sts2_bench/requirements.txt
```

Show benchmark options:

```bash
python -m sts2_bench.run_benchmark --help
```

Run a random baseline:

```bash
python -m sts2_bench.run_benchmark ^
  --agent random ^
  --character Ironclad ^
  --ascension 0 ^
  --count 1 ^
  --out results/random_ironclad_a0.jsonl
```

Run an OpenAI-compatible LLM endpoint:

```bash
python -m sts2_bench.run_benchmark ^
  --agent llm ^
  --base-url http://127.0.0.1:11434/v1 ^
  --model qwen2.5:14b ^
  --character Ironclad ^
  --ascension 0 ^
  --count 1 ^
  --out results/qwen25_ironclad_a0.jsonl
```

Local secrets belong in `.env`, not in tracked files. Non-secret defaults live
in `sts2_bench/benchmark.yaml`.

## Experiment Discipline

Every serious comparison should record:

- runtime commit and benchmark commit;
- backend type and version;
- model name, endpoint family, temperature, and retry settings;
- character, ascension, seed list, max step cap, and truncation policy;
- prompt style, memory mode, context-management mode, and run-summary settings;
- JSONL traces plus a compact metrics summary;
- failure labels that distinguish strategy errors from backend/runtime errors.

When changing prompts or agent methods, keep random and simple LLM baselines
available so changes can be compared against a stable floor.

## Testing

Build the headless adapter:

```bash
.\.tools\dotnet\dotnet.exe build src\Sts2Headless\Sts2Headless.csproj
```

Run focused benchmark tests:

```bash
python -m pytest tests/test_sts2_bench_actions.py tests/test_sts2_bench_context.py tests/test_sts2_bench_config.py -q
```

Run focused CLI/runtime tests:

```bash
python -m pytest tests/test_event.py tests/test_dynamic_card_stats.py tests/test_combat.py -q
```

On Windows, run build-oriented tests separately from long runtime tests to avoid
temporary `.dll` file locks:

```bash
python -m pytest tests/test_godot_stubs.py -q
python -m pytest -m "not slow" --ignore=tests/test_godot_stubs.py -q
```

## Publication Boundary

This repository is private research infrastructure. Public CLI/runtime changes
should be ported to `CyberBarbarian/sts2-cli` when they improve the headless
adapter itself. Research-only methods, traces, prompt experiments, and benchmark
reports should remain here unless explicitly prepared for publication.
