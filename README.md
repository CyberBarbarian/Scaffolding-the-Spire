# STS2 Agent Bench

[Chinese version](README.zh-CN.md)

`sts2-agent-bench` is the private BG-Agent research repository for comparing
LLM and hybrid-agent methods on Slay the Spire 2. The repository includes the
`sts2_bench` benchmark scaffold, experiment configuration, traces, metrics,
and failure-analysis material. The game runtime is not duplicated here: every executable benchmark path uses the
adjacent authoritative `sts2-cli` checkout, or a frozen copy made from it.

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

## Runtime Boundary

The authoritative runtime is the adjacent
[CyberBarbarian/sts2-cli](https://github.com/CyberBarbarian/sts2-cli) checkout.
This repository deliberately contains no second copy of its terminal UI,
headless adapter, Godot stubs, launchers, or autoplay helpers. Set
`STS2_CLI_ROOT` when the checkout is not located at `../sts2-cli`.

You must own and install Slay the Spire 2 through Steam. This repository does
not contain or redistribute game DLLs.

## Layout

- `sts2_bench/` - benchmark loop, action generation, state rendering, agents,
  configuration loading, and RL-style environment scaffolding.
- `localization_eng/` and `localization_zhs/` - engine-derived localization
  data used by reporting and prompts.
- `tests/` - benchmark contracts plus a narrow integration check against the
  authoritative runtime.
- `docs/` - benchmark plans and research notes.
- `logs/`, `results/`, and `artifacts/` - local experiment artifacts.

## Quick Start

Requirements:

- Slay the Spire 2 installed through Steam.
- Python 3.9 or newer.
- A working sibling `sts2-cli` checkout, prepared according to its README.

Clone the private repository:

```bash
git clone https://github.com/CyberBarbarian/sts2-agent-bench.git
cd sts2-agent-bench
```

Clone or place `sts2-cli` next to this repository. Prepare and test the runtime
there, then return here to run benchmark commands. Frozen formal launchers copy
the exact runtime artifacts they need and bind them through `STS2_CLI_ROOT`.

## Running Benchmarks

Install benchmark dependencies when needed:

```bash
python -m pip install -r sts2_bench/requirements.txt
```

Show benchmark options:

```bash
python -m sts2_bench run --help
```

Run a random baseline:

```bash
python -m sts2_bench run ^
  --agent random ^
  --character Ironclad ^
  --ascension 0 ^
  --count 1 ^
  --out results/random_ironclad_a0.jsonl
```

Run an OpenAI-compatible LLM endpoint:

```bash
python -m sts2_bench run ^
  --agent llm ^
  --base-url http://127.0.0.1:11434/v1 ^
  --model qwen2.5:14b ^
  --character Ironclad ^
  --ascension 0 ^
  --count 1 ^
  --out results/qwen25_ironclad_a0.jsonl
```

Local secrets belong in `.env`, not in tracked files. Non-secret defaults live
in `sts2_bench/benchmark.yaml`; the tracked default is the coordinator
chain-of-thought (CoT) route/combat baseline, while random remains available
through `--agent random` for no-credential smoke runs.

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

Run the full benchmark repository suite:

```bash
python -m pytest -q
```

That suite covers `sts2_bench` policy/action/context/trace behavior and keeps
only a small integration surface for the authoritative headless runtime. The exhaustive
CLI/runtime regression suite is maintained in the separate `sts2-cli` checkout,
where public runtime adapter changes should be made and validated.

Run focused benchmark tests while iterating:

```bash
python -m pytest tests/test_sts2_bench_actions.py tests/test_sts2_bench_context.py tests/test_sts2_bench_config.py -q
```

Run the retained runtime integration checks separately when touching process
startup behavior:

```bash
python -m pytest tests/test_local_tools.py tests/test_process_reuse.py tests/test_runtime_smoke.py -q
```

Before porting or publishing runtime adapter changes, run the full runtime gate
from `C:\WorkSpace\BG-Agent\sts2-cli` instead of duplicating those detailed
tests in this benchmark repository.

## Publication Boundary

This repository is private research infrastructure. Public CLI/runtime changes
should be ported to `CyberBarbarian/sts2-cli` when they improve the headless
adapter itself. Research-only methods, traces, prompt experiments, and benchmark
reports should remain here unless explicitly prepared for publication.
