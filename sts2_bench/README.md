# sts2_bench

Python benchmark and RL scaffolding for the STS2 headless JSON protocol.

This package deliberately avoids the HTTP bridge under `agent/`.  It starts
`src/Sts2Headless` directly as a subprocess and communicates through
stdin/stdout JSON.  That keeps benchmark runs reproducible, easier to parallelize,
and closer to an RL environment.

## Layout

- `process.py` - owns one `Sts2Headless` subprocess.
- `actions.py` - converts each decision point into a dynamic legal-action list.
- `context.py` - renders state text/compact observations inspired by `python/play.py`.
- `agents.py` - random baseline, prompt construction, and OpenAI-compatible local LLM agent.
- `runner.py` - programmatic benchmark loop.
- `run_benchmark.py` - command-line benchmark runner.
- `env.py` - dependency-free RL-style environment with action masks.

## Quick Smoke Test

Install the benchmark Python dependency:

```bash
python3 -m pip install -r sts2_bench/requirements.txt
```

Show CLI options without starting the game:

```bash
python3 -m sts2_bench.run_benchmark --help
```

Start the headless process and read the ready message:

```bash
python3 -m sts2_bench.process
```

## Random Baseline

Run one random-policy game:

```bash
python3 -m sts2_bench.run_benchmark \
  --agent random \
  --character Ironclad \
  --ascension 0 \
  --count 1 \
  --out results/random_ironclad_a0.jsonl
```

The output JSONL detail is controlled by `--log-level`.  Use `metrics` for the
smallest benchmark traces, `decisions` for lightweight state/action summaries,
and `full` when debugging exact prompts and model responses.

## Local LLM Benchmark

The LLM agent expects an OpenAI-compatible chat completions endpoint.  Examples:
Ollama, LM Studio, vLLM, llama.cpp server.

Non-secret defaults live in `sts2_bench/benchmark.yaml`.  The file is tracked
by git and documents each setting inline.  Local secrets still come from `.env`;
copy the template and fill in keys locally:

```bash
cp .env.example .env
```

Supported `.env` keys:

- `DEEPSEEK_API_KEY`
- `OPENAI_API_KEY`

Supported `benchmark.yaml` keys include `agent`, `base_url`, `model`,
`llm_timeout`, `llm_max_retries`, `llm_temperature`, `prompt_style`,
`include_json_state`, `memory_enabled`, `memory_window`, `memory_mode`,
`run_summary_enabled`, `context_management`, `character`, `ascension`, `lang`,
`seeds`, `count`, `max_steps`, `out`, `log_level`, `print_prompts`,
`print_model_output`, `include_full_map`, and `allow_repeat_views`.  The file
is loaded with OmegaConf; most settings are top-level scalar keys, while agent
context-management settings live under `context_management`.

Precedence is: CLI arguments, shell environment variables, `.env`,
`sts2_bench/benchmark.yaml`, then code defaults.  Shell overrides can still use
the old `STS2_BENCH_*` names for one-off runs.

Ollama example:

```bash
ollama serve
ollama pull qwen2.5:14b
```

Run benchmark:

```bash
python3 -m sts2_bench.run_benchmark \
  --agent llm \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen2.5:14b \
  --character Ironclad \
  --ascension 0 \
  --seeds sts2_bench/seeds_ironclad_a0.json \
  --out results/qwen25_14b_ironclad_a0.jsonl
```

At startup the runner prints a redacted `run_config` JSON event with the
resolved parameters, including `llm_temperature`.  When `out` is set, the same
event is also written as the first JSONL trace record.  API keys are not logged.

DeepSeek smoke test via `.env` and `benchmark.yaml`:

```bash
# In .env, set DEEPSEEK_API_KEY=...
# In sts2_bench/benchmark.yaml, set agent/model/count/max_steps/print_*.
python3 -m sts2_bench.run_benchmark
```

Prompt styles:

- `default` asks for a compact `{"action_id": ..., "reason": ...}` response.
- `analysis` asks for JSON with situation analysis, calculations, candidate
  action comparison, and final `action_id`.

Set `include_json_state: true` in `sts2_bench/benchmark.yaml` to include the
compact state JSON in the model prompt.  The benchmark-oriented default is
`false` because the human-readable state already carries the current decision
context and the JSON can substantially increase prompt length.

Agent context management:

- `context_management.mode: single_turn` keeps the current behavior: each LLM
  decision is an independent prompt.
- `context_management.mode: turn_chat` keeps a short chat transcript only within
  the current player combat turn, using the nested `turn_chat.window`,
  `turn_chat.update_mode`, and `turn_chat.assistant_history` settings.
- `context_management.turn_chat.plan.enabled: true` asks the first prompt of
  each player turn to include a hand-level `turn_plan`.  Compact assistant
  history keeps that plan for later actions in the same turn; the runner still
  executes exactly one `action_id` per request.

`turn_chat` is parallel to the existing episode memory method.  Disable
`memory_enabled` when using `turn_chat` so benchmark runs measure one context
method at a time.

Experience memory:

- `experience_memory.enabled: true` switches the LLM agent to
  `ExperienceOpenAICompatAgent`, implemented in `experience/agent.py`.
- The first version is read-only/frozen: it retrieves reviewed JSONL experience
  notes from `experience_memory.store_path` and injects a short advisory section
  before the current game state.
- It can optionally write a compact event trace with
  `experience_memory.event_trace_path`; the trace stores action/reason/delta
  summaries for offline combat/act/run reflection, not full state snapshots.
- `experience_memory.auto_learn.enabled: true` reflects the current Act's
  combat spans. `trigger: act_end` runs only after advancing to the next Act;
  `trigger: act_end_or_death` also runs when the character dies, which lets
  cold-start Act 1 failures still produce candidate memories. Death-triggered
  learning does not promote into `active.jsonl` by default; set
  `auto_learn.promote_on_death: true` only when you want failed-run lessons to
  affect the next run immediately.
- The MVP supports only `context_management.mode: single_turn`.

Minimal active memory entry:

```json
{"id":"exp_jaw_worm_0001","status":"active","memory_type":"risk","scope":{"level":"encounter","character":"Ironclad","act":1,"enemy":"Jaw Worm"},"trigger":"Act 1 Jaw Worm combat","recommendation":"Balance early damage with enough block when Jaw Worm is attacking.","quality":{"confidence":0.7},"retrieval_hints":{"keywords":["jaw_worm","combat_play","act1"],"priority":0.8}}
```

In-run Act-end learning loop:

```bash
python3 -m sts2_bench.run_benchmark \
  --experience-memory-enabled \
  --experience-event-trace-path logs/experience_events.jsonl \
  --experience-auto-learn-enabled \
  --experience-auto-learn-trigger act_end_or_death \
  --no-experience-auto-learn-promote-on-death \
  --experience-auto-learn-candidate-path sts2_bench/experience/memory/candidates.jsonl \
  --experience-memory-store-path sts2_bench/experience/memory/active.jsonl
```

At Act transition, the agent uses the same OpenAI-compatible model settings as
normal decisions to reflect only the completed Act's in-memory event trace.  It
writes `candidates.jsonl`, promotes only the newly generated candidate ids into
`active.jsonl`, skips duplicate ids/content, and reloads the active store before
continuing.  On death, the same reflection writes candidates but skips promotion
unless `promote_on_death` is enabled.

For an existing trace, run the same learning step manually:

```bash
python3 -m sts2_bench.experience.learn \
  --trace logs/experience_events.jsonl \
  --candidate-out sts2_bench/experience/memory/candidates.jsonl \
  --active sts2_bench/experience/memory/active.jsonl
```

JSONL logging uses `log_level`:

- `metrics` records only start/action summaries/result.
- `decisions` records lightweight state/action summaries without prompts or raw
  model responses.
- `full` records compact state, full legal actions, prompts, and raw model
  responses.

For a floor benchmark, keep `max_steps` high enough that it acts as a safety
cap rather than the main stopping condition:

```bash
python3 -m sts2_bench.run_benchmark \
  --count 10 \
  --max-steps 10000 \
  --no-print-prompts \
  --no-print-model-output \
  --log-level metrics \
  --out results/deepseek_floor_benchmark.jsonl
```

The summary reports `max_floor`, `median_floor`, `avg_floor`, truncation/error
rates, game-action counts, view-action counts, repeated-view counts, and model
fallback counts.  Treat floors from truncated runs as safety-cap results rather
than natural death/victory outcomes.

The prompt asks the model to return only:

```json
{"action_id": 0, "reason": "short reason"}
```

The model never needs to invent raw game commands.  `actions.py` generates all
legal commands from the current state, and the model picks one by id.
`context.py` only renders the current state; policy instructions and the legal
action prompt wrapper live in `agents.py`.

Two benchmark-local view actions are available at every non-terminal state:

- `view deck` expands the player's full deck with card descriptions.
- `view map and current position` fetches and renders the full map when the
  simulator has map data.

These actions do not advance the game process.  They are information-gathering
actions: the next benchmark prompt/observation keeps the current decision
context and appends the requested viewed information.  Once the policy chooses
a real game action, the game advances and viewed information is cleared by the
new engine state.  By default, each view action is exposed at most once per
decision point; set `allow_repeat_views: true` to allow repeated viewing at
the same decision.

## RL Environment

`env.py` exposes a minimal dependency-free interface:

```python
from sts2_bench.env import Sts2Env

env = Sts2Env(character="Ironclad", ascension=0)
obs, info = env.reset(seed="rl_seed_0001")
mask = info["action_mask"]
legal = info["legal_actions"]

obs, reward, terminated, truncated, info = env.step(0)
env.close()
```

For Gymnasium/SB3 integration, wrap `Sts2Env` and use:

- `obs` from `compact_state()`, or your own numeric encoder.
- `info["action_mask"]` for dynamic valid actions.
- `info["legal_actions"][idx]["cmd"]` as the raw command mapping.

## Design Notes

The current LLM context keeps the important human-facing information from
`python/play.py`, but formats it for models:

- act, floor, room, boss
- player HP, block, gold, deck size, relics, potions
- combat energy and draw/discard pile counts
- enemy HP, block, intents, moves, powers
- hand cards with cost, type, playability, target type, and exposed stats
- map, full-deck view, card reward, event, shop, rest, treasure, and selection
  options

This is intentionally a benchmark scaffold, not a solved policy.  Serious RL
work should add a numeric encoder and tune rewards separately from the
terminal win/floor metrics used for benchmark comparison.
