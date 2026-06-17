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

Non-secret defaults live in `sts2_bench/benchmark.yaml`.  The tracked defaults
run the random baseline with generated seeds so a fresh checkout is safe to
smoke test before any LLM credentials are configured.  Local secrets still come
from `.env`; copy the template and fill in keys locally:

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
is loaded with OmegaConf when available; a small top-level scalar fallback keeps
basic commands usable before dependencies are installed. Agent
context-management settings live under `context_management`, so those nested
settings require OmegaConf.

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
# In sts2_bench/benchmark.yaml, set agent=llm, base_url, model,
# count, max_steps, and print_* as needed.
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
- `--agent coordinator` records `context_management.mode: coordinator`: CA uses
  combat `turn_chat`; OCC uses `reference_turn`, which combines the current
  out-of-combat state, the RM/RP slice, and a bounded short history of recent
  OCC decisions.  The OCC inner LLM call is still one request per decision, but
  it is not a memoryless `single_turn` policy.

`turn_chat` is parallel to the existing episode memory method.  Disable
`memory_enabled` when using `turn_chat` so benchmark runs measure one context
method at a time.

LLM failure handling:

- `llm_max_retries` controls how many correction requests are sent after an LLM
  timeout, parse failure, or illegal `action_id`.  The tracked benchmark config
  keeps this at `1` so strict runs get one correction attempt before stopping.
- `llm_fallback_mode: error` is the benchmark default.  If every retry fails,
  the runner stops the current run and records the current floor/HP/error
  instead of sending a fallback game action.
- `llm_fallback_mode: action` preserves the old exploratory behavior: after all
  retries fail, the agent chooses the first non-view legal action and logs
  `fallback: true`.  Do not use this for strict CA-vs-coordinator comparisons.

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

## Trace Analysis And Replay Artifacts

After a run, generate a compact replay and failure-analysis bundle from its
JSONL trace:

```bash
python3 -m sts2_bench.trace_analysis \
  results/deepseek_floor_benchmark.jsonl \
  --out-dir artifacts/trace_analysis/deepseek_floor_benchmark
```

The analyzer is offline: it does not change benchmark execution or the runner
hot path.  It writes:

- `summary.json` - machine-readable route, room, card reward, shop, rest-site,
  HP, and high-risk route events.
- `replay.md` - human-readable compact replay, including card offers and
  selected/skip choices, Rest/Smith choices, Smith upgrade targets, shop
  purchases, and shop removals.
- `hp_by_action.svg` - HP curve by real game action, colored by room type, with
  room bands and direct HP-drop labels; benchmark-local view actions are not
  counted as game actions.
- `hp_by_room.svg` - room-sequence HP route.  Each room column shows entry HP,
  exit HP, direct `floor/type/loss` labels, room indexes, and loss-severity
  colors.
- `map_path.svg` - selected map route highlighted against full logged
  `map_snapshot` topology when traces include `--include-full-map`; older logs
  without snapshots fall back to all logged legal map choices, rendered in
  separate panels per act.
- `route_hp_replay.svg` - route-vs-HP overview.  When traces include
  `map_snapshot`, the map panel renders the real map topology horizontally
  with map rows on the x-axis and original map columns on the y-axis.  It is
  clipped to rows already reached in the run and includes room-to-room edges,
  unchosen rooms, the selected path, and selected-room HP-loss badges.  Older
  logs without snapshots fall back to same-floor candidate rooms from logged
  legal choices.  This is meant for manual failure inspection rather than
  automatic attribution.
- `large_loss_rooms/` - raw JSONL slices and short summaries for rooms that
  lose at least 20 HP, lose at least 25 percent of max HP, or end in death.

High-risk route events currently include:

- low HP route choices that select an Elite while safer alternatives exist;
- low HP route choices where a Rest Site or Shop is available but ignored;
- low HP forced-Elite entries, linked back to the nearest earlier branch that
  locked the run into that route.

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
