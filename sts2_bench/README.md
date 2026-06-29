# sts2_bench

Python benchmark and LLM-policy harness for the STS2 headless JSON protocol.

This package is the benchmark/research layer.  It deliberately avoids the
legacy HTTP bridge under `legacy/agent-http-bridge/`: benchmark runs start
`src/Sts2Headless` as a subprocess and communicate through stdin/stdout JSON.
The runtime adapter remains in
`sts2-cli`; this package owns policy prompting, legal-action presentation,
validation, trace logging, and offline analysis.

The current default research architecture is evidence-first and pure LLM:
deterministic modules assemble facts and candidate-action evidence, but they do
not score, rank, recommend, vote, or choose.  A single CA/OCC LLM policy still
selects exactly one current legal `action_id`.

## Layout

- `process.py` - owns one `Sts2Headless` subprocess.
- `actions.py` - converts each decision point into the dynamic legal-action list.
- `context.py` - renders human-readable state text and compact observations.
- `agents.py` - prompt headers, OpenAI-compatible LLM agent, random baseline,
  turn-chat context, JSON parsing, action-label validation, and fallback policy.
- `routing.py` - routes each decision point to combat CA or out-of-combat OCC.
- `multi_agent.py` - coordinator facade, OCC prompt wrapper, stale-action guard,
  and RunContext patch application.
- `evidence.py` - evidence-only context modules and schema validation.
- `posterior.py` - evidence-only LLM deck-state posterior evaluator for card
  reward decisions.
- `run_context.py` - per-run RunPlan/RunMemory store used by OCC.
- `feedback.py` - uniform validation-feedback event envelope and per-decision
  feedback bus.
- `route_plan.py` - runner-local route-plan event validation.
- `sequence_validation.py` - engine-backed custom sequence probe.
- `runner.py` - benchmark loop, action validation/retry bus, execution, metrics,
  and JSONL trace logging.
- `run_benchmark.py` - implementation behind the `python -m sts2_bench run` command.
- `trace_analysis.py` - implementation behind the `python -m sts2_bench trace-analysis` command.
- `env.py` - dependency-free RL-style environment with action masks.

## Runtime Flow

One benchmark decision follows this path:

1. `Sts2Process` returns the current headless game state.
2. `actions.build_legal_actions()` builds all currently legal benchmark actions.
   Action ids are local to that one state.
3. `evidence.build_evidence_bundle()` builds an advisory fact bundle from the
   same state and legal actions.
4. `routing.route_state()` chooses the active policy surface:
   `combat_play` and combat-owned `card_select` go to CA; reward, shop, route,
   event, rest, treasure, and non-combat selection states go to OCC.
5. `CoordinatorAgent` delegates to CA or OCC.
6. The selected LLM returns JSON containing one `action_id` and the exact
   `selected_action_label`.
7. The runner validates evidence refs, OCC context contract, route-plan events,
   and custom combat sequences before execution.
8. Valid benchmark-local view actions update only the benchmark state.  Valid
   game actions are sent to `Sts2Headless`.
9. The runner logs the decision/action/result trace and records transition
   context for the agent.

Current legal actions are always the execution authority.  Prompts, evidence,
RunContext, route plans, and previous chat turns are reference context only.

## Legal Actions

`actions.py` is the action source of truth.  The model never invents raw engine
commands; it chooses one item from the current legal-action list.

Each `LegalAction` contains:

- `action_id` - zero-based id for the current state only.
- `label` - human-readable label shown to the LLM and trace readers.
- `kind` - benchmark action category, such as `card_reward` or `map`.
- `command` - either an engine JSON command or a benchmark-local command.

Important action categories:

- `bench_view` - `view deck`, `view map and current position`, `view draw pile`,
  `view discard pile`, and `view exhaust pile`.  These do not advance the game.
- `card_reward_skip`, `combat_reward_skip`, `card_select_skip`,
  `shop_leave`, and event leave actions - skipping or leaving is represented as
  a first-class legal action when the engine exposes it.
- `combat_card_sequence` - deterministic precomputed card sequences used by
  internal/non-LLM action surfaces and by validated custom sequence execution;
  LLM agents do not see these as legal actions.
- `combat_custom_sequence` - a model-proposed deterministic sequence bound to
  the initial visible hand/enemy snapshot.  The runner validates it with an
  engine-backed probe before converting it into `bench_sequence`; cards with
  draw, random, modal choice, generated-card, or other state-changing uncertainty
  should be handled as single legal actions instead.

By default, view actions are exposed at most once per decision point.  Set
`allow_repeat_views: true` to expose repeated views at the same decision.

## Evidence-First Context

`evidence.py` defines the evidence layer.  Its purpose is to organize facts, not
to implement policy.

`EvidenceBundle` contains:

- `decision_type`
- `state_facts`
- `candidate_evidence`
- `missing_information`
- `memory_slice`
- `feedback_context`
- `deck_summary`
- `posterior_audit`
- `reward_posterior_evaluation`
- visibility flags such as `deck_summary_visible`,
  `full_deck_details_visible`, `deck_details_visible`, and
  `skip_action_visible`

`CandidateEvidence` contains:

- `action_id`
- `label`
- `kind`
- `evidence_for`
- `evidence_against`
- `requires_information`

Evidence modules are prohibited from emitting policy fields:

- `score`
- `rank`
- `recommendation`
- `recommended_action`
- `preferred_action`
- `best_action`
- `selected_action`
- `selected_action_label`
- `chosen_action`
- `should_choose`
- `should_pick`
- `should_skip`

If any evidence fragment contains those fields, `validate_no_policy_fields()`
raises.  This keeps the harness from becoming a heuristic scorer while still
giving the LLM structured context.

Current modules:

- `DeckBuildContextModule` - exposes the deck as grouped inventory when
  `player.deck` is present.
- `CombatReflectionModule` - exposes recent observed transition changes from
  `last_action_result`.
- `RewardEvidenceModule` - presents pick/claim/skip reward actions at the same
  candidate level.
- `ShopEvidenceModule` - presents gold, buy/remove/leave candidates, and shop
  action evidence.
- `RouteEvidenceModule` - presents legal map nodes and missing map information.
- `GenericEvidenceModule` - falls back to one candidate evidence item per
  non-view legal action.

### Deck Summary Policy

Deck context is now fixed evidence when the state exposes `player.deck`.  The
deck is compressed as a complete grouped inventory:

```json
{
  "representation": "complete_grouped_inventory",
  "deck_size": 31,
  "grouping_key": "id,name,upgraded,type,cost,description,enchantment,affliction,keywords",
  "groups": [
    {"name": "Strike", "count": 5, "upgraded": 0, "type": "Attack", "cost": 1}
  ]
}
```

This avoids hiding basic deck-building information behind a voluntary `view
deck` action, while keeping large decks compact.  `view deck` remains useful as
a full-detail information action and is still logged through
`full_deck_details_visible`.

### Reward Posterior Evaluator

For `card_reward`, the coordinator can run a separate LLM posterior evaluator
before OCC chooses.  This evaluator is not a second policy and does not execute
or select actions.  It receives the current grouped deck, current legal reward
actions, candidate evidence, recent combat facts, and the deterministic
`posterior_audit`.  It must return one deck-state analysis per candidate:

- `deck_delta`
- `immediate_current_deck_effect`
- `redundancy_or_dilution_risks`
- `visible_support`
- `missing_or_uncertain_support`
- `comparison_to_unchanged_deck`
- `uncertainty`

The evaluator must cover every current `candidate_evidence.action_id` exactly
once and copy labels exactly.  The no-policy-field validator rejects scores,
ranks, recommendations, preferred actions, selected actions, and similar
strategy fields.  OCC receives the resulting
`reward_posterior_evaluation` as evidence inside the card reward prompt; final
selection still comes from the single OCC action-selection response.

The CLI enables this by default for `--agent coordinator`.  Use
`--no-reward-posterior-evaluator` to run A/B comparisons against the
history-free reward prompt without the second analysis pass.

## CA, OCC, And RunContext

`--agent coordinator` creates the current multi-surface LLM harness:

- CA is an `OpenAICompatAgent` in `turn_chat` mode.  It handles combat turns and
  may keep compact chat history within the current player turn.
- OCC is an `OutOfCombatContextAgent` wrapper around an `OpenAICompatAgent`.
  It handles rewards, shops, routes, events, rest sites, treasure, and
  non-combat selections.
- Both surfaces are exposed through one public `CoordinatorAgent`.

The OCC prompt is assembled from:

1. the base state/legal-actions prompt;
2. the advisory `EvidenceBundle`;
3. a `RunContextStore` slice;
4. a bounded OCC short history of recent out-of-combat decisions.

Card reward is a stricter single-step mode.  For `card_reward`, OCC receives the
base prompt plus current evidence, grouped deck inventory, recent combat facts,
deterministic posterior audit, and optional reward posterior deck-state
analysis.  It does not receive OCC short history, prior reward reasons, or
mutable build-plan context, and any returned build-plan or memory patch is
ignored.

`RunContextStore` contains:

- `RunPlan` - `route_plan`, `build_plan`, `resource_policy`, `open_threads`,
  and `delayed_effects`.
- `RunMemory` - recent facts, combat reports, and pending mandates.

The LLM cannot mutate this store directly.  It may return:

- `run_plan_patch`
- `run_memory_facts`
- `combat_reports`

The coordinator validates allowed keys, decorates entries with floor/action
metadata, applies accepted patches, and logs rejected patches in agent metadata.

For OCC non-view actions, an empty `run_plan_patch` is valid and does not need a
separate explanation field.  The action-level `reason` explains why the selected
legal action is being taken.

## LLM Response Contract

The prompt header in `agents.py` asks the model to return JSON with the current
schema.  The minimal important fields are:

```json
{
  "action_id": 0,
  "selected_action_label": "copy exact current legal action label",
  "sequence": [{"card_index": 0, "target_index": 0}],
  "route_plan_event": {
    "type": "route_plan_created",
    "new_plan": [
      {"col": 2, "row": 1, "type": "Unknown"},
      {"col": 1, "row": 2, "type": "RestSite"}
    ],
    "reason_tags": ["example"],
    "reason": "why this route"
  },
  "evidence_refs": [3],
  "run_plan_patch": {},
  "run_memory_facts": [],
  "reason": "short final reason"
}
```

Only `action_id`, `selected_action_label`, and `reason` are broadly relevant to
every decision.  `sequence` is only meaningful for `combat_custom_sequence`.
`route_plan_event` is required when creating or changing a map route plan and
must contain a complete `new_plan` starting at the selected legal map node and
continuing through visible connected nodes to the visible Boss, or to the visible
terminal node when Boss is not present in the route graph.  Copy each submitted
plan node's exact `col`, `row`, and `type` from that graph; the ASCII map is
human-readable context only.  Omit `route_plan_event` when executing
`active_route_plan.expected_next`, but still include the normal action-level
`reason`.  `route_plan_created` and `route_plan_updated` events must include a
non-empty route `reason`.  With Winged Boots, submitted route jumps may use only
the remaining forward-jump charges shown in `route_graph.constraints`.
`evidence_refs` must refer to candidate action ids present in the current
evidence bundle.

Action validation is strict:

- `action_id` must be in range for the current legal list.
- `selected_action_label` must exactly match the selected current action label,
  after normalization.
- If `action_id` and label disagree, the label can remap the action only when it
  matches exactly one current legal action; the remap is logged.
- If the reason appears to describe a different non-view legal action, the agent
  receives a correction request.
- With `llm_fallback_mode: error`, exhausted retries raise `ModelActionError`
  instead of silently executing a fallback action.

## FeedbackEvent And FeedbackBus

`FeedbackEvent` is the runner's uniform envelope for validation failures that
can be sent back to the active agent before retrying the same decision.

It is defined in `feedback.py` as:

```python
@dataclass(frozen=True)
class FeedbackEvent:
    kind: str
    message: str
    retry: int
    retryable: bool
    code: str | None = None
    step: int | None = None
    decision: str | None = None
    action_id: int | None = None
    action_label: str | None = None
    validator: str | None = None
    evidence_bundle_hash: str | None = None
    max_retries: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
```

Fields:

- `kind` - validation family, such as `evidence_validation`,
  `route_plan_validation`, or `custom_sequence_validation`.
- `message` - natural-language correction sent to the model.
- `retry` - zero-based retry count at the time this event is created.
- `retryable` - whether the runner still has retry budget for this failure.
- `code` - optional machine-readable reason, such as `invalid_evidence_refs`
  or a route-plan error code.
- `step`, `decision`, `action_id`, `action_label`, `validator`, and
  `evidence_bundle_hash` - trace linkage for the invalid attempt.
- `max_retries` - retry budget for this validation family.
- `details` - structured validator-specific context, such as the original
  route-plan event, failed sequence payload, or precise evidence-ref error.

`FeedbackBus` owns the feedback lifecycle within one decision:

- tracks retry counts per validation family;
- creates enriched `FeedbackEvent` objects with the current step, decision,
  action, validator, evidence hash, and retry budget;
- writes a uniform `feedback_event` payload into validation-error trace rows;
- appends prior feedback events to the successful retry action's agent metadata;
- increments result retry counters such as
  `evidence_validation_retry_count`;
- dispatches retryable feedback through `agent.receive_external_feedback()`;
- reports exhausted retry budget without dispatching another model retry.

Current uses:

- Evidence validation checks bad `evidence_refs`.
- Route-plan validation checks whether emitted route-plan events match current
  legal map actions and the active route plan.
- Custom-sequence validation checks model-proposed combat sequences against an
  engine-backed probe before execution.

For a single-turn LLM, dispatched feedback is appended to the next retry
request.  For a turn-chat LLM, it is recorded as user feedback in the short chat
buffer.

`FeedbackEvent` and `FeedbackBus` do not choose an action, mutate game state, or
apply fallback.  They are not policy modules.  They are the common traceable
shape and lifecycle for "your previous response was invalid; here is why; choose
again from the current legal actions."

## Validation And Retry Semantics

The runner validates in layers:

1. LLM parse/action validation inside `OpenAICompatAgent`.
2. Coordinator stale-action guard.
3. Evidence/OCC contract validation in the runner.
4. Route-plan event validation through `RoutePlanTracker`.
5. Custom-sequence validation through `validate_custom_sequence()`.
6. Engine execution through `Sts2Headless`.

Strict benchmark mode should use:

```yaml
llm_fallback_mode: error
```

Exploratory mode may use:

```yaml
llm_fallback_mode: action
```

Fallback mode keeps long exploratory batches moving, but fallback traces are not
clean policy samples.  They are logged with `fallback: true` and counted in
summary metrics.

## JSONL Trace Logging

`log_level` controls trace size:

- `metrics` - start/action/result summaries only.
- `decisions` - compact decision state, legal-action summaries, evidence
  summary, action summaries, and agent metadata.
- `full` - compact state, full legal actions, full evidence bundle, prompt, raw
  model response, and full agent metadata.

Decision and action logs include evidence-related metadata when available:

- `evidence_bundle_hash`
- `deck_summary_visible`
- `full_deck_details_visible`
- `deck_details_visible`
- `skip_action_visible`
- `feedback_events`
- `run_context_patch`
- `run_context_hash`

Route validation errors, custom sequence errors, evidence validation errors, and
fallback metadata are trace-visible.

## Quick Smoke Test

Install the benchmark Python dependency:

```bash
python -m pip install -r sts2_bench/requirements.txt
```

Show CLI options without starting the game:

```bash
python -m sts2_bench run --help
```

Start the headless process and read the ready message:

```bash
python -m sts2_bench process-smoke
```

Run one random-policy game:

```bash
python -m sts2_bench run \
  --agent random \
  --character Ironclad \
  --ascension 0 \
  --count 1 \
  --out results/random_ironclad_a0.jsonl
```

## Local LLM Benchmark

The LLM agent expects an OpenAI-compatible chat completions endpoint.  Examples
include Ollama, LM Studio, vLLM, and llama.cpp server.

Non-secret defaults live in `sts2_bench/benchmark.yaml`.  The tracked defaults
run the random baseline with the preconfigured fixed seeds in
`sts2_bench/seeds_ironclad_a0.json` so a fresh checkout is reproducible before
LLM credentials are configured.  The default model is DeepSeek V4 Pro, written
as the provider model name `deepseek-v4-pro`; local provider URLs and secrets
still come from `.env`.

Supported `.env` keys:

- `DEEPSEEK_API_KEY`
- `OPENAI_API_KEY`

Supported `benchmark.yaml` keys include:

- agent/model: `agent`, `base_url`, `model`, `llm_timeout`,
  `llm_max_retries`, `llm_temperature`, `llm_fallback_mode`
- prompt/context: `prompt_style`, `include_json_state`, `memory_enabled`,
  `memory_window`, `memory_mode`, `run_summary_enabled`,
  `context_management`
- run setup: `character`, `ascension`, `lang`, `seeds`, `count`, `max_steps`
- logging: `out`, `log_level`, `print_prompts`, `print_model_output`,
  `trace_analysis`, `trace_analysis_dir`
- benchmark action surface: `include_full_map`, `allow_repeat_views`,
  `allow_custom_sequences`

Precedence is: CLI arguments, shell environment variables, `.env`,
`sts2_bench/benchmark.yaml`, then code defaults.

When `--seeds` is omitted, runs use the preconfigured fixed seed pool and take
the first `--count` seeds.  If `--count` exceeds that pool, pass an explicit
`--seeds` file.

Ollama example:

```bash
ollama serve
ollama pull qwen2.5:14b
```

Run a coordinator benchmark:

```bash
python -m sts2_bench run \
  --agent coordinator \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen2.5:14b \
  --character Ironclad \
  --ascension 0 \
  --seeds sts2_bench/seeds_ironclad_a0.json \
  --out results/qwen25_14b_coordinator_ironclad_a0.jsonl
```

For strict floor benchmarks, keep `max_steps` high enough that it acts as a
safety cap rather than the main stopping condition:

```bash
python -m sts2_bench run \
  --agent coordinator \
  --count 10 \
  --max-steps 10000 \
  --no-print-prompts \
  --no-print-model-output \
  --no-allow-custom-sequences \
  --log-level metrics \
  --out results/coordinator_floor_benchmark.jsonl
```

The summary reports floors, win rate, truncation/error rates, game-action
counts, view-action counts, repeated-view counts, model fallback counts, and
wall time.

## Card Choice Evaluation

`card_choice_eval.py` is an offline synthetic diagnostic for LLM card-reward
choices.  It does not start `Sts2Headless` and it is not a win-rate benchmark.
Each scenario is a controlled `card_reward` state with three pick actions and a
first-class `skip card reward` action.

For a Chinese design and usage walkthrough, see
`docs/evaluations/card-choice/card-choice-eval-zh.md`.

The evaluation still uses the real harness path:

```text
scenario state -> build_legal_actions() -> build_evidence_bundle() -> agent.choose()
```

The hidden oracle is used only after the model selects an action.  It is not
included in the prompt, evidence bundle, or agent metadata.

Run a local coordinator model:

```bash
python -m sts2_bench card-choice \
  --agent coordinator \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen2.5:14b \
  --out results/card_choice_eval.jsonl
```

Useful filters:

```bash
python -m sts2_bench card-choice \
  --agent coordinator \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen2.5:14b \
  --category skip_is_best \
  --category defense_gap_indirect \
  --limit 8 \
  --log-level full \
  --out results/card_choice_eval_debug.jsonl
```

The v1 fixed suite covers:

- `defense_gap_indirect`
- `skip_is_best`
- `synergy_pick`
- `anti_synergy_trap`
- `overweight_attack`
- `resource_curve`
- `redundancy`
- `ambiguous_good_pick`

The JSONL output contains `start`, `scenario`, `choice`, and `summary` rows.
Summary metrics include overall accuracy, per-category accuracy, skip
precision/recall, trap failure rate, and invalid/error counts.  `decisions`
logs evidence summaries; `full` logs full evidence and the prompt.

## Trace Analysis And Replay Artifacts

After a run, generate a compact replay and failure-analysis bundle:

```bash
python -m sts2_bench trace-analysis \
  results/coordinator_floor_benchmark.jsonl \
  --out-dir artifacts/trace_analysis/coordinator_floor_benchmark
```

Benchmark runs generate these artifacts by default after the JSONL trace is
closed.  Unless `--trace-analysis-dir <dir>` is provided, `run_benchmark` writes
them under `artifacts/trace_analysis/<trace-stem>`.  Multi-run traces are split
per seed under that directory and indexed by `index.json`.  Use
`--no-trace-analysis` only for runs that intentionally skip report generation.

The analyzer is offline.  It does not change benchmark execution or the runner
hot path.  It writes:

- `summary.json` - machine-readable route, room, card reward, shop, rest-site,
  HP, and high-risk route events.
- `replay.md` - human-readable compact replay with card offers, selected/skip
  choices, rest choices, shop purchases, and removals.
- `hp_by_action.svg` - HP curve by real game action.
- `hp_by_room.svg` - room-sequence HP route.
- `map_path.svg` - selected map route highlighted against logged topology when
  map snapshots are present.
- `route_hp_replay.svg` - route-vs-HP overview for manual failure inspection.
- `large_loss_rooms/` - raw JSONL slices and short summaries for severe HP-loss
  rooms or deaths.

High-risk route events currently include low-HP elite selections, ignored
rest/shop alternatives, and forced-elite entries linked back to earlier route
branches when trace evidence is available.

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

The RL environment is separate from the pure-LLM coordinator harness.  It should
not inherit evidence modules or LLM prompt assumptions unless a later experiment
explicitly chooses to expose those features.

## Prompt Contract Boundaries

Model-facing prompts, trace logs, and engine commands are separate interfaces.
`LegalAction.to_prompt_dict()` is the model surface and exposes only
`action_id`, `kind`, and `label`.  Full traces, eval artifacts, and RL `info`
use the log surface with the raw `cmd` mapping so command tapes remain
replayable without leaking engine protocol details into prompts.

`agents.py` derives a `PromptContract` for every decision from the current
state and legal action list.  Global feature switches are only capabilities:
route-plan schema is exposed only on `map_select`, custom sequence schema is
exposed only when the current `combat_play` legal actions include the
`combat_custom_sequence` gateway, and `turn_check` is combat-only.  Experiment
wrappers should add instructions by decision type instead of changing an
agent-wide prompt style when only one decision surface needs the behavior.

## Design Boundaries

This package is intentionally a benchmark scaffold, not a solved STS2 policy.

Kept deterministic:

- legal action generation
- state/context rendering
- evidence-only fact organization
- route/custom-sequence validation
- stale action guard
- trace logging

Kept LLM-owned:

- final action choice
- combat tactical judgment
- reward/shop/build reasoning
- route preference under current legal actions
- RunContext patch content

Not currently part of this harness:

- heuristic scoring/ranking modules
- supervised reward pickers
- learned value functions
- non-LLM voting policies
- automatic hidden information actions
