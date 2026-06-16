# Learnable Experience Memory Implementation Plan

本文把 `docs/ChatGPT-LLM Agent 深度研究.md` 中的“经验库 + 优先回放 + 反思重写”方案，收缩成适配当前 `sts2_bench` 代码库的工程实现计划。

核心目标不是做一个会在每一步随意写入的 run memory，而是做一个**可学习、可冻结、可审计**的经验记忆库：

- training / reflection 阶段可以从历史 run 中生成候选经验。
- evaluation 阶段只允许检索冻结后的经验库，绝不写入。
- 每条经验必须有证据、作用域、置信度和反例边界。
- 当前 state 永远优先于经验库；经验库只提供策略建议，不作为事实来源。

## Why This Fits STS2

当前 STS2 benchmark 已经把可见状态完整文本化，并且每一步只允许选择一个合法 `action_id`。因此 agent 的主要短板不是“看不到当前牌/遗物/敌人”，而是：

- 跨楼层目标不稳定：奖励、商店、路线、火堆决策容易漂移。
- 失败模式会重复：例如低血贪精英、缺成长硬进 Boss、牌组膨胀、药水过度囤积。
- 模型会把局部错误推理带入后续：尤其是 enemy intent、card damage/block 的二次计算。
- 单 seed 经验不能直接当规则：需要证据、置信度、反例和冻结评估。

所以长期记忆不应该保存当前状态摘要，而应该保存“未来 run 仍可能复用的策略约束”。

## Current Codebase Anchors

当前代码库已有几个适合接入的点：

| Area | Current file | Existing role | Experience memory integration |
|---|---|---|---|
| Prompt construction | `sts2_bench/agents.py` | `build_llm_prompt()`、`OpenAICompatAgent.build_prompt()` | 在 rules 后、state 前注入 retrieved experience notes。 |
| LLM response | `sts2_bench/agents.py` | `OpenAICompatAgent.choose()` 解析 JSON 并返回 meta | 在 meta 中记录 retrieved ids、store digest、可选 `used_memory_ids`。 |
| Turn chat | `sts2_bench/agents.py` | `TurnChatBuffer` 只保留同一玩家出牌回合 history | experience memory 与 turn_chat 平行，可叠加但先单独实验。 |
| Short memory | `sts2_bench/agents.py` | `ShortTermMemory` 保存当前 episode 最近事实 | experience memory 不替代它；两者时间尺度不同。 |
| Run loop | `sts2_bench/runner.py` | `run_one()` 每步 build actions、choose、execute、log | 不在主循环做 LLM 反思；只记录检索和行动 meta。 |
| JSONL logging | `sts2_bench/runner.py` | `log_level=decisions/full` 输出 decision/action/result | 反思脚本优先读取 JSONL trace，不依赖 terminal tee log。 |
| Config / CLI | `sts2_bench/run_benchmark.py` | YAML -> env -> argparse -> run_config | 增加 `experience_memory.*` 嵌套配置，并记录 store digest。 |

设计原则：MVP 尽量不改 runner 行为，只在 agent prompt 前增加检索层，并把学习过程放到离线脚本。

## Proposed Modes

经验库要区分四种模式，避免 training 和 evaluation 混在一起。

| Mode | Retrieval | Write active store | Write candidates | Use case |
|---|---:|---:|---:|---|
| `disabled` | no | no | no | baseline。 |
| `frozen` | yes | no | no | 正式评估，经验库只读。 |
| `observe` | yes | no | no | 调试检索和 prompt 注入，不更新。 |
| `reflect` | optional | no | yes | 离线从 log 生成候选经验。 |

第一版只实现 `disabled`、`frozen`、离线 `reflect`。不要做在线写 active memory。

## Configuration

建议在 `sts2_bench/benchmark.yaml` 增加独立层级，不与 `memory_enabled` 或 `context_management` 混用：

```yaml
experience_memory:
  enabled: false
  mode: frozen
  store_path: sts2_bench/experience/memory/active.jsonl
  candidate_path: sts2_bench/experience/memory/candidates.jsonl
  top_k: 5
  token_budget: 700
  include_global: true
  allow_cross_character: false
  allow_cross_ascension: true
  min_confidence: 0.35
  max_notes_per_type:
    rule: 2
    risk: 2
    episode: 1
    playbook: 1
  reflection:
    enabled: false
    candidate_only: true
    trigger: offline
```

`run_config` 必须记录：

- resolved config。
- active store path。
- active store sha256 digest。
- top_k / token budget。
- mode 是否 frozen。

这样同一批 benchmark 结果可以回放和解释。

## Storage Layout

建议新增目录：

```text
sts2_bench/experience/memory/
  active.jsonl
  candidates.jsonl
  deprecated.jsonl
  reflections/
    bench_0000_20260602.json
  README.md
```

第一版使用 JSONL，不引入数据库和向量库。原因：

- 当前条目量会很小。
- JSONL 易审查、易 diff、易 pin digest。
- 先把 schema、证据和冻结评估跑通，比复杂检索更重要。

## Event Trace Granularity

学习系统应该**每步记录，但记录内容必须压缩**。完整 state 只用于 `log_level=full` 的 debug / replay，不应直接作为经验学习的默认输入。

原因：

- STS2 的 state 很长，完整记录每步会让反思输入迅速膨胀。
- 当前手牌、敌人、遗物、牌组等信息大多只是当步事实，不是长期经验。
- 反思器真正需要的是“发生了什么决策、造成了什么变化、是否暴露了可复用失败模式”。

建议新增一个 compact event trace，和现有 JSONL benchmark trace 平行：

```json
{
  "run_id": "bench_0000_20260602",
  "step": 87,
  "act": 1,
  "floor": 12,
  "decision": "combat_play",
  "room_type": "Monster",
  "encounter": "Jaw Worm",
  "hp_bucket_before": "0.50-0.75",
  "hp_bucket_after": "0.50-0.75",
  "deck_size_bucket": "16-20",
  "action": {
    "kind": "combat_play_card",
    "label": "play card 2: Shrug It Off"
  },
  "outcome": {
    "decision_after": "combat_play",
    "hp_delta": 0,
    "block_delta": 8,
    "enemy_hp_delta": -0,
    "energy_delta": -1,
    "important_changes": ["draw_pile: 8 -> 7", "hand_count: 5 -> 5"]
  },
  "agent": {
    "reason": "Need block before continuing damage.",
    "fallback": false,
    "retrieved_memory_ids": [],
    "used_memory_ids": []
  }
}
```

记录规则：

- 每步只保存 bucket、delta、action kind/label、简短 reason、fallback、memory ids。
- `important_changes` 最多 5-8 条，来自现有 `last_action_result`，不要保存整段 state。
- 非战斗节点保存 reward/shop/map/rest/event 的选择摘要和资源变化。
- 如果某一步需要 debug，可通过 `trace_ref` 回到 full JSONL，而不是把 full state 内联进学习 trace。

这样做的效果是：学习输入从“几百步完整状态”变成“几十到几百条短事件”，反思器更容易抽取稳定经验。

## Memory Entry Schema

第一版只接受五类经验：

| Type | Meaning | Example |
|---|---|---|
| `rule` | 可复用策略规则。 | 缺成长时 Boss 前优先补 scaling 或保血。 |
| `risk` | 失败模式和禁忌。 | 低血且无药水时不要贪可选精英。 |
| `episode` | 一个关键历史片段的压缩教训。 | 某 run 因牌组膨胀在 Act 1 Boss 失血过多。 |
| `playbook` | 更高层的路线/构筑策略片段。 | Ironclad A0 early shop remove + block density plan。 |
| `counterfactual` | 明确反事实边界。 | 如果已有可靠 draw/scaling，则可放宽跳牌规则。 |

JSONL item：

```json
{
  "id": "exp_ironclad_a0_lagavulin_0001",
  "status": "active",
  "memory_type": "risk",
  "scope": {
    "level": "encounter",
    "character": "Ironclad",
    "ascension_min": 0,
    "ascension_max": 5,
    "act": 1,
    "enemy": "Lagavulin Matriarch",
    "boss": "Lagavulin Matriarch",
    "archetype": "low_scaling"
  },
  "trigger": "Act 1 boss planning or Lagavulin fight with weak scaling",
  "lesson": "Long Lagavulin fights punish low scaling decks because repeated debuffs reduce later block and damage quality.",
  "recommendation": "Before the boss, prefer frontloaded damage, scaling, or potion value over marginal deck expansion.",
  "anti_pattern": "Taking several medium cards without improving damage density or scaling, then entering boss with no clear kill plan.",
  "counterfactual": "If the deck already has reliable scaling or strong potion support, this risk is lower.",
  "evidence": [
    {
      "trace": "results/deepseek-chat-analysis-2026-06-01.jsonl",
      "seed": "bench_0000",
      "step": 202,
      "act": 1,
      "floor": 17,
      "outcome": "death"
    }
  ],
  "quality": {
    "confidence": 0.4,
    "support_count": 1,
    "contradiction_count": 0,
    "retrieval_count": 0,
    "used_count": 0
  },
  "retrieval_hints": {
    "keywords": ["lagavulin", "boss", "act1", "scaling", "deck_bloat"],
    "priority": 0.7
  },
  "created_at": "2026-06-02",
  "updated_at": "2026-06-02"
}
```

Validation rules:

- `lesson` / `recommendation` 不能只是当前状态摘要。
- 禁止保存具体当前战斗算式，例如 `Attack 11x2 + Strength = ...`。
- 禁止无证据经验进入 `active`。
- 初始候选 `confidence <= 0.5`。
- 同一条经验必须能说明适用边界和反例边界。
- `global` 经验必须非常少，且只能是稳定 prompt 规则或引擎语义。

## Retrieval Design

第一版用结构化过滤 + 简单关键词打分，不上向量库。

### Feature Extraction

新增 `sts2_bench/experience/features.py`：

```python
@dataclass
class StateFeatures:
    character: str
    ascension: int
    decision: str
    act: int | None
    floor: int | None
    room_type: str
    boss: str
    enemy_names: list[str]
    hp_bucket: str
    deck_size_bucket: str
    card_names: list[str]
    relic_names: list[str]
    potion_names: list[str]
    action_kinds: list[str]
    tags: set[str]
```

来源：

- 当前 `state`。
- `context`。
- `legal_actions`。
- 如果已有 `view_deck`，可使用已展示 deck 信息；否则不要假装知道完整牌表。

### Scoring

检索步骤：

1. 过滤 `status=active`。
2. 按 character / ascension / act / boss / enemy / decision 做硬过滤。
3. `global` 条目低权重保留。
4. 按关键词和 scope 计算分数。
5. 按 type 做配额，避免全被同一类 memory 占满。
6. 在 `token_budget` 内渲染 top K。

建议分数：

```text
score =
  2.0 * exact_enemy_or_boss_match
+ 1.2 * character_match
+ 0.8 * act_match
+ 0.7 * decision_match
+ 0.6 * keyword_overlap
+ 0.8 * confidence
+ 0.3 * priority
- 0.5 * contradiction_penalty
```

### Prompt Rendering

注入位置：rules 之后、state 之前。

```text
Experience notes, advisory only. Current game state and legal actions override these notes.
- [risk/id=exp_.../conf=0.40] Trigger: Act 1 Lagavulin with weak scaling. Recommendation: ...
- [rule/id=exp_.../conf=0.55] Deck rewards: skip marginal cards when the deck already has enough low-impact attacks; prefer draw/block/scaling.
```

不要注入 evidence 全量，只注入短 recommendation。完整 evidence 留在日志和 memory store。

## Prompt / Response Changes

当 `experience_memory.enabled=true` 时，action response schema 可以允许一个可选字段：

```json
{
  "action_id": 5,
  "reason": "short reason",
  "used_memory_ids": ["exp_ironclad_a0_lagavulin_0001"]
}
```

`used_memory_ids` 只用于分析，不影响行动解析。规则：

- 只能填本步实际 retrieved ids。
- 如果没有用到经验，返回空数组或省略。
- 不能输出新的 memory。

这样可以估计“记忆是否真的参与了决策”，而不是只知道它被塞进了 prompt。

## New Modules

建议新增：

```text
sts2_bench/experience/memory/
  __init__.py
  schema.py
  store.py
  features.py
  retriever.py
  render.py
  digest.py
  reflection.py
  validate.py
```

职责：

| Module | Responsibility |
|---|---|
| `schema.py` | dataclass / TypedDict 定义，JSON parse / serialize。 |
| `store.py` | 读写 JSONL，按 status 分流 active/candidate/deprecated。 |
| `features.py` | 从 state + legal_actions 提取检索特征。 |
| `retriever.py` | 结构化过滤、打分、top K。 |
| `render.py` | token budget 内渲染 prompt section。 |
| `digest.py` | active store sha256，写入 run_config。 |
| `reflection.py` | 离线从 trace 生成 candidate entries。 |
| `validate.py` | 拒绝低质量、无证据、状态摘要、战斗算式。 |

## Code Integration Plan

### `run_benchmark.py`

增加：

- `NESTED_CONFIG_ENV_KEYS` for `experience_memory.*`。
- CLI:
  - `--experience-memory-enabled`
  - `--experience-memory-mode {frozen,observe}`
  - `--experience-memory-store-path`
  - `--experience-memory-candidate-path`
  - `--experience-memory-top-k`
  - `--experience-memory-token-budget`
  - `--experience-memory-min-confidence`
- `build_run_config()` 增加 resolved config 和 store digest。
- 将配置传入 `agent_from_args()`。

### `runner.py`

最小改动：

- `agent_from_args()` 接收 experience memory 参数，构造 `OpenAICompatAgent`。
- `_agent_meta_summary()` 白名单加入 `experience_memory`，至少记录：
  - `enabled`
  - `mode`
  - `retrieved_ids`
  - `used_memory_ids`
  - `store_digest`
  - `rendered_chars`

不建议在 `run_one()` 里直接调用反思器。反思走离线脚本。

### `agents.py`

新增 agent 字段：

```python
experience_memory_enabled: bool = False
experience_memory_mode: Literal["frozen", "observe"] = "frozen"
experience_memory_retriever: ExperienceMemoryRetriever | None = field(init=False)
```

修改：

- `build_llm_prompt(..., experience_text="", experience_enabled=False)`。
- `build_turn_chat_system_message()` 不需要知道完整 memory，只需要允许 optional `used_memory_ids`。
- `OpenAICompatAgent.build_prompt()`：
  - 根据 state + legal_actions 检索。
  - 渲染 experience section。
- `OpenAICompatAgent.choose()`：
  - 和 build_prompt 使用同一组 retrieved ids。
  - meta 写入 `experience_memory`。
  - parse optional `used_memory_ids`，非法则忽略。

注意：如果 runner 先调用 `build_prompt()` 用于日志，再调用 `choose(prompt=prompt)`，要避免重复检索导致 prompt 和 meta 不一致。可以采用两种实现之一：

1. `build_prompt()` 返回 preview 时也缓存本步 retrieval，`choose()` 使用缓存。
2. 给 `choose()` 新增内部 `_build_prompt_with_experience()`，当外部传入 prompt 时，meta 只记录重新计算的 retrieval ids。

推荐第一种，但要用 state signature 做缓存 key，避免跨步复用错误。

### Offline reflection script

新增命令：

```bash
python -m sts2_bench.experience.reflect \
  --trace logs/experience_events.jsonl \
  --candidate-out sts2_bench/experience/memory/candidates.jsonl \
  --model deepseek-chat \
  --base-url https://api.deepseek.com \
  --max-candidates-per-span 2
```

自动化命令可以把 reflect 和本次候选 promote 合并：

```bash
python -m sts2_bench.experience.learn \
  --trace logs/experience_events.jsonl \
  --candidate-out sts2_bench/experience/memory/candidates.jsonl \
  --active sts2_bench/experience/memory/active.jsonl \
  --max-candidates-per-span 2
```

`learn` 默认只 promote 本次反思新增的 candidate ids，不会把 `candidates.jsonl` 里历史未审核候选一并激活。需要人工审核时用 `--no-promote`。

当前实验实现还支持 in-run act-end learning：

```bash
python -m sts2_bench.run_benchmark \
  --experience-memory-enabled \
  --experience-event-trace-path logs/experience_events.jsonl \
  --experience-auto-learn-enabled \
  --experience-auto-learn-trigger act_end
```

触发时机：state 的 `act` 从 N 变成 N+1 后，agent 只反思当前 episode 内 act=N 的 combat events。它不会从 `event_trace_path` 重读历史同 seed 事件，避免复用 trace 文件时混入旧 run。反思完成后只 promote 本次新增候选，并刷新 retriever，让下一 Act 可以直接检索刚学到的 active memory。

输入使用 `experience_memory.event_trace_path` 写出的 compact event trace。当前可跑 MVP 默认只切分 combat span；`log_level=full` 的 benchmark JSONL trace 可作为后续证据回查来源，但不是当前默认反思上下文。

脚本步骤：

1. 读取 `run_config`、compact event trace、必要时回查 full JSONL。
2. 切分 spans：
   - MVP：each combat。
   - 后续：act end。
   - 后续：full run。
   - 后续：boss / elite combat，作为高优先级 combat span。
   - 后续：shop。
   - 后续：card reward series。
   - 后续：death final 20 steps。
3. 给 reflection LLM 生成候选。
4. 本地 validator 过滤。
5. append 到 `candidates.jsonl`。

### Reflection frequency

推荐的学习频率：

| Trigger | Input span | Purpose | Write target |
|---|---|---|---|
| 每步 action 后 | compact event only | 记录证据，不调用 LLM 反思。 | compact event trace |
| 每场 combat 结束 | 当前 combat 的 compact events | 复盘局内操作、药水使用、出牌顺序、伤害/格挡节奏。 | candidate experience |
| 每个 Act 结束 | 当前 Act 的 compact events + combat summaries | 复盘路线、奖励、商店、火堆、Boss 准备。 | candidate experience |
| 死亡 / 胜利 / run 结束 | full run compact summaries | 汇总主要成功/失败模式，生成高层 playbook/risk。 | candidate experience |

这样比“每步反思”简单，也比“只在终局反思”更细。combat-level reflection 关注战术质量，act-level reflection 关注宏观策略，run-level reflection 负责最终归因和去重。

### Candidate promotion script

第一版可以不做复杂 UI，只加一个脚本：

```bash
python -m sts2_bench.experience.promote \
  --candidates sts2_bench/experience/memory/candidates.jsonl \
  --active sts2_bench/experience/memory/active.jsonl
```

当前脚本会校验 candidate schema、confidence 阈值，并按 id/content 去重后 append 到 active store。需要更严格人审时，可以先把候选标成 `status=reviewed`，再用 `--reviewed-only` promote。关键是 active store 必须可审查。

## Reflection Prompt

离线反思 prompt 应严格避免泛泛总结：

```text
You are extracting reusable STS2 benchmark experience for future runs.

Rules:
- Current state facts are not long-term lessons.
- Do not invent hidden mechanics.
- Do not store tactical arithmetic from one combat state.
- Separate observation, diagnosis, reusable recommendation, anti-pattern, and counterfactual boundary.
- Each candidate must cite evidence from the trace.
- Initial confidence must be low unless the trace gives repeated evidence.
- Output JSON only.

Allowed memory_type: rule, risk, episode, playbook, counterfactual.

Trace span:
{{trace_span}}

Similar existing memories:
{{similar_memories}}
```

Output:

```json
{
  "candidates": [
    {
      "memory_type": "risk",
      "scope": {"level": "encounter", "character": "Ironclad", "enemy": "Lagavulin Matriarch"},
      "trigger": "...",
      "lesson": "...",
      "recommendation": "...",
      "anti_pattern": "...",
      "counterfactual": "...",
      "evidence_refs": ["step=199", "step=202"],
      "confidence": 0.35,
      "keywords": ["lagavulin", "scaling", "boss"]
    }
  ]
}
```

## Learning Loop

MVP training loop：

```text
run benchmark on training seeds with log_level=full
      |
      v
build compact event trace from benchmark JSONL
      |
      v
combat-end / act-end / run-end reflection creates candidates
      |
      v
human review promotes selected candidates to active
      |
      v
run frozen evaluation on paired seeds
      |
      v
compare baseline vs experience_memory
```

不要在同一 evaluation run 中边反思边更新 active store。

## Metrics

必须同时记录结果指标和记忆指标。

Result metrics:

- win rate。
- floor / act reached。
- boss / elite survival。
- avg HP loss by floor。
- model fallback count。
- wall time / tokens。

Memory metrics:

- retrieved ids per decision。
- `used_memory_ids` per decision。
- retrieved-but-unused ratio。
- memory type distribution。
- token overhead。
- repeated failure avoidance rate。
- candidate -> active promotion rate。
- contradiction count。

最重要的是 paired seed 比较：

- `single_turn` baseline。
- `single_turn + experience_memory(frozen)`。
- `turn_chat` baseline。
- `turn_chat + experience_memory(frozen)`。

严肃实验里不要同时打开 `memory_enabled`、`turn_chat`、`experience_memory` 三个变量再比较。

## Phase Plan

### Phase 0: Log hygiene

目标：保证离线反思有可靠输入。

- 每次 benchmark 使用唯一 `out` 路径。
- 反思用 compact event trace，不用 terminal tee log。
- full JSONL 只作为 evidence 回查，不作为默认反思输入。
- `run_config` 记录 temperature、model、prompt_style、context mode。
- fallback meta 保存足够 raw/error 信息，便于反思失败原因。
- compact event trace 只记录 action、reason、bucket、delta、重要变化和 memory ids。

### Phase 1: Read-only retrieval MVP

目标：先证明经验检索能稳定注入 prompt 且不破坏行动。

- 手写 5-10 条 active memory。
- 实现 JSONL store、features、retriever、renderer。
- prompt 注入 experience section。
- action meta 记录 retrieved ids。
- 不做 reflection。

验收：

- 单元测试覆盖 schema、检索、token budget、prompt injection。
- benchmark run_config 包含 store digest。
- `frozen` mode 下不会写任何文件。

### Phase 2: Offline reflection candidates

目标：从 run trace 自动产生候选经验，但不自动激活。

- 实现 compact event trace builder。
- 实现 `sts2_bench.experience.reflect`。
- 实现 validator。
- 写 `candidates.jsonl`。
- 人审提升到 active。

验收：

- 候选经验都有 evidence。
- validator 能拒绝状态摘要、战斗算式、无边界泛化。
- active store 只通过人工或明确 promote 命令更新。
- 当前 MVP：combat span reflection 能生成带 evidence 的候选并通过 promote 命令进入 active。
- 后续扩展：act-end、run-end reflection 分别生成宏观路线/整局归因候选。

### Phase 3: Priority replay and quality update

目标：让记忆库从 append-only 变成可维护。

- 支持 dedupe / merge。
- 支持 confidence、support、contradiction 更新。
- 支持 deprecated。
- 记录 retrieval/use stats。

验收：

- 相似经验不会无限增殖。
- 反例会降低 confidence 或生成 counterfactual。
- prompt 中不会出现互相矛盾的高权重经验。

### Phase 4: Playbook layer

目标：把零散 rule/risk 压缩成策略片段。

- 从 active memory 中生成 playbook。
- playbook 有版本和适用条件。
- evaluation 可固定 playbook digest。

验收：

- playbook 可以回滚。
- 不同角色/ascension 有 namespace。
- 对比 active memory only vs playbook + memory。

## Test Plan

Unit tests:

- `ExperienceMemoryEntry` parse / serialize。
- invalid schema rejection。
- state feature extraction from combat / reward / shop / map。
- retrieval scoring by boss/enemy/character/decision。
- token budget rendering。
- prompt injection only when enabled and retrieved notes non-empty。
- `used_memory_ids` only accepts retrieved ids。
- store digest changes when active store changes。

Integration tests:

- fake active store + fake LLM response -> prompt contains ids and meta records retrieval。
- `frozen` mode run does not mutate active/candidate files。
- `sts2_bench.experience.reflect` with tiny fake trace produces candidate JSONL。
- validator rejects tactical calculation candidate.
- `run_config` records experience_memory settings.

Regression tests:

- baseline agent without experience memory remains unchanged。
- random agent rejects or ignores experience memory settings。
- turn_chat still resets on end turn and does not store full experience text in assistant history。

## Risk Controls

| Risk | Control |
|---|---|
| 记忆污染 | candidate-only reflection, manual promote, confidence threshold。 |
| 证据和推断混淆 | schema separates evidence, lesson, recommendation, anti_pattern, counterfactual。 |
| 当前状态被复述 | validator rejects HP/gold/deck_size/enemy intent/current arithmetic summaries。 |
| prompt 过载 | top_k, token_budget, per-type quota。 |
| 负迁移 | character / ascension / boss / archetype scoped retrieval。 |
| 评估泄漏 | evaluation mode requires frozen store digest and no writes。 |
| 难以归因 | meta records retrieved ids, used ids, store digest, rendered chars。 |
| 自演化失控 | no online active writes in MVP; playbook versioning only in later phase。 |

## Recommended First Implementation

最小可控版本：

1. 新增 JSONL active store，先手写少量经验。
2. 实现 read-only retriever。
3. 在 `OpenAICompatAgent` prompt 中注入 retrieved experience。
4. meta 记录 retrieved ids 和 store digest。
5. 跑 paired seeds，看 prompt、fallback、floor 是否变化。
6. 再实现离线 `sts2_bench.experience.reflect`，只写 candidates。

这个顺序比“先做在线可写 memory”更慢，但更可控，也更适合当前代码库和 benchmark 评估。
