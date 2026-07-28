# Self-Improving Experience Memory Plan

> Historical design and experiment-outline document. Its three-study research questions remain useful, but its older Top-K, frozen-store, deterministic semantic-gating, and implementation sections are superseded by `docs/structured-memory-reflection-release-todo-20260721.md`. Do not implement or run from this file without reconciling against the current recovery plan.

本文记录一个新的长期研究方向：让模型在 benchmark 游玩过程中维护一个可检索、可更新的经验库，并在后续决策中利用这些经验优化策略。这个方向类似 text gradient / textual feedback 的思想：模型不仅输出动作，还从结果中提炼文字形式的策略改进，持续更新一个外部知识库。

这个方案与 `turn_chat`、`short_term_memory_enabled` 的关系不同：

- `turn_chat`: 当前玩家出牌回合内的短 chat history，解决同一回合多步操作的局部连续性。
- `short_term_memory_enabled`: 单局 episode 内的短期动作/事实历史，解决最近若干动作和状态变化的可见性。
- `experience_memory`: 跨局、跨 seed、可冻结评估的长期经验库，解决“这个角色/敌人/遗物/卡牌组合应该怎么打”的策略知识积累。

因此 `experience_memory` 可以与 `single_turn` 或 `turn_chat` 同时存在。它不是 prompt 组织方法，而是一个检索增强的长期策略层。

## Research Questions

自 2026-07-20 起，记忆方向不再以“实现一个更复杂的记忆系统”为目标，而是拆成三个可独立证伪的 benchmark 研究问题：

| ID | Formal question | Why it matters |
|---|---|---|
| `RQ1 Experience Induction` | 长时序、多层级游戏轨迹会如何影响经验抽取的忠实度、覆盖率和可迁移性？ | 不能只凭几条错误经验宣称“反思很难”，必须对退化现象做可量化比较。 |
| `RQ2 Exposure Gating` | 经验暴露的覆盖率、特异性、重复度和上下文成本如何影响长程表现？ | 把“门控会影响表现”改写为可验证的暴露—效用权衡，而不是比较两个任意实现。 |
| `RQ3 Cross-Character Transfer` | 哪些经验能够跨角色迁移，哪些经验会导致系统性负迁移？ | 利用 STS2 共享敌人/地图与角色私有牌组并存的特点，研究经验的可迁移边界。 |

以下问题作为上述三个 RQ 的辅助分析，不单独宣称为创新点：同角色跨难度迁移、在线更新对评估的污染、模型自写经验的错误积累、以及 frozen evaluation 与 online training 的边界。

## Method Naming

建议把这个方向暂时命名为 `experience_memory`，也可以作为后续实验矩阵中的 M3：

| Name | Meaning |
|---|---|
| `M0 no_memory` | 原始 single-turn baseline。 |
| `M1 episode_memory` | 单局内 factual/action memory。 |
| `M2 turn_chat` | 玩家回合内 chat history。 |
| `M3 experience_memory` | 跨局经验库和自我改进。 |

`M3` 可以叠加在 `M0/M1/M2` 上，例如：

- `single_turn + experience_memory`
- `turn_chat + experience_memory`
- `turn_chat + experience_memory + frozen_eval`

但严肃对比时应一次只打开一个主要变量。

## Core Design

### Runtime Loop

```text
state + legal actions
      |
      v
build retrieval query
      |
      v
retrieve relevant experience entries
      |
      v
agent prompt = rules + state + legal actions + retrieved experience notes
      |
      v
model chooses one atomic action
      |
      v
environment transition
      |
      v
optional reflection/update pass writes candidate experience entries
```

关键原则：

- 经验库只提供建议，不是事实来源；当前 state 永远优先。
- 经验库不能包含当前 evaluation seed 的未来信息。
- evaluation 时经验库应冻结，不能边评估边更新。
- 更新经验库时必须记录 evidence，例如 source log、floor、encounter、death cause、action result。

### Experience Entry Schema

第一版建议用 JSONL 存储，便于程序读取；同时可以生成 Markdown 视图供人审阅。

```json
{
  "id": "exp_ironclad_lagavulin_0001",
  "status": "active",
  "scope": {
    "level": "character",
    "character": "Ironclad",
    "ascension_min": 0,
    "ascension_max": 20,
    "encounter": "Lagavulin Matriarch",
    "boss": null,
    "cards": ["Rampage", "Shrug It Off"],
    "relics": []
  },
  "trigger": "Act 1 boss Lagavulin Matriarch with Strength/Dexterity debuffs active",
  "lesson": "Do not wait until heavy debuffs stack before switching to defense or scaling damage.",
  "recommendation": "Prioritize early damage setup and preserve potions for the boss. If debuffs already reduce block/damage to near zero, recognize the fight is likely lost.",
  "anti_pattern": "Continuing low-impact attacks while ignoring that future turns will be debuffed below survival thresholds.",
  "evidence": [
    {
      "run_id": "bench_20260529_194446",
      "seed": "bench_0000",
      "act": 1,
      "floor": 17,
      "outcome": "death"
    }
  ],
  "confidence": 0.4,
  "updated_at": "2026-05-30"
}
```

字段说明：

| Field | Purpose |
|---|---|
| `scope.level` | `global`, `character`, `encounter`, `card`, `relic`, `archetype` 等。 |
| `trigger` | 检索时匹配当前状态的短描述。 |
| `lesson` | 从历史结果中学到的核心经验。 |
| `recommendation` | 决策时可执行的建议。 |
| `anti_pattern` | 应避免的错误模式。 |
| `evidence` | 经验的来源，防止无根据经验无限累积。 |
| `confidence` | 根据重复证据、胜负结果、人审或反例调整。 |
| `status` | `active`, `candidate`, `deprecated`, `contradicted`。 |

## Knowledge Scopes

经验库应分层，避免 Ironclad 专属知识错误污染其他角色。

| Scope | Example | Retrieval rule |
|---|---|---|
| Global | “不要对 engine preview damage/block 二次应用 Strength/Dexterity。” | 所有角色都可检索。 |
| Character | “Ironclad 可以更积极利用 Burning Blood 的战后回复。” | 只给对应角色，或迁移实验中显式允许。 |
| Ascension band | “高 ascension 下低 HP 事件风险更高。” | 按难度区间检索。 |
| Encounter | “Tunneler burrowed block 不清除，破 block 可 stun。” | 当前敌人匹配时检索。 |
| Boss | “Lagavulin Matriarch 的 debuff 会让后期 block/damage 失效。” | 当前 boss 或 boss path planning 时检索。 |
| Card/relic | “Pael's Eye extra-turn 条件是未出牌 end turn，不是普通存能。” | 当前 deck/relic 或手牌匹配时检索。 |
| Archetype | “高 deck size 时优先高影响卡，避免继续膨胀。” | deck/relic/card pattern 匹配时检索。 |

## Retrieval Design

第一版不需要复杂向量库。建议从可解释的检索开始：

1. 从当前 state 构造 query terms：
   - character, ascension, act, floor, room type
   - enemies, boss, elite/event/shop/rest context
   - current HP band, deck size band
   - visible hand cards, relics, potions
   - current decision type
2. 对 JSONL entry 做 keyword/BM25-like scoring：
   - exact enemy/boss match 权重最高
   - character match 次之
   - card/relic match 次之
   - global entries 作为低权重常驻候选
3. 取 top K 条，按 token budget 压缩成 prompt section。

Prompt section 示例：

```text
Experience notes, advisory only. Current game state overrides these notes.
- [global/conf=0.8] Card damage/block values are engine previews; do not apply visible Strength/Dexterity/Frail/Weak again.
- [encounter/conf=0.5] Tunneler keeps Burrowed block between turns; if you cannot remove block to stun, prioritize survival over small damage.
- [character/conf=0.4] Ironclad can trade some HP in hallway fights because Burning Blood heals after combat, but do not enter boss fights below a safe HP threshold.
```

控制项：

| Config | Default | Meaning |
|---|---:|---|
| `experience_memory.enabled` | `false` | 是否启用经验库检索。 |
| `experience_memory.store_path` | `sts2_bench/experience_memory/` | 经验库路径。 |
| `experience_memory.top_k` | `5` | 每次最多注入几条经验。 |
| `experience_memory.token_budget` | `800` | 经验库 prompt 最大预算。 |
| `experience_memory.include_global` | `true` | 是否总是允许 global 条目。 |
| `experience_memory.allow_cross_character` | `false` | 是否允许跨角色检索。 |
| `experience_memory.allow_cross_ascension` | `true` | 是否允许同角色跨难度检索。 |

## Update Design

经验库更新应分两种模式。

### Offline Reflection

先从 offline reflection 开始，风险最低。

```text
completed run log
      |
      v
reflection prompt
      |
      v
candidate experience entries
      |
      v
dedupe / merge / confidence update
      |
      v
human-readable diff for review
```

优点：

- 不影响当前 run。
- 可以用完整 log 做更准确的总结。
- 方便人审查候选经验，避免错误知识直接进入库。

触发点：

- 每局结束后。
- 每次死亡后。
- 每个 boss/elite 结束后。
- 出现 fallback、invalid action、明显 tactical blunder 后。

### Online Update

在线更新是后续阶段，不建议第一版直接用于严肃评估。

适用场景：

- training runs。
- agent 自我改进演示。
- 观察是否能快速修正重复错误。

限制：

- evaluation mode 必须冻结经验库。
- 在线写入的 entry 初始状态应为 `candidate`，不能默认 `active`。
- 只有重复证据或人审后才提升 confidence。

## Text-Gradient Style Reflection Prompt

反思 prompt 不应泛泛而谈，而应要求模型输出可检索、可验证、可执行的经验。

示例输入：

```text
You are updating an experience library for future STS2 benchmark decisions.
Extract only lessons that are likely to help future runs.
Do not invent hidden mechanics.
Each lesson must cite evidence from the run.
Separate global lessons from character-specific lessons.
Mark uncertain lessons as low confidence.
```

示例输出 schema：

```json
{
  "new_entries": [
    {
      "scope": {"level": "encounter", "encounter": "Tunneler"},
      "trigger": "Tunneler is Burrowed with high block and attacking",
      "lesson": "Removing all block can stun Tunneler, but if the current hand cannot remove the block, survival should be prioritized.",
      "recommendation": "Compare total available block-break damage to enemy block before spending attacks. Use redraw or block if break is impossible.",
      "anti_pattern": "Spending all energy on small attacks that leave block intact while taking full attack damage.",
      "confidence": 0.3,
      "evidence_refs": ["run=bench_20260529_183627 step=178"]
    }
  ],
  "updates": [
    {
      "id": "exp_preview_values_0001",
      "confidence_delta": 0.1,
      "reason": "Another run showed the model double-counted visible modifiers."
    }
  ],
  "deprecated_entries": []
}
```

## Generalization Experiments

不需要立刻跑大规模实验，但计划上要把 train/eval 拆清楚。

### Stage A: Same Character, Same Difficulty

| Train | Eval | Purpose |
|---|---|---|
| Ironclad A0 seeds 0-4 | Ironclad A0 seeds 5-9 | 验证经验库能否改善同分布表现。 |

比较：

- baseline: no experience memory
- frozen library: train 后冻结经验库再 eval
- online: 边 eval 边更新，单独记录但不作为严肃结论

### Stage B: Same Character, Higher Difficulty

| Train | Eval | Purpose |
|---|---|---|
| Ironclad A0 | Ironclad A5/A10 | 验证同角色跨 ascension 迁移。 |

注意：

- A0 学到的高风险路线策略可能在高 ascension 下过于激进。
- 需要记录哪些 entry 在高难度下失效，并降低 confidence 或加 ascension scope。

### Stage C: Cross Character Transfer

| Train | Eval | Purpose |
|---|---|---|
| Ironclad A0 global-only entries | Silent/Defect/Regent/Necrobinder A0 | 验证全局机制知识能否迁移。 |
| Ironclad full library | Other characters A0 | 故意测试污染风险，不作为默认设置。 |

默认只允许 `global` 和明确角色无关的 encounter/mechanic entries 迁移。角色专属卡牌、遗物、HP tradeoff 经验默认不迁移。

## Formal Three-Study Experimental Outline (2026-07-20)

三项实验彼此独立，但按“产生经验 → 使用经验 → 迁移经验”的顺序共享数据和审计基础设施。任何一项失败都应产生独立的否定性结论，不应靠继续增加记忆系统复杂度来回避。

### Study 1 / RQ1: Long-Horizon Experience Induction

#### Hypothesis

整条超长轨迹的单次反思容易产生稀疏、过度概括或无法由原始证据支持的经验；局部分段能提高证据忠实度和决策覆盖，但可能丢失战斗、选牌、路线与整局结果之间的跨层关联。分段抽取加整局汇总应在两者之间取得更好平衡。

#### Conditions

| Arm | Extraction policy | Constraint |
|---|---|---|
| `E1 monolithic` | 把完整 run 轨迹一次性交给模型反思。 | 不预先按决策域分段。 |
| `E2 segmented-local` | 分别处理战斗、选牌和路线片段。 | 不提供整局后验汇总。 |
| `E3 hierarchical` | 先局部抽取，再用整局结果审计、合并和标注跨层冲突。 | 局部证据引用必须保留。 |

每个 arm 使用相同模型、完整卡牌/敌人信息、源轨迹、输出 schema 和最大候选数。在多条完整轨迹上重复抽取，不得只用一条 run 或一次模型采样下结论。

#### Primary Metrics

- `evidence_fidelity`: 经验的核心主张能否由它引用的原始 transition 直接支持。
- `hallucination_rate`: 误解卡牌、敌人意图、事件、伤害或资源变化的经验比例。
- `domain_coverage`: 战斗、卡牌价值、选牌和路线经验的覆盖与均衡度。
- `nontriviality`: 经验是否超出卡牌文本、当前状态的浅层推理或无条件常识。
- `cross-trajectory_confirmation`: 经验在不同轨迹中获得后续支持、反驳或仍无法判定的比例。
- `cross-level_conflict`: 局部有利但与整局构筑、路线或资源结果冲突的经验比例。

保留模型的全部原始候选，不允许为了结果好看而人工删除。人工或独立模型可以作为评估标注者，但标注与运行时自动筛选必须分开。

#### Success and Stop Criteria

- 成功不等于 `E3` 必须更好；只要三种方法在预注册指标上出现跨轨迹可重复的差异，就能形成 benchmark 分析结论。
- 若标注者之间无法就“证据忠实/平凡/幻觉”达到可接受一致性，先修订评估规则，不进入 RQ2。
- 若所有抽取方法都只产生浅层知识且无法后续验证，则把它作为经验抽取失败结论，而不继续增加复杂记忆模块。

### Study 2 / RQ2: Exposure Gating and Long-Horizon Performance

#### Hypothesis

经验暴露不是“越多越好”。宽门控可能提高召回但导致重复污染；精确门控可能降低污染但产生零暴露。门控特异性、暴露频率和单位上下文成本之间可能存在非单调的效用曲线。

#### Conditions

RQ2 必须冻结同一批已经 RQ1 证据审计的经验，只改变暴露策略：

| Arm | Exposure policy | Purpose |
|---|---|---|
| `G0 no-memory` | 不注入经验。 | Baseline。 |
| `G1 broad` | 只要决策域大致相关就注入。 | 测量高召回和重复污染。 |
| `G2 exact` | 要求敌人/卡牌/路线实体与状态条件精确匹配。 | 测量高特异性与零暴露风险。 |
| `G3 compiled` | 同门控经验先合并压缩，再按实体、状态和当前可执行动作暴露。 | 测量在不使用硬 `top-k` 截断时的中间方案。 |
| `G4 mismatched-dose-control` | 按照与正常门控相同的频率和长度注入错配经验。 | 区分经验内容效果和单纯增加 prompt 的效果。 |

#### Primary Metrics

- 每局和每个决策域的注入次数、零暴露率与触发覆盖。
- 每条经验的触发次数、连续重复率、单门控压缩率和实际适用率。
- 经验载荷字符/token 成本、时延、fallback 和格式失败。
- 经验注入后的动作改变率，以及改变后的即时与 run-level 结果。
- act/floor、通关率、死亡位置以及每千经验 token 的进度变化。

#### Identification Requirements

- 严格配对实验必须共享并冻结“首次暴露之前”的完整 agent/OCC 上下文，之后才分叉。
- 若使用独立模型采样，必须在多 seed、多重复上作分布级比较，不得把单个同 seed pair 标记为因果提升。
- 所有 arm 保留完整状态、卡牌、敌人和合法动作上下文；不允许为了加速而缩减信息。

#### Success and Stop Criteria

- 只有当不同门控在多 seed 上形成可重复的暴露—行为—结果差异时，才将 RQ2 写成正式 benchmark 结论。
- 若任一主要决策域始终零暴露，不允许声称已评估该决策域。
- 若处理前策略上下文已经分叉，严格配对因果分析终止，只保留明确标注的描述性分析。

### Study 3 / RQ3: Cross-Character Experience Transfer

#### Hypothesis

敌人机制、事件后果和地图资源等经验可能跨角色迁移；依赖特有卡牌、遗物、生命恢复和牌组节奏的经验若直接迁移，可能导致负迁移。显式区分通用经验与角色/牌组条件经验，可能改善迁移边界。

#### Transfer Scope Taxonomy

| Scope | Example | Default transfer policy |
|---|---|---|
| `universal_fact` | 敌人机制、事件已观测后果、地图结构。 | 允许迁移，但保留证据与不确定性。 |
| `universal_strategy` | 成长型敌人、多目标威胁、资源风险的一般处理。 | 待跨角色验证，不默认为真。 |
| `character_conditioned` | 依赖角色固有机制、恢复或资源的策略。 | 默认禁止跨角色。 |
| `deck_conditioned` | 依赖牌组费用、抽牌、遗物或某套构筑的经验。 | 只在可执行条件匹配时允许。 |
| `character_private` | 特有卡牌、遗物和套路知识。 | 禁止迁移。 |

#### Conditions

| Arm | Target-character memory | Purpose |
|---|---|---|
| `T0 no-memory` | 无经验。 | Target baseline。 |
| `T1 target-self` | 来自目标角色其他训练 seed 的冻结经验。 | 同角色参照上界。 |
| `T2 naive-cross-character` | 不区分 scope，直接注入源角色全部经验。 | 测量朴素迁移和负迁移。 |
| `T3 typed-cross-character` | 只迁移 `universal_*` 或条件真正匹配的经验。 | 测量作用域过滤的价值。 |
| `T4 mismatched-transfer` | 使用数量/长度匹配但与目标场景无关的迁移经验。 | 负对照。 |

源角色的经验生成 seed 与目标角色 evaluation seed 必须完全分离。整个实验必须使用 RQ2 选定后冻结的暴露策略，避免同时改变经验内容和门控。

#### Primary Metrics

- `transfer_exposure_rate`: 迁移经验在目标角色决策中实际触发的比例。
- `portable_validation_rate`: 迁移经验在目标轨迹中被支持、反驳或仍无法判定的比例。
- `negative_transfer_rate`: 迁移经验触发后产生明确不适用建议、有害动作或表现下降的比例。
- 不同 scope 的触发覆盖、动作改变率、act/floor、通关率和单位上下文成本。

#### Success and Stop Criteria

- RQ3 不要求跨角色迁移必须为正。若朴素迁移稳定导致负迁移，且能定位到具体 scope，仍然是有效 benchmark 分析结论。
- 只有当 `T3` 在多角色、多 seed 上相对 `T2` 减少负迁移或改善效用时，才能把“作用域类型化迁移”作为轻量方法候选。
- 若目标轨迹根本不遇到可迁移实体，则该 seed 不具有 treatment exposure，必须在运行前预检并更换，不得把零暴露当作迁移无效。

### Relationship to the 2026-07-20 Five-Pair Pilot

当前 `gate_compiled_memory_fullgame_v2` 五组同 seed 运行只作为 RQ2 的先导证据，不是上述任一正式实验的最终结果：

- 抽取链将一条完整轨迹分成 27 个片段，生成 39 条原始候选，结构验证保留 16 条；这只能证明已建立可审计抽取链，不能证明经验质量。
- 每局实际注入 44–102 次，说明投递链已经不再过严；但约四分之一的动作携带经验，且宽泛 `Strike` 经验占主导，提示重复污染问题。
- 路线经验每局只触发 1–3 次，选牌经验触发 0 次，因此不得宣称已验证战斗、选牌和路线三域。
- 描述性进度为 `2 better / 3 equal / 0 worse`，但五组在首次注入前的完整策略上下文已因模型随机采样分叉，故严格配对因果审计不通过。
- 该先导实验只证明“经验能够被真实注入并改变行为”，不证明反思改善表现。

### Execution Order and Paper Boundary

1. 先做 RQ1：使用现有完整轨迹离线运行，成本最低，并产生 RQ2 的冻结经验池。
2. 再做 RQ2：经验内容固定，只比较暴露策略，不把经验质量与门控效果混在一起。
3. 最后做 RQ3：使用 RQ2 冻结的暴露策略，只改变迁移内容和 scope 过滤。

三项实验本身是 benchmark 研究问题，不自动构成方法创新。只有当 RQ3 的类型化迁移在正式实验中稳定减少负迁移时，才考虑将其提升为轻量方法贡献；否则三者均只作为超长序列 benchmark 的系统分析。

## Evaluation Metrics

Run-level:

- act / floor reached
- win rate
- death floor
- wall time
- total prompt/completion tokens
- fallback count
- invalid state count
- view action rate

Experience-specific:

- retrieved entry count per decision
- prompt token overhead from experience notes
- entries used in model reason
- entries contradicted by final result
- new candidate entries per run
- active/deprecated/confidence distribution

Generalization:

- source-trained vs target-eval delta
- same-character high-ascension delta
- cross-character delta
- negative transfer cases

## Implementation Plan

### Phase 1: Read-only Experience Retrieval

- Add structured store loader for JSONL entries.
- Add query builder from compact state.
- Add deterministic keyword scorer.
- Inject top K experience notes into prompt.
- Log retrieved entry ids and token count.
- No writing yet.

### Phase 2: Offline Reflection Writer

- Add script that reads benchmark logs and proposes candidate entries.
- Write candidates to a review file, not directly into active store.
- Add dedupe/merge by normalized trigger + scope.
- Add human-readable Markdown report.

### Phase 3: Frozen Evaluation Mode

- Add config:

```yaml
experience_memory:
  enabled: true
  mode: frozen
  store_path: sts2_bench/experience_memory/active.jsonl
  top_k: 5
  token_budget: 800
  allow_cross_character: false
  allow_cross_ascension: true
```

- Refuse to write updates in `frozen` mode.
- Store `experience_memory.store_digest` in `run_config` for reproducibility.

### Phase 4: Online Training Mode

- Add config:

```yaml
experience_memory:
  enabled: true
  mode: online_train
  candidate_path: sts2_bench/experience_memory/candidates.jsonl
  update_on: ["run_end", "death", "boss_end", "fallback"]
```

- Write candidate entries with source run id.
- Keep them inactive until promoted.

### Phase 5: Promotion / Deprecation

- Promote entries after:
  - repeated evidence across seeds, or
  - manual review, or
  - positive eval correlation.
- Deprecate entries after:
  - repeated contradictions, or
  - scoped failure on higher ascension / other characters.

## Tests

Unit tests:

- JSONL entry parsing and validation.
- Scope matching for character / ascension / encounter.
- Retrieval scoring order.
- Token budget truncation.
- Prompt injection includes “advisory only”.
- `frozen` mode never writes.
- `online_train` writes only candidates.
- Store digest changes when active store changes.

Integration tests:

- Run benchmark with a tiny fake experience store and assert retrieved ids appear in action meta/logs.
- Verify `run_config` records experience memory settings and store digest.
- Verify `allow_cross_character=false` excludes character-specific entries.

## Risks

| Risk | Mitigation |
|---|---|
| 错误经验被反复检索，放大坏策略。 | 记录 evidence/confidence/status；默认 candidate 需审核或重复证据后激活。 |
| evaluation 污染。 | 严格区分 `online_train` 和 `frozen`；eval 时记录 store digest。 |
| 经验库 token 过大。 | top K + token budget + compact notes。 |
| 跨角色负迁移。 | 默认只迁移 global entries；角色专属经验需显式开启。 |
| 模型把经验当成当前事实。 | prompt 中明确 “current state overrides experience notes”。 |
| 难以归因提升来源。 | 单独做 read-only retrieval / online update / turn_chat 组合消融。 |

## First Small Experiment

不需要先跑大规模实验。可以先做一个小验证：

1. 从已有三条 log 手工或 offline reflection 生成 5-10 条 candidate experience entries。
2. 只启用 read-only retrieval，不启用更新。
3. 使用同一个 seed 做 smoke rerun，观察：
   - prompt 是否正确注入经验；
   - 模型 reason 是否引用经验；
   - fallback/JSON 格式是否变差；
   - token overhead 是否可接受。
4. 再换一个未见 seed 做一次 sanity check，避免只记住 `bench_0000`。

第一批高价值经验候选：

- 当前 state 中展示的 card damage/block 和 enemy intent 是 engine preview，不要二次应用可见 modifiers。
- 一次只执行一个 atomic action；不要依赖后续计划，除非当前 action 本身创造出必要状态。
- Tunneler 的 Burrowed block 不会自动清除；若无法破 block stun，应优先生存或使用 redraw。
- Lagavulin Matriarch 的 Strength/Dexterity debuff 会让后期防御和输出急剧变差；需要更早建立输出/防御路线。
- view actions 不推进游戏；只有在信息能改变当前决策时才使用。
