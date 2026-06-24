# 当前代码库改动总结

日期：2026-06-24

本文简要总结当前工作区尚未提交的改动，主题集中在 STS2 benchmark agent 侧的上下文管理、路线规划、诊断方法和局内出牌控制。

## 改动范围

当前改动主要发生在 `sts2_bench/` 下的 agent benchmark 框架，不涉及 STS2 游戏引擎本身。整体目标是让不同决策上下文方法更容易做对比实验：

- 局内 `turn_chat` 加轻量状态检查；
- route plan 支持关闭、软校验和强校验；
- self-consistency 作为可选诊断方法；
- reflection verifier 作为可选诊断/覆盖方法；
- combat 可以在“多卡 sequence”和“每次决策只打一张牌”之间切换。

## 主要改动

### 1. Combat 多卡 Sequence 开关

新增局内多卡 sequence 的配置开关。

- CLI：`--combat-card-sequences` / `--no-combat-card-sequences`
- 配置：`context_management.combat.card_sequences.enabled`

关闭后：

- `combat_play` 的 legal actions 不再暴露 `combat_card_sequence` 和 `combat_custom_sequence`；
- prompt 中也会移除 sequence schema；
- 模型每次只能选择一张当前合法牌、一个药水、一个 view 动作或 `end turn`。

这个改动用于验证：如果每打一张牌后都重新读取最新状态，模型是否能减少对后续抽牌、随机效果和能量变化的错误假设。

### 2. 从 Turn Plan 改为 Turn Check

原先探索过“局内先做多步 `turn_plan`”，现在改为更轻量的 `turn_check`。

`turn_check` 不是多步脚本，而是当前动作前的状态检查。它只要求模型简短检查：

- 本回合是否需要优先生存；
- 是否已有斩杀；
- 当前第一步动作是否安全；
- 打出当前动作后是否必须重新计算。

后续同一回合的每一步都必须基于最新 state 重新判断，不能自动 follow 旧的 `turn_check`。

这样更符合 STS2 的局内特点：每打一张牌，手牌、抽牌堆、弃牌堆、能量、随机生成牌、敌人状态都可能变化。

### 3. Route Plan 拆成两层控制

路线规划现在拆成两个相对独立的层次。

第一层是 `map_select` 的显式 route plan event：

```yaml
context_management:
  route_plan:
    enabled: false
    enforcement: soft
```

- `enabled` 控制是否要求模型输出 `route_plan_event`；
- `enforcement: hard` 会在路线计划不一致时重问；
- `enforcement: soft` 只记录错误，不阻塞当前动作。

第二层是 OCC 的软性长期路线字段：

```yaml
context_management:
  run_plan:
    route_plan:
      enabled: true
```

它控制 `RunPlan.route_plan` 是否展示在 prompt 中，以及是否接受 `run_plan_patch.route_plan` 更新。

这样可以区分两件事：

- 强约束的路线校验是否真的有帮助；
- 软性的长期路线叙事是否能帮助模型保持策略一致性。

### 4. Self-Consistency 诊断

新增 self-consistency 采样配置：

```yaml
diagnostics:
  self_consistency:
    enabled: false
    samples: 3
    decisions: map_select,card_reward
    mode: diagnostic
```

支持两种模式：

- `diagnostic`：只重复采样并记录投票分布，不改变实际动作；
- `majority_vote`：在允许的非战斗决策上，用多数票结果覆盖 primary action。

为了避免局内动作被破坏，combat action 即使误配进 self-consistency，也不会被 majority vote 覆盖。

这个功能主要用于观察模型是否对某些选择有稳定偏好，比如总是走精英、总是抓牌、总是选择某类路线。

### 5. Reflection Verifier

新增 reflection verifier，用于与 self-consistency 并行探索。

当前设计是“平等评估所有候选项”：

1. 对每个非 view 候选 action 单独生成一份 forced counterfactual report；
2. 每份报告都假设“请选择这个候选项，并说明理由”；
3. verifier 只基于这些候选报告做横向比较；
4. `diagnostic` 模式只记录 verifier 的判断；
5. `verifier_vote` 模式在置信度和分数门槛通过后，才允许覆盖 primary action。

当前默认只建议先用于 `card_reward`，因为更宽范围的 reflection 在实验中成本较高，而且容易把“牌组更瘦”误当作唯一目标。

### 6. Trace 与可视化分析增强

runner 现在会在 compact state 之外额外记录 `enemy_snapshot`，方便后续分析敌人名称、意图和状态，而不是只看到 `enemy_count`。

trace analyzer 当前可以生成：

- `hp_by_action.svg`
- `hp_by_room.svg`
- `map_path.svg`
- `route_hp_replay.svg`
- 中文终局牌组与选牌报告；
- 大失血房间的截取日志。

这些产物用于复盘模型在每个阶段的路线、血量变化、选牌和局内操作风格。

## 当前默认 Baseline

当前 `benchmark.yaml` 中，大多数诊断方法默认关闭，只保留轻量上下文方法：

```yaml
context_management:
  route_plan:
    enabled: false
    enforcement: soft
  run_plan:
    route_plan:
      enabled: true
  turn_chat:
    turn_check:
      enabled: true
  combat:
    card_sequences:
      enabled: true

diagnostics:
  self_consistency:
    enabled: false
  reflection:
    enabled: false

reward_posterior_evaluator: false
```

如果要跑“每次决策只打一张牌”的局内实验，需要额外加：

```bash
--no-combat-card-sequences
```

## 最近一次 One-Card 实验结论

最近一次关闭 combat sequence 的实验结果：

- 没有死亡；
- 因 `max_steps=500` 截断；
- 截断位置：Act 3 floor 12 普通战；
- 血量：66/97；
- 牌组大小：43；
- `sequence_actions=0`；
- `validation_errors=0`；
- `model_fallback_count=0`。

主要观察：

- 单卡模式让局内行为更容易审计；
- 每打一张牌后重新读状态，减少了多卡 sequence 中“提前假设后续状态”的问题；
- 模型围绕 `Offering / Dark Embrace / Feel No Pain / Second Wind` 形成了明确的 exhaust/draw engine；
- 但模型仍然非常偏爱 `Offering`，低血量时也容易过度相信付血抽牌；
- 非战斗选牌仍然偏积极抓牌，牌组容易变厚。

## 测试状态

之前已跑过聚焦的 actions/config 测试：

```bash
PYTHONPATH=. pytest -q tests/test_sts2_bench_actions.py tests/test_sts2_bench_config.py
```

结果：

```text
80 passed
```

但在当前全部改动完成后，还没有重新跑完整测试集。

## 当前涉及文件

主要实现文件：

- `sts2_bench/actions.py`
- `sts2_bench/agents.py`
- `sts2_bench/runner.py`
- `sts2_bench/run_benchmark.py`
- `sts2_bench/run_context.py`
- `sts2_bench/multi_agent.py`
- `sts2_bench/benchmark.yaml`
- `sts2_bench/trace_analysis.py`

测试文件：

- `tests/test_sts2_bench_actions.py`
- `tests/test_sts2_bench_config.py`
- `tests/test_sts2_bench_multi_agent.py`
- `tests/test_route_plan.py`
- `tests/test_trace_analysis.py`

设计文档：

- `docs/combat-chat-context-plan.md`

