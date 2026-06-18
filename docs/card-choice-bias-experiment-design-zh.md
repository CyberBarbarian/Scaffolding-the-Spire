# LLM 选牌偏置控制实验设计

本文档设计一组新的离线 synthetic card reward 实验，用于确认模型是否存在稳定的“攻杀倾向”和“低 skip 意愿”。它不是实现计划，也不修改 prompt；它是待审阅的实验方案。

## 1. 背景

上一轮 `deepseek-v4-flash` 在 24 个综合选牌场景上的结果是：

- 总体准确率：13 / 24，54.17%。
- `skip_recall=16.67%`，6 个应 skip 的场景只跳过 1 个。
- `redundancy=0/3`，所有冗余/污染场景都没有 skip。
- 多个错题选择了 `Bludgeon`、`Anger`、`Twin Strike`、`Pommel Strike`、`Havoc`。
- 运行链路干净：0 invalid action，0 runtime/model error。

这些结果支持一个初步怀疑：模型不是看不到合法动作，也不是无法输出 JSON，而是策略先验偏向“拿一张看起来有用的攻击/高伤害牌”，并且 skip 阈值过高。

但 24 题综合 suite 还不足以把这个判断写死。下一步应该做控制变量实验。

## 2. 实验目标

本实验验证三条假设：

1. **Attack bias**：当正确动作是补防御、补能量或 skip 时，模型是否仍高频选择攻击牌。
2. **Skip aversion**：当奖励牌低协同、重复或低于当前牌组质量线时，模型是否仍倾向拿牌。
3. **Redundancy blindness**：当同类功能牌已经足够时，模型是否无法随重复度上升而降低拿牌倾向。

如果这三条成立，那么问题更像模型原始选牌策略偏置，而不是 evidence 层或 legal action 暴露问题。

## 3. 设计原则

- 不改 prompt。
- 不加入“不要贪攻击”“skip 很重要”等策略提示。
- 不引入启发式评分器、监督模型或投票器。
- 继续使用当前真实 harness 路径：

```text
synthetic card_reward state
  -> build_legal_actions()
  -> build_evidence_bundle()
  -> coordinator/OCC LLM
  -> hidden oracle scoring
```

- 所有题都暴露 `skip card reward`。
- oracle 只用于事后评分，不进入 state、evidence 或 prompt。
- 所有卡牌必须存在于本地 `localization_eng/cards.json`。
- 场景中避免使用当前已发现不可靠的旧 STS1 名称：`Cleave`、`Flex`、`Heavy Blade`。
- 第一版以 DeepSeek 为主要诊断对象，但格式应支持横向比较其它模型。

## 4. 总体方案

新增一个独立 suite，建议命名为 `card_choice_bias_eval`。第一版 42 题，分成 6 组：

| 组别 | 题数 | 主要测什么 |
| --- | ---: | --- |
| Attack Temptation Ladder | 8 | 攻击诱饵强度上升时，模型是否放弃正确防御/skip |
| Skip Threshold Matrix | 9 | 牌组质量和大小变化时，模型何时愿意 skip |
| Redundancy Dose Response | 10 | 同类牌重复度上升时，模型是否降低拿牌概率 |
| Rare Damage Trap | 6 | `Bludgeon` 等 high-damage rare 是否过度吸引模型 |
| Energy vs Damage Curve | 5 | 模型是否真正理解费用曲线，而不是固定偏好伤害 |
| Havoc Semantic Trap | 4 | 模型是否误解 `Havoc` 的随机性和 off-plan 风险 |

第一版重点不是覆盖所有选牌能力，而是确认偏置是否稳定存在。

## 5. 实验组 A：Attack Temptation Ladder

目的：固定牌组与生存压力，只改变攻击诱饵强度，观察模型何时从正确防御牌转向攻击。

基础牌组模板：

- `Strike x4`
- `Defend x4`
- `Bash`
- `Pommel Strike`
- `True Grit`
- `Blood Wall`

该牌组攻击略多，但不是极端缺防。最近战斗反馈为“上一战 5 回合掉 18-22 HP”。

| ID | Offer | Hidden acceptable | 设计意图 |
| --- | --- | --- | --- |
| `bias_attack_001` | `Twin Strike`, `Shrug It Off`, `Havoc` | `Shrug It Off` | 低强度攻击诱饵。 |
| `bias_attack_002` | `Pommel Strike`, `Shrug It Off`, `Havoc` | `Shrug It Off` | 攻击带 draw，看是否仍能补防。 |
| `bias_attack_003` | `Anger`, `Shrug It Off`, `Havoc` | `Shrug It Off` | 0 费攻击诱饵。 |
| `bias_attack_004` | `Bludgeon`, `Shrug It Off`, `Havoc` | `Shrug It Off` | high-damage rare 诱饵。 |
| `bias_attack_005` | `Twin Strike`, `Stone Armor`, `Cinder` | `Stone Armor` | 防御牌换成 Plating，测试防御类型变化。 |
| `bias_attack_006` | `Pommel Strike`, `Stone Armor`, `Cinder` | `Stone Armor` | 攻击+抽牌对 Plating 的诱惑。 |
| `bias_attack_007` | `Anger`, `Stone Armor`, `Cinder` | `Stone Armor` | 0 费攻击对 Plating 的诱惑。 |
| `bias_attack_008` | `Bludgeon`, `Stone Armor`, `Cinder` | `Stone Armor` | rare damage 对 Plating 的诱惑。 |

关键指标：

- `attack_pick_rate`
- `attack_overpick_rate`
- `defense_pick_rate_by_attack_strength`

预期判读：

- 如果 `Bludgeon` / `Anger` 题明显错得更多，说明 attack strength 对模型选择有强影响。
- 如果 8 题中攻击诱饵被选超过 4 次，可以初步确认 attack bias。

## 6. 实验组 B：Skip Threshold Matrix

目的：确认模型是否只有在奖励极差时才 skip，还是能在“中等可用但不该拿”的情况下 skip。

变量：

- 牌组大小：small / medium / large。
- 牌组质量：basic / improved / coherent。
- Offer 质量：统一为低协同、重复或 off-plan，但不是完全不可解释。

| ID | Deck type | Offer | Hidden acceptable | 设计意图 |
| --- | --- | --- | --- | --- |
| `bias_skip_001` | small-basic | `Strike`, `Defend`, `Havoc` | `Skip` | 低价值基础牌测试。 |
| `bias_skip_002` | small-improved | `Strike`, `Defend`, `Cinder` | `Skip` | 小牌组但已有核心改良。 |
| `bias_skip_003` | small-coherent | `Strike`, `Havoc`, `Cinder` | `Skip` | 小牌组已有 block-plan，奖励 off-plan。 |
| `bias_skip_004` | medium-basic | `Strike`, `Defend`, `Havoc` | `Skip` | 中等牌组的基础牌污染。 |
| `bias_skip_005` | medium-improved | `Strike`, `Defend`, `Cinder` | `Skip` | 中等高质量牌组的质量线测试。 |
| `bias_skip_006` | medium-coherent | `Strike`, `Havoc`, `Anger` | `Skip` | 有明确思路时拒绝诱惑牌。 |
| `bias_skip_007` | large-basic | `Strike`, `Defend`, `Havoc` | `Skip` | 大牌组的继续污染测试。 |
| `bias_skip_008` | large-improved | `Strike`, `Defend`, `Cinder` | `Skip` | 大而质量较高时应更严格。 |
| `bias_skip_009` | large-coherent | `Strike`, `Havoc`, `Anger` | `Skip` | 大且成型时最应该 skip。 |

建议 deck 模板：

- `small-basic`：`Strike x4`, `Defend x4`, `Bash`, `Pommel Strike`
- `small-improved`：`Strike x4`, `Defend x4`, `Bash`, `Shrug It Off`, `Stone Armor`, `Bloodletting`
- `small-coherent`：`Strike x4`, `Defend x4`, `Bash`, `Barricade`, `Blood Wall`, `True Grit`
- `medium-basic`：small-basic 加 `Twin Strike`, `Anger`, `Pommel Strike`
- `medium-improved`：small-improved 加 `Offering`, `Bludgeon`, `Armaments`
- `medium-coherent`：small-coherent 加 `Impervious`, `Armaments`, `Entrench`
- `large-basic`：medium-basic 加重复 `Strike`, `Defend`, `Twin Strike`
- `large-improved`：medium-improved 加重复 `Shrug It Off`, `Stone Armor`, `Pommel Strike`
- `large-coherent`：medium-coherent 加重复 `Blood Wall`, `True Grit`, `Impervious`

关键指标：

- `skip_recall`
- `take_when_skip_best_rate`
- `skip_rate_by_deck_size`
- `skip_rate_by_deck_quality`

预期判读：

- 如果牌组越大、质量越高，skip rate 没有上升，说明模型没有学到质量线。
- 如果 coherent deck 仍频繁拿 `Havoc` / `Anger`，说明它忽略 deck plan。

## 7. 实验组 C：Redundancy Dose Response

目的：确认模型是否能随着重复牌数量上升而降低拿牌概率。

设计方式：固定 offer 中有一张“单看不错”的牌，逐步增加 deck 中同类角色牌数量。

### C1：攻击循环牌重复

Offer 固定：`Pommel Strike`, `Havoc`, `Strike`

| ID | Deck 中已有攻击循环密度 | Hidden acceptable |
| --- | --- | --- |
| `bias_redundancy_001` | 0 张 `Pommel Strike`，攻击不足 | `Pommel Strike` |
| `bias_redundancy_002` | 1 张 `Pommel Strike` | `Pommel Strike` / `Skip` |
| `bias_redundancy_003` | 2 张 `Pommel Strike` | `Skip` |
| `bias_redundancy_004` | 3 张 `Pommel Strike` | `Skip` |
| `bias_redundancy_005` | 4 张 `Pommel Strike` | `Skip` |

### C2：防御抽牌重复

Offer 固定：`Shrug It Off`, `Twin Strike`, `Havoc`

| ID | Deck 中已有防御抽牌密度 | Hidden acceptable |
| --- | --- | --- |
| `bias_redundancy_006` | 0 张 `Shrug It Off`，防御不足 | `Shrug It Off` |
| `bias_redundancy_007` | 1 张 `Shrug It Off` | `Shrug It Off` / `Skip` |
| `bias_redundancy_008` | 2 张 `Shrug It Off` + `Stone Armor` | `Skip` |
| `bias_redundancy_009` | 3 张 `Shrug It Off` + `Stone Armor x2` | `Skip` |
| `bias_redundancy_010` | 4 张 `Shrug It Off` + `Stone Armor x2` | `Skip` |

关键指标：

- `redundant_take_rate_by_count`
- `dose_response_slope`
- `skip_rate_after_threshold`

预期判读：

- 如果 0/1 张时拿、3/4 张时 skip，说明模型能感知冗余。
- 如果 3/4 张仍拿，说明存在 redundancy blindness。

## 8. 实验组 D：Rare Damage Trap

目的：隔离 `Bludgeon` 这类高伤害 rare 对模型的吸引力。

| ID | Deck context | Offer | Hidden acceptable | 设计意图 |
| --- | --- | --- | --- | --- |
| `bias_rare_001` | 防御略缺，已有足够攻击 | `Bludgeon`, `Shrug It Off`, `Havoc` | `Shrug It Off` | rare damage vs block-draw。 |
| `bias_rare_002` | 高费牌已多 | `Bludgeon`, `Bloodletting`, `Stone Armor` | `Bloodletting` | rare damage vs energy patch。 |
| `bias_rare_003` | block-plan 有多张未升级牌 | `Bludgeon`, `Armaments`, `Cinder` | `Armaments` | rare damage vs upgrade support。 |
| `bias_rare_004` | 大而成型高质量牌组 | `Bludgeon`, `Strike`, `Havoc` | `Skip` | rare damage vs quality line。 |
| `bias_rare_005` | 低曲线缺伤害 payoff | `Bludgeon`, `True Grit`, `Havoc` | `Bludgeon` | 正例：rare damage 确实该拿。 |
| `bias_rare_006` | 已有 `Bludgeon x2`，曲线很重 | `Bludgeon`, `Pommel Strike`, `Havoc` | `Pommel Strike` | 第三张高费伤害诱饵。 |

关键指标：

- `rare_attack_pick_rate`
- `rare_attack_trap_rate`
- `rare_attack_context_sensitivity`

预期判读：

- 如果 `bias_rare_005` 对，而其它 trap 也拿 `Bludgeon`，说明它不是不会识别好牌，而是对 rare damage 缺乏上下文抑制。

## 9. 实验组 E：Energy vs Damage Curve

目的：确认模型是否真的理解费用曲线。

| ID | Deck context | Offer | Hidden acceptable | 设计意图 |
| --- | --- | --- | --- | --- |
| `bias_curve_001` | `Bludgeon`, `Barricade`, `Impervious` 已在 deck | `Bloodletting`, `Bludgeon`, `Havoc` | `Bloodletting` | 高费过载，需要能量。 |
| `bias_curve_002` | 高费牌多且 HP 健康 | `Offering`, `Bludgeon`, `Stone Armor` | `Offering` | HP 可支付时，能量+抽牌优于伤害。 |
| `bias_curve_003` | 高费牌多但 HP 很低 | `Offering`, `Stone Armor`, `Havoc` | `Stone Armor` | 低 HP 时不应盲拿 HP-cost 能量。 |
| `bias_curve_004` | 低曲线、缺 payoff | `Bludgeon`, `Bloodletting`, `True Grit` | `Bludgeon` | 正例：高费伤害合理。 |
| `bias_curve_005` | 已有 `Bloodletting x2`, `Offering` | `Bloodletting`, `Pommel Strike`, `Havoc` | `Pommel Strike` / `Skip` | 能量牌冗余。 |

关键指标：

- `curve_sensitive_accuracy`
- `energy_pick_when_high_curve`
- `damage_pick_when_low_curve`
- `hp_cost_overpick_rate`

预期判读：

- 如果高费过载场景仍拿 `Bludgeon`，说明伤害偏置压过费用曲线。
- 如果低 HP 仍拿 `Offering`，说明模型没有把 HP cost 纳入上下文。

## 10. 实验组 F：Havoc Semantic Trap

目的：确认模型是否误把 `Havoc` 当作稳定 deck thinning / draw manipulation。

| ID | Deck context | Offer | Hidden acceptable | 设计意图 |
| --- | --- | --- | --- | --- |
| `bias_havoc_001` | 无 exhaust payoff，关键牌多 | `Havoc`, `Shrug It Off`, `Strike` | `Shrug It Off` | 无支撑时不该拿。 |
| `bias_havoc_002` | 无 exhaust payoff，大而高质量 | `Havoc`, `Strike`, `Defend` | `Skip` | `Havoc` 不是自动 thinning。 |
| `bias_havoc_003` | 有 `Feel No Pain`, `Dark Embrace` | `Havoc`, `Twin Strike`, `Defend` | `Havoc` | 正例：有 exhaust payoff 时可拿。 |
| `bias_havoc_004` | 有 exhaust payoff 但 HP/防御压力大 | `Havoc`, `Stone Armor`, `Cinder` | `Stone Armor` / `Havoc` | 允许多解，测试权衡。 |

关键指标：

- `havoc_pick_rate_without_support`
- `havoc_pick_rate_with_support`
- `havoc_context_sensitivity`

预期判读：

- 如果无支撑场景仍高频选 `Havoc`，说明模型对卡牌语义存在系统性误解。

## 11. 运行设计

建议每个模型至少跑 3 次：

```powershell
python -m sts2_bench.card_choice_bias_eval `
  --agent coordinator `
  --model deepseek-v4-flash `
  --log-level decisions `
  --out results/card_choice_bias_deepseek_v4_flash_run1.jsonl
```

然后重复 `run2`、`run3`。如果成本可接受，再对错题类别用 `--log-level full` 复跑。

温度固定为 `0`，但仍按多次运行处理，因为远程模型和服务端推理可能有输出抖动。

## 12. 指标汇总

建议 summary 输出新增这些字段：

| 指标 | 含义 |
| --- | --- |
| `accuracy` | 总体 hidden oracle 命中率 |
| `attack_pick_rate` | 模型选择攻击牌的比例 |
| `attack_overpick_rate` | 正确答案不是攻击时，模型仍选攻击的比例 |
| `skip_recall` | 应该 skip 的场景中实际 skip 的比例 |
| `skip_precision` | 模型 skip 时确实应该 skip 的比例 |
| `take_when_skip_best_rate` | 应 skip 却拿牌的比例 |
| `redundant_take_rate_by_count` | 重复度 0/1/2/3/4 下的拿牌比例 |
| `rare_attack_trap_rate` | high-damage rare 诱饵错误率 |
| `curve_sensitive_accuracy` | 费用曲线题准确率 |
| `havoc_overpick_rate` | 无支撑时错选 `Havoc` 的比例 |
| `support_sensitive_accuracy` | 有/无支撑对照题中选择是否随上下文变化 |

## 13. 判定标准

如果 DeepSeek 在 3 次重复运行中满足多数条件，可以确认存在稳定偏置：

- `attack_overpick_rate > 50%`
- `skip_recall < 40%`
- `take_when_skip_best_rate > 60%`
- `redundant_take_rate_by_count` 在重复度上升时没有明显下降
- `rare_attack_trap_rate > 40%`
- `havoc_pick_rate_without_support > 50%`
- `curve_sensitive_accuracy < 60%`

如果只在单次运行满足，不能下强结论，只能说“有倾向信号”。

## 14. 结果解释边界

这个实验只能证明模型在当前 evidence-first harness 和 synthetic card reward 任务中的行为倾向。它不能直接证明真实游戏胜率，也不能证明某个 prompt 改法有效。

它可以支持的结论包括：

- 模型是否有稳定攻击倾向。
- 模型是否不愿意 skip。
- 模型是否不能识别冗余。
- 模型是否误解某些卡牌语义。
- 当前 evidence 是否足以暴露这些问题。

它不应该用于：

- 直接训练模型。
- 反向调 prompt 到过拟合这些题。
- 替代真实游戏胜率 benchmark。
- 把某个单次输出当作模型稳定能力结论。

## 15. 推荐执行顺序

1. 先实现 `card_choice_bias_eval` 的数据集和 runner，不改 prompt。
2. 用 `random` agent 做 smoke，确认 42 题都有 3 pick + 1 skip。
3. 跑 `deepseek-v4-flash` 3 次，`log-level=decisions`。
4. 对错误集中的组别用 `log-level=full` 复跑一次。
5. 生成中文报告，按组展示均值、方差、典型错题和 reason 模式。
6. 再选 1-2 个其它模型横向比较。

## 16. 审阅问题

需要确认的主要问题：

1. 第一版 42 题是否太多，是否先缩到 24-30 题。
2. `Skip Threshold Matrix` 是否应该占更大比例。
3. 是否要把 `Havoc Semantic Trap` 独立成单独小实验，而不是放进 bias suite。
4. `Redundancy Dose Response` 中是否允许 1 张重复时既接受 pick 也接受 skip。
5. 是否先只测 DeepSeek，还是第一轮就加入一个对照模型。

我的建议是：第一版保留 42 题，但运行时先支持 `--category`，这样可以先跑 `skip_threshold` 和 `redundancy_dose` 两组，快速确认最核心的偏置。
