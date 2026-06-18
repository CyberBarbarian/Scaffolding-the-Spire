# DeepSeek 选牌偏差评测报告（run1）

## 实验配置

- 评测入口：`python -m sts2_bench.card_choice_bias_eval`
- Agent：`coordinator`
- 模型：`deepseek-v4-flash`
- 温度：`0`
- 严格模式：`llm_fallback_mode=error`
- 日志级别：`decisions`
- 结果文件：`results/card_choice_bias_deepseek_v4_flash_20260618_run1.jsonl`

这个评测不是游戏胜率 benchmark，而是离线合成诊断。每题都是 `card_reward`，都有 3 张候选牌和 `skip card reward`，并走真实 harness 路径：`state -> build_legal_actions() -> build_evidence_bundle() -> agent.choose()`。隐藏 oracle 只用于事后评分，不进入 prompt、evidence 或 agent meta。

## 题集结构

本轮新增 bias suite 一共 42 题：

| 类别 | 题数 | 测什么 |
| --- | ---: | --- |
| `attack_temptation_ladder` | 8 | 面对不同强度攻击诱惑时，是否会忽略更合适的防御/工具牌 |
| `skip_threshold_matrix` | 9 | 中高质量牌组或明确思路牌组里，是否愿意跳过低价值 offer |
| `redundancy_dose_response` | 10 | 面对已有 0-4 张同类/同名牌时，是否能随冗余增加而提高跳过或转选其他角色的倾向 |
| `rare_damage_trap` | 6 | 面对 `Bludgeon` 这类强稀有攻击时，是否会压过更契合当前牌组的牌 |
| `energy_vs_damage_curve` | 5 | 高费曲线是否选能量，低费曲线是否选终结手段，已有能量时是否避免继续拿 HP 换能量 |
| `havoc_semantic_trap` | 4 | 是否理解 `Havoc` 的随机顶牌并消耗语义，而不是只看到“消耗协同”就拿 |

## 总体结果

| 指标 | 结果 |
| --- | ---: |
| 总题数 | 42 |
| 正确 | 24 |
| 错误 | 18 |
| invalid/error | 0 / 0 |
| 总正确率 | 57.1% |

分类正确率：

| 类别 | 正确率 |
| --- | ---: |
| `energy_vs_damage_curve` | 5/5 = 100.0% |
| `havoc_semantic_trap` | 4/4 = 100.0% |
| `attack_temptation_ladder` | 5/8 = 62.5% |
| `rare_damage_trap` | 3/6 = 50.0% |
| `redundancy_dose_response` | 4/10 = 40.0% |
| `skip_threshold_matrix` | 3/9 = 33.3% |

## 偏差指标

### Skip aversion

- 期待 skip 的题：13
- 实际选择 skip：3
- skip recall：3/13 = 23.1%
- skip precision：3/3 = 100.0%

这说明模型不是乱 skip。相反，它一旦 skip 基本是对的，但明显不愿意 skip。10 道应该跳过的题里，它选择了拿牌。

典型错误：

- `bias_sk_002`：高质量大牌组，offer 是 `Strike / Defend / Havoc`，模型选 `Havoc`。理由是 Havoc 可以“消耗基础牌”，但这里核心问题是没有必要继续污染牌组。
- `bias_sk_009`：已有完整消耗体系，offer 是 `Havoc / Strike / Defend`，oracle 期待 skip；模型选 `Havoc`，理由是它和 `Dark Embrace / Feel No Pain / Burning Pact / Second Wind` 有协同。这暴露了“看到协同词就拿”的倾向，没有充分比较边际收益和牌组厚度。

### 攻击牌倾向

- 有攻击 offer 的题：38
- 实际选攻击：17，选择率 44.7%
- `attack_trap` 题：14
- `attack_trap` 中实际选攻击：5，选择率 35.7%
- `rare_damage_trap` 中选稀有攻击：3/6 = 50.0%

攻击倾向是存在的，但不是全局失控。模型在普通攻击诱惑里有 5/8 正确；真正明显的问题集中在“强攻击/稀有攻击”上。

典型错误：

- `bias_rt_001`：应选 `Shrug It Off`，模型选 `Bludgeon`，理由是“高爆发能缩短战斗”。这正是我们想验证的攻杀叙事。
- `bias_rt_006`：应选 `True Grit`，模型仍选 `Bludgeon`，理由是和 `Bash` 易伤配合、适合 Act 1 精英/Boss。
- `bias_at_006`：应选 `Stone Armor`，模型选 `Pommel Strike`，理由是低费攻击加抽牌提升一致性和伤害。

### 冗余敏感性

- 有明确冗余 offer 的题：10
- 实际选择冗余 offer：5
- 冗余选择率：50.0%

按已有同名/同类数量拆分：

| 已有冗余数 | 选择冗余 offer |
| ---: | ---: |
| 0 | 1/1 |
| 1 | 1/2 |
| 2 | 1/3 |
| 3 | 1/2 |
| 4 | 1/2 |

这个曲线不够单调。已有 3-4 张时，模型仍会拿重复攻击或重复功能牌。

典型错误：

- `bias_rd_010`：已有 4 张 `Twin Strike`，offer 里还有 `Twin Strike / Strike / Havoc`，oracle 期待 skip；模型选第 5 张 `Twin Strike`，理由是 Act 1 伤害好，甚至承认已有 4 张但仍说“adding a 5th is acceptable”。
- `bias_rd_008` / `bias_rd_009`：已有多张 `Pommel Strike` 后仍继续选 `Pommel Strike`。

### 能量曲线与 Havoc 语义

这两类表现很好：

- `energy_vs_damage_curve`：5/5 正确。高费曲线里能选 `Bloodletting` / `Offering`，低费曲线里能选 `Bludgeon`，已有能量和 HP 压力时能转选 `Shrug It Off`。
- `havoc_semantic_trap`：4/4 正确。无消耗支持时没有拿 `Havoc`，有完整消耗支持时能接受 `Havoc`。

这说明模型不是完全不会读上下文。它能处理清晰的资源曲线和语义陷阱；主要弱点更像是 skip 阈值、边际收益和冗余控制。

## 一个额外异常：信息动作

`bias_at_004` 选择了 `view map and current position`。这是 legal action，不是 invalid，但它没有完成本题的“立即选牌/跳过”目标。当前评分把它记为 incorrect。

这提醒我们后续可以给离线选牌评测增加一个单独指标，例如 `deferred_decision_rate`，把“选择 view deck/view map 等信息动作”从普通错误里拆出来。真实 harness 仍应保留信息动作，但离线 reward-choice 诊断可以单独标记它。

## 初步结论

这次结果支持两个判断：

1. DeepSeek 的 skip aversion 很明显。它不是不知道 skip 合法，而是不太愿意在“offer 有一点点理由可拿”的时候跳过。
2. 攻击倾向存在，但更集中在 `Bludgeon` / 高爆发 / Act 1 伤害叙事上，而不是所有攻击牌都会无脑拿。

因此我不建议先用提示词工程强行修正 DeepSeek 的策略。更干净的下一步是把这个 suite 作为模型诊断工具，横向比较不同模型、不同 evidence 版本、是否隐藏/展示完整 deck、是否允许信息动作，并观察偏差指标是否稳定。
