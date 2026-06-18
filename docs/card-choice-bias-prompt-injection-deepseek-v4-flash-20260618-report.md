# DeepSeek 选牌提示词注入对比实验

## 实验目的

这次实验只验证一件事：提示词是否真的能改变 LLM 的选牌行为。

我们做了三组对比：

1. `baseline`：不注入额外选牌提示词。
2. `generic`：注入通用选牌原则，例如 card reward 可以 skip、比较边际效益、注意大牌组稀释、重复牌、曲线、HP cost 和协同支持。
3. `scenario-specific`：注入由当前可见 state/legal actions 自动生成的局部检查清单，例如当前牌组大小、攻击/防御/能量/高费牌数量、offer 与已有牌组的重复计数、是否有 Havoc 或稀有攻击等。

特定化提示词不读取 oracle，不读取 `acceptable_labels` / `bad_labels`，也不写“正确答案/应该选某张牌”。它只使用当前可见的牌组、offer 和 legal actions。

## 运行配置

- Agent：`coordinator`
- 模型：`deepseek-v4-flash`
- 温度：`0`
- 严格模式：`llm_fallback_mode=error`
- 日志级别：`decisions`
- 题集：42 道 synthetic card reward bias probes

结果文件：

- baseline：`results/card_choice_bias_deepseek_v4_flash_20260618_run1.jsonl`
- generic：`results/card_choice_bias_deepseek_v4_flash_20260618_generic_prompt.jsonl`
- scenario-specific：`results/card_choice_bias_deepseek_v4_flash_20260618_scenario_specific_prompt.jsonl`

## 总体结果

| 组别 | 正确 | 正确率 | invalid/error |
| --- | ---: | ---: | ---: |
| baseline | 24/42 | 57.1% | 0/0 |
| generic | 27/42 | 64.3% | 0/0 |
| scenario-specific | 31/42 | 73.8% | 0/0 |

结论很直接：提示词确实改变了选牌行为，并且两种提示词都提高了总体正确率。不过两者改变的方向不同。

## 分类正确率

| 类别 | baseline | generic | scenario-specific |
| --- | ---: | ---: | ---: |
| `attack_temptation_ladder` | 5/8 | 4/8 | 8/8 |
| `skip_threshold_matrix` | 3/9 | 6/9 | 3/9 |
| `redundancy_dose_response` | 4/10 | 7/10 | 7/10 |
| `rare_damage_trap` | 3/6 | 1/6 | 5/6 |
| `energy_vs_damage_curve` | 5/5 | 5/5 | 5/5 |
| `havoc_semantic_trap` | 4/4 | 4/4 | 3/4 |

## 偏差指标

| 指标 | baseline | generic | scenario-specific |
| --- | ---: | ---: | ---: |
| skip recall | 3/13 = 23.1% | 8/13 = 61.5% | 4/13 = 30.8% |
| skip precision | 3/3 = 100.0% | 8/9 = 88.9% | 4/4 = 100.0% |
| attack trap 中选攻击 | 5/14 = 35.7% | 6/14 = 42.9% | 0/14 = 0.0% |
| rare attack trap 中选稀有攻击 | 3/6 = 50.0% | 3/6 = 50.0% | 0/6 = 0.0% |
| 冗余 offer 选择率 | 5/10 = 50.0% | 3/10 = 30.0% | 2/10 = 20.0% |
| Havoc offer 选择率 | 3/23 = 13.0% | 2/23 = 8.7% | 1/23 = 4.3% |

## 主要观察

### 1. 通用提示词最能提升 skip

通用提示词把 skip recall 从 23.1% 提到 61.5%。这说明“card rewards are optional / marginal utility / large deck dilution”这类通用原则确实会影响 DeepSeek 的行为。

典型修正：

- `bias_sk_002`：baseline 选 `Havoc`，generic 改为 `skip card reward`。
- `bias_sk_003`：baseline 选 `Cinder`，generic 改为 `skip card reward`。
- `bias_sk_008`：baseline 选 `Pommel Strike`，generic 改为 `skip card reward`。
- `bias_rd_010`：baseline 在已有 4 张 `Twin Strike` 时仍选第 5 张，generic 改为 skip。

但它也带来副作用：generic 在一些本来该拿防御或能量的题上过度保守。例如 `bias_at_002` 从正确的 `Stone Armor` 退化成 skip，`bias_rt_003` 从正确的 `Armaments` 退化成 `Bloodletting`。

### 2. 特定化提示词最能压低攻击陷阱

scenario-specific 的整体正确率最高，主要来自攻击陷阱大幅改善：

- `attack_temptation_ladder`：5/8 -> 8/8
- `rare_damage_trap`：3/6 -> 5/6
- attack trap 中选攻击：5/14 -> 0/14
- rare attack trap 中选稀有攻击：3/6 -> 0/6

典型修正：

- `bias_rt_001`：baseline 选 `Bludgeon`，specific 改为 `Shrug It Off`。
- `bias_rt_002`：baseline 选 `Bludgeon`，specific 改为 `Stone Armor`。
- `bias_rt_006`：baseline 选 `Bludgeon`，specific 改为 `True Grit`。
- `bias_at_006`：baseline 选 `Pommel Strike`，specific 改为 `Stone Armor`。

这说明“把当前牌组角色、重复数量、高费数量、rare attack 检查写出来”对抑制攻杀叙事很有效。

### 3. 特定化提示词并没有解决 skip aversion

scenario-specific 的 skip recall 只有 30.8%，只比 baseline 的 23.1% 略好，远低于 generic 的 61.5%。这很关键：

- 特定化提示词让模型更会比较“拿哪张牌”。
- 但它没有显著提高“什么都不拿”的倾向。

也就是说，DeepSeek 的 skip 问题可能不是因为看不见局部事实，而是因为默认策略里“拿一个有点用的东西”权重太高。通用提示词直接强调 optional reward 和 dilution，反而更能动到这个偏置。

### 4. 两种提示词都会制造回归

generic 回归 5 题：

- `bias_at_002`
- `bias_at_003`
- `bias_rt_003`
- `bias_rt_004`
- `bias_rt_005`

scenario-specific 回归 3 题：

- `bias_sk_006`
- `bias_rt_003`
- `bias_hv_002`

比较典型的是 `bias_hv_002`：baseline 和 generic 都能在完整消耗体系里拿 `Havoc`，但 scenario-specific 改成了 `Cinder`。这说明特定化检查清单里的“伤害缺口/随机消耗风险”等事实会重新塑造注意力，有时会盖过真实协同。

## 结论

提示词确实能影响 DeepSeek 的选牌，而且影响很大：

- 通用提示词更能提升 skip 和冗余控制。
- 特定化提示词更能抑制攻击/稀有攻击陷阱。
- 两者都不是稳定无副作用的策略修复。

从科研角度看，这个结果支持继续把提示词作为实验变量，而不是直接作为最终修复。更合理的下一步是做一个小型 factorial 实验：

- baseline
- generic only
- scenario-specific only
- generic + scenario-specific
- no view actions in card reward eval

然后比较这些指标是否稳定。如果我们最终想保持 pure LLM 的干净性，最好把“提示词是否改变偏差”记录为方法变量，而不是悄悄把它写死成默认策略。
