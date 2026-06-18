# LLM 选牌能力离线评测说明

这份文档解释 `sts2_bench.card_choice_eval` 的用途、数据流、场景设计、输出格式和结果解读方式。它面向研究和调试，不是给游戏玩家看的攻略，也不是胜率 benchmark。

## 1. 这个评测在测什么

我们想单独测试 LLM 在 `card_reward` 三选一场景里的判断能力，尤其是：

- 能不能在牌组防御不足时，从上下文中主动推断应该补防御牌；
- 能不能识别“跳过拿牌”是合理动作，而不是每次都拿一张；
- 能不能避免被强单卡诱惑，转而选择更适配当前牌组的牌；
- 能不能理解费用曲线、重复牌、攻击/防御密度和简单协同；
- 能不能在多个可接受答案之间给出合理选择。

评测不直接测整局胜率，因为整局胜率会混入战斗操作、路线、商店、事件、随机性和模型稳定性。这里把问题收窄到“当前牌组 + 最近经历 + 三张奖励牌 + skip action”下的单步选牌。

## 2. 它和真实 harness 的关系

每个评测场景都是合成的 `card_reward` state，但执行路径使用真实 benchmark harness：

```text
synthetic card_reward state
  -> build_legal_actions(state)
  -> build_evidence_bundle(state, legal_actions)
  -> agent.choose(state, legal_actions)
  -> hidden oracle scoring
```

也就是说，LLM 看到的不是一个专门为评测写的自然语言题目，而是当前架构会提供给 OCC/LLM 的 state、legal actions 和 evidence。这样能测试我们现在的 evidence-first 结构是否足够支持选牌判断。

第一版不启动 `Sts2Headless`，也不从真实游戏随机采样。这样做是为了让场景可控，先测清楚模型在已知困难点上的能力边界。

## 3. 什么会给 LLM 看，什么不会

LLM 可以看到：

- 当前 `card_reward` state；
- `player.deck` 生成的 grouped deck summary；
- 三张候选牌；
- `skip card reward` 这个合法动作；
- evidence bundle；
- 如果使用 coordinator/OCC，还会看到 OCC prompt、RunContext slice 和 OCC history；
- 某些场景中的最近战斗反馈，例如 `previous fight lost 34 HP over 5 turns`。

LLM 不会看到：

- `oracle.acceptable_labels`；
- `oracle.bad_labels`；
- `expected_behavior`；
- 场景类别是否是 `skip_is_best`、`defense_gap_indirect` 等隐藏评分含义；
- 人类备注 `notes_for_humans` 中的评价标准。

隐藏 oracle 只在模型选完动作后用于评分。它不会进入 prompt、evidence，也不会进入 agent meta。

## 4. 为什么要设计“隐性”场景

这个评测不是要检查模型能不能服从“请补防御牌”这种明显提示，而是要检查它能否从实际上下文里推理。

例如 `defense_gap_indirect` 场景不会直接写“防御牌不足”。它只会给：

- 一个攻击牌偏多、技能偏少的 deck summary；
- 较低 HP；
- 最近战斗损失很多 HP 的 `last_action_result`；
- 三张奖励牌，其中一张防御牌、一张攻击牌、一张诱饵牌。

如果模型能选防御牌，说明它不是靠显式提示，而是在读牌组结构和战斗反馈。

同理，`skip_is_best` 场景不会写“应该跳过”。它只构造一个已经足够完整或已经臃肿的牌组，再给三张低影响、重复或不协同的牌。正确模型应该意识到不拿牌也是一等动作。

## 5. 场景类别

当前固定 suite 有 24 个场景，覆盖 8 类：

| 类别 | 主要测试点 |
| --- | --- |
| `defense_gap_indirect` | 从牌组和掉血反馈推断需要补防御，而不是继续拿攻击 |
| `skip_is_best` | 识别三张牌都不值得拿，选择 `skip card reward` |
| `synergy_pick` | 根据已有牌组选择协同牌，例如 Strike 密度、Block payoff |
| `anti_synergy_trap` | 避免强单卡或稀有牌诱饵，选择更适配当前牌组的牌 |
| `overweight_attack` | 牌组攻击已经过多时，避免继续堆攻击 |
| `resource_curve` | 根据费用曲线选择低费补丁或高费 payoff |
| `redundancy` | 避免拿已有太多的重复牌，即使单卡本身不错 |
| `ambiguous_good_pick` | 允许多个合理答案，避免强行单一 oracle |

每个场景都有 3 个 pick action 和 1 个 skip action。oracle 用 action label 做评分，例如：

```json
{
  "acceptable_labels": ["pick card 1: Shrug It Off"],
  "bad_labels": ["pick card 0: Twin Strike", "pick card 2: Havoc", "skip card reward"],
  "expected_behavior": "defense_gap_indirect"
}
```

评分只比较模型实际选择的 legal action label，不会根据 reason 反推正确性。

## 6. 具体场景清单

下面是当前 v1 suite 的固定场景。牌组用数量压缩展示；“隐藏可接受答案”只用于评测完成后的评分，不会进入 prompt 或 evidence。所有场景都暴露 `skip card reward` 这个 legal action；没有把 `Skip` 写进隐藏可接受答案时，模型选择 skip 会被判错。

### `defense_gap_indirect`

这组场景不直接告诉模型“缺防御”。它只给牌组结构和上一战掉血反馈，测试模型能不能自己推断应该补 block。

- `cc_001`
  - 牌组：`Strike x4`, `Defend x4`, `Bash`, `Twin Strike`, `Blood Wall`
  - 奖励三选一：`Twin Strike`, `Shrug It Off`, `Havoc`
  - 隐藏可接受答案：`Shrug It Off`
  - 最近经验：上一战 5 回合掉 18 HP
  - 考点：防御只比攻击少一张，测试模型能否在不极端的缺口里重视优质防御抽牌。

- `cc_002`
  - 牌组：`Strike x4`, `Defend x3`, `Bash`, `Pommel Strike`, `True Grit`, `Stone Armor`
  - 奖励三选一：`Stone Armor`, `Cinder`, `Havoc`
  - 隐藏可接受答案：`Stone Armor`
  - 最近经验：上一战 5 回合掉 20 HP
  - 考点：牌组已经有一些防御，但继续补高质量防御比拿随机消耗攻击更合理。

- `cc_003`
  - 牌组：`Strike x4`, `Defend x4`, `Body Slam`, `Bash`, `True Grit`
  - 奖励三选一：`Flame Barrier`, `Bludgeon`, `Havoc`
  - 隐藏可接受答案：`Flame Barrier`
  - 最近经验：上一战 5 回合掉 22 HP
  - 考点：已有 `Body Slam`，但仍更需要强 block 来源，而不是继续补伤害。

### `skip_is_best`

这组场景测试模型是否理解“不拿牌”也是合理动作。奖励不是显然全坏，但对当前牌组提升很小或会增加污染。

- `cc_004`
  - 牌组：`Strike x4`, `Defend x4`, `Bash`, `Shrug It Off`, `Stone Armor`, `Pommel Strike`, `Bloodletting`
  - 奖励三选一：`Strike`, `Defend`, `Havoc`
  - 隐藏可接受答案：`Skip`
  - 最近经验：无
  - 考点：中等大小牌组已有抽牌、防御和能量，奖励只给基础牌或不稳定顶牌打出。

- `cc_005`
  - 牌组：`Strike x5`, `Defend x5`, `Shrug It Off x2`, `Stone Armor x2`, `Pommel Strike x2`, `Bloodletting`, `Offering`, `Bash`, `Bludgeon`
  - 奖励三选一：`Strike`, `Defend`, `Havoc`
  - 隐藏可接受答案：`Skip`
  - 最近经验：无
  - 考点：牌组偏大但质量高，三张奖励都低于当前单卡质量线。

- `cc_006`
  - 牌组：`Strike x4`, `Defend x4`, `Barricade`, `Impervious`, `Blood Wall`, `True Grit`, `Armaments`, `Bash`
  - 奖励三选一：`Strike`, `Havoc`, `Cinder`
  - 隐藏可接受答案：`Skip`
  - 最近经验：无
  - 考点：牌组已经有明确 block-retention 思路，奖励都不服务这个思路。

### `synergy_pick`

这组场景测试模型是否能根据已有牌组选择协同牌，而不是孤立评价单卡。

- `cc_007`
  - 牌组：`Strike x8`, `Defend x4`, `Bash`, `Twin Strike`, `Pommel Strike`
  - 奖励三选一：`Perfected Strike`, `Havoc`, `Stone Armor`
  - 隐藏可接受答案：`Perfected Strike`
  - 最近经验：无
  - 考点：Strike-name 密度高，`Perfected Strike` 有上下文价值。

- `cc_008`
  - 牌组：`Strike x4`, `Defend x4`, `Shrug It Off x2`, `Stone Armor x2`, `Entrench`
  - 奖励三选一：`Body Slam`, `Havoc`, `Twin Strike`
  - 隐藏可接受答案：`Body Slam`
  - 最近经验：无
  - 考点：block 密度和 `Entrench` 支撑 `Body Slam`。

- `cc_009`
  - 牌组：`Strike x4`, `Defend x4`, `Bash`, `Bludgeon`, `Barricade`, `Impervious`
  - 奖励三选一：`Bloodletting`, `Drum of Battle`, `Havoc`
  - 隐藏可接受答案：`Bloodletting`
  - 最近经验：无
  - 考点：高费牌让 `Bloodletting` 有价值；没有消耗 payoff 时不应选 `Drum of Battle`。

### `anti_synergy_trap`

这组场景故意给强单卡或稀有牌诱饵，测试模型是否会忽略当前牌组需求。

- `cc_010`
  - 牌组：`Strike x4`, `Defend x4`, `Bash`, `Twin Strike`, `Pommel Strike`, `Blood Wall`
  - 奖励三选一：`Bludgeon`, `Shrug It Off`, `Cinder`
  - 隐藏可接受答案：`Shrug It Off`
  - 最近经验：上一战 5 回合掉 19 HP
  - 考点：`Bludgeon` 是强单卡，但曲线和掉血反馈让 `Shrug It Off` 更贴合。

- `cc_011`
  - 牌组：`Strike x4`, `Defend x4`, `Bash`, `Pommel Strike`, `Anger`, `Bloodletting`
  - 奖励三选一：`Barricade`, `Pommel Strike`, `Havoc`
  - 隐藏可接受答案：`Pommel Strike`
  - 最近经验：无
  - 考点：`Barricade` 诱人，但牌组还没有足够稳定的持续 block 支撑。

- `cc_012`
  - 牌组：`Strike x4`, `Defend x4`, `Bash`, `Barricade`, `Impervious`, `Blood Wall`, `True Grit`
  - 奖励三选一：`Bludgeon`, `Armaments`, `Bloodletting`
  - 隐藏可接受答案：`Armaments`
  - 最近经验：无
  - 考点：牌组里有多张未升级后收益很高的 block-plan 牌，因此 `Armaments` 比额外 payoff 更合适。

### `overweight_attack`

这组场景测试模型是否会在攻击牌已经过多时继续堆攻击。

- `cc_013`
  - 牌组：`Strike x4`, `Defend x4`, `Twin Strike x2`, `Pommel Strike`, `Bash`, `True Grit`, `Blood Wall`
  - 奖励三选一：`Twin Strike`, `Shrug It Off`, `Cinder`
  - 隐藏可接受答案：`Shrug It Off`
  - 最近经验：无
  - 考点：并非完全没防御，但攻击仍多于高质量防御，`Shrug It Off` 是更好的补强。

- `cc_014`
  - 牌组：`Strike x4`, `Defend x4`, `Anger x2`, `Twin Strike`, `Pommel Strike`, `Bash`, `True Grit`
  - 奖励三选一：`Stone Armor`, `Anger`, `Havoc`
  - 隐藏可接受答案：`Stone Armor`
  - 最近经验：无
  - 考点：攻击只中等偏多，但增量收益最明确的仍是防御牌。

- `cc_015`
  - 牌组：`Strike x4`, `Defend x4`, `Twin Strike x2`, `Pommel Strike`, `Anger`, `Bash`, `Blood Wall`
  - 奖励三选一：`Armaments`, `Bludgeon`, `Cinder`
  - 隐藏可接受答案：`Armaments`
  - 最近经验：无
  - 考点：牌组需要提升，但另一张攻击不如升级基础牌和防御牌。

### `resource_curve`

这组场景测试模型是否能理解费用曲线，而不是固定偏好高伤害或低费牌。

- `cc_016`
  - 牌组：`Strike x4`, `Defend x4`, `Bash`, `Bludgeon`, `Barricade`, `Impervious`
  - 奖励三选一：`Bloodletting`, `Bludgeon`, `Barricade`
  - 隐藏可接受答案：`Bloodletting`
  - 最近经验：无
  - 考点：高费牌已经偏多，能量牌比继续拿高费 payoff 更能修补曲线。

- `cc_017`
  - 牌组：`Strike x5`, `Defend x5`, `Anger x2`, `Pommel Strike`
  - 奖励三选一：`Bludgeon`, `True Grit`, `Havoc`
  - 隐藏可接受答案：`Bludgeon`
  - 最近经验：无
  - 考点：低费牌组缺少大伤害 payoff，可以承受一张高费强牌。

- `cc_018`
  - 牌组：`Strike x5`, `Defend x4`, `Bludgeon x2`, `Bash`, `Barricade`
  - 奖励三选一：`Pommel Strike`, `Bludgeon`, `Havoc`
  - 隐藏可接受答案：`Pommel Strike`
  - 最近经验：无
  - 考点：已有两张高费 `Bludgeon`，再拿高费不如补低费循环。

### `redundancy`

这组场景测试模型是否能意识到“好牌也会因为已有太多而不该拿”。

- `cc_019`
  - 牌组：`Strike x5`, `Defend x4`, `Shrug It Off x3`, `Stone Armor x2`, `Bash`
  - 奖励三选一：`Shrug It Off`, `Twin Strike`, `Havoc`
  - 隐藏可接受答案：`Skip`
  - 最近经验：无
  - 考点：`Shrug It Off` 通常不错，但这里防御/抽牌角色已经足够。

- `cc_020`
  - 牌组：`Strike x5`, `Defend x4`, `Bloodletting x2`, `Offering`, `Pommel Strike`, `Bash`
  - 奖励三选一：`Bloodletting`, `Havoc`, `Anger`
  - 隐藏可接受答案：`Skip`
  - 最近经验：无
  - 考点：已有多张以 HP 换能量的牌，再拿 `Bloodletting` 边际价值低。

- `cc_021`
  - 牌组：`Strike x7`, `Twin Strike x3`, `Pommel Strike x2`, `Defend x4`, `Bash`
  - 奖励三选一：`Twin Strike`, `Strike`, `Pommel Strike`
  - 隐藏可接受答案：`Skip`
  - 最近经验：无
  - 考点：三张奖励都复制了已经拥挤的攻击角色。

### `ambiguous_good_pick`

这组场景允许多个合理答案，避免把主观的单一最优解硬塞进 oracle。

- `cc_022`
  - 牌组：`Strike x6`, `Defend x2`, `Bash`, `Twin Strike`
  - 奖励三选一：`Shrug It Off`, `Stone Armor`, `Twin Strike`
  - 隐藏可接受答案：`Shrug It Off`, `Stone Armor`
  - 最近经验：上一战 5 回合掉 21 HP
  - 考点：两张防御牌都是合理改善；继续拿攻击不是。

- `cc_023`
  - 牌组：`Strike x7`, `Defend x4`, `Twin Strike`, `Bash`
  - 奖励三选一：`Perfected Strike`, `Pommel Strike`, `Havoc`
  - 隐藏可接受答案：`Perfected Strike`, `Pommel Strike`
  - 最近经验：无
  - 考点：直接 Strike payoff 和低费循环攻击都可辩护。

- `cc_024`
  - 牌组：`Strike x5`, `Defend x5`, `Bash`, `Armaments`, `Pommel Strike`
  - 奖励三选一：`Shrug It Off`, `Pommel Strike`, `Havoc`
  - 隐藏可接受答案：`Shrug It Off`, `Pommel Strike`
  - 最近经验：无
  - 考点：抽牌防御和抽牌攻击都合理；`Havoc` 是低质量诱饵。

## 7. 如何运行

用本地 OpenAI-compatible 服务跑 coordinator：

```powershell
python -m sts2_bench.card_choice_eval `
  --agent coordinator `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen2.5:14b `
  --out results/card_choice_eval.jsonl
```

只跑某一类场景：

```powershell
python -m sts2_bench.card_choice_eval `
  --agent coordinator `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen2.5:14b `
  --category skip_is_best `
  --out results/card_choice_eval_skip.jsonl
```

调试 prompt 和完整 evidence：

```powershell
python -m sts2_bench.card_choice_eval `
  --agent coordinator `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen2.5:14b `
  --limit 4 `
  --log-level full `
  --out results/card_choice_eval_debug.jsonl
```

快速 smoke，不依赖 LLM：

```powershell
python -m sts2_bench.card_choice_eval `
  --agent random `
  --limit 2 `
  --out results/card_choice_eval_random.jsonl
```

## 8. JSONL 输出怎么看

输出包含四类行：

- `start`：本次评测配置和场景数量；
- `scenario`：场景摘要、legal actions、evidence hash、按 log level 选择是否写完整 prompt/evidence；
- `choice`：模型选择、reason、agent meta、verdict、failure tags；
- `summary`：整体指标。

`scenario` 行不会写 oracle。也就是说，打开 trace 看模型输入时，不会泄露答案。

`choice` 行会包含：

- `selected_label`
- `selected_kind`
- `verdict`
- `failure_tags`
- `expected_behavior`
- `rationale_keyword_hits`

`expected_behavior` 在选择完成后写入 choice row，用于分析结果，不参与模型输入。

## 9. 指标解释

`summary` 里主要看这些字段：

- `accuracy`：所有场景中选择 acceptable label 的比例；
- `category_accuracy`：每个类别的准确率；
- `skip_precision`：模型选择 skip 时，有多少次确实应该 skip；
- `skip_recall`：应该 skip 的场景里，模型有多少次选了 skip；
- `trap_failure_rate`：`anti_synergy_trap` 场景中掉进 bad label 的比例；
- `invalid`：agent 返回了不在当前 legal actions 里的动作；
- `errors`：agent 调用或解析过程中发生异常。

reason 里的关键词命中只做辅助诊断。例如模型选对了 `Shrug It Off`，但 reason 完全没有提到 block、HP 或 deck 结构，这说明它可能是偶然选对，不一定真的理解上下文。

## 10. 如何扩展场景

新增场景时要遵守几个规则：

1. `decision` 必须是 `card_reward`。
2. `can_skip` 必须是 `true`。
3. 必须正好有三张奖励牌。
4. 必须提供 `player.deck`，这样 evidence 层可以生成 grouped deck summary。
5. `acceptable_labels` 必须能匹配当前 `build_legal_actions()` 生成的 label。
6. 不要把“正确答案”写进 state、evidence 或 notes 里给模型看到。
7. 如果多个答案都合理，就把它们都放进 `acceptable_labels`。

扩展后应至少跑：

```powershell
python -m pytest -q tests/test_card_choice_eval.py
```

如果改动影响 evidence 或 coordinator，再跑：

```powershell
python -m pytest -q tests/test_card_choice_eval.py tests/test_sts2_bench_evidence.py tests/test_sts2_bench_multi_agent.py
```

## 11. 当前局限

第一版是离线合成评测，所以它不能替代真实游戏胜率：

- 场景由我们设计，覆盖范围有限；
- oracle 是研究判断，不是游戏引擎给出的唯一真理；
- 单步选牌正确不代表整局策略正确；
- 真实 run 里的路线、商店、事件、战斗结果会改变选牌价值；
- 目前只评估 card reward，不评估 boss relic、shop purchase 或 route choice。

但它适合做模型能力诊断。如果某个模型在这些受控场景里都经常拿重复攻击、忽略防御缺口或不会 skip，那么它在真实游戏里大概率也会出现同类问题。

## 12. 建议的实验流程

建议按这个顺序跑：

1. 用 `--agent random` 做 smoke，确认输出文件和统计正常。
2. 用目标 LLM 跑完整 suite，`log-level=decisions`。
3. 看 `summary`，先定位低分类别。
4. 对低分类别用 `--category ... --log-level full` 重跑。
5. 检查 full trace 中的 prompt、evidence、selected action 和 reason。
6. 如果模型错误集中在同一类，判断是模型能力问题、context 表达问题，还是 oracle 场景设计太强。
7. 不要用这套评测直接调 prompt 到过拟合。它应该作为诊断工具，而不是训练集或提示词答案库。

## 13. 和 pure LLM 设计的关系

这个评测不引入启发式评分器，也不让模块推荐动作。Evidence 只提供事实和候选动作证据，最终选择仍由 LLM 完成。

因此，评测结果可以帮助我们回答两个问题：

- 当前 evidence-first context 是否给了模型足够的信息；
- 在信息足够时，模型是否真的能做出合理选牌判断。

如果模型失败，我们先看 full trace：是没有看到关键信息，还是看到了但没有推理出来。前者是 harness/context 问题，后者才是模型能力问题。
