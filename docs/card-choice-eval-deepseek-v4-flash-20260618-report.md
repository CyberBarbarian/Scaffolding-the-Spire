# DeepSeek V4 Flash 选牌能力离线评测报告

本报告分析 `deepseek-v4-flash` 在 synthetic card reward suite 上的单步选牌表现。评测使用当前 evidence-first pure-LLM harness 路径：

```text
synthetic card_reward state
  -> build_legal_actions()
  -> build_evidence_bundle()
  -> coordinator/OCC LLM
  -> hidden oracle scoring
```

这不是胜率 benchmark，也不是真实游戏采样。它只诊断模型在可控三选一/可跳过 card reward 场景下的判断能力。

## 运行配置

- 日期：2026-06-18
- Agent：`coordinator`
- Model：`deepseek-v4-flash`
- Base URL：`https://api.deepseek.com`
- Temperature：`0`
- Retry：`llm_max_retries=1`
- Fallback：`llm_fallback_mode=error`
- Log level：`full`
- 结果文件：`results/card_choice_eval_deepseek_v4_flash_20260618_full.jsonl`
- 场景数：24

运行命令等价于：

```powershell
python -m sts2_bench.card_choice_eval `
  --agent coordinator `
  --out results/card_choice_eval_deepseek_v4_flash_20260618_full.jsonl `
  --log-level full `
  --llm-timeout 120 `
  --llm-max-retries 1 `
  --llm-fallback-mode error `
  --llm-temperature 0
```

`.env` 提供 `DEEPSEEK_API_KEY`、`STS2_BENCH_BASE_URL` 和 `STS2_BENCH_MODEL`。

## 总体结果

| 指标 | 结果 |
| --- | ---: |
| Total | 24 |
| Correct | 13 |
| Incorrect | 11 |
| Accuracy | 54.17% |
| Invalid actions | 0 |
| Runtime/model errors | 0 |
| Skip precision | 100.00% |
| Skip recall | 16.67% |
| Trap failure rate | 33.33% |
| Selected skip count | 1 / 24 |
| Total tokens | 85,655 |
| Reasoning tokens | 16,480 |
| Average model latency | 12.66s |

核心结论：模型的格式服从和 legal action 选择能力是干净的，但选牌策略能力只有中等。它能处理明显的合理牌、部分协同和部分费用曲线问题；最大短板是几乎不会主动跳过奖励，也很容易被“再拿一张攻击/高伤害/0 费攻击”这种泛化启发误导。

## 分类表现

| Category | Correct / Total | Accuracy | 观察 |
| --- | ---: | ---: | --- |
| `ambiguous_good_pick` | 3 / 3 | 100.00% | 多个答案都可接受时表现最好，能选到合理牌。 |
| `defense_gap_indirect` | 2 / 3 | 66.67% | 能补防御，但在 `Bludgeon` 诱惑下会转向爆发伤害。 |
| `synergy_pick` | 2 / 3 | 66.67% | 能识别 `Body Slam` / `Bloodletting`，但漏掉 Strike density payoff。 |
| `anti_synergy_trap` | 2 / 3 | 66.67% | 多数能避开诱饵，但仍会被 rare/high damage 单卡吸引。 |
| `resource_curve` | 2 / 3 | 66.67% | 能理解低费循环和低曲线拿高费，但高费过载时仍偏向第二张 `Bludgeon`。 |
| `skip_is_best` | 1 / 3 | 33.33% | 只在最显然的基础牌/`Havoc` 组合里 skip。 |
| `overweight_attack` | 1 / 3 | 33.33% | 攻击过量但不极端时，仍倾向拿 `Anger` 或 `Bludgeon`。 |
| `redundancy` | 0 / 3 | 0.00% | 完全没有识别“好牌/可用牌也可能因冗余而该跳过”。 |

## 逐题结果

| ID | Category | Hidden acceptable | Model selected | Result | 简要原因 |
| --- | --- | --- | --- | --- | --- |
| `cc_001` | `defense_gap_indirect` | `Shrug It Off` | `Shrug It Off` | OK | 识别 block + draw 能改善稳定性。 |
| `cc_002` | `defense_gap_indirect` | `Stone Armor` | `Stone Armor` | OK | 识别 Plating / 防御补强。 |
| `cc_003` | `defense_gap_indirect` | `Flame Barrier` | `Bludgeon` | MISS | 用“快速杀敌减少掉血”合理化高爆发攻击，忽略 block 需求。 |
| `cc_004` | `skip_is_best` | `Skip` | `Skip` | OK | 能拒绝基础牌和不稳定 `Havoc`。 |
| `cc_005` | `skip_is_best` | `Skip` | `Havoc` | MISS | 把 `Havoc` 误判为 deck thinning / utility。 |
| `cc_006` | `skip_is_best` | `Skip` | `Havoc` | MISS | 继续高估 `Havoc`，认为它能帮助打出高费牌。 |
| `cc_007` | `synergy_pick` | `Perfected Strike` | `Stone Armor` | MISS | 过度偏向防御，漏掉 Strike-name 密度带来的 payoff。 |
| `cc_008` | `synergy_pick` | `Body Slam` | `Body Slam` | OK | 正确识别 block deck 支撑 `Body Slam`。 |
| `cc_009` | `synergy_pick` | `Bloodletting` | `Bloodletting` | OK | 正确理解高费牌组需要能量补丁。 |
| `cc_010` | `anti_synergy_trap` | `Shrug It Off` | `Shrug It Off` | OK | 避开 `Bludgeon`，选择 block + draw。 |
| `cc_011` | `anti_synergy_trap` | `Pommel Strike` | `Pommel Strike` | OK | 避开缺支撑的 `Barricade`。 |
| `cc_012` | `anti_synergy_trap` | `Armaments` | `Bludgeon` | MISS | 被 rare/high damage 诱导，忽略已有高价值可升级 block-plan 牌。 |
| `cc_013` | `overweight_attack` | `Shrug It Off` | `Shrug It Off` | OK | 能在攻击偏多时补防御抽牌。 |
| `cc_014` | `overweight_attack` | `Stone Armor` | `Anger` | MISS | 高估 0 费攻击和循环复制，忽略防御比例问题。 |
| `cc_015` | `overweight_attack` | `Armaments` | `Bludgeon` | MISS | 再次偏好高伤害 rare，忽略升级基础牌/防御的上下文价值。 |
| `cc_016` | `resource_curve` | `Bloodletting` | `Bludgeon` | MISS | 高费牌已多时仍拿第二张高费伤害牌。 |
| `cc_017` | `resource_curve` | `Bludgeon` | `Bludgeon` | OK | 低曲线缺 payoff 时选择高费大伤害合理。 |
| `cc_018` | `resource_curve` | `Pommel Strike` | `Pommel Strike` | OK | 已有高费重复时选择低费循环。 |
| `cc_019` | `redundancy` | `Skip` | `Twin Strike` | MISS | 泛化为“Act 1 需要攻击密度”，未识别攻击角色已拥挤。 |
| `cc_020` | `redundancy` | `Skip` | `Anger` | MISS | 高估 0 费攻击和“能量协同”，忽略牌组膨胀和已有能量牌。 |
| `cc_021` | `redundancy` | `Skip` | `Pommel Strike` | MISS | 对重复攻击/循环牌仍倾向“拿一张可用牌”。该题 reason 为空，响应质量较低。 |
| `cc_022` | `ambiguous_good_pick` | `Shrug It Off` / `Stone Armor` | `Shrug It Off` | OK | 选择可接受的防御抽牌。 |
| `cc_023` | `ambiguous_good_pick` | `Perfected Strike` / `Pommel Strike` | `Perfected Strike` | OK | 识别 Strike-heavy payoff。 |
| `cc_024` | `ambiguous_good_pick` | `Shrug It Off` / `Pommel Strike` | `Shrug It Off` | OK | 选择稳定 block + draw。 |

## 能力判断

### 做得好的部分

1. **格式和安全性干净。**
   24 个场景没有 invalid action，也没有 parse/model error。所有选择都来自 legal actions，说明 OCC schema 和 action validator 对这个模型足够有效。

2. **能处理显性合理牌。**
   `Shrug It Off`、`Stone Armor`、`Body Slam`、`Bloodletting`、`Pommel Strike` 这些在上下文里较直接的选择，模型大多能讲出合理理由。

3. **能处理多个合理答案。**
   `ambiguous_good_pick` 3/3，说明如果 oracle 允许多个合理解，模型基本能落在可接受区域。

4. **能理解部分费用曲线。**
   `cc_009` 和 `cc_018` 表明它能在高费牌组里识别能量/低费循环的价值，但这种能力不稳定。

### 主要短板

1. **不会 skip。**
   6 个隐藏期望 skip 的场景里只 skip 了 1 个。它的 `skip_precision=100%`，说明它 skip 时没错；但 `skip_recall=16.67%`，说明阈值太高。实际行为是：只要候选牌看起来“有一点用”，它就倾向拿牌。

2. **冗余判断很弱。**
   `redundancy` 0/3 是最明显短板。它没有把“当前牌组已经有足够同类功能”转化成 skip，而是套用通用理由，例如 “Act 1 needs attack density” 或 “0-cost attack is efficient”。

3. **高估高伤害 rare。**
   `Bludgeon` 相关错题包括 `cc_003`、`cc_012`、`cc_015`、`cc_016`。模型经常用“elite/boss burst damage”合理化选择，哪怕当前问题是防御、升级或费用曲线。

4. **高估 `Havoc`。**
   `cc_005` 和 `cc_006` 都错选 `Havoc`，而且 reason 把它描述成 deck thinning、draw manipulation 或帮助打高费牌。这暴露了卡牌语义理解不稳：它知道 `Havoc` 看起来和消耗/顶牌有关，但低估了不稳定性和 off-plan 风险。

5. **过度依赖泛化启发。**
   许多错误理由不是胡说，而是“常见但不够上下文相关”的泛化：Act 1 需要伤害、0 费攻击高效、rare card 强、快速杀敌减少掉血。这些启发在真实游戏里有时对，但本评测正是在测它能否超越这些表层规律。

6. **没有稳定引用 evidence。**
   24 个响应中有 15 个 `evidence_refs` 为空。虽然这不直接影响 action legality，但说明模型没有稳定地把选择绑定到 evidence bundle 里的候选证据。`cc_021` 甚至给出空 reason，后续可以把非空 reason / evidence_refs 做成更严格的响应质量校验。

## 对当前架构的含义

这次结果更像“模型策略能力不足”，不是 harness 没把动作暴露出来：

- `skip card reward` 全程可见，且模型确实能在 `cc_004` 选择 skip。
- deck summary 可见，模型能在部分题里引用牌组结构。
- legal action / JSON schema 没有失败，说明执行链路本身干净。

因此，当前 evidence-first 架构至少能诊断出问题：LLM 在信息足够时仍然偏好拿牌、偏好伤害、偏好看起来有价值的单卡。这个问题不能只靠“暴露 skip action”解决；它需要更系统的评测、模型比较，或者在 pure-LLM 范围内改进上下文组织和自检方式。

## 建议

1. **把这个模型标记为选牌能力不稳定。**
   54.17% 的离线题准确率不足以支撑“能可靠做 card reward 决策”的结论。它可以作为 baseline，但不能作为强策略样本。

2. **优先扩充 skip/redundancy 场景。**
   当前最大错误集中在 skip 与冗余。建议下一版把 24 题扩到 40-60 题，其中至少一半围绕 skip、冗余、off-plan offer、大牌组质量线。

3. **增加多次重复运行。**
   即使 temperature 为 0，远程模型仍可能存在输出抖动。建议对同一模型跑 3-5 次，报告 mean/std，而不是只看单次 24 题。

4. **增加响应质量校验。**
   可以在不引入启发式策略的前提下，要求非空 `reason`，并校验 `evidence_refs` 至少引用一个存在的 candidate/action。它不会告诉模型选什么，但能减少空泛理由和无证据选择。

5. **单独审计卡牌语义误解。**
   `Havoc` 被误解为稳定 deck thinning/draw manipulation 是典型问题。可以增加一个小型 “card semantics trap” 标签，专门测模型是否读懂候选牌描述。

6. **与其它模型横向比较。**
   这次报告只能说明 `deepseek-v4-flash` 在当前 suite 上的表现。下一步应同样跑一到两个本地/远程模型，比较 skip recall、redundancy accuracy 和 trap failure rate。

## 结论

`deepseek-v4-flash` 在这个 suite 上表现为：**动作格式可靠，基础上下文理解尚可，但选牌策略远未可靠**。它能选对明显好牌，也能处理部分协同；但面对“不要拿牌”“牌已冗余”“高伤害诱饵”“off-plan 低稳定性牌”时，错误率很高。

如果把这个结果映射到真实杀戮尖塔式流程，最可能出现的问题是牌组逐渐变胖、攻击/高伤害牌过量、Havoc/Anger/Bludgeon 这类看起来有价值的牌被过度拿取，最终导致防御与稳定性不足。
