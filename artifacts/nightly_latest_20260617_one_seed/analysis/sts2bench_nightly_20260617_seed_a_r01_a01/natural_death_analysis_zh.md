# a01 自然死亡分析

## 结论

本次 attempt 是有效自然死亡，不是截断或人工终止：

- run_id: `sts2bench_nightly_20260617_seed_a_r01_a01`
- 终局: `game_over`
- 位置: Act 2 Floor 6 `Monster`
- 终局血量: `0/77`
- `truncated=false`
- `error=null`
- `returncode=0`, `analysis_returncode=0`
- 当前 manifest 只把 `r01_a01` 计为 valid；`r02_a01` 是后续误启动留下的 partial trace，没有 attempts 行，不计入本次分析

本次死亡的核心证据不是 floor 分数，而是 Act 2 中连续普通怪战损后，在 `54/77` 血量进入第 6 层普通怪并自然死亡。自动 `high_risk_events` 没有命中，但 map topology 手动对照发现一个重要路线风险：在 A2F5、`54/77` 血量时，planner 选择了 `Monster(3,5)`，该节点如果存活下来下一层只连到 `Elite(3,6)`；另一个合法选项是 `Unknown(2,5)`，后续连到普通怪。这说明当前 route-plan 风险识别还没有把“下一跳被迫精英”作为显式惩罚。

## 关键证据文件

- `route_hp_replay.svg`: 横向完整路线和房间血量回放
- `hp_by_room.svg`: 按房间的 HP 曲线
- `hp_by_action.svg`: 按 action 的 HP 曲线
- `map_path.svg`: 地图路线图
- `replay.md`: 可读回放，包括 map choices、route-plan events、card rewards
- `summary.json`: 结构化分析数据
- `large_loss_rooms/room_017_Boss.md`: Act 1 Boss 大战损日志
- `large_loss_rooms/room_021_Monster.md`: Act 2 Floor 4 大战损日志
- `large_loss_rooms/room_023_Monster.md`: Act 2 Floor 6 死亡战斗日志

## 战损路径

| Room # | 位置 | 房间 | 进入 HP | 离开 HP | 战损 | 说明 |
|---:|---|---|---:|---:|---:|---|
| 17 | A1F17 | Boss | 62/77 | 38/77 | 24 | Act 1 Boss 大战损，但打赢后 Act 2 起始恢复到满血 |
| 18 | A2F1 | Event | 77/77 | 77/77 | 0 | Act 2 开始 |
| 19 | A2F2 | Monster | 77/77 | 77/77 | 0 | 普通怪无损 |
| 20 | A2F3 | Monster | 77/77 | 76/77 | 1 | 小损 |
| 21 | A2F4 | Monster | 76/77 | 54/77 | 22 | 第二个关键大战损 |
| 22 | A2F5 | Shop | 54/77 | 54/77 | 0 | 花光大部分金币，未回血 |
| 23 | A2F6 | Monster | 54/77 | 0/77 | 54 | 自然死亡房间 |

死亡前的路径状态是：`A2F4 Monster -22 -> A2F5 Shop -> A2F6 Monster -54 death`。这比单看最终 floor 更有解释力。

## 低血量高风险路线检查

发现一个值得标记的严重路线风险：

- step 282，A2F5，HP `54/77`
- legal routes:
  - `Unknown(2,5)`
  - `Monster(3,5)`
- selected: `Monster(3,5)`
- selected node 的 children: 只有 `Elite(3,6)`
- alternative `Unknown(2,5)` 的 children 是 `Monster(1,6)`
- reason: `Monster fight to test newly acquired Whirlwind, One-Two Punch, and Stone Armor; earn gold/card reward to further scale for act 2 boss.`

问题在于 reason 只解释了“测试新卡/拿战斗奖励”，没有提到该 Monster 后面会被强制接到 Elite。虽然这局实际死在 `Monster(3,5)`，还没走到 Elite，但如果没有死亡也会进入 forced elite 线路。因此 route replay 上应当把这个点标成“高风险后继路线”，而不是只看当前房间类型。

对照项：

- step 54，A1F7，HP `77/77` 也选择了 `Monster(4,7)`，下一跳只连 `Elite(4,8)`；但当时满血，且后续精英只损 9 HP，不属于同等严重问题。
- step 275，A2F4，HP `54/77` 选择 `Shop(2,4)`，legal 里还有 `Unknown(4,4)`；这一步本身不是强制精英，但 shop 后下一步选择才暴露风险。

## Route-plan 事件

本次 trace 有 22 个 route-plan events，1 个 route-plan validation retry：

- 大部分事件是 `route_plan_created`，少数是 `route_plan_followed`。
- step 15 出现 `partial_invalid_suffix_trimmed`，原计划里的 `RestSite(4,6)` 后缀被修剪。
- step 50 有一次 `route_plan_missing_created_event`，随后 retry 成功写入 `route_plan_created`。
- step 86 的 reason_tags 包含 `map_changed`。
- A2 死亡前的关键事件是 step 282 `route_plan_created`，reason_tags 为 `test_new_cards, safe_choice`，但没有记录 forced-elite 后继风险。

这说明 route-plan event 机制已经能记录“规划路线/执行路线/重新规划”的日志基础，但当前还需要把 topology risk 写进 reason 或单独字段，例如：

- selected node 的 immediate children 类型
- 是否存在 forced elite / forced monster chain
- alternative legal route 的后继风险对比
- 当前 HP 百分比和最近一房间战损

## 卡牌和商店决策

死亡前重要资源变化：

- Act 2 Boss 奖励选了 `Conflagration`
- A2F2 选 `Summon Forth`
- A2F3 选 `Monologue`
- A2F4 选 `Demesne`
- A2F5 商店购买：
  - `One-Two Punch` for 143 gold
  - `Stone Armor` for 78 gold
  - `Whirlwind` for 37 gold
- 商店后金币从 `266` 降到 `8`

商店购买看起来是在强化战斗包，但没有解决 A2F4 之后的生存风险。随后选择普通怪并死亡，说明“买完牌所以可以测试”的 reason 对 HP 和路线后继风险估计不足。

## 死亡战斗细节

最终战斗不是一进入就立刻崩盘，而是 `Ovicopter` 召唤后无法快速清掉衍生敌人，导致 HP 被连续压低：

| Step | HP | 关键变化 | 评估 |
|---:|---:|---|---|
| 282 | 54/77 | 进入 `Monster(3,5)` | 这是路线层面的关键选择 |
| 283-285 | 54/77 | 用 `Howl from Beyond` + `Conflagration` 打 Ovicopter，理由认为敌人 `Summon` 不攻击 | 首回合判断本身可解释，但后续没有体现召唤风险的预案 |
| 286-287 | 54/77 | 双 `Defend` + `Vicious`，只掉 3 HP | 防守尚可 |
| 288-291 | 51/77 -> 32/77 | 选择 `Primal Force` 后打出 `Demesne`，只用一张 `Giant Rock` 把一个 Hatchling 打到 4 HP，回合结束掉 19 HP | 这是战斗层面的主要转折：为了长期收益保留 `Demesne`，没有优先削减场面压力 |
| 292-294 | 32/77 | custom sequence 首次因超过 4 步被打回；随后清掉一个 Hatchling、获得 19 block，未掉血 | 还能救回一部分局面，但场上仍有 3 个敌人 |
| 295-299 | 32/77 -> 0/77 | 查看 draw pile 后尝试 `Bash`、`Dominate`、`Drum of Battle`，最终承认无法击杀或防住，end turn 后死亡 | 末回合已经是被动求解，失败根因更早 |

所以 r01 的失败不是单一“路线错”或单一“战斗错”。更准确地说，是路线层先在 `54/77` 选择了一个高后继风险普通怪；进入战斗后又用长期收益牌和抽牌赌法处理高压多敌局面，导致最终无解。

## 本次分析建议

1. benchmark 指标里应保留自然死亡为一等事件：`game_over + hp=0 + truncated=false + error=null`。
2. replay 里需要新增或突出 forced-risk 标记：不是只标当前房间，而是标“选择该节点后下一跳是否被迫 Elite/Monster/无 Rest”。
3. route-plan event 的 reason 应被约束为包含风险对照：当前 HP、最近一房间战损、selected route 后继、alternative route 后继。
4. 当前自动 `high_risk_events=[]` 不够，至少应补一类规则：`hp_ratio <= threshold` 且 selected node 的唯一 child 是 Elite，或 selected path 比 legal alternative 明显更危险。
5. 死亡战斗 log 应继续保留逐步 action/reason，因为 r01 暴露出 route-plan 和 combat-turn 两层风险会叠加，单看 floor 或最终 HP 不够。
