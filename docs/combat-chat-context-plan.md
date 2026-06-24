# Combat Chat Context Plan

本文记录 benchmark 上下文管理实验的一次收缩版设计：不再先做“整场战斗内多轮记忆”，而是先做更小的 **player-turn scoped chat**，也就是只在同一个玩家出牌回合内保留短多轮对话。

核心原则：

- benchmark 仍然坚持一张牌一张牌出，每次只选择一个合法 `action_id`。
- 多轮对话只改变 LLM 收到消息的组织方式，不改变环境动作粒度。
- 每个玩家出牌回合是一个短 conversation；一旦结束回合或进入下一回合，这段 history 就清空。
- `Burning Pact`、`Second Wind`、药水发现牌、选择弃牌/烧牌等需要环境反馈的 modal 状态，也当成同一出牌回合里的普通状态消息；仍然每次只输出一个合法 action。

## 设计动机

原先设想是“单场战斗内记忆”：同一个怪或 Boss 战中保留 `user -> assistant -> user -> assistant` 历史，离开战斗后清空。这个方向更接近 chat agent，但第一版会遇到两个问题：

- Boss 战可能有 8 到 15 个决策步，历史容易变长。
- 很多历史计算会过期，例如上一回合的手牌、能量、敌人 intent、block 都已经改变。
- 如果 assistant 的旧推理有错，整场战斗级 history 会持续污染后续回合。

对照 `python/play.py` 的 CLI 行为，人在一个出牌回合内可以连续进行多个操作；但 benchmark 不能把多个操作压成一个动作，因为部分卡牌会中途触发环境交互，例如烧牌、选牌、发现牌、药水选择。这说明更自然的实验单位不是“整场战斗”，而是“一个玩家出牌回合”。

所以本设计把 chat history 缩小到：

```text
one player turn = round N 中，从进入玩家出牌阶段开始，到模型选择 end_turn、战斗结束、或进入非战斗状态为止
```

这比整场战斗记忆更短，也更贴近当前 benchmark 的 atomic action 接口。

## 目标

- 构造一个比当前单轮 prompt 更接近真实 chat agent 的 tactical memory baseline。
- 只保留同一个玩家出牌回合内的短历史，避免跨回合保留过期信息。
- 保持 benchmark 的“一次一个 action”结构，不实现 CLI 式的一次输入多个操作。
- 让模型在同一个回合内记得刚刚的行动和计算，例如“已经打出 Unrelenting，下一张攻击变成 0 费”“刚刚 Battle Trance 抽了哪些牌”。
- 避免把完整 prompt 原样堆进 history，控制 token 在开源模型可承受范围内。
- 作为严肃实验中的一个明确 baseline：naive turn chat history，而不是人工长期摘要或可查询记忆系统。

## 当前单轮模式的问题

当前 `OpenAICompatAgent.choose()` 每一步都会重新构造：

```text
system: stable instruction
user: full prompt = rules + run summary + episode memory + state + legal actions
```

这意味着模型每一步都像重新开始。即使同一个玩家回合内刚刚打出了一张牌，它也只能从新的状态和 `Last game action result` 中恢复上下文。

当前 `factual_diff` memory 是压缩后的伪记忆，不是真正多轮对话。它适合记录长期事实，但不适合保留同一回合内的局部计划，例如：

- 先打 `Unrelenting`，再打免费攻击。
- 先打 `Battle Trance` 看新手牌，再继续判断是否防御。
- 先查看 draw/discard pile，再决定当前回合是否赌抽牌。
- 先打低费牌触发某个临时效果，再根据新状态继续行动。

如果直接把完整 prompt 作为每轮 `user` message 保留，token 会增长过快。因此第一版只保留当前玩家回合的短窗口，并且后续轮次使用 delta/update message。

## 实验模式命名

建议新增配置项：

```yaml
conversation_mode: single_turn
```

可选值：

- `single_turn`: 当前行为。每一步都是独立单轮 prompt。
- `turn_chat`: 只在同一个玩家出牌回合内保留短 chat history。

后续可扩展但不必第一版实现：

- `combat_chat`: 整场战斗内保留历史，作为更长上下文的后续实验。
- `turn_chat_full_window`: 同一出牌回合内最近 K 轮保留完整 user prompt 和 assistant response，作为 naive 但高 token 的对照。
- `turn_chat_delta`: 第一轮完整状态，后续只给 turn delta 和 legal actions。建议作为 `turn_chat` 的默认实现。

## 回合边界

`turn_chat` 的有效范围是同一个玩家出牌回合。

建议 active 条件：

- 当前状态属于战斗内的玩家回合决策，例如 `combat_play` 或由当前出牌触发的 `card_select`。
- 当前 combat round 与 buffer 中记录的 round 相同。
- 当前房间仍是同一场 combat，例如 `context.act/floor/room_type/boss` 没变。

建议重置条件：

- 模型选择了 `combat_end_turn`，下一次再进入 `combat_play` 时清空旧 turn history。
- `round` 发生变化。
- 离开当前战斗或进入非战斗流程，例如：
  - `combat_reward`
  - `card_reward`
  - `map_select`
  - `shop`
  - `rest_site`
  - 普通事件
  - `game_over`
- 进入新房间或新楼层。

这意味着同一场 Boss 战中的每个玩家回合都是独立短 conversation。跨回合信息如果需要保留，应由 `run_summary` 或未来的长期 memory 负责，而不是由 `turn_chat` 负责。

## Modal 状态处理

`card_select` 这类 modal 状态第一版直接当成同一出牌回合里的普通状态处理，不额外切换 single-turn prompt。

原因：

- benchmark 本来就是每次只输出一个 `action_id`，所以 modal 并不会迫使模型一次规划多个动作。
- modal 的状态也是当前回合的一部分，例如 `Burning Pact` 触发的“选择烧哪张牌”会影响同一回合后续手牌和能量判断。
- 把 modal 放进同一个 turn chat 可以让模型记住它为什么打出触发 modal 的牌，以及 modal 解决后继续沿用当前回合的短期上下文。

示例：

```text
assistant:
{"action_id":5,"action":"play Burning Pact","reason":"Need to exhaust Injury and draw into block."}

user:
Turn modal.

Decision: card_select
source_card: Burning Pact
prompt: Exhaust 1 card.
Current turn reminder:
Player: hp=19/96 block=0 energy=2/3
Enemy: Lagavulin Matriarch hp=57/222 intent=Attack 21
Selectable cards:
  [0] Strike
  [1] Defend
  [2] Injury

Legal actions:
...
```

模型仍然只回答一个 action，例如选择 `Injury`。modal 解决后，如果环境回到同一 round 的 `combat_play`，继续使用同一个 turn chat。

这个设计不会让模型输出多动作计划；它只是把 modal 的 user/assistant 消息也放进当前 turn history。

## View Action 处理

`view deck`、`view map`、`view draw pile`、`view discard pile` 等动作不推进游戏，但在 `turn_chat` 中很有价值。

它们应作为同一玩家回合内的信息扩展，而不是替换当前战斗上下文。

示例：

```text
assistant:
{"action_id":0,"action":"view draw pile","reason":"Need to know whether Battle Trance can find block."}

user:
Requested information: draw pile.

Draw pile:
  [0] Defend cost=1 block=3
  [1] Strike cost=1 damage=5
  ...

Current turn reminder:
Player: hp=19/96 block=0 energy=3/3
Enemy: Lagavulin Matriarch hp=57/222 intent=Attack 21
Hand:
  ...

Legal actions:
...
```

这样模型看完信息后仍然处在同一个 turn conversation 中，不会只看到一个孤立的 deck/pile view。

## 消息结构

### System Message

系统消息只放稳定规则：

```text
You are playing Slay the Spire 2 through a headless benchmark environment.
Choose exactly one legal action.
Use only an action_id from the legal action list.
Return only valid JSON.
Each legal action is atomic.
Do not invent actions.
Do not output a multi-action plan.
```

战斗数值语义也应放在稳定规则里：

```text
Enemy intent damage shown in the state is the engine-displayed damage after currently visible modifiers; do not add enemy Strength or other visible modifiers to that intent damage a second time.
Card damage and block values shown in the state are engine preview values after currently visible player modifiers such as Strength, Dexterity, Frail, Weak, and card-specific temporary effects; do not apply those visible modifiers to shown card damage or block a second time.
```

### Turn Initial User Message

进入一个新的玩家出牌回合时，给完整当前回合状态：

```text
New player turn.

Run summary:
...

Game state:
Decision: combat_play
Context: act=1 floor=17 room=Boss boss=Lagavulin Matriarch
Player: hp=19/96 block=0 gold=202 deck_size=20
Relics:
...
Potions:
...
Combat: round=6 energy=3/3 draw_pile=5 discard_pile=10
Player powers:
...
Enemies:
...
Hand:
...

Legal actions:
...
```

这个 initial message 可以接近当前 `build_llm_prompt()`，但不应加入整局 verbose episode memory。建议保留 `run_summary`，关闭普通 `memory_enabled`，让实验更干净。

### Turn Update User Message

同一玩家回合内，后续每个 `combat_play` 决策只给当前回合 update：

```text
Turn update.

Last game action result:
  action: play card 4: Unrelenting on enemy 0: Lagavulin Matriarch
  energy: 3/3 -> 1/3
  enemy Lagavulin Matriarch[0]: hp 57/222 -> 46/222

Current turn state:
Player: hp=19/96 block=0 energy=1/3 potions=0/3
Player powers:
  Strength(-1)
  Dexterity(-2)
Enemies:
  [0] Lagavulin Matriarch hp=46/222 block=0 intent=Attack 21 powers=Demise(9),Strength(2)
Hand:
  [0] Strike cost=0 can_play=True damage=5 target=AnyEnemy
  [1] Defend cost=1 can_play=True block=3
Piles: draw=5 discard=11 exhaust=0

Legal actions:
...
```

必要字段：

- `last_action_result`
- player HP / block / energy
- player powers
- potion slots and potion names
- enemy HP / block / intent / powers
- hand cards with cost, can_play, damage/block, target
- draw/discard/exhaust pile counts
- legal actions

不建议每轮重复：

- 完整地图
- 完整 deck
- 所有 relic 长描述
- 所有 card upgrade description
- 整局 Episode memory
- 旧回合的完整推理

### Assistant History

传给模型的 assistant 历史不应保留完整 CoT 或完整 `analysis` JSON，否则 token 会增长，也会把旧错误计算带入当前决策。

建议写进 chat history 的 assistant message 是 compact action record：

```json
{"action_id":9,"action":"play card 4: Unrelenting on enemy 0: Lagavulin Matriarch","reason":"Set next attack to 0 cost while dealing best immediate damage."}
```

完整 raw response 仍然写进 benchmark log，但不进入下一轮 chat history。

## 配置建议

初始配置：

```yaml
conversation_mode: turn_chat
turn_chat_window: 4
turn_chat_update_mode: delta
turn_chat_assistant_history: compact
turn_chat_reset_on_end_turn: true
```

建议第一组实验关闭现有短期 memory：

```yaml
memory_enabled: false
run_summary_enabled: true
prompt_style: analysis
include_json_state: false
```

理由：

- `memory_enabled: false` 可以让实验更干净，只测同一玩家回合内 chat history 的影响。
- `run_summary_enabled: true` 仍然提供宏观资源提醒，例如低 HP、无药水、Boss 名称。
- `include_json_state: false` 减少重复 token。
- modal 状态默认进入同一 turn chat；复杂性由环境的 atomic action 边界承担，而不是由 prompt 一次规划多个动作。

如果使用 8k 上下文开源模型：

```yaml
turn_chat_window: 3
```

如果使用 16k 到 32k 模型：

```yaml
turn_chat_window: 4
```

如果使用 DeepSeek API 或更大上下文模型：

```yaml
turn_chat_window: 6
```

## Token 预算估计

当前单轮模式：

- 每步 prompt 约 2k 到 3.5k tokens。
- 每步没有 chat history。
- 单次请求通常小于 4k 到 6k tokens。

整场 combat chat：

- Boss 战 8 到 15 步很容易超过 20k tokens。
- 旧回合的 intent、hand、block 很快过期。
- 不适合作为第一版默认实验。

推荐 `turn_chat + delta`：

- 每个新玩家回合 initial full state: 2k 到 3k tokens。
- 同回合后续 delta: 500 到 1000 tokens。
- compact assistant: 50 到 120 tokens。
- 普通回合 2 到 5 个动作，总上下文通常 3k 到 7k。
- 抽牌链或能量链回合可能 6 到 10 个动作，用 `turn_chat_window` 裁剪即可。

## 实验矩阵

建议先做三个 baseline：

| Method | 说明 | 主要问题 |
| --- | --- | --- |
| `single_turn` | 当前方法，每步独立 prompt | 无同回合历史 |
| `turn_chat_delta` | 同一玩家回合内 first full + later delta | 目标方法 |
| `turn_chat_full_window` | 同一玩家回合内保留最近 K 轮完整 prompt/response | token 更高的 naive 对照 |

后续再扩展：

| Method | 说明 |
| --- | --- |
| `combat_chat_delta` | 整场战斗内 first full + later delta |
| `combat_summary_memory` | 跨回合只保留自动摘要 |
| `queryable_memory` | 模型可主动查询 deck/map/pile/history |
| `subagent_turn_planner` | 子 agent 做同回合 tactical calculation，主 agent 只选 action |

## 评估指标

Run-level:

- max floor / act
- win rate
- death floor
- wall time
- prompt tokens / completion tokens
- invalid action fallback count
- engine error count

Turn-level tactical metrics:

- 是否错过本回合 lethal。
- 是否在有必要防御时误判 block。
- 是否对已预览 `damage/block` 二次应用 Strength/Dexterity/Frail/Weak。
- 是否在仍有明显可用牌时过早 end turn。
- 是否在无攻击 intent 时浪费能量打纯 block。
- 是否因为 view action 后丢失当前战斗状态。

Boss-specific:

- Act 1 boss 是否通过。
- Boss 战回合数。
- Boss 战中低 HP 回合是否选择正确防御线。
- Lagavulin Matriarch 的 `Demise`、Strength/Dexterity debuff、反伤击杀等情况是否正确处理。

## 实现计划

### 1. Agent 配置

在 `sts2_bench/agents.py` 中新增类型：

```python
ConversationMode = Literal["single_turn", "turn_chat"]
```

为 `OpenAICompatAgent` 增加：

- `conversation_mode`
- `turn_chat_window`
- `turn_chat_update_mode`
- `turn_chat_assistant_history`

第一版可以只支持 `OpenAICompatAgent`，`RandomAgent` 保持原样。

### 2. Turn Chat Buffer

新增一个 agent 内部对象，例如：

```python
@dataclass
class TurnChatBuffer:
    max_turns: int
    messages: list[dict[str, str]]
    active_turn_key: tuple[Any, ...] | None
```

`active_turn_key` 可以包含：

```python
(seed, act, floor, room_type, round)
```

职责：

- 判断是否进入新 player turn。
- 模型选择 `end_turn`、离开当前 combat、或 round 变化时清空。
- 把同一 round 内的 `combat_play`、`card_select`、view 信息更新都纳入同一个 turn history。
- 保留最近 K 个 `user/assistant` 对。
- 生成 OpenAI-compatible `messages`。
- 只把 compact assistant message 写进 history。

### 3. Prompt Builder 拆分

保持 `build_llm_prompt()` 不变，继续服务 `single_turn`。

为 turn chat 新增：

- `build_turn_initial_user_message(state, legal_actions, run_summary_text)`
- `build_turn_update_user_message(state, legal_actions)`
- `build_view_update_user_message(state, legal_actions)`
- `build_compact_assistant_message(action, parsed_response)`

这些函数仍然放在 `agents.py`。`context.py` 继续只负责 state text rendering，不写 instruction。

### 4. Runner 兼容

`runner.py` 当前通过 `agent.build_prompt()` 生成日志 prompt，再调用 `agent.choose(..., prompt=prompt)`。

为保持兼容：

- `single_turn` 下行为不变。
- `turn_chat` 下，`build_prompt()` 可以返回 transcript preview，用于日志和终端打印。
- `OpenAICompatAgent.choose()` 内部根据 `conversation_mode` 生成真实 chat messages。
- 日志里记录真实发送的 message preview，而不是只记录单条 prompt。

日志里建议新增：

- `conversation_mode`
- `message_count`
- `turn_chat_history_turns`
- `turn_key`
- `conversation_prompt_chars`

### 5. CLI 和 YAML 配置

在 `sts2_bench/benchmark.yaml` 中新增：

```yaml
conversation_mode: single_turn
turn_chat_window: 4
turn_chat_update_mode: delta
turn_chat_assistant_history: compact
```

在 `run_benchmark.py` 中新增环境变量和 CLI 参数：

- `STS2_BENCH_CONVERSATION_MODE`
- `STS2_BENCH_TURN_CHAT_WINDOW`
- `STS2_BENCH_TURN_CHAT_UPDATE_MODE`
- `STS2_BENCH_TURN_CHAT_ASSISTANT_HISTORY`

### 6. Tests

需要添加测试：

- 非战斗状态不会保留 turn chat history。
- 同一 `combat_play` round 会保留多轮 user/assistant。
- `end_turn` 后下一轮 `combat_play` 会清空。
- `round` 变化会清空。
- `turn_chat_window` 会裁剪旧消息。
- assistant history 不包含完整 calculations/candidates。
- `view deck/draw/discard/exhaust` 后仍然能在 messages 中看到 current turn reminder。
- `card_select` 会作为同一回合的普通 user message 进入 turn chat。
- `card_select` 解决后回到同一 round 的 `combat_play` 时，history 仍然保留。

## 前置语义约束

在正式评估 `turn_chat` 之前，应锁定这些 state 语义，避免模型把数值二次计算：

- enemy intent damage 是 engine-displayed damage，已经包含当前可见 enemy modifiers。
- hand/card 中展示的 `damage` 和 `block` 是 engine preview value，已经包含当前可见 player modifiers。
- 负数 Strength/Dexterity 应明确渲染成 debuff 语义，例如：
  - `Strength(-1): Decreases attack damage by 1.`
  - `Dexterity(-2): Reduces Block gained from cards by 2.`
- prompt 中应明确禁止对展示出来的 card damage/block 二次应用 Strength/Dexterity/Frail/Weak。

最新 `logs/bench_20260529_002806.log` 暴露的问题正是这里：模型把 `Defend block=3` 又按 `Dexterity(-2)` 扣了一次，误判为只能挡 1 点，选择直接 `end_turn` 后死亡。

## 后续记录空间

### Ideas

- 继续探索 `chat history` 分支时，不再使用多步 `turn_plan`，改成 **current-state turn_check**：
  - 每次 `combat_play` response 都可以给出一个短 `turn_check`，同时仍然只选择一个当前合法 `action_id`。
  - `turn_check` 是当前动作前的 tactical guardrail，不是后续动作计划；它只检查生存、斩杀、当前第一步是否安全。
  - 不建议为 check 单独增加一次 LLM call。额外 call 会增加成本，而且抽牌、随机 exhaust、发现牌、modal 状态都会让任何多步计划快速过期。
  - 历史里的 `turn_check` 只能作为参考，后续 `Turn update` 必须根据最新 state 重新计算，不能自动 follow 旧 check。
  - 设计目标是避免模型写出“看似完整但能量/随机性上不可执行”的手牌脚本。

建议 schema 草案：

```json
{
  "turn_check": {
    "primary_goal": "survive",
    "incoming_damage": "enemy attacks for 23",
    "survival_requirement": "need 23 block, lethal, stun, or potion line from the current state",
    "lethal_available": false,
    "first_action_rule": "redraw is safe because the current hand cannot block or kill and redraw is a current legal action",
    "must_recompute_after": [
      "card draw changes hand",
      "random exhaust/discard happens",
      "energy changes",
      "modal decision opens"
    ]
  },
  "action_id": 9,
  "reason": "Current hand cannot break 32 block; redraw is the best chance to find enough damage or block."
}
```

设计约束：

- `turn_check` 不能替代合法动作列表，最终仍必须只选择一个当前合法 `action_id`。
- 不允许让模型输出“本回合完整操作序列”并要求 runner 执行；runner 仍然逐动作 re-prompt。
- `turn_check` 进入 history 时应 compact，避免把完整 calculations/candidates 带入后续轮次。
- 如果旧 `turn_check` 与最新 `Last game action result` 冲突，最新真实 state 永远优先。

### Experiment Results

- 2026-05-29 单 seed smoke test 结果已记录在 `docs/sts2-benchmark-experiment-2026-05-29.md`。
  - `no_memory` baseline：`single_turn + memory_enabled=false`，死于 Act 1 floor 17。
  - `turn_chat`：`context_management.mode=turn_chat + memory_enabled=false`，死于 Act 2 floor 6。
  - `memory`：`single_turn + memory_enabled=true + factual_diff`，死于 Act 2 floor 7。
- 阶段性判断：`turn_chat` 相比最原始 `no_memory` baseline 有正向信号，但当前只有 `bench_0000` 一个 seed，不能写成统计结论。
- `turn_chat` 没有超过 `factual_diff memory`；因此更准确的定位是“有潜力的 chat-history 替代方法”，不是已验证优于 memory 的方法。
- 三次运行都使用了相同 `logging.out` 路径，结构化 JSONL 可能被后一次覆盖；本轮分析以 `logs/bench_*.log` 为准。后续实验应让每个 variant 使用唯一 `out` 路径。

### Current Preferred Setting

2026-06-18 单 seed 对比后，当前阶段先采用这组轻约束 baseline：

```bash
--turn-check \
--no-route-plan \
--run-plan-route-plan \
--route-plan-enforcement soft \
--no-reward-posterior-evaluator
```

含义：保留 combat 的 current-state `turn_check`，关闭 `map_select` 的 `route_plan_event` 强校验/事件机制，保留 OCC `RunPlan.route_plan` 作为软性路线叙事字段，并关闭选牌 posterior evaluator，减少方法混杂。现有观察里这组能到 Act 2 floor 7，但仍只是 `bench_0000` 的阶段性 baseline，不是统计最优结论。

下一步偏好诊断：在 `map_select` 和 `card_reward` 上开启 self-consistency，只重复采样并记录投票分布，不改变实际执行的 primary action，用来观察模型是否对“攻杀/精英路线”或某类选牌存在稳定偏好。

并行探索 `reflection` verifier：先收窄到 `card_reward`，为每个非 view 候选分别发起一次 forced counterfactual report，再由 verifier 只基于这些平等生成的候选报告横向选择最终动作。`diagnostic` 模式只记录 verifier 结论，`verifier_vote` 只有在高置信、分数差足够大、且没有高价值可拿牌被 skip 掉时才执行 verifier 选中的 action。当前不默认覆盖 `combat_reward/event/shop/rest_site`，因为三组 `bench_0000` 复跑显示宽范围 reflection 成本高、schema 错误集中在 `combat_reward`，并且 verifier 容易把“牌组更瘦”误当作目标，导致 Act 1 Boss 前资源不足。新的 rubric 要求优先判断选牌是否补足前期输出、稳定格挡、过牌、能量利用、AoE、精英/Boss 解法或明确协同；只有所有选项都低影响/冗余/太慢时才 skip。

### Bugs Found

- 2026-05-29 `turn_chat` 运行中出现 1 次 fallback：
  - log: `logs/bench_20260529_183627.log`
  - step: `178`
  - decision: `combat_play`
  - error: `Invalid control character at: line 2 column 7086 (char 7087)`
  - fallback action: `action_id=5`, `play card 0: Defend`
- 该 fallback 不是 action_id 越界，而是模型输出了严格 JSON 无法解析的内容。最可能原因是 JSON 字符串字段中出现未转义换行、tab 或其他控制字符。
- 当前 fallback 分支没有保存 `raw_response`，因此无法复原具体是哪一个字段破坏了 JSON。后续应在 fallback meta 中保存截断后的 raw response，例如前 1000 到 2000 字符，便于定位格式失败。
- 因为该实验设置 `llm_max_retries=0`，第一次 parse 失败后直接进入 fallback。后续调试可以考虑在小规模 run 中打开 1 次 retry，观察格式纠错是否能消除这类 fallback。
- 当前 fallback 策略是 `first_non_view_action`，在该局面选择了 `Defend`。这能保证继续跑，但不一定是策略最优。后续可以考虑区分 parse failure 与 model bad action，对 parse failure 做一次严格 JSON 修复重试优先于策略 fallback。
