# 终局牌组与选牌事件（中文）

- Source trace: `artifacts\nightly_latest_20260617_one_seed\traces\sts2bench_nightly_20260617_seed_a_r01_a01.jsonl`
- 说明: trace 的 `view_deck` 只记录 deck_size 和 viewed 标记，没有展开完整牌组；本文件用初始牌组和后续 `deck:` 变更重构终局牌组。
- 中文名来源: `localization_eng/cards.json` title 反查 key，再从 `localization_zhs/cards.json` 读取官方简中 title。

## 终局牌组（重构）

- 初始牌组张数: 10
- 终局牌组张数: 29
- result deck_size: 29
- 与 result deck_size 对齐: True
- 终局不同牌名数: 22
- 未映射牌名: 无

| 中文牌名 | English | 数量 |
|---|---|---:|
| 主宰 | Dominate | 1 |
| 凶恶 | Vicious | 2 |
| 原始力量 | Primal Force | 1 |
| 头槌 | Headbutt | 1 |
| 岩石铠甲 | Stone Armor | 2 |
| 彼岸咆哮+ | Howl from Beyond+ | 1 |
| 征召上前 | Summon Forth | 1 |
| 战鼓 | Drum of Battle | 1 |
| 打击 | Strike | 4 |
| 拆卸 | Dismantle | 1 |
| 旋风斩 | Whirlwind | 1 |
| 欺凌 | Bully | 1 |
| 焚烧 | Conflagration | 1 |
| 狂宴 | Feed | 1 |
| 狂怒+ | Rage+ | 1 |
| 独白 | Monologue | 1 |
| 痛击 | Bash | 1 |
| 连环拳 | One-Two Punch | 1 |
| 闪电霹雳 | Thunderclap | 1 |
| 防御 | Defend | 3 |
| 防御+ | Defend+ | 1 |
| 领域 | Demesne | 1 |

### 终局完整列表

1. 打击 / Strike
2. 打击 / Strike
3. 打击 / Strike
4. 打击 / Strike
5. 防御 / Defend
6. 防御 / Defend
7. 防御 / Defend
8. 痛击 / Bash
9. 原始力量 / Primal Force
10. 战鼓 / Drum of Battle
11. 防御+ / Defend+
12. 头槌 / Headbutt
13. 凶恶 / Vicious
14. 岩石铠甲 / Stone Armor
15. 狂怒+ / Rage+
16. 闪电霹雳 / Thunderclap
17. 欺凌 / Bully
18. 凶恶 / Vicious
19. 狂宴 / Feed
20. 主宰 / Dominate
21. 拆卸 / Dismantle
22. 彼岸咆哮+ / Howl from Beyond+
23. 焚烧 / Conflagration
24. 征召上前 / Summon Forth
25. 独白 / Monologue
26. 领域 / Demesne
27. 连环拳 / One-Two Punch
28. 岩石铠甲 / Stone Armor
29. 旋风斩 / Whirlwind

## Card Reward 候选与选择

| # | Step | 位置 | HP | 候选项（中文 / English） | 选择 |
|---:|---:|---|---:|---|---|
| 1 | 13 | A1F2 Monster | 80/80 | 1. 灰烬打击 / Ashen Strike<br>2. 战鼓 / Drum of Battle<br>3. 拆卸 / Dismantle | **战鼓 / Drum of Battle** |
| 2 | 24 | A1F4 Monster | 77/77 | 1. 头槌 / Headbutt<br>2. 坚毅 / True Grit<br>3. 熔融之拳 / Molten Fist | **头槌 / Headbutt** |
| 3 | 45 | A1F5 Monster | 77/77 | 1. 完美打击 / Perfected Strike<br>2. 惊逃 / Stampede<br>3. 凶恶 / Vicious | **凶恶 / Vicious** |
| 4 | 65 | A1F8 Monster | 77/77 | 1. 头槌 / Headbutt<br>2. 闪电霹雳 / Thunderclap<br>3. 火焰屏障 / Flame Barrier | **闪电霹雳 / Thunderclap** |
| 5 | 84 | A1F9 Elite | 68/77 | 1. 欺凌 / Bully<br>2. 铁斩波 / Iron Wave<br>3. 与我一战！ / Fight Me! | **欺凌 / Bully** |
| 6 | 102 | A1F11 Monster | 68/77 | 1. 凶恶 / Vicious<br>2. 突破 / Breakthrough<br>3. 愤怒 / Anger | **凶恶 / Vicious** |
| 7 | 125 | A1F12 Monster | 61/77 | 1. 巨像 / Colossus<br>2. 狂宴 / Feed<br>3. 剑柄打击 / Pommel Strike | **狂宴 / Feed** |
| 8 | 143 | A1F13 Monster | 58/77 | 1. 主宰 / Dominate<br>2. 火焰屏障 / Flame Barrier<br>3. 岩石铠甲 / Stone Armor | **主宰 / Dominate** |
| 9 | 162 | A1F14 Monster | 63/77 | 1. 放血 / Bloodletting<br>2. 彼岸咆哮 / Howl from Beyond<br>3. 无情猛攻 / Unrelenting | **彼岸咆哮 / Howl from Beyond** |
| 10 | 181 | A1F15 Monster | 62/77 | 1. 灰烬打击 / Ashen Strike<br>2. 拆卸 / Dismantle<br>3. 怨恨 / Spite | **拆卸 / Dismantle** |
| 11 | 220 | A1F17 Boss | 38/77 | 1. 地狱狂徒 / Hellraiser<br>2. 恶魔之焰 / Fiend Fire<br>3. 焚烧 / Conflagration | **焚烧 / Conflagration** |
| 12 | 236 | A2F2 Monster | 77/77 | 1. 剑柄打击 / Pommel Strike<br>2. 征召上前 / Summon Forth<br>3. 预判 / Anticipate | **征召上前 / Summon Forth** |
| 13 | 252 | A2F3 Monster | 76/77 | 1. 人工合成 / Synthesis<br>2. 高速脱离 / Boost Away<br>3. 独白 / Monologue | **独白 / Monologue** |
| 14 | 274 | A2F4 Monster | 54/77 | 1. 领域 / Demesne<br>2. 热修复 / Hotfix<br>3. 连续反弹 / Ricochet | **领域 / Demesne** |

## 其他入牌/改牌事件

| Step | 位置 | 类型 | 动作 | 牌组变化（中文 / English） |
|---:|---|---|---|---|
| 1 | A1F1 Event | 事件/选择改牌 | `select card 0: Strike` | -打击 / Strike<br>+原始力量 / Primal Force |
| 13 | A1F2 Monster | 战斗奖励选牌 | `pick card 1: Drum of Battle` | +战鼓 / Drum of Battle |
| 16 | A1F3 Event | 事件改牌 | `choose option 0: Decipher` | -防御 / Defend<br>+防御+ / Defend+ |
| 24 | A1F4 Monster | 战斗奖励选牌 | `pick card 0: Headbutt` | +头槌 / Headbutt |
| 45 | A1F5 Monster | 战斗奖励选牌 | `pick card 2: Vicious` | +凶恶 / Vicious |
| 47 | A1F6 Shop | 商店购买 | `buy card 4: Stone Armor for 73 gold` | +岩石铠甲 / Stone Armor |
| 48 | A1F6 Shop | 商店购买 | `buy card 2: Rage for 78 gold` | +狂怒 / Rage |
| 53 | A1F7 RestSite | 营火升级 | `select card 13: Rage` | -狂怒 / Rage<br>+狂怒+ / Rage+ |
| 65 | A1F8 Monster | 战斗奖励选牌 | `pick card 1: Thunderclap` | +闪电霹雳 / Thunderclap |
| 84 | A1F9 Elite | 战斗奖励选牌 | `pick card 0: Bully` | +欺凌 / Bully |
| 102 | A1F11 Monster | 战斗奖励选牌 | `pick card 0: Vicious` | +凶恶 / Vicious |
| 125 | A1F12 Monster | 战斗奖励选牌 | `pick card 1: Feed` | +狂宴 / Feed |
| 143 | A1F13 Monster | 战斗奖励选牌 | `pick card 0: Dominate` | +主宰 / Dominate |
| 162 | A1F14 Monster | 战斗奖励选牌 | `pick card 1: Howl from Beyond` | +彼岸咆哮 / Howl from Beyond |
| 181 | A1F15 Monster | 战斗奖励选牌 | `pick card 1: Dismantle` | +拆卸 / Dismantle |
| 185 | A1F16 RestSite | 营火升级 | `select card 18: Howl from Beyond` | -彼岸咆哮 / Howl from Beyond<br>+彼岸咆哮+ / Howl from Beyond+ |
| 220 | A1F17 Boss | 战斗奖励选牌 | `pick card 2: Conflagration` | +焚烧 / Conflagration |
| 236 | A2F2 Monster | 战斗奖励选牌 | `pick card 1: Summon Forth` | +征召上前 / Summon Forth |
| 252 | A2F3 Monster | 战斗奖励选牌 | `pick card 2: Monologue` | +独白 / Monologue |
| 274 | A2F4 Monster | 战斗奖励选牌 | `pick card 0: Demesne` | +领域 / Demesne |
| 277 | A2F5 Shop | 商店购买 | `buy card 2: One-Two Punch for 143 gold` | +连环拳 / One-Two Punch |
| 279 | A2F5 Shop | 商店购买 | `buy card 4: Stone Armor for 78 gold` | +岩石铠甲 / Stone Armor |
| 280 | A2F5 Shop | 商店购买 | `buy card 1: Whirlwind for 37 gold` | +旋风斩 / Whirlwind |

## 直接可读结论

- 本局终局 deck_size 为 29，重构牌组为 29。
- 所有 card reward 的候选项都优先从 trace state.cards 读取；compact state 缺失时，从 legal action label 恢复。
- 最后一次 card reward 是 step 274，候选为 领域 / Demesne、热修复 / Hotfix、连续反弹 / Ricochet，选择 领域 / Demesne。
