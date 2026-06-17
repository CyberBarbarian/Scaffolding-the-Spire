# STS2 Trace Replay

- Source: `artifacts\nightly_latest_20260617_one_seed\traces\sts2bench_nightly_20260617_seed_a_r01_a01.jsonl`
- Seed: `sts2bench_nightly_20260617_seed_a`

## Route And Room HP

| # | Act | Floor | Room | Node | Entry HP | Exit HP | Loss | Entered by |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 1 | 1 | 2 | Monster | (4,1) | 80/80 | 80/80 | 0 | go to Monster at col=4 row=1 |
| 2 | 1 | 3 | Event | (4,2) | 80/80 | 77/77 | 3 | go to Unknown at col=4 row=2 |
| 3 | 1 | 4 | Monster | (4,3) | 77/77 | 77/77 | 0 | go to Monster at col=4 row=3 |
| 4 | 1 | 5 | Monster | (4,4) | 77/77 | 77/77 | 0 | go to Monster at col=4 row=4 |
| 5 | 1 | 6 | Shop | (4,5) | 77/77 | 77/77 | 0 | go to Shop at col=4 row=5 |
| 6 | 1 | 7 | RestSite | (3,6) | 77/77 | 77/77 | 0 | go to RestSite at col=3 row=6 |
| 7 | 1 | 7 | RestSite | (3,6) | 77/77 | 77/77 | 0 | go to RestSite at col=3 row=6 |
| 8 | 1 | 8 | Monster | (4,7) | 77/77 | 77/77 | 0 | go to Monster at col=4 row=7 |
| 9 | 1 | 9 | Elite | (4,8) | 77/77 | 68/77 | 9 | go to Elite at col=4 row=8 |
| 10 | 1 | 10 | Treasure | (4,9) | 68/77 | 68/77 | 0 | go to Treasure at col=4 row=9 |
| 11 | 1 | 11 | Monster | (4,10) | 68/77 | 68/77 | 0 | go to Unknown at col=4 row=10 |
| 12 | 1 | 12 | Monster | (4,11) | 68/77 | 61/77 | 7 | go to Monster at col=4 row=11 |
| 13 | 1 | 13 | Monster | (5,12) | 61/77 | 58/77 | 3 | go to Monster at col=5 row=12 |
| 14 | 1 | 14 | Monster | (5,13) | 58/77 | 63/77 | 0 | go to Monster at col=5 row=13 |
| 15 | 1 | 15 | Monster | (5,14) | 63/77 | 62/77 | 1 | go to Monster at col=5 row=14 |
| 16 | 1 | 16 | RestSite | (5,15) | 62/77 | 62/77 | 0 | go to RestSite at col=5 row=15 |
| 17 | 1 | 17 | Boss | (3,16) | 62/77 | 38/77 | 24 | go to Boss at col=3 row=16 |
| 18 | 2 | 1 | Event | (3,0) | 77/77 | 77/77 | 0 | go to Ancient at col=3 row=0 |
| 19 | 2 | 2 | Monster | (1,1) | 77/77 | 77/77 | 0 | go to Monster at col=1 row=1 |
| 20 | 2 | 3 | Monster | (2,2) | 77/77 | 76/77 | 1 | go to Monster at col=2 row=2 |
| 21 | 2 | 4 | Monster | (3,3) | 76/77 | 54/77 | 22 | go to Monster at col=3 row=3 |
| 22 | 2 | 5 | Shop | (2,4) | 54/77 | 54/77 | 0 | go to Shop at col=2 row=4 |
| 23 | 2 | 6 | Monster | (3,5) | 54/77 | 0/77 | 54 | go to Monster at col=3 row=5 |

## Map Decisions

- step 2 A1F1 HP 80/80: selected `go to Monster at col=4 row=1`
  - legal routes: `go to Monster at col=1 row=1`, `go to Monster at col=2 row=1`, `go to Monster at col=4 row=1`, `go to Monster at col=6 row=1`
  - reason: Choosing col4 path for early event, shop, and rest to strengthen deck.
- step 15 A1F2 HP 80/80: selected `go to Unknown at col=4 row=2`
  - legal routes: `go to Unknown at col=4 row=2`
  - reason: Only available map node; follow planned col4 path for event, shop, rest.
- step 18 A1F3 HP 77/77: selected `go to Monster at col=4 row=3`
  - legal routes: `go to Monster at col=4 row=3`
  - reason: Move to Monster as planned.
- step 26 A1F4 HP 77/77: selected `go to Monster at col=4 row=4`
  - legal routes: `go to Monster at col=4 row=4`
  - reason: Only available map node; continue along planned col4 path for shop and rest.
- step 46 A1F5 HP 77/77: selected `go to Shop at col=4 row=5`
  - legal routes: `go to Shop at col=4 row=5`
  - reason: Only available map node matches planned route; shop visit to spend gold (151) and use potion slot space.
- step 50 A1F6 HP 77/77: selected `go to RestSite at col=3 row=6`
  - legal routes: `go to RestSite at col=3 row=6`
  - reason: Proceed to RestSite to heal and rest after shop purchases; no other map choices available.
- step 50 A1F6 HP 77/77: selected `go to RestSite at col=3 row=6`
  - legal routes: `go to RestSite at col=3 row=6`
  - reason: Only available map node; go to RestSite to heal.
- step 54 A1F7 HP 77/77: selected `go to Monster at col=4 row=7`
  - legal routes: `go to Unknown at col=3 row=7`, `go to Monster at col=4 row=7`
  - reason: Choose Monster for safe gold/card reward; path from col=4 row=7 offers flexibility.
- step 66 A1F8 HP 77/77: selected `go to Elite at col=4 row=8`
  - legal routes: `go to Elite at col=4 row=8`
  - reason: Only legal map node; go to Elite to advance act 1. Updated route plan with new node.
- step 86 A1F9 HP 68/77: selected `go to Treasure at col=4 row=9`
  - legal routes: `go to Treasure at col=4 row=9`
  - reason: Only legal map node; create new route plan with Treasure.
- step 88 A1F10 HP 68/77: selected `go to Unknown at col=4 row=10`
  - legal routes: `go to Unknown at col=4 row=10`
  - reason: Only legal map node; advance to the next floor.
- step 103 A1F11 HP 68/77: selected `go to Monster at col=4 row=11`
  - legal routes: `go to Monster at col=4 row=11`
  - reason: Proceed to the only available node after defeating The Kin. Plan to rest at the next opportunity.
- step 126 A1F12 HP 61/77: selected `go to Monster at col=5 row=12`
  - legal routes: `go to RestSite at col=4 row=12`, `go to Monster at col=5 row=12`
  - reason: Choosing Monster to gain rewards and feed HP; Rest is safer but less beneficial for scaling.
- step 144 A1F13 HP 58/77: selected `go to Monster at col=5 row=13`
  - legal routes: `go to Monster at col=5 row=13`
  - reason: Proceed to the only available Monster node; set route plan for another monster and rest before the boss for Max HP scaling and safety.
- step 163 A1F14 HP 63/77: selected `go to Monster at col=5 row=14`
  - legal routes: `go to Monster at col=5 row=14`
  - reason: Advancing to the Monster node to gain combat rewards and HP scaling from Feed; preparing for the boss fight with a Rest afterwards.
- step 182 A1F15 HP 62/77: selected `go to RestSite at col=5 row=15`
  - legal routes: `go to RestSite at col=5 row=15`
  - reason: Only available map node is RestSite; plan was to rest after the boss fight to heal and potentially upgrade a key card for act 2.
- step 186 A1F16 HP 62/77: selected `go to Boss at col=3 row=16`
  - legal routes: `go to Boss at col=3 row=16`
  - reason: Advancing to act 1 boss The Kin; deck has Feed, Dominate, Howl from Beyond+ for AoE; 62 HP with Explosive Ampoule potion for burst.
- step 221 A2FNone HP 38/77: selected `go to Ancient at col=3 row=0`
  - legal routes: `go to Ancient at col=3 row=0`
  - reason: Proceed to the only available map node to begin act 2 exploration.
- step 223 A2F1 HP 77/77: selected `go to Monster at col=1 row=1`
  - legal routes: `go to Monster at col=1 row=1`, `go to Monster at col=6 row=1`
  - reason: Choosing left Monster route for immediate combat and card rewards.
- step 239 A2F2 HP 77/77: selected `go to Monster at col=2 row=2`
  - legal routes: `go to Unknown at col=0 row=2`, `go to Monster at col=2 row=2`
  - reason: Monster provides combat rewards to scale deck with attacks; preferable to Unknown event.
- step 253 A2F3 HP 76/77: selected `go to Monster at col=3 row=3`
  - legal routes: `go to Monster at col=3 row=3`, `go to Monster at col=1 row=3`
  - reason: Monster provides combat rewards to continue scaling attack deck; right path may lead to shop.
- step 275 A2F4 HP 54/77: selected `go to Shop at col=2 row=4`
  - legal routes: `go to Shop at col=2 row=4`, `go to Unknown at col=4 row=4`
  - reason: Shop to spend gold on cards or relics for act 2 scaling.
- step 282 A2F5 HP 54/77: selected `go to Monster at col=3 row=5`
  - legal routes: `go to Unknown at col=2 row=5`, `go to Monster at col=3 row=5`
  - reason: Monster fight to test newly acquired Whirlwind, One-Two Punch, and Stone Armor; earn gold/card reward to further scale for act 2 boss.

## Route plan events

- step 2 `route_plan_created`: Col4 path offers early event at row2, shop at row5, and rest at row6.
  - selected node: Monster(4,1)
  - reason tags: early_shop_rest
  - route validity: topological
  - new plan: Monster(4,1)
- step 15 `route_plan_created`: Col4 path leads to shop and rest for strengthening deck; prioritize resources over early elites.
  - selected node: Unknown(4,2)
  - reason tags: col4_path, shop_rest_strategy
  - route validity: partial_invalid_suffix_trimmed
  - new plan: Unknown(4,2) -> Monster(4,3) -> Monster(4,4) -> Shop(4,5)
  - active remaining: Monster(4,3) -> Monster(4,4) -> Shop(4,5)
- step 18 `route_plan_followed`: Following planned path to Monster at col=4 row=3 then shop and rest.
  - selected node: Monster(4,3)
  - active remaining: Monster(4,4) -> Shop(4,5)
- step 26 `route_plan_followed`: Following planned route: Monster at col=4 row=4, then shop at col=4 row=5, then rest at col=4 row=6.
  - selected node: Monster(4,4)
  - active remaining: Shop(4,5)
- step 46 `route_plan_followed`: Follow planned route to shop to spend gold and acquire cards/relics, then rest at planned RestSite on col=4 row=6.
  - selected node: Shop(4,5)
- step 50 `route_plan_created`: Proceed to RestSite to heal after shop purchases; no other map choices available.
  - selected node: RestSite(3,6)
  - reason tags: heal, only_choice
  - route validity: topological
  - new plan: RestSite(3,6)
- step 54 `route_plan_created`: Full HP, upgraded deck; normal fight gives gold and card reward to strengthen further before later elites/boss.
  - selected node: Monster(4,7)
  - reason tags: gold_and_cards, test_deck
  - route validity: topological
  - new plan: Monster(4,7)
- step 66 `route_plan_created`: Choosing the only available path from current position; Elite is a risk but our deck with Thunderclap, Rage+, and strong attacks can handle it. Plan to proceed to elite for relic and reward.
  - selected node: Elite(4,8)
  - reason tags: route_created
  - route validity: topological
  - new plan: Elite(4,8)
- step 86 `route_plan_created`: After elite, take Treasure for gold/card reward; act 1 boss approaching.
  - selected node: Treasure(4,9)
  - reason tags: map_changed
  - route validity: topological
  - new plan: Treasure(4,9)
- step 88 `route_plan_created`: Only legal map node; create new route plan with Unknown.
  - selected node: Unknown(4,10)
  - route validity: topological
  - new plan: Unknown(4,10)
- step 103 `route_plan_created`: Only legal map node; Monster leads to a Rest site which is valuable before the next elite or boss.
  - selected node: Monster(4,11)
  - reason tags: only_path
  - route validity: topological
  - new plan: Monster(4,11)
- step 126 `route_plan_created`: Take Monster node to farm max HP with Feed and build deck for act 2; HP is sufficient at 61/77.
  - selected node: Monster(5,12)
  - reason tags: hp_stable, feed_farming
  - route validity: topological
  - new plan: Monster(5,12)
- step 144 `route_plan_created`: Route to take two monsters for Feed Max HP scaling and rewards, then rest before the Act 1 boss.
  - selected node: Monster(5,13)
  - reason tags: max_hp_scaling, feed_priority
  - route validity: topological
  - new plan: Monster(5,13) -> Monster(5,14) -> RestSite(5,15)
  - active remaining: Monster(5,14) -> RestSite(5,15)
- step 163 `route_plan_followed`: Following the pre-existing route plan: Monster (row 13) -> Monster (row 14) -> Rest (row 15) before the boss. This Monster node is the next step to gain HP scaling from Feed and card rewards before the Rest.
  - selected node: Monster(5,14)
  - active remaining: RestSite(5,15)
- step 182 `route_plan_followed`: Following the existing route plan to rest before entering act 2, which provides HP recovery and upgrade opportunity.
  - selected node: RestSite(5,15)
- step 186 `route_plan_created`: Only available node is the Boss; proceed to act 1 boss fight with strong deck and HP.
  - selected node: Boss(3,16)
  - reason tags: proceed_to_boss
  - route validity: topological
  - new plan: Boss(3,16)
- step 221 `route_plan_created`: Start of act 2; only available node from current position.
  - selected node: Ancient(3,0)
  - reason tags: start
  - route validity: topological
  - new plan: Ancient(3,0)
- step 223 `route_plan_created`: Starting act 2 with a Monster node to gain card rewards; deck has strong AoE with Conflagration and Howl from Beyond+, and we want to build on our attack synergy.
  - selected node: Monster(1,1)
  - reason tags: route_created
  - route validity: topological
  - new plan: Monster(1,1)
- step 239 `route_plan_created`: Choosing a Monster for combat rewards and card scaling; full HP allows safe combat.
  - selected node: Monster(2,2)
  - reason tags: farm_rewards
  - route validity: topological
  - new plan: Monster(2,2)
- step 253 `route_plan_created`: Choosing right Monster path to gain card rewards and continue building attack package; potential shop at next floor.
  - selected node: Monster(3,3)
  - reason tags: attack_build, combat_rewards
  - route validity: topological
  - new plan: Monster(3,3)
- step 275 `route_plan_created`: Shop with 266 gold offers chance to buy useful cards or relics.
  - selected node: Shop(2,4)
  - reason tags: high_gold
  - route validity: topological
  - new plan: Shop(2,4)
- step 282 `route_plan_created`: Choosing Monster over Unknown to test new cards and gain combat rewards.
  - selected node: Monster(3,5)
  - reason tags: test_new_cards, safe_choice
  - route validity: topological
  - new plan: Monster(3,5)

## Route plan validation errors

- step 50 `route_plan_missing_created_event`: No active route plan exists; emit route_plan_created before selecting a map node.
  - retry: 0

## Card Rewards

- step 13 offered ['灰烬打击 / Ashen Strike', '战鼓 / Drum of Battle', '拆卸 / Dismantle'] -> `战鼓 / Drum of Battle`
- step 24 offered ['头槌 / Headbutt', '坚毅 / True Grit', '熔融之拳 / Molten Fist'] -> `头槌 / Headbutt`
- step 45 offered ['完美打击 / Perfected Strike', '惊逃 / Stampede', '凶恶 / Vicious'] -> `凶恶 / Vicious`
- step 65 offered ['头槌 / Headbutt', '闪电霹雳 / Thunderclap', '火焰屏障 / Flame Barrier'] -> `闪电霹雳 / Thunderclap`
- step 84 offered ['欺凌 / Bully', '铁斩波 / Iron Wave', '与我一战！ / Fight Me!'] -> `欺凌 / Bully`
- step 102 offered ['凶恶 / Vicious', '突破 / Breakthrough', '愤怒 / Anger'] -> `凶恶 / Vicious`
- step 125 offered ['巨像 / Colossus', '狂宴 / Feed', '剑柄打击 / Pommel Strike'] -> `狂宴 / Feed`
- step 143 offered ['主宰 / Dominate', '火焰屏障 / Flame Barrier', '岩石铠甲 / Stone Armor'] -> `主宰 / Dominate`
- step 162 offered ['放血 / Bloodletting', '彼岸咆哮 / Howl from Beyond', '无情猛攻 / Unrelenting'] -> `彼岸咆哮 / Howl from Beyond`
- step 181 offered ['灰烬打击 / Ashen Strike', '拆卸 / Dismantle', '怨恨 / Spite'] -> `拆卸 / Dismantle`
- step 220 offered ['地狱狂徒 / Hellraiser', '恶魔之焰 / Fiend Fire', '焚烧 / Conflagration'] -> `焚烧 / Conflagration`
- step 236 offered ['剑柄打击 / Pommel Strike', '征召上前 / Summon Forth', '预判 / Anticipate'] -> `征召上前 / Summon Forth`
- step 252 offered ['人工合成 / Synthesis', '高速脱离 / Boost Away', '独白 / Monologue'] -> `独白 / Monologue`
- step 274 offered ['领域 / Demesne', '热修复 / Hotfix', '连续反弹 / Ricochet'] -> `领域 / Demesne`

## Rest Sites

- step 52 A1F7 HP 77/77: Smith; smith `Rage`
- step 184 A1F16 HP 62/77: Smith; smith `Howl from Beyond`

## Shops

- A1F6:
  - buy card 4: Stone Armor for 73 gold
  - buy card 2: Rage for 78 gold
  - leave shop
- A2F5:
  - buy card 2: One-Two Punch for 143 gold
  - buy card 4: Stone Armor for 78 gold
  - buy card 1: Whirlwind for 37 gold
  - leave shop

## High Risk Route Events
