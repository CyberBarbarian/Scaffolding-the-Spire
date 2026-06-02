# sts2-agent-bench

`sts2-agent-bench` is the private BG-Agent research repository for comparing
LLM and hybrid-agent methods on Slay the Spire 2. It currently carries the
headless `sts2-cli` runtime snapshot plus the `sts2_bench` benchmark scaffold so
agent experiments, traces, metrics, and failure analysis can run from one
checkout.

The public CLI-only runtime is maintained separately at
[CyberBarbarian/sts2-cli](https://github.com/CyberBarbarian/sts2-cli). Keep
CLI/headless adapter fixes there; keep LLM policy, benchmark, trace, and
research-method work here.

> Runtime lineage: this repository started from the
> [CyberBarbarian/sts2-cli](https://github.com/CyberBarbarian/sts2-cli) fork of
> [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli), then became the
> private research home for LLM agent and benchmark work.
>
> The fork focuses on CLI, JSON protocol, headless adapter, state export, and
> terminal presentation fixes. It does not intentionally change Slay the Spire 2
> card, relic, enemy, event, combat, or reward semantics.

## English

- [What This Is](#what-this-is)
- [Quick Start](#quick-start)
- [Play](#play)
- [JSON Protocol](#json-protocol)
- [Logs And Bug Reports](#logs-and-bug-reports)
- [Testing](#testing)
- [Release Notes](#release-notes)
- [中文说明](#中文说明)

## What This Is

The included `sts2-cli` runtime runs the real Slay the Spire 2 engine
headlessly and exposes it as an interactive terminal game, a stdin/stdout JSON
protocol for agents, and a reproducible logging surface for
CLI/export/headless debugging.

You must own and install Slay the Spire 2 through Steam. This repository does
not contain or redistribute game DLLs.

![English demo](docs/demo_en.gif)

This fork is a rolling BG-Agent build. It includes additional fixes for
Windows/headless launch stability, pending selections, target-specific combat
preview exports, shop/rest/treasure/map/potion/reward affordances, bilingual
startup paths, and focused pytest workflows.

The fork's `main` branch is the recommended branch for users. The active
development branch may also exist, but `main` is kept fast-forwarded after
tested fix batches.

## Quick Start

Requirements:

- Slay the Spire 2 installed through Steam.
- Python 3.9 or newer.
- .NET 9 SDK or newer.
- On Unix-like systems, `bash` for `setup.sh` / `copy_dlls.sh`.

Windows:

```cmd
git clone https://github.com/CyberBarbarian/sts2-cli.git
cd sts2-cli
.\sts2-cli.bat
```

PowerShell requires the `.\` prefix for commands in the current directory. For
the Chinese UI, run `.\sts2-cli-zh.bat`. Double-clicking either launcher still
works.

Windows entry points:

| File | Purpose |
| --- | --- |
| `sts2-cli.bat` | English game UI launcher; use `.\sts2-cli.bat` in PowerShell or double-click the file |
| `sts2-cli-zh.bat` | Chinese game UI launcher; use `.\sts2-cli-zh.bat` in PowerShell or double-click the file |
| `setup-windows.bat` | Setup/build only, no game start |

macOS, Linux, or Git Bash:

```bash
git clone https://github.com/CyberBarbarian/sts2-cli.git
cd sts2-cli
./setup.sh
python3 python/play.py
```

The first run tries to locate the Steam install, copy required DLLs into
`lib/`, and build the headless adapter. Do not commit or redistribute `lib/`.
For cross-platform DLL details, see
[docs/cross-platform-usage.md](docs/cross-platform-usage.md).

## Play

```bash
python3 python/play.py                         # menu: language, character, ascension
python3 python/play.py --lang zh               # Chinese UI mode
python3 python/play.py --lang both             # bilingual display where supported
python3 python/play.py --character Silent      # choose character
python3 python/play.py --ascension 10          # choose ascension
python3 python/play.py --continue saves/run.save
python3 python/play.py --load saves/replay.json
python3 python/play.py --no-log
```

Supported characters:

- `Ironclad`
- `Silent`
- `Defect`
- `Regent`
- `Necrobinder`

Supported ascension range is currently `0-10`.

Common in-game commands:

```text
help                 show help
map                  show map
deck                 show deck
draw                 show draw pile
discard              show discard pile
exhaust              show exhaust pile
potions              show potions
relics               show relics
quit                 quit and offer save

Map                  enter a visible path number
Combat               0, 0@1, 0>1, e, p0
Reward               reward index, card pick, skip where allowed
Rest                 option index
Event                option index / leave when exposed by the engine
Shop                 c0, r0, p0, rm, leave
Treasure             exposed option index
Card selection       comma/space separated indices, or skip when optional
```

Combat input forms:

```text
0                  play card [0]
0@1 or 0>1        play card [0] on enemy [1]
e                  end turn
p0, p1             use potion slot
seq 1 2 4          play cards by the hand snapshot currently shown
play 1 2 4         same as seq
1,2,4              short queued-play form
seq 1@0 4@2        queued targeted cards for multi-enemy fights
```

Queued play is bound to the original hand and enemy list. AOE and no-target
cards do not need targets. If a later action requires a new selection, has an
invalid target, loses the card, lacks energy, or becomes unplayable, the queue
stops instead of guessing.

## JSON Protocol

For programmatic control, start the headless process and send JSON lines:

```bash
dotnet run --project src/Sts2Headless/Sts2Headless.csproj
```

```json
{"cmd": "start_run", "character": "Ironclad", "seed": "test", "ascension": 0}
{"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 0}}
{"cmd": "action", "action": "end_turn"}
{"cmd": "action", "action": "select_map_node", "args": {"col": 3, "row": 1}}
{"cmd": "action", "action": "skip_card_reward"}
{"cmd": "quit"}
```

Each command returns a decision/state payload such as `map_select`,
`combat_play`, `card_select`, `bundle_select`, `card_reward`, `combat_reward`,
`treasure`, `rest_site`, `event_choice`, `shop`, or `game_over`.

## Logs And Bug Reports

Interactive runs write JSONL logs under `logs/` by default. Logs include state
snapshots and actions with timestamps.

```bash
python3 python/play.py --no-log
```

When reporting a CLI/headless bug, include seed, character, ascension, exact
visible decision state, command entered, relevant `logs/*.jsonl` file when
available, and stderr/exception text if the headless process failed.

Normal game outcomes are not CLI bugs unless the CLI hides, corrupts, or blocks
an engine decision.

## Testing

Use focused pytest selections while iterating:

```bash
python -m pytest tests/test_event.py -q
python -m pytest tests/test_dynamic_card_stats.py tests/test_combat.py -q
python -m pytest tests/test_treasure.py tests/test_shop.py tests/test_potions.py -q
python -m pytest tests/test_process_reuse.py -q
```

For broad local coverage without the slowest runs:

```bash
python -m pytest -m "not slow" -q
```

Use full `python -m pytest -q` for pre-release or overnight validation.

## Release Notes

There is currently no packaged binary release. This is intentional for now:
game DLLs are local Steam files and must not be redistributed.

A safe public release should be a source/tag release that contains scripts,
docs, tests, and source code only. Users still run `sts2-cli.bat`,
`setup-windows.bat`, or `./setup.sh` locally to copy their own game files and
build the adapter.

## 中文说明

- [这是什么](#这是什么)
- [快速开始](#快速开始)
- [启动游戏](#启动游戏)
- [JSON 协议](#json-协议-1)
- [日志和 bug 报告](#日志和-bug-报告)
- [测试](#测试)
- [发布说明](#发布说明)

## 这是什么

`sts2-cli` 使用真实的杀戮尖塔 2 游戏引擎，并把它以无头命令行形式暴露出来：可以直接在终端里玩，也可以通过 stdin/stdout JSON 协议给 agent 或 benchmark 使用。

你需要自己在 Steam 中拥有并安装 Slay the Spire 2。本仓库不包含、也不会重新分发游戏 DLL。

![中文演示](docs/demo_zh.gif)

本 fork 是 BG-Agent 使用的滚动版本，重点补全 Windows/headless 稳定性、卡牌/事件/奖励选择、不同目标下的战斗预览、商店/休息点/宝箱/地图/药水/奖励交互、双语启动入口和 focused pytest 流程。

推荐普通用户使用本 fork 的 `main` 分支。开发分支可能同时存在，但经过验证的修复会同步推进到 `main`。

## 快速开始

依赖：

- 通过 Steam 安装 Slay the Spire 2。
- Python 3.9 或更新版本。
- .NET 9 SDK 或更新版本。
- macOS/Linux/Git Bash 下，`setup.sh` 和 `copy_dlls.sh` 需要 `bash`。

Windows：

```cmd
git clone https://github.com/CyberBarbarian/sts2-cli.git
cd sts2-cli
.\sts2-cli.bat
```

PowerShell 需要用 `.\` 前缀运行当前目录下的脚本。中文界面入口是
`.\sts2-cli-zh.bat`；直接双击 bat 文件仍然可以启动。

Windows 入口：

| 文件 | 用途 |
| --- | --- |
| `sts2-cli.bat` | 英文游戏界面入口；PowerShell 中运行 `.\sts2-cli.bat`，也可双击 |
| `sts2-cli-zh.bat` | 中文游戏界面入口；PowerShell 中运行 `.\sts2-cli-zh.bat`，也可双击 |
| `setup-windows.bat` | 只做安装和构建，不进入游戏 |

macOS、Linux 或 Git Bash：

```bash
git clone https://github.com/CyberBarbarian/sts2-cli.git
cd sts2-cli
./setup.sh
python3 python/play.py
```

首次运行会尝试自动定位 Steam 游戏目录，把所需 DLL 复制到 `lib/`，并构建 headless adapter。不要提交或分发 `lib/`。跨平台复制 DLL 的细节见
[docs/cross-platform-usage.md](docs/cross-platform-usage.md)。

## 启动游戏

```bash
python3 python/play.py                         # 菜单：语言、角色、进阶
python3 python/play.py --lang zh               # 中文界面模式
python3 python/play.py --lang both             # 支持位置显示中英双语
python3 python/play.py --character Silent      # 指定角色
python3 python/play.py --ascension 10          # 指定进阶
python3 python/play.py --continue saves/run.save
python3 python/play.py --load saves/replay.json
python3 python/play.py --no-log
```

当前支持角色：`Ironclad`、`Silent`、`Defect`、`Regent`、`Necrobinder`。当前支持进阶范围是 `0-10`。

游戏内常用命令：

```text
help                 显示帮助
map                  显示地图
deck                 查看牌组
draw                 查看抽牌堆
discard              查看弃牌堆
exhaust              查看消耗堆
potions              查看药水
relics               查看遗物
quit                 退出并提示保存

地图                 输入可见路线编号
战斗                 0、0@1、0>1、e、p0
奖励                 奖励编号、选牌、允许时跳过
休息点               选项编号
事件                 引擎暴露的选项编号 / leave
商店                 c0、r0、p0、rm、leave
宝箱                 暴露出来的选项编号
选牌                 逗号或空格分隔的编号；可选时可以 skip
```

战斗输入形式：

```text
0                  打出卡牌 [0]
0@1 或 0>1        对敌人 [1] 打出卡牌 [0]
e                  结束回合
p0, p1             使用对应药水槽
seq 1 2 4          按当前显示的手牌快照连续出牌
play 1 2 4         与 seq 相同
1,2,4              连续出牌短写
seq 1@0 4@2        多敌人战斗中为每张单体牌指定目标
```

连续出牌会绑定输入时的原始手牌和敌人列表。AOE 或无目标牌不用写目标。如果后续动作需要新的选择、目标非法、卡牌不在手牌、费用不足或不能打出，剩余队列会停止，不会替玩家猜。

## JSON 协议

如需程序化控制，启动 headless 进程并发送 JSON lines：

```bash
dotnet run --project src/Sts2Headless/Sts2Headless.csproj
```

```json
{"cmd": "start_run", "character": "Ironclad", "seed": "test", "ascension": 0}
{"cmd": "action", "action": "play_card", "args": {"card_index": 0, "target_index": 0}}
{"cmd": "action", "action": "end_turn"}
{"cmd": "action", "action": "select_map_node", "args": {"col": 3, "row": 1}}
{"cmd": "action", "action": "skip_card_reward"}
{"cmd": "quit"}
```

每个命令会返回一个决策/状态 payload，例如 `map_select`、`combat_play`、`card_select`、`bundle_select`、`card_reward`、`combat_reward`、`treasure`、`rest_site`、`event_choice`、`shop` 或 `game_over`。

## 日志和 bug 报告

交互运行默认把 JSONL 日志写入 `logs/`，其中包含状态快照、动作和时间戳。

```bash
python3 python/play.py --no-log
```

报告 CLI/headless bug 时，请尽量提供 seed、角色、进阶、当时可见的 decision state、输入命令、可用的 `logs/*.jsonl`，以及 headless 进程失败时的 stderr 或异常文本。

正常游戏结果本身不是 CLI bug。只有当 CLI 隐藏、破坏或阻塞了原引擎决策时，才应当作为 CLI/headless bug 处理。

## 测试

开发时优先使用 focused pytest：

```bash
python -m pytest tests/test_event.py -q
python -m pytest tests/test_dynamic_card_stats.py tests/test_combat.py -q
python -m pytest tests/test_treasure.py tests/test_shop.py tests/test_potions.py -q
python -m pytest tests/test_process_reuse.py -q
```

需要较宽覆盖但不跑最慢测试时：

```bash
python -m pytest -m "not slow" -q
```

发布前或夜间验证再使用完整 `python -m pytest -q`。

## 发布说明

当前还没有打包好的二进制 release。这是有意保持的：游戏 DLL 来自本地 Steam 安装，不能重新分发。

安全的公开发布应当是源码/tag release，只包含脚本、文档、测试和源码。用户仍然需要在本地运行 `sts2-cli.bat`、`setup-windows.bat` 或 `./setup.sh`，用自己的游戏文件完成构建。
