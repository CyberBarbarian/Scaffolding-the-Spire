# 直接 CLI 零脚手架基线

这条实验路径让大模型像人类一样，通过 `sts2-cli` 的终端画面完成一局游戏。模型不会收到引擎 JSON、Benchmark 合法动作对象、状态摘要、CA/OCC 分工、规划器、技能、经验或反思。

## 实验接口

- 观察：人类 CLI 已经打印到终端的原始文本。
- 动作：系统提示明确要求模型只输出当前要输入的一行命令，不附带分析、解释、Markdown、引号或 `command:` 等前缀；需要直接回车时输出 `<ENTER>`。
- 历史：原始终端 transcript 的固定字符窗口，超出后从最旧文本开始截断，不生成摘要。
- 非法输入：控制器只把 `<ENTER>` 转换为空输入，不截取多行内容，也不修正格式、策略或命令。格式违规会单独记入日志，随后由人类 CLI 按原有逻辑处理并显示真实错误。
- 自动行为：游戏已经胜利或失败时输入 `q` 退出；达到决策上限时安全关闭。这两类输入不计为模型决策。

## 运行

在 `C:\WorkSpace\BG-Agent\sts2-agent-bench` 中运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONUTF8='1'
python -m sts2_bench.direct_cli_baseline `
  --model <OpenAI兼容模型名称> `
  --base-url <OpenAI兼容接口的v1地址> `
  --seed ironclad_a0_3407 `
  --character Ironclad `
  --ascension 0 `
  --lang en `
  --history-chars 24000 `
  --omit-max-tokens `
  --out-dir results/direct_cli/example
```

输出上限必须显式选择：使用 `--omit-max-tokens` 交给 provider 默认值，或使用 `--max-tokens N` 指定上限；入口不会再静默采用任意默认上限。

若 `.env` 已配置 `STS2_BENCH_MODEL`、`STS2_BENCH_BASE_URL` 和 API Key，可省略相应参数。默认从同一 BG-Agent 目录下的 `sts2-cli` 仓库加载 `python/play.py`；也可通过 `--sts2-cli-root` 显式指定。

每次运行生成：

- `config.json`：冻结后的接口、模型、运行配置以及控制器与人类 CLI 源文件哈希，不记录 API Key；
- `terminal.txt`：模型看到并参与的完整原始终端记录；
- `events.jsonl`：逐次记录真实 provider 请求尝试及其成功/失败状态，并记录最终模型响应、原样命令、Token、延迟和对应终端字符区间；可从 `terminal.txt` 精确重建并核对每次请求；
- `engine_trace.jsonl`：仅供离线评测使用的完整引擎状态与动作日志；不会进入模型上下文；
- `benchmark_trace.jsonl`：由引擎日志转换得到的 benchmark 兼容轨迹；
- `trace_analysis/`：沿用 benchmark 的战报接口，包含 `replay.md`、`deck_and_card_rewards_zh.md`、`summary.json`、生命曲线和路线回放；
- `summary.json`：胜负、可见最高层数、决策数、无效输入数、成本统计以及离线分析产物状态。

正式实验需要在运行前冻结模型版本、系统提示、历史字符数、角色、难度、种子集合、温度、输出上限和决策上限。不同方法不得使用不同的 CLI 展示内容。

正式批量实验直接从活动工作树运行。`--lock-only` 只生成并验证
`source_lock.json`，不会复制代码、DLL 或本地化资源，也不会生成第二套运行时：

```powershell
python -m sts2_bench.direct_cli_experiment `
  --protocol experiments/direct_cli_baseline_rules_v2.json `
  --lock-only
```

若未提前执行 `--lock-only`，正式启动可直接使用同一入口；若已经生成
`source_lock.json`，首次实际运行也应加 `--resume`，因为输出根已存在：

```powershell
python -m sts2_bench.direct_cli_experiment `
  --protocol experiments/direct_cli_baseline_rules_v2.json `
  --env-file C:\WorkSpace\BG-Agent\.secrets\formal-v7-20260723.env `
  --resume

# 后续同样只在这一 source_lock 下恢复未完成的 (seed, replicate)
python -m sts2_bench.direct_cli_experiment `
  --protocol experiments/direct_cli_baseline_rules_v2.json `
  --env-file C:\WorkSpace\BG-Agent\.secrets\formal-v7-20260723.env `
  --resume
```

环境文件只读取三个凭据键：`STS2_BENCH_API_KEY`、`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`；其他变量一律忽略。API Key 不会写入命令、配置或结果产物。非本地模型接口在缺少有效 Key 时会在创建 attempt 前停止。

`source_lock.json` 绑定正式控制器、协议、规则、Human CLI、Headless DLL、
`lib`、本地 .NET 运行时、C# 源码以及中英文本地化资源。每个 attempt 启动前、
结束后和整批结束时都会重新验证这些文件；活动工作树发生任何锁定范围内的变化，
当前批次都会 fail-close，不会混入不同版本样本。调度保持三个种子并行，同一种子的
五个 replicate 严格串行。

## 正式实验状态

Rules-v2 五职业正式实验已经完成：5 个职业、3 个固定种子、每种子 5 次重复，
共 75 个唯一自然终局；通关 0/75，平均累计楼层 10.09，中位数 8，最高 30，
其中 2/75 进入 Act 2。自然终局修正汇总与完整性审计分别位于：

- `results/direct_cli_rules_v2_active_r4_natural_75_summary_20260724.json`
- `results/direct_cli_rules_v2_active_r4_integrity_audit_20260724.json`

与 Default r10b 的派生比较记录位于
`results/default_r10b_vs_direct_cli_rules_v2_comparison_20260725.json`。该比较明确保留
ruleset 哈希和集成 treatment 差异，不将总体提升错误归因于单一接口或组件。

### 历史 no-rules/FIFO 结果

以下是 rules-v2 重跑之前的历史 Direct CLI 工程结果，只保留作旧口径证据，
不得与新的 rules-v2 Default 主实验直接混算。2026-07-21 的 Ironclad 15 局
实际从活动 `sts2-cli` 运行，本地化文本完整。2026-07-22 的四职业扩展在复制式
恢复环境中运行，其中一部分缺失本地化目录：75 局历史池中 43 局文本正常、32 局
出现原始本地化键。因此下面的旧汇总只能用于诊断，不能再称为 75 个论文级有效样本：

| 职业 | 有效局数 | 通关 | 平均最高楼层 | 最高楼层 |
| --- | ---: | ---: | ---: | ---: |
| Ironclad | 15 | 0 | 9.47 | 16 |
| Silent | 15 | 0 | 6.67 | 14 |
| Defect | 15 | 0 | 10.00 | 16 |
| Regent | 15 | 0 | 8.20 | 14 |
| Necrobinder | 15 | 0 | 8.47 | 16 |
| **总体** | **75** | **0（0%）** | **8.56** | **16** |

旧池 75 局共记录 8,252 次模型决策。模型没有发生外层命令格式违规，但运行器记录了
1,248 次 `invalid_cli_inputs`；该字段主要统计界面返回的 `Invalid. Options`，
是全部语义性动作拒绝的下界。平均每局耗时 11.00 分钟，记录的平均总 Token 数约为
1.40M。上述数值包含受本地化缺失影响的样本。

所有历史结果和失败 attempt 继续保留供审计，不再恢复或与新批次混合。新正式计数
除自然终止、身份和完整终态要求外，还必须通过模型可见文本本地化检查，并确认运行
日志没有 `Localization dir not found:`。

统一完成报告和派生表格位于：

`C:\WorkSpace\BG-Agent\sts2-agent-bench\results\direct_cli_baseline_v1_multichar_20260722`

其中 `AGENT-01-MC_completion_report.zh-CN.md` 给出完整统计口径、职业表、种子表和来源说明；`valid_runs.csv`、`aggregate_by_character.csv`、`aggregate_by_seed.csv` 与 `source_verification.json` 可直接用于后续论文制表和复核。Ironclad 的逐轨迹详细分析仍保留在 `results/direct_cli_baseline_v1_20260721/AGENT-01_completion_report.zh-CN.md`。
