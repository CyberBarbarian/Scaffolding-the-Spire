# STS2 Agent Bench

[英文版](README.md)

`sts2-agent-bench` 是 BG-Agent 的私有研究仓库，用来比较多种大语言模型与混合智能体方法在 Slay the Spire 2 中的表现。这个仓库包含无头 `sts2-cli` 运行时、`sts2_bench` 评测脚手架、实验配置、轨迹、指标和失败分析材料。

公开的纯命令行运行时单独维护在 [CyberBarbarian/sts2-cli](https://github.com/CyberBarbarian/sts2-cli)。命令行与无头适配器的修复放在那里；大语言模型策略、评测、轨迹、提示词、记忆、规划、搜索和研究方法工作放在这里。

## 项目范围

这个仓库用于：

- 比较基于大语言模型的 STS2 游戏方法；
- 运行可复现的随机、大语言模型、记忆、聊天历史、规划和混合智能体基线；
- 记录轨迹、运行配置、指标和失败标签；
- 把失败案例转成回归场景和评测用例；
- 维护 STS2 智能体评估相关的私有研究流程。

这个仓库不用于：

- 重新分发 Slay the Spire 2 的游戏 DLL；
- 修改游戏本身的卡牌、遗物、敌人、事件、战斗或奖励语义；
- 维护公开命令行运行时的清晰边界。那部分属于 `sts2-cli`。

## 运行时来源

这里的运行时来自 [CyberBarbarian/sts2-cli](https://github.com/CyberBarbarian/sts2-cli)，而它是 [wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli) 的派生仓库。运行时会无头启动真实的 Slay the Spire 2 引擎，并暴露：

- 通过 `python/play.py` 使用的交互式终端界面；
- 通过 `src/Sts2Headless` 使用的标准输入/标准输出 JSON 协议；
- 评测可复用的状态导出、本地化文本和动作接口。

你必须自己通过 Steam 拥有并安装 Slay the Spire 2。本仓库不包含、也不会重新分发游戏 DLL。

## 目录结构

- `sts2_bench/`：评测循环、合法动作生成、状态渲染、智能体、配置加载和强化学习风格环境脚手架。
- `src/Sts2Headless/`：围绕真实 STS2 引擎的 C# 无头适配器。
- `src/GodotStubs/`：无头适配器需要的窄 Godot 呈现层桩件。
- `python/`：交互式命令行玩家、日志工具和完整运行冒烟工具。
- `localization_eng/` 与 `localization_zhs/`：运行时和提示词使用的引擎导出本地化数据。
- `tests/`：评测回归覆盖，以及少量运行时冒烟测试。
- `docs/`：运行时说明、评测计划和研究笔记。
- `logs/`、`results/`、`artifacts/`、`lib/`、`.tools/`、`saves/`：本地运行时或实验产物。除非明确提升为交付物，否则不要提交。

## 快速开始

依赖：

- 通过 Steam 安装 Slay the Spire 2。
- Python 3.9 或更新版本。
- .NET 9 SDK，或仓库本地的 `.tools/dotnet` 运行时。
- 在类 Unix 系统中，`setup.sh` 和 `copy_dlls.sh` 需要 `bash`。

克隆私有仓库：

```bash
git clone https://github.com/CyberBarbarian/sts2-agent-bench.git
cd sts2-agent-bench
```

Windows：

```cmd
.\setup-windows.bat
.\sts2-cli.bat
```

macOS、Linux 或 Git Bash：

```bash
./setup.sh
python3 python/play.py
```

首次配置会定位 Steam 安装目录，把需要的本地 DLL 复制到 `lib/`，并构建无头适配器。不要提交或分发 `lib/`。

## 运行评测

按需安装评测依赖：

```bash
python -m pip install -r sts2_bench/requirements.txt
```

查看评测参数：

```bash
python -m sts2_bench.run_benchmark --help
```

运行随机基线：

```bash
python -m sts2_bench.run_benchmark ^
  --agent random ^
  --character Ironclad ^
  --ascension 0 ^
  --count 1 ^
  --out results/random_ironclad_a0.jsonl
```

运行兼容 OpenAI 的 LLM 接口：

```bash
python -m sts2_bench.run_benchmark ^
  --agent llm ^
  --base-url http://127.0.0.1:11434/v1 ^
  --model qwen2.5:14b ^
  --character Ironclad ^
  --ascension 0 ^
  --count 1 ^
  --out results/qwen25_ironclad_a0.jsonl
```

本地密钥放在 `.env`，不要写进受 Git 跟踪的文件。非敏感默认参数放在 `sts2_bench/benchmark.yaml`。

## 实验纪律

每次严肃比较都应该记录：

- 运行时提交和评测提交；
- 后端类型和版本；
- 模型名、接口类型、采样温度和重试设置；
- 角色、进阶、种子列表、最大步数上限和截断策略；
- 提示词风格、记忆模式、上下文管理模式和运行摘要设置；
- JSONL 轨迹和紧凑指标摘要；
- 区分策略错误与后端/运行时错误的失败标签。

修改提示词或智能体方法时，要保留随机基线和简单大语言模型基线，这样新方法能和稳定下限对比。

## 测试

运行本仓库的完整评测测试：

```bash
python -m pytest -q
```

这组测试覆盖 `sts2_bench` 的策略、动作、上下文、轨迹分析等行为，只保留少量嵌入式无头运行时冒烟测试。完整的 CLI/运行时详细回归测试维护在单独的 `sts2-cli` 仓库中；公开运行时适配器改动应在那里修改并验证。

迭代时运行聚焦的评测测试：

```bash
python -m pytest tests/test_sts2_bench_actions.py tests/test_sts2_bench_context.py tests/test_sts2_bench_config.py -q
```

触碰本地工具链或进程启动逻辑时，运行保留的运行时冒烟检查：

```bash
python -m pytest tests/test_local_tools.py tests/test_process_reuse.py tests/test_runtime_smoke.py tests/test_godot_stubs.py -q
```

移植或发布运行时适配器改动前，应到 `C:\WorkSpace\BG-Agent\sts2-cli` 运行完整 runtime gate，而不是在这个评测仓库里重复维护详细 runtime 测试。

## 发布边界

这个仓库是私有研究基础设施。如果某个改动改善了无头适配器本身，应当把它移植到 `CyberBarbarian/sts2-cli`。研究方法、轨迹、提示词实验和评测报告默认留在这里，除非明确准备对外发布。
