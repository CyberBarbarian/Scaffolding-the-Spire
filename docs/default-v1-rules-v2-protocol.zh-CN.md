# Default-v1 rules-v2 正式协议说明

`default-v1-rules-v2-active-vs-cli-20260723` 是加入基础规则后的非反思
Default 基线。旧的 rules-v1 和 copied-snapshot 协议与结果只保留作废审计，
不得续跑或与新结果混合。

## 上下文与历史口径

- `--no-short-term-memory` 关闭的是通用 Agent 短期动作历史；协议以 `generic_short_term_action_history: false` 明确记录。
- `--occ-context-mode reference_turn` 仍会保留 OCC 的 episode 内短回合历史，供后续决策引用；协议以 `occ_reference_turn_history: true` 明确记录。
- episode 内的结构化 run context 继续启用；反思、跨局经验、人工战斗技能提示、自一致性和 reward posterior evaluator 均关闭。

这三类状态彼此独立，因此不能再用 `short_term_action_history: false` 概括整个 Default 上下文链路。

## 规则暴露审计

每个正式 action、rejected action 和最终模型错误都必须记录其全部真实模型请求，包括 transport retry、自一致性请求和格式纠错请求。验证器会重算完整 messages 哈希与规则暴露；任一请求缺失或不一致，整局不得计为正式样本，也不得被 `--resume` 跳过。

正式协议位于 `experiments/default_v1_rules_v2_formal_20260723.json`。旧的
2026-07-23 copied-snapshot 身份和结果只保留作废审计，不能续跑或与新样本混合。

当前正式入口直接执行活动 `sts2-agent-bench` 和同级 `sts2-cli` 工作树，不再复制
第二套代码或运行时。启动时生成 `source_lock.json`，其执行闭包包含 Default 控制器、
正式协议、规则集、种子文件、Human CLI、Headless DLL、`lib`、本地 .NET 运行时、
C# 源码和中英文本地化资源；`.env`、Reflection 和跨局经验模块不属于该闭包。

每个 `(seed, replicate)` 启动前和结束后都复验同一 source lock，整批结束后再复验
一次。锁定文件发生变化时 fail-close；纯基础设施失败可以在同一 source lock 下
恢复。三个 seed lane 并行，每条 lane 内五局严格串行。

正式启动。若输出根已经预先生成 source lock，则同时传入 `-Resume`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\run_default_v1_rules_v2_formal.ps1 `
  -EnvFile C:\WorkSpace\BG-Agent\.secrets\formal-v7-20260723.env `
  -Resume
```

验证器还会检查实际模型请求中是否出现未解析的 `.title`、`.description` 或 `.name`
键，以及 stderr 是否出现本地化目录缺失警告。旧 copied-snapshot Default 包缺少
本地化资源，不属于可用论文样本。
