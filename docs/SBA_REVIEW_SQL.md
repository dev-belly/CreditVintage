# 用 SQL 重算 SBA 复核预算 / Replay review budgets in SQLite

这是一项可运行的面试练习：从保存的逐笔预测重算三个方法、三个固定预算，
核对全部九行指标和 3,000 行名单。SQL 使用 Python 标准库的 SQLite，无需数据库服务。
先完成冻结证据的完整重拟合验证，再执行 SQL；输入目录保持不变，结果只写到标准输出。

The repository example independently ranks saved SBA scores in SQLite, then
compares every policy row and queue member with the fully verified evidence.
It requires this checkout's adjacent Python and SQL files; it is not a new
installed command or a feature bundled in the published v0.8.2 wheel.

[两分钟交互速览](https://dev-belly.github.io/CreditVintage/sba-brief/) ·
[完整研究与限制](SBA_CREDIT.md) ·
[SQL](../examples/sba_review_budget.sql) ·
[执行器](../examples/sba_review_sql.py)

## 先运行，再解释

从当前源码 checkout 的项目根目录运行，先按 [安装说明](../README.md#run-and-verify)
建立 Python 3.12+ 虚拟环境并安装项目。以下命令假定虚拟环境已激活：

```text
python -m creditvintage sba-benchmark --out outputs/sba-sql
python examples/sba_review_sql.py outputs/sba-sql
```

未激活时，Linux/macOS 将 `python` 换成 `.venv/bin/python`；Windows PowerShell
换成 `.\.venv\Scripts\python.exe`。重复运行第一步须选择新目录；第二步可对同一份
完整证据重复核对。生成和重拟合使用随包提供的冻结样本，无需下载官方大文件或安装 PITBridge。
已发布 v0.8.2 可提供研究和验证器，但单独安装 wheel 不会创建本练习的 `examples/` 文件。

成功返回退出码 0，JSON 包括：

```json
{
  "status": "matched_saved_evidence",
  "prediction_rows": 10000,
  "policy_rows": 9,
  "queue_rows": 3000,
  "labels_used_for_ranking": false
}
```

完整输出还保留重拟合结果、SQLite 版本、三份比较文件的 SHA256 和全部九行指标。
`queue_rows` 是三个方法各 1,000 行的总和，并非 3,000 名不同借款人。
证据缺失、被改动或 SQL 与保存结果不一致时，退出码为 2，诊断写到标准错误，
不会输出成功 JSON。哈希标识文件字节，不证明来源真实或有人独立使用。

## 15 分钟练习

1. **前 5 分钟：运行。** 找出 `logistic`、`capacity: 0.1` 的一行，
   口算精确率与事件捕获率，并解释为什么分母不同。
2. **中间 5 分钟：读 SQL。** 找到 `ROW_NUMBER()`、`PARTITION BY method`、
   `score DESC` 和 `record_id COLLATE BINARY`；说明相同分数为什么也只有一个确定顺序。
   再指出结果标签在哪一步关联，为什么没有参与排序。
3. **最后 5 分钟：检查口径。** 说明为什么总体事件分母从每笔贷款一行的
   `outcomes` 表计算，而不能将三个方法的分数长表相加；解释零事件时的 `NULLIF`。
   最后用下面三种分母各讲一句话。

| 问题 | 原始 Logistic 的 10% 预算结果 | 含义 |
| --- | ---: | --- |
| 名单里有多少记录事件？ | 26 / 1,000 = 2.60% | 精确率；分母是复核名额 |
| 全部记录事件中覆盖多少？ | 26 / 89 ≈ 29.21% | 事件捕获率；分母是测试事件 |
| 整个测试期事件比例？ | 89 / 10,000 = 0.89% | 测试集基准事件率 |

还有 **63 / 89** 笔事件没有进入该名单。26 笔是历史记录事件落在固定名单中的数量，
不表示复核阻止了 26 笔坏账。同样的 1,000 个名额，独立 Isotonic 为 24 / 89，
常数基线为 7 / 89；Brier 改善不保证名单更好。保留全部预算，不能从测试结果挑部署阈值。

## SQL 中的四个容易出错的地方

| 控制 | 实现与检查 |
| --- | --- |
| 先排序后评估 | `ranked_scores` 只读取 `scores`，评估再关联 `outcomes`；改变全部标签不能改变名单 |
| 相同分数的名额 | 按分数降序、ID 升序确定唯一排名；常数基线也是 ID 顺序，不是随机抽样 |
| 百分比向上取整 | `(loans * percent + 99) / 100` 使用整数运算；21 笔时 5%、10%、20% 分别取 2、3、5 笔 |
| 保持每笔贷款的口径 | 总体只有 10,000 笔、89 个事件；三个方法共用这一个总体，不能把分母重复三次 |

测试还覆盖打乱输入顺序、零事件分母、非法概率/重复 ID，以及重新计算哈希后的篡改。
篡改指标即使更新 manifest 和 ZIP，也必须先被原有重拟合验证器拒绝。
SQL 比较使用保存的预测，数值容差为绝对值 `1e-12`；它不放宽原有研究验证规则。

## 能说明什么

35,000 笔来源贷款按真实放款期划分：2006–2010 年训练 20,000 笔，2013 年
校准 5,000 笔，2016–2017 年测试 10,000 笔。目标是放款后 24 个自然月内的
**记录核销事件**，前一阶段的观察期在下一阶段开始前成熟。

模型只有四项审批输入：`log1p` 总审批金额、SBA 担保比例、初始利率和处理方式；
填补、缺失指示、缩放和编码参数仅从训练期学习。最终状态、核销日期、借款人 ID
和分组日期不作为预测变量。但“审批字段”这一名称不能证明它在历史决策时已经公开：
最终来源快照缺少首次公开时间和历史修订。这里仍是回溯研究，不是部署评估、
因果收益估计或中国信贷人群验证。合成事件流的时间可见性控制是另一个实验。

面试时，先实际运行并理解这些步骤，再按真实参与程度介绍。自己完成的改动、
复现与 AI 或其他协助应如实区分；这份练习不会替代个人对 SQL 和研究限制的理解。
