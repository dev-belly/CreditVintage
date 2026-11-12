# 信贷批次分析：从业务问题到可复核的 SQL

这份案例回答一个具体问题：**在相同的 180 天观察口径下，哪些放款批次和客群值得进一步复核？有限的人工复核名额覆盖了多少风险事件？**

使用原始评估的 480 笔合成测试申请，和 PITBridge 联合案例的 120 笔测试预测分开。所有百分比都来自逐笔预测的 SQL 聚合；案例不代表真实银行客户或实际风险水平。

[在线分析](https://dev-belly.github.io/CreditVintage/analysis/) · [SQL 聚合](../src/creditvintage/business_groups.sql) · [复核清单 SQL](../src/creditvintage/business_queue.sql) · [原始逐笔预测](demo/predictions.csv)

## 一次运行，得到四种视角

安装方法见 [README](../README.md#run-and-verify)。从仓库根目录运行，输出目录须尚不存在：

```bash
creditvintage analyze docs/demo --out outputs/business-case
creditvintage verify-analysis outputs/business-case
```

Windows PowerShell：

```powershell
.\.venv\Scripts\creditvintage.exe analyze docs/demo --out outputs/business-case
.\.venv\Scripts\creditvintage.exe verify-analysis outputs/business-case
```

打开 `outputs/business-case/index.html`。输出包含 21 个聚合组、48 笔复核清单、原始报告的完整副本、三份实际执行的 SQL、摘要和文件哈希。整个输出目录可以搬到另一台已安装 CreditVintage 的电脑，直接核验；无需保留原目录或重新训练模型。公开案例在 Pages 构建时生成并核验，不把第二份逐笔数据再提交到 Git 历史。

| 视角 / `scope` | 分组数 | 回答的问题 |
| --- | ---: | --- |
| 全体 / `portfolio` | 1 | 整体风险事件率和复核效果怎样？ |
| 客群 / `segment` | 2 | 零售与小微客群的数量、预测及表现有何差异？ |
| 放款月 / `vintage` | 6 | 相同表现期下，哪一个放款月的事件率较高？ |
| 月份 × 客群 / `vintage_segment` | 12 | 月度变化是否集中在某一个客群？ |

同一申请会分别进入四种视角。**只能在同一视角内合计**；把四种视角一起相加会重复计算四遍。

## 正确解释这组结果

| 指标 | 合成结果 | 分子 / 分母或计算方式 |
| --- | ---: | --- |
| 风险事件率 | 22.29% | 107 个事件 / 480 笔成熟申请 |
| 平均校准后 PD | 18.78% | 每笔校准后概率的算术平均 |
| 事件率与平均 PD 的差 | +3.51 个百分点 | 22.29% − 18.78%；描述整体偏差，不是正式校准检验 |
| 本金加权 PD | 18.49% | `SUM(principal_cents * calibrated_pd) / SUM(principal_cents)` |
| 人工复核占比 | 10.00% | 48 笔复核申请 / 480 笔申请 |
| 复核精确率 | 52.08% | 复核组的 25 个事件 / 48 笔复核申请 |
| 风险事件捕获率 | 23.36% | 复核组的 25 个事件 / 全部 107 个事件 |

复核精确率回答“挑出的申请里有多少发生了事件”；捕获率回答“全部事件里抓到了多少”。两者不能互换。这是回顾性排序评价；没有对照干预，不能声称“降低了 23.36% 坏账”。

客群比较中，零售为 63 / 352 = **17.90%**，小微为 44 / 128 = **34.38%**。下一步应查看“月份 × 客群”，同时核对样本量、客群构成和预测偏差。不能仅凭这张合成汇总表断言某个行业、渠道或政策导致了差异，数据也没有提供那些字段。

复核容量是在**全体测试申请中固定为 10%**，不是每个月、每个客群都抽 10%。因此零售进入复核组的比例为 15 / 352 = 4.26%，小微为 33 / 128 = 25.78%。这说明名额分配的实际结果，需要展示分组覆盖；它不构成公平性或最优策略结论。

## 可以从这条 SQL 开始手写

下面是对已加载到 `predictions` 表的成熟测试申请做客群汇总。它是完整聚合 SQL 中最小的一步：

```sql
SELECT
    segment,
    COUNT(*) AS loans,
    SUM(label) AS events,
    1.0 * SUM(label) / COUNT(*) AS event_rate,
    AVG(calibrated_pd) AS mean_pd
FROM predictions
GROUP BY segment
ORDER BY segment;
```

`1.0 *` 让除法得到小数；否则 SQLite 的整数相除可能截断。完整查询再加入条件聚合、`NULLIF` 的零分母处理和 `LAG` 的前一观察月份比较。没有事件的组，捕获率为未定义；没有复核申请的组，复核精确率为未定义，输出空值而非 0%。
