# 两分钟交互速览 / A short entry into the SBA study

[打开交互速览](https://dev-belly.github.io/CreditVintage/sba-brief/) ·
[完整报告](https://dev-belly.github.io/CreditVintage/sba-credit/) ·
[研究协议、来源与限制](SBA_CREDIT.md)

无需登录、安装 Python 或上传数据。先选 500、1,000、2,000 个相同的复核名额，
比较三个固定方法。原始 Logistic 在 1,000 个名额中捕获 26 笔记录核销事件：
精确率为 26/1,000，事件捕获率为 26/89；未进入名单的 63 笔事件也保留显示。
独立校准改善 Brier，却不保证更好的排序或固定容量名单。

这三个容量已经写在冻结研究协议中。按钮只显示已保存的九组汇总结果；浏览器
不训练模型、不读取逐笔借款人记录、不选择“最佳”测试容量。没有分析统计或外部
前端依赖。复制按钮只写入系统剪贴板；链接的片段保存容量。禁用 JavaScript 时，
默认 10% 卡片和全部九组静态表格仍可阅读。

## 从证据生成页面

`scripts/build_public_site.py` 先生成全部研究报告，再调用
`scripts/build_credit_brief.py`。后者先运行原有 `verify_sba_benchmark`，
检查冻结来源、协议、模型重拟合、预测、指标、队列、HTML 和 ZIP，再提取汇总。
页面的内嵌 JSON 与同目录 `data.json` 来自同一份对象，并包含四份证据文件的
SHA256：summary、policy metrics、predictions 和 protocol。
它们可追溯字节身份，不等于外部来源真实性或独立采用。

从源码安装开发依赖后，也可单独构建到一个尚不存在的目录：

```text
python -m creditvintage sba-benchmark --out outputs/sba-credit
python scripts/build_credit_brief.py outputs/sba-credit outputs/sba-brief
```

本页由当前网站构建，未修改已发布 v0.8.0 的软件或证据资产。
原有研究报告、CLI、源样本、模型设置、数据协议与校验格式不变。

## 检查交互与数字

新测试从逐笔预测独立排序并累计事件，检查全部九组汇总的分子、分母及来源摘要；
损坏证据会在速览输出目录创建之前失败。Pages 的真实 Chrome 检查在 1,440px
和 390px 视口切换三个容量，将三张卡片与逐笔预测计算值逐项比对；还检查链接
重载、键盘操作、复制成功与受限剪贴板回退、页面溢出，以及无 JavaScript 的表格。

## 解释范围

35,000 笔来源贷款中，有 10,000 笔后期测试记录和 89 笔 24 月记录核销事件。
来源为最终公开快照，缺少历史首次公开及修订记录。这里展示的是回溯放款队列
研究与历史事件集中程度，不能推断避免坏账、因果收益、真实部署效果或当前
中国信贷人群表现。完整报告保留所有方法、两个测试年份和不确定性区间。
