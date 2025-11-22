# Oracle 技术报告 / Oracle Technical Report

本报告基于当前 `Oracle` 目录下的代码，概述系统设计、预测结果对比、评估指标与未来展望。所有内容同时提供中文与英文说明，便于双语读者理解和复现。

## 1. 系统设计与实现方式 / Design & Implementation

### 中文
- **架构概览**：系统以 `RouterAgent` 负责任务分发，`CompanyAgent` 负责公司级分析，均继承自 `BaseAgent`，通过 OpenAI 客户端调用大模型完成推理。主要逻辑定义在 `agent.py`。
- **数据处理**：`data_process.py` 提供投资与政策数据的预处理，包括日期标准化、金额清洗、行业/来源国抽取，并按年份划分训练与预测样本。
- **提示模板**：`prompt_template.py` 定义任务路由与数据分析的系统提示，约定严格的 XML 风格标签（`<thought>`、`<action>`、`<observation>`、`<final_answer>`）以约束模型交互格式。
- **评估工具**：`test_tool.py` 通过按年聚合投资次数与金额，对模型生成的预测与真实基准进行模糊匹配，输出匹配度百分比。
- **执行流程**：路由代理接收用户问题→分配给对应公司代理→公司代理按提示模板循环调用工具（读取政策、测试预测、更新参数）→满足准确度门槛后返回最终答案。

### English
- **Architecture**: `RouterAgent` dispatches tasks while `CompanyAgent` performs company-level analysis; both inherit from `BaseAgent` and call LLMs via the OpenAI client. Core logic resides in `agent.py`.
- **Data Processing**: `data_process.py` preprocesses investment and policy data—normalizing dates, cleaning amounts, extracting sectors/source countries, and splitting records into training vs. prediction sets by year.
- **Prompt Templates**: `prompt_template.py` defines routing and analysis prompts with strict XML-like tags (`<thought>`, `<action>`, `<observation>`, `<final_answer>`) to constrain model interaction.
- **Evaluation Tooling**: `test_tool.py` aggregates yearly investment counts and amounts to fuzzily compare generated predictions against ground truth, returning a percentage match score.
- **Execution Flow**: Router agent receives a user query → routes to relevant company agents → company agents iteratively call tools (policy lookup, prediction testing, parameter updates) per the template → once accuracy threshold is met, a final answer is produced.

## 2. 预测结果与原始数据的重叠度 / Overlap Between Predictions and Baseline

> 请在此处填写：当前预测结果与历史/基准数据的重叠分析（例如匹配度百分比、逐年对比、差异摘要）。

> Fill here: analysis of overlap between current predictions and baseline/historical data (e.g., match percentage, year-by-year comparison, delta summary).

## 3. 评估指标设计 / Evaluation Metric Design

### 中文
- **按年聚合**：仅考虑与目标国家匹配的记录，将投资次数与金额按年份聚合，确保对时序趋势的对比。
- **数量得分**：依据真实与预测的年度投资次数差异计算比率，差异越小得分越高（归一化于 0-1）。
- **金额得分**：对年度总金额的相对差异进行归一化评估，避免规模差异影响结果。
- **综合匹配度**：数量得分与金额得分等权平均后求多年均值，再转换为百分比，形成最终“预测投资信息匹配度”。
- **鲁棒性处理**：对缺失、非标准格式、零金额等情况做防御性转换，避免异常数据导致评估失真。

### English
- **Yearly Aggregation**: Only records matching the target country are considered; counts and amounts are aggregated per year to compare temporal trends.
- **Count Score**: Normalized (0–1) score based on the difference between real vs. predicted yearly investment counts; smaller gaps yield higher scores.
- **Amount Score**: Normalized evaluation of relative differences in yearly total amounts, reducing bias from scale disparities.
- **Overall Match Rate**: Equal-weight average of count and amount scores across years, converted to a percentage labeled as “prediction match rate.”
- **Robustness**: Defensive handling of missing fields, irregular formats, and zero values to prevent distorted metrics.

## 4. 未来预测展望 / Forward-Looking Insights

### 中文
- 在当前匹配度基础上，可调整 `CompanyAgent` 的个性化参数（风险偏好、模仿、政策影响、扩张）以模拟不同战略情境。
- 引入更多政策特征（如税收变化、市场开放度、资本管制强度）将提升预测灵敏度。
- 结合更细粒度的行业分类与地区属性，可提升路由和分析的精度。
- 通过迭代更新历史投资数据与政策数据，支持滚动预测与回测。

### English
- Based on current match rates, tuning `CompanyAgent` personalization parameters (risk preference, imitation, policy impact, expansion) can simulate alternative strategies.
- Incorporating richer policy features (tax shifts, market openness, capital controls) would improve predictive sensitivity.
- Finer-grained sector and regional attributes can enhance routing accuracy and analytical depth.
- Continuously updating investment and policy datasets enables rolling forecasts and backtesting.

---

本报告旨在为后续实验、复现与扩展提供清晰的技术脉络。若需进一步补充数据结果，可在第 2 节直接填写最新分析。

This report provides a clear technical trace for future experiments, reproduction, and extensions. To add fresh results, populate Section 2 with the latest overlap analysis.
