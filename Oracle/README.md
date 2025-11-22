# Oracle 技术报告 / Oracle Technical Report

本报告基于当前 `Oracle` 目录下的代码，概述系统设计、预测结果对比、评估指标与未来展望。所有内容同时提供中文与英文说明，便于双语读者理解和复现。

## 1. 系统设计与实现方式 / Design & Implementation

### 中文

#### 1.1 整体架构
系统采用**双层代理架构**：上层 `RouterAgent` 根据用户查询识别相关公司并分配任务；下层 `CompanyAgent` 针对单个公司进行迭代式分析。两者均继承自 `BaseAgent` 基类，通过 OpenAI 兼容接口调用大模型（支持代理商 API）。

#### 1.2 核心类与方法说明（agent.py）

**BaseAgent 类（基础代理）**
- `__init__(tools, model)`: 初始化代理，将工具函数转换为名称到函数的映射字典，建立 OpenAI 客户端。
- `get_api_key()`: 静态方法，从环境变量读取 API 密钥。使用 `dotenv` 加载 `.env` 文件中的 `OPENROUTER_API_KEY`。
- `get_tool_list()`: 遍历 `self.tools` 字典，使用 `inspect` 模块提取每个函数的签名（参数列表）和文档字符串（docstring），生成格式化的工具描述清单，供提示模板填充。
- `add_tool(tool)`: 动态注册新工具函数，将其函数名作为 key 添加到 `self.tools` 字典。
- `call_model(messages)`: 调用 OpenAI 兼容 API（通过 `self.client.chat.completions.create()`），传入消息历史，返回模型响应内容，并自动追加到消息列表。

**RouterAgent 类（任务路由代理）**
- 继承 `BaseAgent`；新增属性 `company_agents`（公司代理字典）、`previous_context`（对话上下文）、`message_history`（消息历史）。
- `run(user_input)`: 
  1. 构建系统提示和用户问题消息对。
  2. 调用 `call_model()` 获取模型对任务的分析结果。
  3. 用正则表达式 `<agent>...<name>...<question>...</agent>` 抽取路由决策：哪些公司、对应什么问题。
  4. 为每个公司代理调用其 `run()` 方法处理具体任务。
  5. 返回 `{company_name: result}` 字典。
- `render_system_prompt(system_prompt_template)`: 使用 `string.Template` 替换模板中的 `${sub_agents}` 占位符为公司列表。

**CompanyAgent 类（公司级分析代理）**
- 继承 `BaseAgent`；新增属性：`company_name`、`company_country`、`company_sector`（行业集合）、四个个性化参数分数（`risk_preference_score` 等）、历史投资数据、预测投资数据。
- `run(user_input)`: 核心执行循环，实现"思考-行动-观察"（Thought-Action-Observation）范式：
  1. 初始化消息列表，包含系统提示和用户问题。
  2. 进入无限循环，每次调用 `call_model()` 获取模型回复。
  3. 用正则表达式提取回复中的 `<thought>`、`<action>`、`<final_answer>` 标签。
  4. 若检测到 `<final_answer>` 标签，直接返回答案并退出循环。
  5. 若检测到 `<action>` 标签，调用 `parse_action()` 解析函数名、位置参数、关键字参数。
  6. 执行对应工具函数，捕获返回值或异常作为 `<observation>`。
  7. 将 observation 追加到消息列表，同时更新系统提示中的参数分数（动态提示）。
  8. 循环至模型产生最终答案。
- `render_system_prompt(system_prompt_template)`: 使用 `Template.substitute()` 填充提示模板中的所有变量：
  - `${tool_list}`: 通过 `get_tool_list()` 生成的工具描述。
  - `${company_name}`、`${company_country}`、`${company_sector}`: 公司信息。
  - `${risk_preference_score}` 等：四个个性化分数。
  - `${history_investments}`: 历史投资数据（用于上下文）。
- `parse_action(code_str)`: 使用正则表达式和状态机解析 LLM 输出的函数调用字符串（如 `company_test("...", destination_country="China")`）：
  1. 匹配函数名和参数字符串。
  2. 初始化状态机变量（`in_string`, `paren`, `bracket`, `brace`）以追踪嵌套层级。
  3. 逐字符扫描参数字符串，用逗号和括号/方括号/大括号匹配确定参数边界。
  4. 对于每个参数令牌，调用 `flush_token()` 判断是关键字参数（含 `=`）还是位置参数，并调用 `_parse_single_arg()` 解析值。
  5. 返回三元组 `(func_name, args, kwargs)`。
- `_parse_single_arg(arg_str)`: 解析单个参数值：
  1. 若为字符串字面量（`"..."` 或转义形式 `\\"...\\"`)，去除引号和转义符，返回字符串。
  2. 否则用 `ast.literal_eval()` 尝试解析 Python 字面量（数字、列表、字典等）。
  3. 解析失败时返回原字符串。
- `flush_context()`: 清空 `previous_context`，用于重置对话历史。
- `compress_context(max_length)`: 若上下文超过 `max_length`，保留前后各 `max_length/2` 个字符，中间插入 "…（省略）…" 标记，用于长对话场景的内存管理。

**全局函数**
- `update_company_parameters(company_agent, risk_preference_score, imitation_score, policy_impact_score, expansion_score)`: 更新代理的四个个性化分数，用于在优化预测时调整公司行为偏好。
- `build_company_regex(company_name)`: 为公司名生成**鲁棒正则表达式**：
  1. 提取公司名的第一个单词（如 "Microsoft"）和全称。
  2. 对每个单词用 `\s*` 连接字符，允许字符间出现空白（匹配 "M i c r o s o f t" 形式）。
  3. 返回不区分大小写的模式，用于在政策文本中模糊匹配公司名称。
- `read_policy(policy_dict, year, company)`: 从政策字典中按 `(year, company)` 键查询并返回相应政策记录列表。
- `main()`: 程序入口，完整工作流：
  1. 调用 `process_investment_csv()` 和 `process_policy_csv()` 读取并预处理数据。
  2. 遍历每个公司，为其创建 `CompanyAgent` 实例，初始化公司信息和参数分数。
  3. 定义局部工具函数（`company_test`, `company_read_policy`, `company_update_parameters`）绑定公司数据，添加到代理工具集。
  4. 创建 `RouterAgent` 实例，设置所有公司代理为其下属。
  5. 启动交互循环，接收用户任务并调用路由代理执行。

#### 1.3 数据处理管道（data_process.py）

**format_company_analysis(result)**: 格式化公司分析结果
- 输入可为 dict 或字符串形式的 dict。
- 使用 `ast.literal_eval()` 安全解析字符串类型的 Python 对象。
- 返回格式化的多行文本，每个公司占一段（公司名 + 两个换行 + 分析内容）。

**normalize_date(date_str)**: 日期标准化
- 使用 `pd.to_datetime()` 解析多种日期格式（如 "2023-01-15", "20230115" 等）。
- 返回统一的 "YYYY-MM-DD" 格式字符串；失败或 NA 值返回空字符串。

**extract_amount(amount_raw)**: 金额提取
- 输入可为字符串（如 "18.03（百万美元）"）或数值。
- 用正则表达式 `[^0-9.\-]` 删除所有非数字、小数点、负号的字符。
- 用 `float()` 转换，异常或空值返回 0.0。

**process_investment_csv(input_csv_path)**: 投资数据处理（关键）
- 读取 CSV 文件，验证必需列：`Parent_Company`, `Source_Country`, `Project_Date`, `Destination_Country`, `Capital_Investment`, `Sector`。
- 按 `Parent_Company` 分组，对每个公司：
  1. 遍历其所有投资记录，调用 `normalize_date()` 和 `extract_amount()` 清洗数据。
  2. 按年份切分：year >= 2020 为 `pred` 预测集，否则为 `train` 训练集。
  3. 收集所有 sector 和 source_country 信息。
- 返回 `{company: (investments_dict, source_country, sector_list)}`，其中 `investments_dict = {"train": [...], "pred": [...]}`。

**process_policy_csv(csv_path)**: 政策与财务数据处理（关键）
- 读取 Compustat 或类似格式的财务/政策 CSV。
- 列名标准化（小写 + 下划线）。
- 通过候选列名自适应识别关键字段：
  - 年份列：`fyear`, `datadate` 等；调用 `extract_year()` 统一提取四位年份。
  - 公司列：Compustat 的 `conm`（公司名）。
  - 其他可选字段：GICS 行业代码、国家代码、资产、收入、利润等。
- 建立索引 `{(year, company): [record_list]}` 的形式，每条记录包含上述字段。
- 这些"政策"实际上是公司的财务与行业环境指标，用于 `CompanyAgent` 决策分析。

#### 1.4 提示模板（prompt_template.py）

**data_analysis_system_prompt_template**: 公司代理系统提示
- 角色定义：财经政策专家，帮助分析政策和投资数据进行预测。
- 工具说明：`${tool_list}` 占位符由 `get_tool_list()` 填充。
- 交互格式要求：严格的 XML 标签（`<thought>`, `<action>`, `<observation>`, `<final_answer>`）。
- 输出示例：演示完整的"思考-行动-观察"循环和参数更新流程。
- 环境上下文：填充公司名、来源国、行业、四个参数分数、历史投资信息。

**task_router_system_prompt_template**: 路由代理系统提示
- 角色定义：任务路由专家，识别相关公司并生成子任务。
- 输出格式：`<sub_agents>` 列表和 `<sub_agent_questions>` 详细映射。
- 核心约束：不修改或扩展用户问题的含义。

#### 1.5 评估与工具调用（test_tool.py）

**test_predicted_investments(gen_pred_context, pred, destination_country="China")**: 预测匹配度评估
- 详见下方第 3 节评估指标。

### English

#### 1.1 Overall Architecture
The system employs a **dual-layer agent architecture**: the upper `RouterAgent` identifies relevant companies and distributes tasks based on user queries; the lower `CompanyAgent` performs iterative analysis for individual companies. Both inherit from the `BaseAgent` base class and call LLMs via an OpenAI-compatible interface (supporting API proxies).

#### 1.2 Core Classes & Methods (agent.py)

**BaseAgent Class**
- `__init__(tools, model)`: Initializes the agent, converting tool functions into a name→function dict, establishes OpenAI client.
- `get_api_key()`: Static method, loads API key from environment variable via `python-dotenv`.
- `get_tool_list()`: Iterates through `self.tools`, uses `inspect` module to extract function signature and docstring, generates formatted tool descriptions for prompt templating.
- `add_tool(tool)`: Dynamically registers a new tool function to the `self.tools` dict.
- `call_model(messages)`: Invokes OpenAI-compatible API, appends response to message history, returns content.

**RouterAgent Class**
- Adds attributes: `company_agents` (dict of company agents), `previous_context`, `message_history`.
- `run(user_input)`: 
  1. Constructs system prompt and user question.
  2. Calls `call_model()`, parses routing decisions via regex on `<agent>` tags.
  3. Invokes each company agent's `run()` method.
  4. Returns `{company_name: result}` dict.
- `render_system_prompt()`: Uses `string.Template` to substitute `${sub_agents}` placeholder.

**CompanyAgent Class**
- Adds: `company_name`, `company_country`, `company_sector`, four personality score attributes, history/predicted investment data.
- `run(user_input)`: Core Thought-Action-Observation loop:
  1. Initializes message list with system prompt and user query.
  2. Enters infinite loop, calls `call_model()` each iteration.
  3. Uses regex to extract `<thought>`, `<action>`, `<final_answer>` tags.
  4. If `<final_answer>` found, returns and exits.
  5. If `<action>` found, calls `parse_action()` to extract function name, args, kwargs.
  6. Executes tool, captures observation or exception.
  7. Appends observation to messages, updates system prompt with current parameter scores.
  8. Loop until final answer.
- `render_system_prompt()`: Fills template placeholders via `Template.substitute()`.
- `parse_action(code_str)`: Parses LLM function call string using regex + state machine:
  1. Extracts function name and arguments via regex.
  2. Initializes state machine to track string, parenthesis, bracket, brace nesting.
  3. Scans argument string character-by-character, identifies parameter boundaries via comma separators at nesting depth 0.
  4. For each parameter token, calls `flush_token()` to distinguish keyword vs. positional args.
  5. Returns `(func_name, args, kwargs)` tuple.
- `_parse_single_arg(arg_str)`: Parses individual argument value:
  1. If string literal (`"..."` or escaped `\\"...\\"`), strips quotes/escapes, returns string.
  2. Else tries `ast.literal_eval()` for Python literals (numbers, lists, dicts).
  3. Falls back to returning original string on parse failure.
- `flush_context()`: Clears `previous_context`.
- `compress_context(max_length)`: Truncates context to head + "..." + tail for long conversations.

**Global Functions**
- `update_company_parameters()`: Updates agent's four personality scores.
- `build_company_regex(company_name)`: Generates **fuzzy regex** for robust company name matching:
  1. Extracts first word and full name.
  2. Joins characters with `\s*` to allow whitespace between letters.
  3. Returns case-insensitive pattern for policy text matching.
- `read_policy()`: Looks up policy records by `(year, company)` key.
- `main()`: Entry point orchestrating full workflow:
  1. Loads and preprocesses data via CSV processors.
  2. Creates `CompanyAgent` for each company with localized tool functions.
  3. Instantiates `RouterAgent` managing all company agents.
  4. Runs interactive loop accepting user tasks.

#### 1.3 Data Processing Pipeline (data_process.py)

**format_company_analysis(result)**: Formats analysis results
- Parses dict or string-encoded dict input via `ast.literal_eval()`.
- Returns multi-line text with company sections.

**normalize_date(date_str)**: Standardizes dates
- Uses `pd.to_datetime()` to parse multiple formats.
- Returns "YYYY-MM-DD" or empty string on failure.

**extract_amount(amount_raw)**: Extracts numeric amount
- Regex removes non-numeric characters except `.` and `-`.
- Returns float or 0.0 on failure.

**process_investment_csv(input_csv_path)**: Processes investment records (key)
- Validates required columns, groups by company.
- For each company, cleans date/amount and splits by year (≥2020 = prediction, <2020 = training).
- Returns `{company: (investments_dict, source_country, sector_list)}`.

**process_policy_csv(csv_path)**: Processes policy/financial data (key)
- Reads Compustat-format CSV, normalizes column names.
- Auto-detects key fields (year, company, GICS sectors, country codes, financial metrics).
- Builds index `{(year, company): [record_list]}` representing policy/financial environment.

#### 1.4 Prompt Templates (prompt_template.py)

**data_analysis_system_prompt_template**: Company agent system prompt
- Role: Financial/policy analyst assisting in investment prediction.
- Tool list: Filled by `get_tool_list()` substitution.
- Format spec: Strict XML tags for model interaction.
- Examples: Demonstrates complete Thought-Action-Observation loops with parameter updates.
- Context: Company info, personality scores, history investments.

**task_router_system_prompt_template**: Router agent system prompt
- Role: Task router identifying relevant companies and generating sub-tasks.
- Format: `<sub_agents>` list and `<sub_agent_questions>` mapping.
- Constraint: Do not modify user intent.

#### 1.5 Evaluation & Tool Invocation (test_tool.py)

**test_predicted_investments()**: Prediction accuracy scoring
- See Section 3 below for detailed metric design.

## 2. 预测结果与原始数据的重叠度 / Overlap Between Predictions and Baseline

### 中文

使用我们的评估方法，评估预测投资数据，重叠度指标在1-shot情况下（应用场景）数值在20%-60%之间。在n-shot情况下（训练场景）会上升至40%-70%的重叠度。
需要注意的一点是训练过程中的个性化参数对于上下文的输出影响有限，对于模型输出更偏向定型而非定量的影响。所以在训练过程中更新的个性化参数，并不能完全解释重叠度逐步上升的现象，我将其归结为大模型基于上下文的试错机制。

### English
Using our evaluation method, the overlap metric for predicting investment data ranges from 20% to 60% in a 1-shot scenario (application scenario). In an n-shot scenario (training scenario), the overlap increases to 40%-70%.

It's important to note that personalized parameters updated during training have limited impact on the contextual output, having a more qualitative rather than quantitative effect on the model output. Therefore, personalized parameters updated during training cannot fully explain the gradual increase in overlap; I attribute this to the large model's context-based trial-and-error mechanism.

## 3. 评估指标设计 / Evaluation Metric Design

### 中文

#### 3.1 评估流程概述
预测评估通过 `test_predicted_investments(gen_pred_context, pred, destination_country="China")` 函数完成，该函数对比**模型生成的预测投资记录** vs **真实基准投资记录**，计算匹配度百分比。评估核心思想是：**按年聚合、同时考虑投资次数与金额**，形成模糊而鲁棒的对比。

#### 3.2 函数输入参数详解

1. **gen_pred_context** (预测内容)
   - 类型：`str` 或 `list[dict]`
   - 说明：模型生成的投资预测，可以是字符串形式的 Python 对象或 JSON，也可以是列表。
   - 格式示例：`[{"Date": "2020-01-15", "tar_country": "China", "amount": 50.0, "sector": "Technology"}, ...]`

2. **pred** (基准/真实数据)
   - 类型：`list[dict]`
   - 说明：已知的真实或基准投资记录，通常来自历史数据或测试集。
   - 格式：同上。

3. **destination_country** (目标国家，默认 "China")
   - 类型：`str`
   - 说明：评估的目标国家，仅匹配此国家的投资记录。

#### 3.3 核心计算逻辑

**第一步：数据解析与清洗**
```python
def to_list_of_dict(x):
    # 1. 若已是 list，直接返回
    # 2. 若是字符串，先尝试 json.loads()，失败则尝试 ast.literal_eval()
    # 3. 其他类型返回空列表
```
- 防御性处理：支持多种输入格式（JSON字符串、Python repr字符串、列表）。
- 目的：将模型输出的非标准格式转换为统一的列表结构。

**第二步：年份提取**
```python
def extract_year(date_str):
    # 使用正则 r"(\d{4})" 从日期字符串中提取四位年份
    # 支持格式：
    #   - "2020-03-15"  → 2020
    #   - "20200315"    → 2020
    #   - "2020/3/15"   → 2020
```
- 关键特性：**不考虑具体日期（月、日）**，只按年份聚合。
- 容错性：返回 `None` 表示无法解析。

**第三步：金额解析**
```python
def parse_amount(amount_val):
    # 1. 将 amount 转为字符串
    # 2. 用正则 r"[^0-9.\-]" 删除所有非数字、小数点、负号的字符
    # 3. 用 float() 转换，失败返回 0.0
    # 例：
    #   - "54.5"          → 54.5
    #   - "50（百万美元）"  → 50.0
    #   - "invalid"       → 0.0
```
- 目的：统一处理多种格式的金额字符串。

**第四步：按年份聚合**
真实数据和预测数据分别按年份、国家聚合，形成：
```python
{
    2020: {"count": 3, "amount": 150.0},   # 该年3笔，总额150百万
    2021: {"count": 2, "amount": 80.0},
    ...
}
```
逻辑：
- 遍历所有投资记录。
- 过滤：仅保留 `tar_country == destination_country` 的记录。
- 对每条记录提取年份和金额，累计到对应年份的 `count` 和 `amount`。
- 若目标国家无记录，返回 0% 匹配度。

**第五步：逐年计算得分**
对真实数据中**每一年**分别计算两个子得分，然后合并：

1. **次数得分** (Count Score)
   $$\text{count\_score} = \begin{cases}
   1.0 & \text{if } \text{real\_cnt} = 0 \text{ and } \text{gen\_cnt} = 0 \\
   \max(0, 1 - \frac{|\text{real\_cnt} - \text{gen\_cnt}|}{\max(\text{real\_cnt}, 1)}) & \text{otherwise}
   \end{cases}$$
   - 解释：投资次数的相对偏差。
   - 例：真实3笔、预测2笔 → diff = 1/3 ≈ 0.33 → score = 0.67
   - 例：真实3笔、预测3笔 → diff = 0 → score = 1.0
   - 例：真实0笔、预测0笔 → score = 1.0（完全匹配）

2. **金额得分** (Amount Score)
   $$\text{amount\_score} = \begin{cases}
   1.0 & \text{if } |\text{real\_amt}| < 10^{-6} \text{ and } |\text{gen\_amt}| < 10^{-6} \\
   \max(0, 1 - \frac{|\text{real\_amt} - \text{gen\_amt}|}{|\text{real\_amt}| + 10^{-6}}) & \text{otherwise}
   \end{cases}$$
   - 解释：总金额的相对偏差。
   - 分母 `abs(real_amt) + 1e-6`：防止除零，`1e-6` 是数值稳定性项。
   - 例：真实100、预测80 → diff = 20/100 = 0.2 → score = 0.8
   - 例：真实100、预测100 → diff = 0 → score = 1.0
   - 例：真实0、预测0 → score = 1.0（完全匹配）

3. **年度综合得分** (Year Score)
   $$\text{year\_score} = 0.5 \times \text{count\_score} + 0.5 \times \text{amount\_score}$$
   - 等权重平均次数和金额得分，范围 [0, 1]。

**第六步：多年平均与百分比转换**
```python
final_score = (sum(year_scores) / len(year_scores)) * 100
return f"预测投资信息匹配度: {final_score:.2f}%"
```
- 对所有年份的 year_score 求平均。
- 乘以 100 转为百分比。
- 格式化为 "XX.XX%" 显示。

#### 3.4 数值稳定性与防御设计

| 情景 | 处理方式 | 理由 |
|------|--------|------|
| 预测/真实数据为 None | 转换为 0.0 | 缺失数据等同于无投资 |
| 真实金额为 0，预测为 0 | amount_score = 1.0 | 完全匹配 |
| 真实金额为 0，预测 > 0 | diff = gen_amt，score 可能较低 | 误预测有投资 |
| 投资次数的相对差异 > 100% | 得分钳制在 0 | 最坏情况 |
| 数值溢出 | `1e-6` 稳定项防止除零 | 浮点精度保障 |

#### 3.5 评估指标的含义与解释

- **匹配度 = 100%**：模型预测与基准完全一致（按年统计）。
- **匹配度 ∈ (80%, 100%)**：预测质量优秀，偏差小于 20%。
- **匹配度 ∈ (60%, 80%)**：预测质量良好，系统允许输出（见提示模板中的 60% 门槛）。
- **匹配度 ∈ (40%, 60%)**：预测需改进，模型应调整参数重试。
- **匹配度 < 40%**：预测质量较差，需要明显调整。
- **匹配度 = 0%**：
  - 真实数据中完全无对应国家的投资，或
  - 预测内容格式错误导致解析失败。

### English

#### 3.1 Evaluation Overview
Prediction accuracy is computed by `test_predicted_investments()`, which compares **model-generated investment predictions** against **ground-truth baseline records**, returning a percentage match score. The core idea is: **aggregate by year while considering both count and amount**, enabling robust fuzzy comparison.

#### 3.2 Function Parameters

1. **gen_pred_context** (Predicted content)
   - Type: `str` or `list[dict]`
   - Format: `[{"Date": "2020-01-15", "tar_country": "China", "amount": 50.0, "sector": "Technology"}, ...]`

2. **pred** (Baseline/ground-truth)
   - Type: `list[dict]`
   - Format: Same as above.

3. **destination_country** (Target country, default "China")
   - Type: `str`
   - Scope: Only records matching this country are evaluated.

#### 3.3 Computation Logic

**Step 1: Robust Input Parsing**
- Converts string representations (JSON, Python repr) to list via `json.loads()` or `ast.literal_eval()`.
- Fallback: returns empty list on parse failure.

**Step 2: Year Extraction**
- Regex `r"(\d{4})"` extracts 4-digit year from date strings.
- Supported formats: "2020-03-15", "20200315", "2020/3/15" all → 2020.
- **Key property**: **ignores day/month, aggregates by year only**.

**Step 3: Amount Parsing**
- Regex `r"[^0-9.\-]"` strips non-numeric characters (except `.` and `-`).
- Converts to float, returns 0.0 on failure.
- Handles: "54.5", "50（百万美元）", "invalid" → 54.5, 50.0, 0.0 respectively.

**Step 4: Yearly Aggregation**
Real and predicted data are separately aggregated by year and target country:
```python
{2020: {"count": 3, "amount": 150.0}, 2021: {"count": 2, "amount": 80.0}}
```

**Step 5: Per-Year Score Calculation**
For each year in real data:

1. **Count Score**: Normalized difference between real and predicted investment counts
   $$\text{count\_score} = \max(0, 1 - \frac{|real\_cnt - pred\_cnt|}{\max(real\_cnt, 1)})$$

2. **Amount Score**: Normalized difference between real and predicted yearly amounts
   $$\text{amount\_score} = \max(0, 1 - \frac{|real\_amt - pred\_amt|}{|real\_amt| + 10^{-6}})$$

3. **Year Score**: Equal-weight average
   $$\text{year\_score} = 0.5 \times \text{count\_score} + 0.5 \times \text{amount\_score}$$

**Step 6: Multi-Year Average & Percentage Conversion**
$$\text{final\_score} = \frac{\sum \text{year\_scores}}{|\text{years}|} \times 100\%$$

#### 3.4 Numerical Stability & Defensive Design

| Scenario | Handling | Rationale |
|----------|----------|-----------|
| Missing amount | Convert to 0.0 | No investment |
| Real & predicted both 0 | amount_score = 1.0 | Perfect match |
| Real 0, predicted > 0 | score < 1.0 | False positive |
| Relative count diff > 100% | score clamped to 0 | Worst case |
| Division by zero | Add 1e-6 term | Numerical stability |

#### 3.5 Metric Interpretation

- **100%**: Perfect match between prediction and baseline (yearly aggregate).
- **80–100%**: Excellent prediction quality.
- **60–80%**: Good quality; system outputs final answer (60% threshold in prompt).
- **40–60%**: Needs improvement; model should adjust parameters and retry.
- **< 40%**: Poor prediction; significant parameter adjustment required.
- **0%**: No matching country in baseline or parse failure.

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
