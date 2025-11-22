import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import ast

def format_company_analysis(result) -> str:
    """
    接受两种情况：
    - result 是 dict，例如 {'Microsoft': '一、...'}
    - result 是 str，例如 "{'Microsoft': '一、...'}"

    返回一个干净的多行字符串：
        Microsoft

        一、基于历史对华投资轨迹的关键信息
        ...
    """

    # 1. 如果已经是 dict，就直接用
    if isinstance(result, dict):
        data = result
    else:
        # 2. 否则，把它当字符串处理
        raw = str(result).strip()
        if not raw:
            return ""

        # 2.1 尝试用 literal_eval 把 "{'Microsoft': '...'}" 变回 dict
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                data = parsed
            else:
                # 解析出来的不是 dict，就当作纯文本返回
                return raw
        except Exception:
            # 完全解析失败，直接当作完整文本返回
            return raw

    # 3. 走到这里，说明我们手里有一个 dict: {company -> content}
    parts = []
    for company, content in data.items():
        # 确保是字符串
        content = str(content)

        # 把字面量的 \n 变成真正换行（如果原本就是正常换行，这一步也安全）
        content = content.replace("\\n", "\n").strip()

        # 组合：公司名 + 两个换行 + 正文
        if company:
            parts.append(f"{company}\n\n{content}")
        else:
            parts.append(content)

    return "\n\n".join(parts)


def normalize_date(date_str: str) -> str:
    """把日期标准化成 YYYY-MM-DD 格式。"""
    if pd.isna(date_str):
        return ""
    dt = pd.to_datetime(str(date_str), errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.strftime("%Y-%m-%d")


def extract_amount(amount_raw) -> float:
    """从 '18.03（百万美元）' 这类字符串中提取数值。"""
    if pd.isna(amount_raw):
        return 0.0

    cleaned = re.sub(r"[^0-9.\-]", "", str(amount_raw))
    if cleaned in ["", "-", "."]:
        return 0.0

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def process_investment_csv(input_csv_path: str) -> dict:
    """
    以 Parent_Company 为 key，
    返回值为 (investments_dict, source_country, company_sector_list)。

    其中 investments_dict 为：
    {
        "train": [  # 2020 年之前的投资记录
            { "Date": ..., "tar_country": ..., "amount": ..., "sector": ... },
            ...
        ],
        "pred": [   # 2020 年及之后的投资记录
            { "Date": ..., "tar_country": ..., "amount": ..., "sector": ... },
            ...
        ]
    }

    需要的 CSV 列：
    - Parent_Company
    - Source_Country
    - Project_Date
    - Destination_Country
    - Capital_Investment
    - Sector
    """
    input_path = Path(input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {input_csv_path}")

    df = pd.read_csv(input_path)

    required_columns = [
        "Parent_Company",
        "Source_Country",
        "Project_Date",
        "Destination_Country",
        "Capital_Investment",
        "Sector",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必须列: {missing}")

    df["Parent_Company"] = df["Parent_Company"].fillna("UNKNOWN_COMPANY")
    df["Source_Country"] = df["Source_Country"].fillna("UNKNOWN_COUNTRY")
    df["Sector"] = df["Sector"].fillna("UNKNOWN_SECTOR")

    result = {}

    for company, group in df.groupby("Parent_Company"):
        train_investments = []
        pred_investments = []
        sectors = set()
        source_countries = set()

        for _, row in group.iterrows():
            date_str = normalize_date(row["Project_Date"])
            amount = extract_amount(row["Capital_Investment"])
            sector = row["Sector"]
            tar_country = row["Destination_Country"]

            record = {
                "Date": date_str,
                "tar_country": tar_country,
                "amount": amount,
                "sector": sector,
            }

            # 记录 sector、source_country
            sectors.add(sector)
            source_countries.add(row["Source_Country"])

            # 按年份切分 train / pred
            year = None
            if date_str:
                try:
                    year = int(date_str[:4])
                except ValueError:
                    year = None

            # 约定：year >= 2020 为预测集，其余为训练集
            if year is not None and year >= 2020:
                pred_investments.append(record)
            else:
                train_investments.append(record)

        # source_country 一般是唯一的；如果不是，就返回列表
        if len(source_countries) == 1:
            source_country = list(source_countries)[0]
        else:
            source_country = list(source_countries)

        investments_dict = {
            "train": train_investments,
            "pred": pred_investments,
        }

        result[company] = (
            investments_dict,   # 聚合的投资行为，已按时间切分
            source_country,     # 来源国
            list(sectors),      # 所有 sector 汇总
        )

    return result

def process_policy_csv(csv_path: str) -> Dict[Tuple[int, str], List[dict]]:
    """
    简化版政策处理：
    - 从 Compustat_Microsoft.csv 中读取数据
    - 通过 (year, company) 进行索引
      * year: 来自 fyear 或 datadate
      * company: 使用 Compustat 的公司名 conm
    - 返回的 value 是一个包含关键政策/环境信息的 dict 列表

    返回结构示意：
    {
        (2021, "MICROSOFT CORP"): [
            {
                "gvkey": "...",
                "conm": "MICROSOFT CORP",
                "fic": "USA",
                "loc": "USA",
                "gsector": "...",
                "gind": "...",
                "gsubind": "...",
                "naics": "...",
                "sic": "...",
                "idbflag": "D/B/I",
                "exchg": 11,
                "at": 123456.0,
                "sale": 98765.0,
                "revt": 98765.0,
                "ni": 10000.0,
                "ib": 9500.0,
                "icapt": 80000.0,
                "lt": 50000.0,
                "ceq": 30000.0,
                "dltt": 10000.0,
                "wcap": 15000.0,
                "raw": {... 原始整行 ...}
            },
            ...
        ],
        ...
    }
    """

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    df = pd.read_csv(csv_file)

    # 1. 标准化列名：全小写 + 空格转下划线
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]

    def find_col(candidates, required: bool = True) -> Any:
        """在若干候选列名中找到第一个存在的列名。"""
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise ValueError(f"CSV 中未找到列: {candidates}")
        return None

    # 2. 年份列：优先用 fyear，其次 datadate，再次 year
    year_col = find_col(["fyear", "datadate", "year", "date"], required=True)

    # 3. 公司列：优先使用 Compustat 的 conm（Company Name）
    company_col = find_col(["conm", "company", "parent_company"], required=True)

    # 4. 从 year_col 抽取年份
    def extract_year(v) -> int:
        if pd.isna(v):
            return None
        s = str(v)
        # fyear: 2021
        if s.isdigit() and len(s) == 4:
            return int(s)
        # datadate: 20231130
        if s.isdigit() and len(s) == 8:
            return int(s[:4])
        # 其他日期形式：寻找 4 位年份
        m = re.search(r"(\d{4})", s)
        return int(m.group(1)) if m else None

    df["__year"] = df[year_col].apply(extract_year)

    # 5. 依据说明文档挑选一些关键字段（存在才会保留）
    # 公司标识/基本信息
    key_fields = [
        "gvkey",     # Global Company Key
        "conm",      # Company Name
        "tic",       # Ticker
        "fic",       # Country of Incorporation
        "loc",       # Headquarters Country
        "idbflag",   # International/Domestic/Both
        "exchg",     # Stock Exchange Code
    ]

    # 行业 & 业务
    key_fields += [
        "gsector",   # GICS Sector
        "ggroup",    # GICS Group
        "gind",      # GICS Industry
        "gsubind",   # GICS SubIndustry
        "sic",       # SIC Code
        "naics",     # NAICS Code
        "busdesc",   # Business Description
    ]

    # 规模、资产负债、盈利等“环境”指标（可当成公司层面的政策/条件）
    key_fields += [
        "at",        # Total Assets
        "sale",      # Sales/Turnover (Net)
        "revt",      # Total Revenue
        "ni",        # Net Income
        "ib",        # Income Before Extraordinary Items
        "icapt",     # Invested Capital - Total
        "lt",        # Total Liabilities
        "ceq",       # Common Equity - Total
        "dltt",      # Long-Term Debt - Total
        "wcap",      # Working Capital
        "txdb",      # Deferred Taxes (Balance Sheet)
        "txt",       # Total Income Taxes
        "govgr",     # Government Grants (若有)
        "tlcf",      # Tax Loss Carry Forward (若有)
    ]

    available_fields = [f for f in key_fields if f in df.columns]

    # 6. 建立索引：(year, company) -> [policy dict]
    index_map: Dict[Tuple[int, str], List[dict]] = {}

    for _, row in df.iterrows():
        year = row["__year"]
        if year is None:
            continue

        company = str(row[company_col]).strip()
        if not company:
            continue

        key = (int(year), company)

        record = {f: row[f] for f in available_fields}
        record["raw"] = row.to_dict()

        index_map.setdefault(key, []).append(record)

    return index_map

# def process_policy_csv(csv_path: str) -> Dict[Tuple[int, str, str, str], List[dict]]:
#     import pandas as pd
#     import re
#     import json
#     import ast
#     from pathlib import Path

#     csv_file = Path(csv_path)
#     if not csv_file.exists():
#         raise FileNotFoundError(f"政策 CSV 文件不存在: {csv_path}")

#     df = pd.read_csv(csv_file)

#     # -------- 1. 标准化列名 --------
#     df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]

#     def find_col(candidates):
#         for c in candidates:
#             if c in df.columns:
#                 return c
#         raise ValueError(f"CSV 中未找到列: {candidates}")

#     # -------- 2. Compustat-friendly 年份字段 --------
#     # Compustat 字段包含：
#     #   datadate: 日期（最标准）
#     #   fyear: 财报年份
#     #   fyr: 财报年结束月份（非必须）
#     year_col = find_col(["datadate", "fyear", "date", "year", "policy_year"])

#     # 行业字段
#     sector_col = find_col(["sector", "industry", "naics", "sic", "gsector", "gind"])

#     # 国家字段（企业所在国 / 投资来源国）
#     src_col = find_col(["source_country", "src_country", "home_country", "fic", "loc"])

#     # 目标国字段
#     dst_col = find_col(["destination_country", "dest_country", "host_country", "target_country"])

#     # -------- 3. 抽取年份 --------
#     def extract_year(v):
#         if pd.isna(v):
#             return None
#         s = str(v)
#         # fyear 是纯数字
#         if s.isdigit() and len(s) == 4:
#             return int(s)
#         # datadate，例如 20230105
#         if s.isdigit() and len(s) == 8:
#             return int(s[:4])
#         # 日期形式
#         m = re.search(r"(\d{4})", s)
#         return int(m.group(1)) if m else None

#     df["__year"] = df[year_col].apply(extract_year)

#     # -------- 4. 建立索引 --------
#     index_map = {}

#     key_fields = [
#         "policy_type", "policy_strength", "tax_change", "subsidy_amount",
#         "regulatory_risk", "market_openness", "capital_control_level",
#         "gdp_growth", "interest_rate", "exchange_rate"
#     ]
#     available_fields = [c for c in key_fields if c in df.columns]

#     for _, row in df.iterrows():
#         year = row["__year"]
#         if year is None:
#             continue

#         sector = str(row[sector_col]).strip()
#         src_country = str(row[src_col]).strip()
#         dst_country = str(row[dst_col]).strip()

#         key = (year, sector, src_country, dst_country)
#         record = {f: row[f] for f in available_fields}
#         record["raw"] = row.to_dict()

#         index_map.setdefault(key, []).append(record)

#     return index_map

