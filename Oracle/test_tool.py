import re
import ast
import json

def test_predicted_investments(gen_pred_context, pred, destination_country="China"):
    """
    模糊对比预测内容（按“年”统计）：
    - 不精确到天，只看每年的投资次数
    - 不精确到单笔，只看每年的总投资金额接近程度

    参数：
    - gen_pred_context: 模型预测的投资记录 list[dict]
    - pred:             真实/基准的投资记录 list[dict]
    每个元素形如：
        {
            "Date": "2020-03-01",
            "tar_country": "China",
            "amount": 54.5 或 "54.5",
            "sector": "Software & IT services"
        }
    """
    def to_list_of_dict(x):
        # 已经是 list，直接用
        if isinstance(x, list):
            return x
        # 字符串：尝试 JSON / Python 表达式解析
        if isinstance(x, str):
            s = x.strip()
            # 先试 json.loads
            try:
                return json.loads(s)
            except Exception:
                pass
            # 再试 ast.literal_eval（支持单引号等 Python repr 格式）
            try:
                return ast.literal_eval(s)
            except Exception:
                # 解析失败就当空列表处理，也可以改成 raise
                return []
        # 其他类型直接当空处理
        return []

    gen_pred_context = to_list_of_dict(gen_pred_context)

    def extract_year(date_str):
        """从日期字符串中提取年份，失败返回 None。"""
        if not date_str:
            return None
        m = re.search(r"(\d{4})", str(date_str))
        return int(m.group(1)) if m else None

    def parse_amount(amount_val):
        """允许 amount 是数字或字符串，统一转成 float。"""
        if amount_val is None:
            return 0.0
        s = str(amount_val)
        cleaned = re.sub(r"[^0-9.\-]", "", s)
        if cleaned in ["", "-", "."]:
            return 0.0
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    # ------- 1. 真实数据按年份聚合 -------
    real_by_year = {}
    for inv in pred:
        if not isinstance(inv, dict):
            continue  # 防御式：避免奇怪数据
        if inv.get("tar_country") != destination_country:
            continue
        year = extract_year(inv.get("Date"))
        if year is None:
            continue
        amt = parse_amount(inv.get("amount"))
        if year not in real_by_year:
            real_by_year[year] = {"count": 0, "amount": 0.0}
        real_by_year[year]["count"] += 1
        real_by_year[year]["amount"] += amt

    # ------- 2. 生成数据按年份聚合 -------
    gen_by_year = {}
    for inv in gen_pred_context:
        if not isinstance(inv, dict):
            continue
        if inv.get("tar_country") != destination_country:
            continue
        year = extract_year(inv.get("Date"))
        if year is None:
            continue
        amt = parse_amount(inv.get("amount"))
        if year not in gen_by_year:
            gen_by_year[year] = {"count": 0, "amount": 0.0}
        gen_by_year[year]["count"] += 1
        gen_by_year[year]["amount"] += amt

    if not real_by_year:
        return "预测投资信息匹配度: 0.00%"

    # ------- 3. 按年计算“次数 + 金额”的模糊得分 -------
    year_scores = []

    for year, real_stat in real_by_year.items():
        real_cnt = real_stat["count"]
        real_amt = real_stat["amount"]

        gen_stat = gen_by_year.get(year, {"count": 0, "amount": 0.0})
        gen_cnt = gen_stat["count"]
        gen_amt = gen_stat["amount"]

        # 次数得分
        if real_cnt == 0 and gen_cnt == 0:
            count_score = 1.0
        else:
            denom_cnt = max(real_cnt, 1)
            diff_cnt = abs(real_cnt - gen_cnt) / denom_cnt
            count_score = max(0.0, 1.0 - diff_cnt)

        # 金额得分
        if abs(real_amt) < 1e-6 and abs(gen_amt) < 1e-6:
            amount_score = 1.0
        else:
            denom_amt = abs(real_amt) + 1e-6
            diff_amt = abs(real_amt - gen_amt) / denom_amt
            amount_score = max(0.0, 1.0 - diff_amt)

        year_score = 0.5 * count_score + 0.5 * amount_score
        year_scores.append(year_score)

    final_score = sum(year_scores) / len(year_scores) * 100 if year_scores else 0.0
    return f"预测投资信息匹配度: {final_score:.2f}%"
