import ast
import inspect
import os
os.environ["OPENROUTER_API_KEY"] = "sk-xtbmmu8JYNTtQHpZ9zW7co27gtzAuQdckXAQzxgpwEk6l7Bc"
import re
import ast
import json
from string import Template
from typing import List, Callable, Tuple, Any, Dict

import click
from dotenv import load_dotenv
from openai import OpenAI

import platform

from prompt_template import data_analysis_system_prompt_template, task_router_system_prompt_template
from data_process import process_investment_csv,process_policy_csv,format_company_analysis
from test_tool import test_predicted_investments


class BaseAgent:
    def __init__(self, 
                 tools: List[Callable], 
                 model: str
                 ):
        self.tools = { func.__name__: func for func in tools }
        self.model = model
        self.client = OpenAI(
            base_url="https://api.chatanywhere.tech/v1",
            api_key=BaseAgent.get_api_key(),
        )

    @staticmethod
    def get_api_key() -> str:
        """Load the API key from an environment variable."""
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。")
        return api_key
    
    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)
    
    def add_tool(self, tool: Callable):
        """添加新工具"""
        self.tools[tool.__name__] = tool

    def render_system_prompt(self, system_prompt_template: str) -> str:
        pass

    def update_system_prompt(self, new_system_prompt: str):
        pass

    def run(self, user_input: str):
        pass

    def call_model(self, messages):
            print("\n\n正在请求模型，请稍等...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            return content
        
    

class RouterAgent(BaseAgent):
    def __init__(self, 
                 tools: List[Callable], 
                 model: str,
                 company_agents: dict[str, 'CompanyAgent']
                 ):
        super().__init__(tools, model)
        self.company_agents = company_agents
        self.previous_context = ""  # 用于存储之前的对话上下文
        self.message_history = []  # 用于存储消息历史记录
    
    def run(self, user_input: str):
        messages = [
            {"role": "system", "content": self.render_system_prompt(task_router_system_prompt_template)},
            {"role": "user", "content": f"<question>{user_input}</question>"}
        ]

        # 请求模型
        content = self.call_model(messages)

        # 提取分配的具体问题
        agent_questions = re.findall(r"<agent>\s*<name>(.*?)</name>\s*<question>(.*?)</question>\s*</agent>", content, re.DOTALL)
        agent_question_map = {name.strip(): question.strip() for name, question in agent_questions}

        results = {}
        for agent_name, question in agent_question_map.items():
            agent = self.company_agents.get(agent_name)
            if agent:
                print(f"\n\n➡️ 正在将任务交给公司代理：{agent_name}")
                result = agent.run(question)
                results[agent_name] = result
            else:
                results[agent_name] = f"未找到名为 {agent_name} 的公司代理。"
        return results
    
    def render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提示模板，替换变量"""
        sub_agents = "\n".join([f"- {name}" for name in self.company_agents.keys()])
        return Template(system_prompt_template).substitute(
            sub_agents=sub_agents
        )



class CompanyAgent(BaseAgent):
    def __init__(self, 
                 tools: List[Callable], 
                 model: str,
                 company_name: str,
                 company_country: str,
                 company_sector: set,
                 risk_preference_score: int,
                 imitation_score: int,
                 policy_impact_score: int,
                 expansion_score: int,
                 history_investments: List[dict],
                 predicted_investments: List[dict]|None = None,
                 ):
        super().__init__(tools, model)
        self.company_name = company_name
        self.company_country = company_country
        self.company_sector = company_sector
        self.risk_preference_score = risk_preference_score
        self.imitation_score = imitation_score
        self.policy_impact_score = policy_impact_score
        self.expansion_score = expansion_score
        self.history_investments = history_investments
        self.predicted_investments = predicted_investments if predicted_investments is not None else []

        self.previous_context = ""  # 用于存储之前的对话上下文
        self.message_history = []  # 用于存储消息历史记录
        

    def run(self, user_input: str):
        messages = [
            {"role": "system", "content": self.render_system_prompt(data_analysis_system_prompt_template)},
            {"role": "user", "content": f"<question>{user_input}</question>"}
        ]

        while True:

            # 请求模型
            content = self.call_model(messages)
            # print(f"\n\n📝 公司模型回复内容：\n{content}")
            # 检测 Thought
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                print(f"\n\n💭 Thought: {thought}")

            # 检测模型是否输出 Final Answer，如果是的话，直接返回
            if "<final_answer>" in content:
                final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                if final_answer:    
                    return final_answer.group(1)
                

            # 检测 Action
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                error_msg = (
                    "上一步的回复缺少 <action> 标签。请严格按照："
                    "<thought>…</thought> 后紧跟 <action>…</action> 的格式重新回答。"
                )
                print(f"\n\n⚠️  Format Warning：{error_msg}")
                messages.append({"role": "user", "content": f"<observation>{error_msg}</observation>"})
                continue
            action = action_match.group(1)
            tool_name, args, kwargs = self.parse_action(action)


            # 打印参数（避免 list/dict 之类 join 出错）
            pos_str = ", ".join(repr(a) for a in args)
            kw_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            all_str = ", ".join(s for s in [pos_str, kw_str] if s)  # 兼容没有 kwargs 的情况

            print(f"\n\n🔧 Action: {tool_name}({all_str})")
            # 只有终端命令才需要询问用户，其他的工具直接执行
            should_continue = input(f"\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。")
                return "操作被用户取消"

            try:
                observation = self.tools[tool_name](*args, **kwargs)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            print(f"\n\n🔍 Observation：{observation}")
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})
            #更新system prompt里的参数
            messages[0]["content"] = self.render_system_prompt(data_analysis_system_prompt_template)
        


    def render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提示模板，替换变量"""
        tool_list = self.get_tool_list()
        return Template(system_prompt_template).substitute(
            tool_list=tool_list,
            company_name=self.company_name,
            company_country=self.company_country,
            company_sector=self.company_sector,
            risk_preference_score=self.risk_preference_score,
            imitation_score=self.imitation_score,
            policy_impact_score=self.policy_impact_score,
            expansion_score=self.expansion_score,
            history_investments=self.history_investments
        )

        
    def parse_action(self, code_str: str) -> Tuple[str, List[Any], Dict[str, Any]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            # raise ValueError("Invalid function call syntax") #made
            return  "Final Answer should be provided instead of action" , [], {}

        func_name = match.group(1)
        args_str = match.group(2).strip()

        args = []
        kwargs = {}

        current = ""
        in_string = False
        string_char = None
        paren = bracket = brace = 0
        i = 0

        def flush_token(token):
            token = token.strip()
            if not token:
                return

            # keyword argument?  key=value
            if '=' in token and not token.startswith(('{"', "{'", "[")):  
                # split only at top-level '='
                key, value = token.split('=', 1)
                key = key.strip()
                kwargs[key] = self._parse_single_arg(value.strip())
            else:
                args.append(self._parse_single_arg(token))

        while i < len(args_str):
            ch = args_str[i]

            if not in_string:
                if ch in ['"', "'"]:
                    in_string = True
                    string_char = ch
                    current += ch
                elif ch == '(':
                    paren += 1; current += ch
                elif ch == ')':
                    paren -= 1; current += ch
                elif ch == '[':
                    bracket += 1; current += ch
                elif ch == ']':
                    bracket -= 1; current += ch
                elif ch == '{':
                    brace += 1; current += ch
                elif ch == '}':
                    brace -= 1; current += ch
                elif ch == ',' and paren == bracket == brace == 0:
                    flush_token(current)
                    current = ""
                else:
                    current += ch
            else:
                current += ch
                if ch == string_char and args_str[i-1] != '\\':
                    in_string = False
                    string_char = None

            i += 1

        if current.strip():
            flush_token(current)

        return func_name, args, kwargs

    def _parse_single_arg(self, arg_str: str):
        arg_str = arg_str.strip()

        # 判断是否为字符串字面量，包括被 LLM 转义成 \"...\" 的情况
        # 情况1: "China"
        # 情况2: \"China\"
        if (
            (arg_str.startswith('"') and arg_str.endswith('"')) or
            (arg_str.startswith('\\"') and arg_str.endswith('\\"'))
        ):
            # 去除最外层引号（处理多次转义）
            s = arg_str

            # Case like \"China\" -> strip first and last \"
            if s.startswith('\\"') and s.endswith('\\"'):
                s = s[2:-2]

            # Case like "China" -> strip quotes normally
            elif s.startswith('"') and s.endswith('"'):
                s = s[1:-1]

            # 把内部的转义字符标准化
            s = s.replace('\\"', '"')
            s = s.replace("\\'", "'")
            s = s.replace('\\\\', '\\')
            s = s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

            return s

        # 其他 literal（数字、dict、list 等）
        try:
            return ast.literal_eval(arg_str)
        except Exception:
            return arg_str


    def flush_context(self):
        """清空之前的对话上下文"""
        self.previous_context = ""
    
    def compress_context(self, max_length: int = 2000):
        """压缩对话上下文以适应最大长度限制"""
        if len(self.previous_context) <= max_length:
            return self.previous_context
        
        # 简单的压缩策略：保留开头和结尾部分，中间省略
        half_length = max_length // 2
        compressed_context = (
            self.previous_context[:half_length] +
            "\n...（中间内容省略）...\n" +
            self.previous_context[-half_length:]
        )
        return compressed_context
    


def update_company_parameters(company_agent: CompanyAgent,
                              risk_preference_score: float,
                              imitation_score: float,
                              policy_impact_score: float,
                              expansion_score: float
                              ):
    company_agent.risk_preference_score = risk_preference_score
    company_agent.imitation_score = imitation_score
    company_agent.policy_impact_score = policy_impact_score
    company_agent.expansion_score = expansion_score
    return "公司参数更新成功"


def build_company_regex(company_name: str) -> re.Pattern:
    """
    为公司名生成鲁棒正则表达式，用于匹配大小写变化、空格变化、连接符等。
    适合匹配政策文本。
    """

    # 去掉多余空白
    company_name = company_name.strip()

    # 主体：公司名第一个单词（如 Microsoft）
    main_word = company_name.split()[0]

    # 把主词拆成字符并允许空白符或符号隔开
    # 例如 "Microsoft" -> M\s*i\s*c\s*r\s*o\s*s\s*o\s*f\s*t
    fuzzy_main = r'\s*'.join(list(main_word))

    # 匹配完整公司名（允许空白/连字符差异）
    # Microsoft Corporation -> Microsoft\s*Corporation
    fuzzy_full = r'\s*'.join(company_name.split())

    # 最终模式：匹配主词 或 全称
    pattern = rf"({fuzzy_main})|({fuzzy_full})"

    return re.compile(pattern, flags=re.IGNORECASE)

def read_policy(policy_dict: Dict[Tuple[int, str], List[dict]], year: int, company: str) -> str:
    key = (year, company)
    return f"{policy_dict.get(key, [])}"

def main():
    investment_tuple = process_investment_csv("/Users/zhangyilin/Desktop/Test Data/fDi Market__Microsoft.csv")
    policy_dict= process_policy_csv("/Users/zhangyilin/Desktop/Test Data/Compustat_Microsoft.csv")
    print(policy_dict.keys())
    Company_group = {}
    for company, (investments, source_country, company_sector) in investment_tuple.items():
        print(f"公司: {company}")
        for inv in investments["train"]:
            print(f"  投资日期: {inv['Date']}, 目标国家: {inv['tar_country']}, 投资金额: {inv['amount']}百万美元, 领域: {inv['sector']}")
        predicted_invs = investments["pred"]
        #筛选出和公司有关的政策
        company_policy_dict = {}
        sorter = build_company_regex(company)
        for (year, comp), policies in policy_dict.items(): 
            if sorter.search(comp):
                company_policy_dict[(year, company)] = policies
        print(f"公司相关政策年份：{list(company_policy_dict.keys())}")

        def company_test(predicted_investments_str: str,destination_country) -> str:
            gen_pred_context = f"{predicted_investments_str}"
            return test_predicted_investments(gen_pred_context, predicted_invs, destination_country)
        
        def company_read_policy(year: int) -> str:
            return read_policy(company_policy_dict, year, company)
        
        tools = [company_test,company_read_policy]
        agent = CompanyAgent(
            tools=tools,
            model="gpt-5.1",
            company_name=company,
            company_country=source_country,
            company_sector=company_sector,
            risk_preference_score=0.52,
            imitation_score=0.56,
            policy_impact_score=0.82,
            expansion_score=0.50,
            history_investments=investments["train"],
            predicted_investments=investments["pred"],
        )
        def company_update_parameters(risk_preference_score: float,
                                      imitation_score: float,
                                      policy_impact_score: float,
                                      expansion_score: float
                                      ) -> str:
            return update_company_parameters(
                agent,
                risk_preference_score,
                imitation_score,
                policy_impact_score,
                expansion_score
            )
        agent.add_tool(company_update_parameters)

        Company_group[company] = agent

    router_agent = RouterAgent(
        tools=tools,
        model="gpt-5.1",
        company_agents=Company_group
    )
    while True:
        task = input("请输入任务：")
        result = router_agent.run(task)
        print(f"\n\n✅ 任务完成，结果如下：\n{format_company_analysis(result)}")
        

if __name__ == "__main__":
    main()
#预测20、21、22年Microsoft对华投资