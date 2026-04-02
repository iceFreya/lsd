import wikipedia
import openai


def search(entity):
    try:
        page = wikipedia.page(entity)
        print(f"搜索结果:{page.summary[:100]+'...'}")
        return page.summary[:500] + "..."
    except Exception as e:
        return f"搜索失败:{e}"

def finish(answer):
    print(answer)
    return answer

class ReActEnv:
    def __init__(self):
        self.tool = {
            "search": search,
            "finish": finish,
        }

    def step(self, action):
        try:
            if "[" not in action or "]" not in action:
                return "格式错误，请使用 工具名[参数] 格式"

            tool_name = action.split("[")[0].strip()
            args = action.split("[")[1].split("]")[0].strip()

            if tool_name not in self.tool:
                return f"未知工具:{tool_name},仅支持 search 和 finish"

            return self.tool[tool_name](args)
        except Exception as e:
            return f"解析失败:{e}"

client = openai.OpenAI(
    api_key="sk-",
    base_url="https://api.deepseek.com/v1"
)

BASE_PROMPT ="""
你是一个善于思考和使用工具的智能助手，你需要严格按照指定格式完成任务，仅能使用以下两个工具：
1. search[实体]：搜索指定实体的信息，返回维基百科摘要
2. finish[答案]：输出最终答案并结束任务

输出格式必须严格遵循（缺一不可）：
Thought: 你的思考过程（说明为什么要调用这个工具）
Action: 工具名[参数]

示例：
问题：居里夫人哪一年发现镭？
Thought: 我需要获取居里夫人发现镭的年份，因此调用 search 工具搜索相关信息
Action: search[居里夫人 镭]
Observation: 1898年7月18日，玛丽·居里与皮埃尔·居里发现钋（polonium）；12月26日，发现镭（Radium）
Thought: 我已经从搜索结果中找到居里夫人发现镭的年份是1898年，现在调用 finish 工具输出最终答案
Action: finish[居里夫人于1898年发现镭]

示例：
问题：爱因斯坦提出相对论是哪一年？ 
Thought: 我需要获取爱因斯坦提出相对论的具体年份，因此调用 search 工具搜索相关信息
Action: search[爱因斯坦 相对论] 
Observation: 1905年，爱因斯坦发表了《论动体的电动力学》，首次提出了狭义相对论；1915年，他又完成了广义相对论的建立。
Thought: 我已经从搜索结果中找到爱因斯坦提出狭义相对论的年份是1905年，现在调用 finish 工具输出最终答案
Action: finish[爱因斯坦于1905年提出狭义相对论]

示例：
问题：谁是第一个踏上月球的宇航员？ 
Thought: 我需要获取第一个踏上月球的宇航员姓名，因此调用 search 工具搜索相关信息
Action: search[第一个踏上月球的宇航员] 
Observation: 1969年7月21日，尼尔·阿姆斯特朗（Neil Armstrong）在执行阿波罗11号任务时，成为第一个踏上月球的宇航员。 Thought: 我已经从搜索结果中找到第一个踏上月球的宇航员是尼尔·阿姆斯特朗，现在调用 finish 工具输出最终答案
Action: finish[第一个踏上月球的宇航员是尼尔·阿姆斯特朗]

现在请回答以下问题：{question}
"""

def react_deepseek(question,max_steps:int = 10):
    env = ReActEnv()
    current_prompt = BASE_PROMPT.format(question=question)

    for step in range(max_steps):
        print(f"\n======步骤{step + 1}======")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": current_prompt}],
            temperature=0.1,
            max_tokens=500
        )

        llm_output = response.choices[0].message.content.strip()
        print(f"模型输出:{llm_output}")

        if "Thought" not in llm_output or "Action" not in llm_output:
            print("模型输出格式错误，重新生成")
            current_prompt += f"\n{llm_output}\nObservation:格式错误，请严格按照要求输出 Thought 和 Action\n"
            continue

        action_line = [line for line in llm_output.split("\n")if line.startswith("Action:")[0]]
        action = action_line.replace("Action:","").strip()

        if action.startswith("finish["):
            answer = action.split("finish[")[1].split("]")[0].strip()
            print(f"\n最终答案:{answer}")
            return answer
        if action.startswith("search["):
            observation = env.step(action)
            print(f"观察结果：:{observation}")

            current_prompt += f"\n{llm_output}\nObservation:{observation}\n"
        else:
            observation = f"无效动作:{action}"
            current_prompt += f"\n{llm_output}\nObservation:{observation}\n"

    print(f"\n 达到最大步数{max_steps}步,任务停止")
    return None
