"""
A/B 测试实验脚本
对比 CoT Only 和 ReAct 两种方法
"""

import openai
import time
import wikipedia
from typing import Dict, List

DEEPSEEK_API_KEY = "sk-"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

def search(entity: str) -> str:
    """维基百科搜索函数"""
    try:
        page = wikipedia.page(entity)
        summary = page.summary[:500] + "..."
        print(f"搜索结果: {summary}")
        return summary
    except wikipedia.exceptions.PageError:
        return f"搜索失败：未找到 '{entity}' 的维基百科页面，请尝试调整关键词（比如简化/补充背景）"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"搜索失败：'{entity}' 存在歧义，可能的结果：{e.options[:3]}"
    except Exception as e:
        return f"搜索失败：{str(e)}，请尝试重新搜索"

def finish(answer: str) -> str:
    """结束函数，返回最终答案"""
    print(f"最终答案: {answer}")
    return answer

def build_cot_prompt(question):
    """构建CoT提示词"""
    prompt = f"""请你一步步思考并回答以下问题，先给出推理过程，最后明确给出答案。
问题:{question}
要求：
1.首先对问题进行分析
2.逐步推理，展示你的思考过程
3.最后给出你的答案"""
    return prompt

class ReActAgent:
    """ReActAgent的实现"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = base_url or DEEPSEEK_BASE_URL
        self.model = model or DEEPSEEK_MODEL

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def get_react_thought(self, question: str, history: List[Dict]):
        """调用LLM生成ReAct的思考"""

        history_str = "\n".join([
            f"{h['type'].upper()}:{h['content']}" for h in history
        ])

        prompt = f"""你必需按照ReAct框架解决问题，步骤为「思考→行动→观察」，仅允许以下两种行动：
1. SEARCH(关键词)：仅当需要验证/获取事实信息时使用，关键词必须精准
2. FINISH(答案)：仅当已获取准确事实信息后使用，答案必须完全基于搜索结果，禁止编造。

步骤：
1.THOUGHT: 思考当前需要做什么
2.ACTION: 执行行动，工具为 SEARCH(实体) 或 FINISH(最终答案)
3.OBSERVATION: 记录行动结果

问题：{question}
历史记录:
{history_str}

请思考并输出下一步行动，格式要求：
首先输出 THOUGHT: 你的思考内容，明确说明当前需要做什么（比如「需要搜索XX来验证XX」）
然后输出 ACTION: 你的行动（只能是 SEARCH 或 FINISH）"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400
        )
        return response.choices[0].message.content

    def run(self, question: str, verbose: bool = False):
        """运行ReAct Agent，返回：(最终答案, 历史记录)"""
        history = []
        max_steps = 10
        final_answer = ""

        for i in range(max_steps):
            try:
                thought_response = self.get_react_thought(question, history)

                thought = ""
                action = ""
                lines = thought_response.strip().split("\n")
                for line in lines:
                    if line.startswith("THOUGHT"):
                        thought = line.replace("THOUGHT:", "").strip()
                    elif line.startswith("ACTION"):
                        action = line.replace("ACTION:", "").strip()

                history.append({"type": "thought", "content": thought})
                if verbose:
                    print(f"第{i+1}步思考: {thought}")

                if action.startswith("SEARCH(") and action.endswith(")"):
                    entity = action.replace("SEARCH(", "").replace(")", "").strip()
                    search_result = search(entity)
                    history.append({"type": "action", "content": action})
                    history.append({"type": "observation", "content": search_result})
                    if verbose:
                        print(f"第{i+1}步行动: 搜索 {entity}")
                        print(f"第{i+1}步观察: {search_result[:100]}...")

                elif action.startswith("FINISH(") and action.endswith(")"):
                    final_answer = action.replace("FINISH(", "").replace(")", "").strip()
                    history.append({"type": "action", "content": action})
                    if verbose:
                        print(f"第{i+1}步行动: 结束")
                        print(f"最终结果: {final_answer}")
                    break

                else:
                    error_msg = f"无效行动格式: {action}，仅支持 SEARCH(entity) 或 FINISH(answer)"
                    history.append({"type": "action", "content": action})
                    history.append({"type": "observation", "content": error_msg})
                    if verbose:
                        print(f"第{i+1}步行动: {error_msg}")

            except Exception as e:
                error_msg = f"调用 LLM 出错: {str(e)}"
                history.append({"type": "observation", "content": error_msg})
                if verbose:
                    print(f"第{i+1}步错误: {error_msg}")

        if not final_answer:
            final_answer = "程序运行超时，未在最大步数内生成最终答案"
            history.append({"type": "observation", "content": final_answer})
            if verbose:
                print(f"最终结果: {final_answer}")

        return final_answer, history

class CoTAgent:
    """CoT Agent，仅使用思维链，不给工具"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """初始化 CoT Agent"""

        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base_url = base_url or DEEPSEEK_BASE_URL
        self.model = model or DEEPSEEK_MODEL

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def run(self, question: str) -> str:
        """运行 CoT Agent"""

        prompt = build_cot_prompt(question)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"调用 LLM 时发生错误：{e}"


class Experiment:
    """A/B 测试实验"""

    def __init__(self):
        """初始化实验"""

        self.react_agent = ReActAgent()
        self.cot_agent = CoTAgent()

        self.test_questions = [
            "「Python」这门编程语言的名称来源是什么？",
            "第一次工业革命开始的标志性发明是什么？",
            "《红楼梦》后 40 回的作者是谁？"
        ]

        self.expected_answers = [
            "Monty Python's Flying Circus",
            "珍妮纺纱机",
            "高鹗"
        ]

    def run_single_experiment(self, question: str, method: str) -> Dict:
        """运行单个实验"""

        start_time = time.time()

        if method == 'cot':
            answer = self.cot_agent.run(question)
            steps = 1
        else:
            answer, history = self.react_agent.run(question, verbose=False)
            steps = len([h for h in history if h['type'] == 'thought'])

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            'question': question,
            'method': method,
            'answer': answer,
            'steps': steps,
            'execution_time': execution_time
        }

    @staticmethod
    def evaluate_answer(answer: str, expected: str) -> bool:
        """评估答案是否正确"""

        answer_lower = answer.lower()
        expected_lower = expected.lower()

        if expected_lower in answer_lower:
            return True

        variants = {
            "Monty Python's Flying Circus": ["monty python","python 喜剧团体","飞行马戏团","flying circus","monty python's flying circus"],
            "珍妮纺纱机": ["珍妮纺纱机", "珍妮机", "哈格里夫斯", "jenny spinning machine", "jenny machine"],
            "高鹗": ["高鹗", "gao e", "高"]
        }

        if expected in variants:
            for variant in variants[expected]:
                if variant.lower() in answer_lower:
                    return True

        return False

    def run_all_experiments(self) -> List[Dict]:
        """运行所有实验"""

        results = []

        print("开始 A/B 测试实验")
        print("=" * 60)

        for i, question in enumerate(self.test_questions):
            expected = self.expected_answers[i]

            print(f"\n问题 {i + 1}: {question}")
            print(f"预期答案: {expected}")
            print("-" * 40)

            # 运行 CoT 实验
            print("运行 CoT 方法...")
            cot_result = self.run_single_experiment(question, 'cot')
            cot_correct = self.evaluate_answer(cot_result['answer'], expected)
            cot_result['correct'] = cot_correct

            print(f"CoT 答案: {cot_result['answer'][:100]}...")
            print(f"CoT 正确: {cot_correct}")
            print(f"CoT 耗时: {cot_result['execution_time']:.2f}秒")

            # 运行 ReAct 实验
            print("\n运行 ReAct 方法...")
            react_result = self.run_single_experiment(question, 'react')
            react_correct = self.evaluate_answer(react_result['answer'], expected)
            react_result['correct'] = react_correct

            print(f"ReAct 答案: {react_result['answer']}")
            print(f"ReAct 正确: {react_correct}")
            print(f"ReAct 步骤: {react_result['steps']}")
            print(f"ReAct 耗时: {react_result['execution_time']:.2f}秒")

            results.append({
                'question': question,
                'expected': expected,
                'cot': cot_result,
                'react': react_result
            })

        return results

    @staticmethod
    def print_summary(results: List[Dict]):
        """打印实验总结"""

        print("\n" + "=" * 60)
        print("实验总结")
        print("=" * 60)

        cot_correct_count = 0
        react_correct_count = 0
        cot_total_time = 0
        react_total_time = 0

        print("\n详细结果：")
        print("-" * 60)
        for i, result in enumerate(results):
            print(f"\n问题 {i + 1}: {result['question']}")
            print(f"预期答案: {result['expected']}")
            print(f"CoT 答案: {result['cot']['answer'][:100]}...")
            print(f"CoT 正确: {result['cot']['correct']}")
            print(f"ReAct 答案: {result['react']['answer']}")
            print(f"ReAct 正确: {result['react']['correct']}")

            if result['cot']['correct']:
                cot_correct_count += 1
            if result['react']['correct']:
                react_correct_count += 1

            cot_total_time += result['cot']['execution_time']
            react_total_time += result['react']['execution_time']

        print("\n" + "=" * 60)
        print("统计结果：")
        print("-" * 60)
        print(f"CoT 方法正确率: {cot_correct_count}/{len(results)} ({cot_correct_count / len(results) * 100:.1f}%)")
        print(
            f"ReAct 方法正确率: {react_correct_count}/{len(results)} ({react_correct_count / len(results) * 100:.1f}%)")
        print(f"CoT 平均耗时: {cot_total_time / len(results):.2f}秒")
        print(f"ReAct 平均耗时: {react_total_time / len(results):.2f}秒")

        print("\n幻觉分析：")
        print("-" * 60)
        for i, result in enumerate(results):
            if not result['cot']['correct'] and result['react']['correct']:
                print(f"问题 {i + 1}: CoT 产生幻觉，ReAct 正确")
            elif result['cot']['correct'] and not result['react']['correct']:
                print(f"问题 {i + 1}: CoT 正确，ReAct 产生幻觉")
            elif not result['cot']['correct'] and not result['react']['correct']:
                print(f"问题 {i + 1}: 两种方法都产生幻觉")
            else:
                print(f"问题 {i + 1}: 两种方法都正确")


def main():
    """主函数"""

    print("ReAct vs CoT A/B 测试实验")
    print("=" * 60)

    experiment = Experiment()

    results = experiment.run_all_experiments()

    experiment.print_summary(results)


if __name__ == "__main__":
    main()