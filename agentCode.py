import openai
import subprocess
import sys
from typing import List,Dict

client = openai.Client(
    api_key="sk-",
    base_url="https://api.deepseek.com/v1"
)
MODEL_NAME = "deepseek-coder"

def code_agent(task_description,max_attempts:int = 5):
    messages:List[Dict] = [
        {
            "role": "system",
            "content": """你是一位专业的 Python 程序员，你非常擅长根据指令和要求写代码。
            代码必须完整可运行，包含所有必要的import。如果代码执行报错，你会收到完整的错误信息，请根据错误信息精准修复代码，只输出修复后的完整纯代码。"""
        },
        {"role": "user","content":f"任务:\n{task_description}"}
    ]

    for i in range(max_attempts):
        print(f"\n\n尝试:{i+1}/{max_attempts}")

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                temperature=0.1, max_tokens=2048)
            code = response.choices[0].message.content.strip()

            if "```python" in code: code = code.split("```python")[1].split("```")[0]

            print("-" * 30, "\n", code, "\n", "-" * 30)

        except openai.APIError as e:
            print(f"API调用失败: {e}")
            return
        except Exception as e:
            print(f"代码生成异常: {e}")
            return

        result = subprocess.run(
            [sys.executable, "-c",code],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print("\n成功，输出:\n",result.stdout)
            return
        else:
            print("\n错误:\n",result.stderr)
            messages.append({"role":"assistant","content":code})
            messages.append({"role":"user","content":f"报错:\n{result.stderr}\n请根据报错信息修复"})

    print("\n已达到最大重试次数，任务执行失败")
#测试
if __name__ == "__main__":
    task = """
    请编写Python程序完成以下商品销售数据统计任务：

1. 数据生成：
   - 生成30条商品销售记录，每条记录包含：
     a. 商品编号：从1001到1030的整数（依次递增）
     b. 商品单价：10-200元之间的随机整数
     c. 销售数量：1-50件之间的随机整数
   - 将所有记录存储在列表中，列表元素为字典，格式示例：{"商品编号": 1001, "单价": 58, "销量": 12}

2. 数据计算：
   - 计算每条记录的销售额（销售额=单价×销量）
   - 统计所有商品的总销售额（整数）
   - 计算平均销售额（保留2位小数）
   - 找出销售额最高的商品（输出其编号和销售额）
   - 统计单价≥100元的高价商品数量（整数）

3. 输出规则：
   - 文件输出：将所有结果写入`sales_report.txt`，要求：
     1. 文件编码必须指定为utf-8
     2. 第一行："商品销售统计报告"
     3. 第二行："总销售额：XXX 元"
     4. 第三行："平均销售额：XX.XX 元"
     5. 第四行："销售额最高商品：编号XXX，销售额XXX 元"
     6. 第五行："高价商品（≥100元）数量：XXX 件"
   - 控制台输出：仅打印核心数值（无中文），格式示例：
     Total Sales: 12500
     Avg Sales: 416.67
     Top Product: 1015 (5280)
     High Price Count: 12

4. 约束条件：
   - 仅使用random库，禁止使用pandas、numpy等第三方库
   - 文件写入必须显式指定encoding='utf-8'
   - 控制台输出严格按要求仅打印数字/英文，避免中文输出
   - 代码中所有中文仅出现在字典键名和文件输出字段，且确保编码兼容

    """

    code_agent(task)