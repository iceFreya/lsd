import requests
import json

DEEPSEEK_API_KEY = "sk-"
API_URL = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

prompt = """
必须严格按以下要求输出：
1.无论用户输入什么内容，只输出JSON字符串，不添加任何自然语言和换行；
2.JSON格式固定为：{"location": "xxx", "time": "xxx", "intent": "xxx"}；
3.location：提取用户输入中的地点，无地点则为空字符串；
4. time：提取用户输入中的时间，无时间则为空字符串；
5. intent：提取用户输入的意图；

示例：用户输入"明天北京天气如何"，输出{"location": "北京", "time": "明天", "intent": "weather"}。
"""
user_input = "后天成都温度如何"

data = {
    "model": "deepseek-chat",
    "messages":[
        {"role":"system","content":prompt},
        {"role":"user","content":user_input}
    ]
}

try:
    response = requests.post(API_URL, headers=headers, json=data)
    result = response.json()

    output = result["choices"][0]["message"]["content"].strip()
    print(f"output:{output}")

except json.decoder.JSONDecodeError:
    print("wrong!")
except Exception as e:
    print(f"wrong!{e}")
