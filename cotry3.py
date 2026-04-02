import requests

DEEPSEEK_API_KEY = "sk-"

API_URL = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

prompt = """
示例1：
问题：头10个，脚28只，鸡兔各有几只？
答案：鸡有6只，兔子有4只。
推导过程：
1. 设鸡x只，兔y只；
2. 列方程：x + y = 10，2x + 4y = 28；
3. 解方程：x=10-y，代入得2(10-y)+4y=28 → 20+2y=28 → y=4，x=6。

示例2：
问题：头15个，脚40只，鸡兔各有几只？
答案：鸡有10只，兔子有5只。
推导过程：
1. 设鸡x只，兔y只；
2. 列方程：x + y = 15，2x + 4y = 40；
3. 解方程：x=15-y，代入得2(15-y)+4y=40 → 30+2y=40 → y=5，x=10。

示例3：
问题：头20个，脚56只，鸡兔各有几只？
答案：鸡有12只，兔子有8只。
推导过程：
1. 设鸡x只，兔y只；
2. 列方程：x + y = 20，2x + 4y = 56；
3. 解方程：x=20-y，代入得2(20-y)+4y=56 → 40+2y=56 → y=8，x=12。

现在回答：头35个，脚94只，鸡兔各有几只?"""

data = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": prompt}]
}

try:
    response = requests.post(API_URL, headers=headers, json=data)
    result = response.json()
    answer = result["choices"][0]["message"]["content"]

    print("======answer======")
    print(answer)

except requests.exceptions.RequestException as e:
    print(e)
    if hasattr(e,'response') and e.response is not None:
        print(f"wrong!{e.response}")
except KeyError as e:
    print(e)
