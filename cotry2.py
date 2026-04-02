import requests
import json

DEEPSEEK_API_KEY = "sk-"

API_URL = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

prompt = "一个农场里有鸡和兔子，头的总数是 15，脚的总数是 40。请问鸡和兔子各有多少只？Let's think step by step."

request_data = {
    "model":"deepseek-chat",
    "messages":[
        {"role":"user","content":prompt}
    ]
}

try:
    response = requests.post(API_URL, json=request_data, headers=headers)
    response.raise_for_status()

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