import requests
import json
import sys

API_KEY = "sk-密钥"
url = "https://api.deepseek.com/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role":"user","content":"Hello World"}
    ],
    "stream":False
}

try:
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print("Full return content")
        print(json.dumps(result,indent=2,ensure_ascii=False))
        print("AI return content")
        print(result['choices'][0]['message']['content'])
    else:
        print(f"Request failed, status code:{response.status_code}")
        print(f"message:{response.text}")
        sys.exit(1)

except Exception as e:
    print(f"Runtime error:{e}")
    sys.exit(1)