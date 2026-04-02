import openai

DEEPSEEK_API_KEY = "sk-"
MODEL_NAME = "deepseek-chat"

client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

def agent(system_prompt,user_input):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_input}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content

WRITER_PROMPT = """你是一位优秀的量子力学科普作家。你的任务是撰写或修改一段关于量子力学的科普文字。
要求：
1. 内容准确，不能有科学错误。
2. 文笔生动，有吸引力。
3. 通俗易懂，拒绝术语堆砌，善用比喻。
如果这是第一轮，请直接写作。如果收到了修改意见，请根据意见认真修改原文，并保留原文的核心意思"""

CRITIC_PROMPT = """你是一位严谨且拥有量子力学知识的编辑。你的任务是专门挑刺并提出建设性意见。
请针对科普文章，从这几个方面提出修改意见：
1. 科学性：有没有事实错误？有没有误导性描述？
2. 可读性：够不够通俗？是不是太啰嗦或太晦涩？
3. 结构性：逻辑通顺吗？重点突出吗？
直接列出具体的修改意见，不要说客套话"""

print("====================")
current_text = ""
critic_opinion = ""

for round_num in range(1,4):
    print(f"第{round_num}轮")
    if round_num == 1:
        write_input = """请写一段关于量子力学的科普短文，500字左右"""
    else:
        write_input = f"原文:\n{current_text}\n\n修改意见:\n{critic_opinion}\n\n请根据意见修改文章"

    print("[Agent A 写作中...]")
    current_text = agent(WRITER_PROMPT,write_input)
    print(f"Agent A完成写作，输出:\n{current_text}")

    if round_num < 3:
        print(f"[Agent B 审阅中...]")
        critic_opinion = agent(CRITIC_PROMPT,current_text)
        print(f"Agent B提出修改意见:\n{critic_opinion}")

print("====================")
print(f"最终成稿:\n{current_text}")