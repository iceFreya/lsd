import openai
import json
from typing import List, Optional
from enum import Enum

from pydantic import BaseModel, Field,ValidationError,field_validator

DEEPSEEK_API_KEY = "sk-"
MODEL_NAME = "deepseek-chat"

client = openai.OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

class Gender(str,Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class AccountStatus(int,Enum):
    INACTIVE = 0
    ACTIVE = 1
    SUSPENDED = 2

class Preferences(BaseModel):
    notifications:bool
    dark_theme:bool
    font_size:str = Field(...,description="字体大小,如 'large','medium','small'")

class UserProfile(BaseModel):
    user_id: int = Field(..., description="必须是整数")
    username: str
    age:int = Field(...,gt=0,lt=110,description="年龄必须是 0-120 之间的整数")
    gender:Gender
    status:AccountStatus
    preferences:Preferences
    favorite_tags:List[str] = Field(default_factory=list,description="内容偏好")

    @field_validator('username')
    @classmethod
    def username_validator(cls, v):
        if not v.isalnum():
            raise ValueError('用户名只能包含字母和数字，不能有空格或特殊字符')
        return v


def llm(prompt):
    test_switch = False

    if test_switch:
        print("模拟LLM收到prompt，正在生成...")
        if "错误的JSON" in prompt:
            return """
                {
                    "user_id": 180423,
                    "username": "zhou7",
                    "age": 17,
                    "gender": "Male",
                    "status": 1,
                    "preferences": {
                        "notifications": true,
                        "dark_theme": true,
                        "font_size": "medium"
                    },
                    "favorite_tags": ["photography", "study"]
                }
                """
        else:
            return """
                {
                    "user_id": "180423",  // 错误：user_id 是字符串
                    "username": "zhou qi", // 错误：含空格
                    "age": 150,            // 错误：超过 110
                    "gender": "Unknown",   // 错误：枚举值非法
                    "status": 3,           // 错误：枚举值非法
                    "preferences": {
                        "notifications": "yes", // 错误：布尔值写成字符串
                        "dark_theme": true,
                        "font_size": "huge"     // 错误：不在允许列表
                    },
                    "favorite_tags": "photography" // 错误：不是列表
                }
                """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role":"system","content":"你是一个专业的JSON数据生成器。请严格按照用户要求的JSON Schema输出，只返回纯JSON，不要包含任何Markdown标记或解释文字。"},
                {"role":"user","content":prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API调用失败:{e}")
        return ""

def refining_pipeline(initial_prompt,max_retries:int = 3,model_class:type=UserProfile):
    current_prompt = initial_prompt
    last_error_json = ""

    for attempt in range(1, max_retries + 1):
        print(f"\n===第 {attempt} 次尝试===")

        llm_output = llm(current_prompt)
        if not llm_output:
            print("LLM 无输出，重试...")
            continue

        print(f"LLM 输出:\n{llm_output[:200]}...")

        try:
            model = model_class.model_validate_json(llm_output)
            print(f"\n校验成功！共尝试 {attempt} 次。")
            return model

        except ValidationError as e:
            print(f"校验失败，错误详情:\n{e.json(indent=2)}")

            if attempt >= max_retries:
                print(f"\n达到最大重试次数 ({max_retries})，放弃修复。")
                return None

            last_error_json = llm_output
            error_message = e.json(indent=2)

            current_prompt = f"""
                你之前生成的JSON数据不符合要求，校验失败了。
                请根据以下错误信息修正JSON，只返回修正后的纯JSON，不要加任何解释。

                错误的JSON：
                {last_error_json}

                Pydantic校验错误信息：
                {error_message}

                要求的JSON Schema
                请严格按照以下字段和类型生成：
                user_id: 整数
                username: 字符串，只能包含字母数字
                age: 整数，0 <= age <= 110
                gender: 枚举，只能是 "Male"/"Female"/"Other"
                status: 枚举，只能是 0(INACTIVE)/1(ACTIVE)/2(SUSPENDED)
                preferences: 对象，包含 notifications(bool), dark_theme(bool), font_size(str, 只能是 'large'/'medium'/'small')
                favorite_tags: 字符串列表
                """
            print(f"准备第 {attempt + 1} 次重试，已将错误信息加入Prompt...")

    return None


if __name__ == '__main__':
    initial_prompt = """
    请生成一个符合以下要求的用户画像JSON数据：
    user_id: 随机整数
    username: 随机用户名
    age: 15-30 之间的整数
    gender: 随机性别
    status: 1 (ACTIVE)
    preferences: 随机偏好设置
    favorite_tags: 2-3 个随机标签
    只返回纯 JSON，不要包含任何其他文字。
    """

    print("启动DeepSeek自动修复Pipeline...")
    result = refining_pipeline(
        initial_prompt=initial_prompt,
        max_retries=3,
        model_class=UserProfile
        )

    if result:
        print("\n最终成功的结果:")
        print(result.model_dump_json(indent=2))
    else:
        print("\n修复失败，请检查Prompt或API配置。")


