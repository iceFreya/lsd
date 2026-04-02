import json
from typing import List,Optional
from enum import Enum

from pydantic import BaseModel,Field,ValidationError,field_validator

class Gender(str,Enum):
    MALE = 'Male'
    FEMALE = 'Female'
    OTHER = 'Other'

class AccountStatus(int,Enum):
    INACTIVE = 0
    ACTIVE = 1
    SUSPENDED = 2

class Preferences(BaseModel):
    notifications: bool
    dark_theme: bool
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
            raise ValueError('用户名只能包含字母和数字')
        return v

def llm_output(json_str) -> Optional[UserProfile]:
    try:
        model = UserProfile.model_validate_json(json_str)
        print("验证成功")
        return model
    except ValidationError as e:
        print("验证失败，错误详情:")
        print(e.json(indent=2))
        return None

if __name__ == '__main__':

    print("---测试1：正确数据---")
    data_one = """
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
    llm_output(data_one)

    print("\n--- 测试 2: 类型错误 (age 是字符串) ---")
    data_two = """
    {
        "user_id": 102471,
        "username": "zhang3",
        "age": "twenty five", 
        "gender": "Female",
        "status": 1,
        "preferences": {
            "notifications": true,
            "dark_theme": false,
            "font_size": "medium"
            },
        "favorite_tags": ["design"]
        
    }
    """
    llm_output(data_two)

    print("\n--- 测试 3: 枚举值错误 ---")
    data_tree = """
        {
            "user_id": 213495,
            "username": "wang5",
            "age": 30,
            "gender": "not_specified", 
            "status": 5,  
            "preferences": {
                "notifications": false, 
                "dark_theme": false,
                "font_size": "small"
                },
            "favorite_tags": ["read", "manager"]
            
        }
        """
    llm_output(data_tree)
