from pydantic import BaseModel

class TaskRequest(BaseModel):
    data: str  # base64 image/video, or other
    task: str
    user_id: str = None

class TaskResult(BaseModel):
    result: dict
    error: str = None

class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserInfo(BaseModel):
    id: int
    username: str
    created_at: str