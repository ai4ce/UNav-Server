from pydantic import BaseModel
from typing import Optional, Dict

class TaskRequest(BaseModel):
    """
    Request schema for submitting a task.
    - data: base64-encoded image, video, or other content.
    - task: task type or command.
    - user_id: (optional) user identifier.
    """
    data: str
    task: str
    user_id: Optional[str] = None

class TaskResult(BaseModel):
    """
    Response schema for returning task results.
    - result: output result as a dictionary.
    - error: (optional) error message, if any.
    """
    result: Dict
    error: Optional[str] = None

class UserCreate(BaseModel):
    """
    Request schema for user registration.
    - username: account name.
    - password: plaintext password.
    """
    username: str
    password: str

class UserLogin(BaseModel):
    """
    Request schema for user login.
    - username: account name.
    - password: plaintext password.
    """
    username: str
    password: str

class UserInfo(BaseModel):
    """
    Response schema for user information.
    - id: unique user ID.
    - username: account name.
    - created_at: registration timestamp (ISO format).
    """
    id: int
    username: str
    created_at: str
