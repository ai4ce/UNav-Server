from pydantic import BaseModel, EmailStr
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

class UserRegister(BaseModel):
    """
    Request schema for user registration.
    - email: user's email address (login name).
    - nickname: display name.
    - password: plaintext password.
    - code: email verification code.
    """
    email: EmailStr
    nickname: str
    password: str
    code: str

class UserLogin(BaseModel):
    """
    Request schema for user login.
    - email: user's email address (login name).
    - password: plaintext password.
    """
    email: EmailStr
    password: str

class UserInfo(BaseModel):
    """
    Response schema for user information.
    - id: unique user ID.
    - email: user's email address (login name).
    - nickname: display name.
    - avatar_url: URL of the user's avatar image.
    - created_at: registration timestamp (ISO format).
    """
    id: int
    email: EmailStr
    nickname: str
    avatar_url: Optional[str] = None
    created_at: str
