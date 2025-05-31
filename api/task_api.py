# api/task_api.py
# FastAPI router for handling generic task execution with authentication and file support.

from fastapi import APIRouter, Request, Depends, UploadFile, File, HTTPException
from fastapi.security import OAuth2PasswordBearer
from core.task_registry import get_task
from api.user_api import decode_access_token
import numpy as np
import cv2
import json

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_user_id_from_token(token: str = Depends(oauth2_scheme)) -> str:
    """
    Decode JWT token to extract user ID.

    Args:
        token (str): JWT bearer token from the Authorization header.

    Returns:
        str: User ID extracted from the token payload.
    """
    payload = decode_access_token(token)
    return str(payload["id"])

@router.post("/run_task")
async def run_task(
    request: Request,
    token: str = Depends(oauth2_scheme),
    file: UploadFile = File(None),  # Optional file for image-based tasks
):
    """
    Universal task endpoint to invoke any registered backend task by name.
    Accepts JSON body or form-data (for file upload support).

    Body format (JSON):
        {
            "task": "task_name",
            "inputs": {...}
        }

    For image tasks, the image file should be sent as 'file' form-data field.

    Args:
        request (Request): Incoming HTTP request.
        token (str): OAuth2 bearer token for authentication.
        file (UploadFile, optional): Uploaded image file.

    Returns:
        dict: Result returned by the executed task.

    Raises:
        HTTPException 404: If the requested task is not found.
        HTTPException 500: If the task raises an error during execution.
    """
    user_id = get_user_id_from_token(token)

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        task_name = data.get("task")
        inputs = data.get("inputs", {})
    else:
        # Fallback for multipart/form-data (with file upload)
        form = await request.form()
        task_name = form.get("task")
        inputs_raw = form.get("inputs", "{}")
        try:
            inputs = json.loads(inputs_raw)
        except Exception:
            inputs = {}

    # Automatically inject authenticated user_id into inputs
    inputs["user_id"] = user_id

    # If file is provided (image), decode to OpenCV BGR ndarray
    if file is not None:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        query_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        inputs["image"] = query_img

    # Retrieve the registered task function by name
    task = get_task(task_name)
    if not task:
        raise HTTPException(status_code=404, detail=f"No such task: {task_name}")

    try:
        result = task(inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task error: {str(e)}")

    return result
