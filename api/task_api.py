# api/task_api.py
# FastAPI router for generic authenticated task execution, supporting JSON and file uploads.
# Logs user, task function, and execution time for every call.

from fastapi import APIRouter, Request, Depends, UploadFile, File, HTTPException
from fastapi.security import OAuth2PasswordBearer
from core.task_registry import get_task
from api.user_api import decode_access_token
import numpy as np
import cv2
import json
import logging
import time

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Set up a dedicated logger for task execution
logger = logging.getLogger("unav.api")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[UNav-API] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    file: UploadFile = File(None),
):
    """
    Universal endpoint to execute any registered backend task.
    - Accepts both JSON and multipart/form-data (for file upload support)
    - Logs user, task name, function, and execution time

    Body (JSON):
        {
            "task": "task_name",
            "inputs": {...}
        }

    Args:
        request (Request): The incoming HTTP request object.
        token (str): Bearer token for authentication.
        file (UploadFile, optional): Uploaded image file for image-based tasks.

    Returns:
        dict: The result of the executed task, with optional execution time.

    Raises:
        HTTPException 404: If the requested task is not found.
        HTTPException 500: If the task raises an error during execution.
    """
    # Extract user ID from the JWT token
    user_id = get_user_id_from_token(token)
    username = None
    payload = decode_access_token(token)
    username = payload.get("sub", "-")
    
    # Parse input data from JSON or form-data
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        task_name = data.get("task")
        inputs = data.get("inputs", {})
    else:
        # Fallback for multipart/form-data (used for file uploads)
        form = await request.form()
        task_name = form.get("task")
        inputs_raw = form.get("inputs", "{}")
        try:
            inputs = json.loads(inputs_raw)
        except Exception:
            inputs = {}

    # Inject authenticated user ID into task inputs
    inputs["user_id"] = user_id

    # If file is provided (e.g. image), decode it to an OpenCV ndarray
    if file is not None:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        query_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        inputs["image"] = query_img

    # Retrieve the registered task function by name
    task = get_task(task_name)
    if not task:
        logger.info(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] User: id={user_id}, username={username} | "
            f"Path: {request.url.path} | Method: {request.method} | Task: {task_name} | Status: 404 | Time: 0.000s"
        )
        raise HTTPException(status_code=404, detail=f"No such task: {task_name}")

    # Execute the task function and record execution time
    try:
        start_time = time.time()
        result = task(inputs)
        elapsed = time.time() - start_time
        logger.info(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] User: id={user_id}, username={username} | "
            f"Path: {request.url.path} | Method: {request.method} | "
            f"Task: {task_name} ({task.__module__}.{task.__name__}) | "
            f"Status: 200 | Time: {elapsed:.3f}s"
        )
    except Exception as e:
        logger.info(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] User: id={user_id}, username={username} | "
            f"Path: {request.url.path} | Method: {request.method} | "
            f"Task: {task_name} ({task.__module__}.{task.__name__}) | "
            f"Status: 500 | Time: 0.000s | Error: {e}"
        )
        raise HTTPException(status_code=500, detail=f"Task error: {str(e)}")

    if isinstance(result, dict):
        result["_exec_time"] = elapsed

    return result
