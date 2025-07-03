# api/task_api.py
# FastAPI router for generic authenticated task execution, supporting JSON and file uploads.
# Logs user, task function, and execution time for every call.

from fastapi import APIRouter, Request, Depends, UploadFile, File, HTTPException
from fastapi.security import OAuth2PasswordBearer
from config import DATA_ROOT
from db.db import save_query_image, log_navigation_record, init_log_db
from core.task_registry import get_task
from core.unav_state import get_session
from api.user_api import decode_access_token
import numpy as np
import cv2
import json
import logging
import time
from datetime import datetime
import traceback
import threading

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

init_log_db()

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

def async_log_navigation_record(*args, **kwargs):
    """Spawn a thread for log_navigation_record (for non-blocking logging)."""
    thread = threading.Thread(target=log_navigation_record, args=args, kwargs=kwargs)
    thread.daemon = True  # 不阻止主程序退出
    thread.start()
    
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

        if task_name == "unav_navigation":
            try:
                navigation_output = result
                status = "success" if navigation_output.get("success") else "failed"

                # Safe get with None fallback for all possible fields
                source_key = navigation_output.get("best_map_key")
                source_place_id = source_key[0] if (source_key and len(source_key) > 0) else None
                source_building_id = source_key[1] if (source_key and len(source_key) > 1) else None
                source_floor_id = source_key[2] if (source_key and len(source_key) > 2) else None

                # Always set timestamp for file and DB
                timestamp = datetime.utcnow()

                # Try to save image if present, else None
                if "image" in inputs:
                    rel_image_path = save_query_image(
                        image=inputs["image"],
                        data_root=DATA_ROOT,
                        user_id=user_id,
                        source_place_id=source_place_id or "unknown",
                        source_building_id=source_building_id or "unknown",
                        source_floor_id=source_floor_id or "unknown",
                        timestamp=timestamp
                    )
                else:
                    rel_image_path = None

                # Build path, cmds, etc for DB (may be None if failed)
                path = navigation_output.get("result") if status == "success" else None
                navigation_commands = navigation_output.get("cmds") if status == "success" else None
                session = get_session(user_id)
                
                async_log_navigation_record(
                    user_id=user_id,
                    query_image_path=rel_image_path,
                    floorplan_pose=navigation_output.get("floorplan_pose"),
                    navigation_commands=navigation_commands,
                    path=path,
                    source_place_id=source_place_id,
                    source_building_id=source_building_id,
                    source_floor_id=source_floor_id,
                    dest_place_id=session.get("target_place"),
                    dest_building_id=session.get("target_building"),
                    dest_floor_id=session.get("target_floor"),
                    destination_id=session.get("selected_dest_id"),
                    status=status,
                    extra_info={
                        "exec_time": elapsed,
                        "user_lang": inputs.get("language"),
                        "unit": inputs.get("unit"),
                        "error": navigation_output.get("error"),
                        "reason": navigation_output.get("reason"),
                        "stage": navigation_output.get("stage"),
                        "timings": navigation_output.get("timings"),
                        "top_candidates": navigation_output.get("top_candidates"),
                        # Optionally: add any other debug info here
                    },
                    timestamp=timestamp,
                )
            except Exception as log_e:
                logger.warning(
                    f"Failed to log navigation record for user {user_id}: {log_e}\n{traceback.format_exc()}"
                )
                
    return result
