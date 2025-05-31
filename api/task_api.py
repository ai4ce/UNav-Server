# api/task_api.py

from fastapi import APIRouter, Request, Depends, UploadFile, File, HTTPException
from fastapi.security import OAuth2PasswordBearer
from core.task_registry import get_task
from api.user_api import decode_access_token
import numpy as np
import cv2

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_user_id_from_token(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    return str(payload["id"])

@router.post("/run_task")
async def run_task(
    request: Request,
    token: str = Depends(oauth2_scheme),
    file: UploadFile = File(None),  # For image-based tasks
):
    user_id = get_user_id_from_token(token)
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        task_name = data.get("task")
        inputs = data.get("inputs", {})
    else:
        # Fallback to form (multipart/form-data, for files)
        form = await request.form()
        task_name = form.get("task")
        inputs_raw = form.get("inputs", "{}")
        try:
            inputs = json.loads(inputs_raw)
        except Exception:
            inputs = {}
    # 2. 自动补充 user_id
    inputs["user_id"] = user_id
    # 3. 图片任务
    if file is not None:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        query_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        inputs["image"] = query_img
    # 4. 查找并执行任务
    task = get_task(task_name)
    if not task:
        raise HTTPException(status_code=404, detail=f"No such task: {task_name}")
    try:
        result = task(inputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task error: {str(e)}")
    return result