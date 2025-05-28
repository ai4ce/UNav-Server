from fastapi import APIRouter, UploadFile, File, Form
from typing import Dict, Any
import numpy as np
import cv2

from server_state import localizer, user_sessions

router = APIRouter()

@router.post("/localize")
async def localize(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    top_k: int = Form(None)
) -> Dict[str, Any]:
    """
    Endpoint for user localization.
    Handles user image upload, calls localization algorithm,
    and maintains a per-user refinement_queue in memory.
    """
    # 1. Retrieve the user's current refinement_queue from session storage
    refinement_queue = user_sessions.get(user_id, None)
    
    # 2. Decode the uploaded image into a NumPy array (OpenCV format)
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    query_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 3. Call the localization module with the image and the current refinement_queue
    output = localizer.localize(query_img, refinement_queue, top_k=top_k)
    
    # 4. Update the user's refinement_queue in session storage with the latest state
    user_sessions[user_id] = output["refinement_queue"]
    
    # 5. Return structured results, including localization outcome, pose, and metadata
    return {
        "success": output["success"],
        "floorplan_pose": output["floorplan_pose"],  # {'xy': (x, y), 'ang': theta}
        "best_map_key": output["best_map_key"],
        "results": output["results"],
        "top_candidates": output["top_candidates"],
        "n_frames": output["n_frames"],
        # Optionally return additional fields from `output` if needed
    }
