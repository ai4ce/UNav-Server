"""Server-side task: localize_only.
Lives at /home/unav/Desktop/UNav_socket/core/tasks/debug/localize_only.py inside
the UNav container. Exposes a debug-only endpoint that runs JUST
localizer.localize() with no navigation/session side effects.

Registered via task_registry: DEBUG_TASKS = {"localize_only": localize_only, ...}
"""
from core.unav_state import localizer
import numpy as np


def _to_list(x):
    if x is None: return None
    if hasattr(x, "tolist"): return x.tolist()
    if isinstance(x, (list, tuple)): return [_to_list(v) for v in x]
    if isinstance(x, dict): return {k: _to_list(v) for k, v in x.items()}
    if isinstance(x, (float, int, str, bool)): return x
    return str(x)


def localize_only(inputs):
    """Minimal localize task. No session, no navigation, no destination.
    Input:
        image: np.ndarray (BGR, from task_api cv2.imdecode)
        top_k: int (optional, default 10)
    Returns:
        success, qvec, tvec, floorplan_pose, top_candidates, best_map_key,
        n_frames, timings, stage, reason
    All numpy → list so FastAPI can JSON-serialize.
    """
    img = inputs.get("image")
    if img is None or not isinstance(img, np.ndarray):
        return {"success": False, "stage": "input", "reason": "missing or invalid 'image'"}
    top_k = int(inputs.get("top_k", 10))
    resp = localizer.localize(img, {}, top_k=top_k)
    return {
        "success": bool(resp.get("success", False)),
        "stage": resp.get("stage"),
        "reason": resp.get("reason"),
        "qvec": _to_list(resp.get("qvec")),
        "tvec": _to_list(resp.get("tvec")),
        "floorplan_pose": _to_list(resp.get("floorplan_pose")),
        "best_map_key": _to_list(resp.get("best_map_key")),
        "top_candidates": _to_list(resp.get("top_candidates")),
        "n_frames": resp.get("n_frames"),
        "timings": _to_list(resp.get("timings")),
    }


DEBUG_TASKS = {
    "localize_only": localize_only,
}
