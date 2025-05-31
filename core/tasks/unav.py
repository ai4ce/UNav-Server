# core/tasks/unav.py
# All UNav-specific tasks: destination, navigation, map, scale, etc.

from core.unav_state import localizer, nav, commander, user_sessions
from config import DATA_ROOT
import numpy as np
import cv2
import os
import base64
import math

def safe_serialize(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(v) for v in obj]
    elif isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()  # numpy对象转list
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)  # 兜底：强转字符串
    
def get_destinations(inputs):
    place = inputs["place"]
    building = inputs["building"]
    floor = inputs["floor"]
    user_id = inputs["user_id"]
    target_key = (place, building, floor)
    pf_target = nav.pf_map[target_key]
    destinations = [
        {"id": did, "label": pf_target.labels[did]}
        for did in pf_target.dest_ids
    ]
    session = user_sessions.setdefault(user_id, {})
    session["target_place"] = place
    session["target_building"] = building
    session["target_floor"] = floor
    return {"destinations": destinations}

def select_destination(inputs):
    user_id = inputs["user_id"]
    dest_id = inputs["dest_id"]
    session = user_sessions.setdefault(user_id, {})
    session["selected_dest_id"] = dest_id
    return {"success": True}

def get_floorplan(inputs):
    user_id = inputs["user_id"]
    session = user_sessions.get(user_id)
    if not session or "target_place" not in session or "target_building" not in session or "target_floor" not in session:
        return {"error": "No floor context set for this user."}
    place, bld, flr = session["target_place"], session["target_building"], session["target_floor"]
    bg_path = os.path.join(DATA_ROOT, place, bld, flr, "floorplan.png")
    if not os.path.exists(bg_path):
        return {"error": "Floorplan not found."}
    img = cv2.imread(bg_path)
    _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_bytes = img_encoded.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return {"floorplan": img_b64}

def get_scale(inputs):
    user_id = inputs["user_id"]
    session = user_sessions.get(user_id)
    if not session or "target_place" not in session or "target_building" not in session or "target_floor" not in session:
        return {"error": "No floor context set for this user."}
    source_key = (session["target_place"], session["target_building"], session["target_floor"])
    scale = nav.scales.get(source_key, 1.0)
    return {"scale": scale}

def unav_navigation(inputs):
    user_id = inputs["user_id"]
    session = user_sessions.setdefault(user_id, {})
    print("Session state:", session)  # Debugging line
    dest_id = session.get("selected_dest_id")
    target_place = session.get("target_place")
    target_building = session.get("target_building")
    target_floor = session.get("target_floor")
    unit = session.get("unit", "feet")
    if not all([dest_id, target_place, target_building, target_floor]):
        return {"error": "Incomplete navigation context. Please select a destination."}
    refinement_queue = session.get("refinement_queue") or {}
    query_img = inputs["image"]
    output = localizer.localize(query_img, refinement_queue)
    if output is None or "floorplan_pose" not in output:
        return {"error": "Localization failed, no pose found."}
    floorplan_pose = output["floorplan_pose"]
    start_xy, start_heading = floorplan_pose["xy"], floorplan_pose["ang"]
    source_key = output["best_map_key"]
    start_place, start_building, start_floor = source_key
    result = nav.find_path(
        start_place, start_building, start_floor, start_xy,
        target_place, target_building, target_floor, dest_id
    )

    cmds = commander(
        nav, result, initial_heading=start_heading, unit=unit
    )

    session["refinement_queue"] = output["refinement_queue"]
    
    return {
        "result": safe_serialize(result),
        "cmds": safe_serialize(cmds)
    }

UNAV_TASKS = {
    "get_destinations": get_destinations,
    "select_destination": select_destination,
    "get_floorplan": get_floorplan,
    "get_scale": get_scale,
    "unav_navigation": unav_navigation,
}
