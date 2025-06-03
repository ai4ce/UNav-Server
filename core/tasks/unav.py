# core/tasks/unav.py
# UNav-specific task implementations:
# - destination query
# - destination selection
# - floorplan retrieval (current floor)
# - scale retrieval (current floor)
# - localization + navigation pipeline

from core.unav_state import localizer, nav, commander, get_session
from config import DATA_ROOT, PLACES
import numpy as np
import cv2
import os
import base64
import math


def safe_serialize(obj):
    """
    Recursively convert non-serializable objects (e.g., numpy arrays) into JSON-serializable formats.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(v) for v in obj]
    elif isinstance(obj, (np.generic, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        # Fallback: convert unknown objects to string
        return str(obj)

def get_places(inputs):
    """
    Returns all available places as [{"id": ..., "name": ...}, ...]
    """
    places = [{"id": place, "name": place} for place in PLACES.keys()]
    return {"places": places}

def get_buildings(inputs):
    """
    Args:
        inputs = {"place": "New_York_City"}
    Returns:
        {"buildings": [{"id": ..., "name": ...}, ...]}
    """
    place = inputs["place"]
    if place not in PLACES:
        return {"buildings": []}
    buildings = [{"id": bld, "name": bld} for bld in PLACES[place].keys()]
    return {"buildings": buildings}

def get_floors(inputs):
    """
    Args:
        inputs = {"place": "New_York_City", "building": "LightHouse"}
    Returns:
        {"floors": [{"id": ..., "name": ...}, ...]}
    """
    place = inputs["place"]
    building = inputs["building"]
    if place not in PLACES or building not in PLACES[place]:
        return {"floors": []}
    floors = [{"id": flr, "name": flr} for flr in PLACES[place][building]]
    return {"floors": floors}

def get_destinations(inputs):
    """
    Retrieve all destination points for the specified floor.

    Args:
        inputs (dict): {
            "place": str,
            "building": str,
            "floor": str,
            "user_id": str,
        }

    Returns:
        dict: {"destinations": [{"id": int, "name": str}, ...]}
    """
    place = inputs["place"]
    building = inputs["building"]
    floor = inputs["floor"]
    user_id = inputs["user_id"]
    target_key = (place, building, floor)
    pf_target = nav.pf_map[target_key]

    destinations = [{"id": str(did), "name": pf_target.labels[did]} for did in pf_target.dest_ids]

    # Cache the user's selected target floor context
    session = get_session(user_id)
    session["target_place"] = place
    session["target_building"] = building
    session["target_floor"] = floor
    
    return {"destinations": destinations}


def select_destination(inputs):
    """
    Store the user's selected destination ID in the session.

    Args:
        inputs (dict): {
            "user_id": str,
            "dest_id": int,
        }

    Returns:
        dict: {"success": True}
    """
    user_id = inputs["user_id"]
    dest_id = inputs["dest_id"]
    session = get_session(user_id)
    session["selected_dest_id"] = int(dest_id)
    return {"success": True}


def get_floorplan(inputs):
    """
    Retrieve the floorplan image of the user's current location floor,
    encoded as base64 JPEG string.

    Args:
        inputs (dict): {
            "user_id": str
        }

    Returns:
        dict: {"floorplan": base64-encoded JPEG string} or {"error": str}
    """
    user_id = inputs["user_id"]
    session = get_session(user_id)
    if not session or "current_place" not in session or "current_building" not in session or "current_floor" not in session:
        return {"error": "No current floor context set for this user."}

    place = session["current_place"]
    bld = session["current_building"]
    flr = session["current_floor"]

    floorplan_path = os.path.join(DATA_ROOT, place, bld, flr, "floorplan.png")
    if not os.path.exists(floorplan_path):
        return {"error": "Floorplan not found."}

    img = cv2.imread(floorplan_path)
    _, img_encoded = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_bytes = img_encoded.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return {"floorplan": img_b64}


def get_scale(inputs):
    """
    Retrieve the scale (meters or feet per pixel) for the user's current floor.

    Args:
        inputs (dict): {
            "user_id": str
        }

    Returns:
        dict: {"scale": float} or {"error": str}
    """
    user_id = inputs["user_id"]
    session = get_session(user_id)
    if not session or "current_place" not in session or "current_building" not in session or "current_floor" not in session:
        return {"error": "No current floor context set for this user."}

    source_key = (session["current_place"], session["current_building"], session["current_floor"])
    scale = nav.scales.get(source_key, 1.0)
    return {"scale": scale}


def unav_navigation(inputs):
    """
    Full localization and navigation pipeline.
    - Performs localization from query image.
    - Updates user's current position and floor context.
    - Plans path to user-selected destination.
    - Generates human-readable navigation commands.

    Args:
        inputs (dict): {
            "user_id": str,
            "image": np.ndarray (BGR image),
            "top_k": Optional[int]
        }

    Returns:
        dict: {
            "result": dict (path info),
            "cmds": list(str) (step-by-step instructions),
            "best_map_key": tuple(str, str, str) (current floor),
            "floorplan_pose": dict (current pose)
        }
        or dict with "error" key on failure.
    """
    user_id = inputs["user_id"]
    session = get_session(user_id)

    # Check for required navigation context
    dest_id = session.get("selected_dest_id")
    target_place = session.get("target_place")
    target_building = session.get("target_building")
    target_floor = session.get("target_floor")
    unit = session.get("unit", "feet")
    user_lang = session.get("language", "en")

    if not all([dest_id, target_place, target_building, target_floor]):
        return {"error": "Incomplete navigation context. Please select a destination."}

    refinement_queue = session.get("refinement_queue") or {}
    query_img = inputs["image"]
    top_k = inputs.get("top_k", None)

    # Perform localization
    output = localizer.localize(query_img, refinement_queue, top_k=top_k)
    if output is None or "floorplan_pose" not in output:
        return {"error": "Localization failed, no pose found."}

    floorplan_pose = output["floorplan_pose"]
    start_xy, start_heading = floorplan_pose["xy"], floorplan_pose["ang"]
    source_key = output["best_map_key"]
    start_place, start_building, start_floor = source_key

    # Update user's current floor context and pose for real-time tracking
    session["current_place"] = start_place
    session["current_building"] = start_building
    session["current_floor"] = start_floor
    session["floorplan_pose"] = floorplan_pose

    # Plan navigation path to destination
    result = nav.find_path(
        start_place, start_building, start_floor, start_xy,
        target_place, target_building, target_floor, dest_id
    )

    # Generate spoken/navigation commands
    cmds = commander(
        nav, result, initial_heading=start_heading, unit=unit, language=user_lang
    )

    # Update refinement queue for subsequent localization calls
    session["refinement_queue"] = output["refinement_queue"]

    # Return all relevant info safely serialized for JSON
    return {
        "result": safe_serialize(result),
        "cmds": safe_serialize(cmds),
        "best_map_key": safe_serialize(source_key),
        "floorplan_pose": safe_serialize(floorplan_pose),
    }


# Register all UNav-related tasks in a dictionary
UNAV_TASKS = {
    "get_places": get_places,
    "get_buildings": get_buildings,
    "get_floors": get_floors,
    "get_destinations": get_destinations,
    "select_destination": select_destination,
    "get_floorplan": get_floorplan,
    "get_scale": get_scale,
    "unav_navigation": unav_navigation,
}
