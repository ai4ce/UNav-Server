# core/tasks/unav.py
# UNav-specific task implementations:
# - destination query
# - destination selection
# - floorplan retrieval (current floor)
# - scale retrieval (current floor)
# - localization + navigation pipeline
# - turn-mode selection (NEW)

from core.unav_state import localizer, nav, commander, get_session, LABELS
from core.i18n_labels import get_label
from config import DATA_ROOT, PLACES

import numpy as np
import cv2
import os
import base64


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

def _user_lang(inputs) -> str:
    """Pick user language from session; fallback to 'en'."""
    uid = inputs.get("user_id") if isinstance(inputs, dict) else None
    if uid:
        try:
            sess = get_session(uid)
            return sess.get("language", "en")
        except Exception:
            pass
    return "en"

def get_places(inputs):
    """
    Returns available places as [{"id": <canonical_id>, "name": <localized>}, ...]
    """
    lang = _user_lang(inputs)
    # PLACES keys are canonical ids, use them for labels lookup
    rows = []
    for place in PLACES.keys():
        name = get_label("places", place, lang, fallback=place)
        rows.append({"id": place, "name": name})
    return {"places": rows}


def get_buildings(inputs):
    """
    Args:
        inputs = {"place": "<place_id>", "user_id": "..."}
    Returns:
        {"buildings": [{"id": ..., "name": ...}, ...]}
    """
    place = inputs["place"]
    lang = _user_lang(inputs)
    if place not in PLACES:
        return {"buildings": []}
    rows = []
    for bld in PLACES[place].keys():
        key = f"{place}/{bld}"
        name = get_label("buildings", key, lang, fallback=bld)
        rows.append({"id": bld, "name": name})
    return {"buildings": rows}


def get_floors(inputs):
    """
    Args:
        inputs = {"place": "<place_id>", "building": "<building_id>", "user_id": "..."}
    Returns:
        {"floors": [{"id": ..., "name": ...}, ...]}
    """
    place = inputs["place"]
    building = inputs["building"]
    lang = _user_lang(inputs)
    if place not in PLACES or building not in PLACES[place]:
        return {"floors": []}
    rows = []
    for flr in PLACES[place][building]:
        key = f"{place}/{building}/{flr}"
        name = get_label("floors", key, lang, fallback=flr)
        rows.append({"id": flr, "name": name})
    return {"floors": rows}


def get_destinations(inputs):
    """
    Retrieve all destination points for the specified floor.
    Adds localization based on labels.json.

    Args:
        inputs (dict): {
            "place": str,
            "building": str,
            "floor": str,
            "user_id": str,
        }
    Returns:
        dict: {"destinations": [{"id": str, "name": str}, ...]}
    """
    place = inputs["place"]
    building = inputs["building"]
    floor = inputs["floor"]
    user_id = inputs["user_id"]
    lang = _user_lang(inputs)

    target_key = (place, building, floor)
    pf_target = nav.pf_map[target_key]

    # pf_target.labels[did] is the fallback (English/id-like)
    rows = []
    for did in pf_target.dest_ids:
        did_str = str(did)
        key = f"{place}/{building}/{floor}/{did_str}"
        fallback = pf_target.labels[did]
        name = get_label("destinations", key, lang, fallback=fallback)
        rows.append({"id": did_str, "name": name})

    # Cache target floor context
    session = get_session(user_id)
    session["target_place"] = place
    session["target_building"] = building
    session["target_floor"] = floor

    return {"destinations": rows}


def select_destination(inputs):
    """
    Store the user's selected destination ID in the session.

    Args:
        inputs (dict): {
            "user_id": str,
            "dest_id": int | str,
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
    _, img_encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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


def select_turn_mode(inputs):
    """
    Persist user's preferred turn mode in session.

    Args:
        inputs (dict): {
            "user_id": str,
            "turn_mode": str   # 'default' | 'deg15'
        }

    Returns:
        dict: {"success": True, "turn_mode": str} or {"error": str}
    """
    user_id = inputs["user_id"]
    turn_mode = str(inputs.get("turn_mode", "")).strip().lower()
    allowed = {"default", "deg15"}
    if turn_mode not in allowed:
        return {"success": False, "error": "Invalid turn_mode. Allowed: 'default' or 'deg15'."}
    session = get_session(user_id)
    session["turn_mode"] = turn_mode
    return {"success": True, "turn_mode": turn_mode}


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
            "top_k": Optional[int],
            "turn_mode": Optional[str]  # 'default' | 'deg15'
        }

    Returns:
        dict: {
            "success": True,
            "result": dict (path info),
            "cmds": list(dict) (step-by-step commands w/ tags & meta),
            "best_map_key": tuple(str, str, str) (current floor),
            "floorplan_pose": dict (current pose),
            "turn_mode": str           # echoed effective turn mode
        }
        or dict with "success": False, "error", "stage", ... on failure.
    """
    # -------- Session & context --------
    user_id = inputs["user_id"]
    session = get_session(user_id)

    # Read navigation context
    dest_id = session.get("selected_dest_id")
    target_place = session.get("target_place")
    target_building = session.get("target_building")
    target_floor = session.get("target_floor")
    unit = session.get("unit", "feet")
    user_lang = session.get("language", "en")

    turn_mode = session.get("turn_mode", "default")
    if turn_mode not in {"default", "deg15"}:
        turn_mode = "default"
        session["turn_mode"] = turn_mode

    if any(x is None for x in [dest_id, target_place, target_building, target_floor]):
        return {
            "success": False,
            "error": "Incomplete navigation context. Please select a destination.",
            "stage": "context_check"
        }

    # -------- Input image & queue --------
    query_img = inputs["image"]
    key = query_img.shape[:2]
    top_k = inputs.get("top_k", None)

    refinement_queue = {}
    if (
        "refinement_queue" in session
        and session["refinement_queue"] is not None
        and key in session["refinement_queue"]
    ):
        refinement_queue = session["refinement_queue"][key]

    # -------- Localization --------
    output = localizer.localize(query_img, refinement_queue, top_k=top_k)

    def format_localization_error(output):
        """Format a standardized error dictionary from localization output."""
        if output is None:
            return {
                "success": False,
                "error": "Localization failed: No output returned from pipeline.",
                "stage": "pipeline"
            }
        return {
            "success": False,
            "error": output.get("reason", "Localization failed for unknown reasons."),
            "stage": output.get("stage", "unknown"),
            "timings": output.get("timings"),
            "top_candidates": output.get("top_candidates"),
            "results": output.get("results"),
            "best_map_key": output.get("best_map_key"),
        }

    if output is None or not output.get("success", False):
        return format_localization_error(output)

    # -------- Pose & session update --------
    floorplan_pose = output["floorplan_pose"]
    start_xy, start_heading = floorplan_pose["xy"], -floorplan_pose["ang"]
    source_key = output["best_map_key"]
    start_place, start_building, start_floor = source_key

    session["current_place"] = start_place
    session["current_building"] = start_building
    session["current_floor"] = start_floor
    session["floorplan_pose"] = floorplan_pose

    # -------- Path planning --------
    result = nav.find_path(
        start_place, start_building, start_floor, start_xy,
        target_place, target_building, target_floor, dest_id
    )

    # -------- Command generation --------
    cmds = commander(
        nav,
        result,
        initial_heading=start_heading,
        unit=unit,
        language=user_lang,
        turn_mode=turn_mode,
        labels=LABELS,
        data_final_root=DATA_ROOT
    )

    # -------- Persist refinement queue --------
    if "refinement_queue" not in session or session["refinement_queue"] is None:
        session["refinement_queue"] = {}
    session["refinement_queue"][key] = output["refinement_queue"]

    # -------- Response --------
    return {
        "success": True,
        "result": safe_serialize(result),
        "cmds": safe_serialize(cmds),
        "best_map_key": safe_serialize(source_key),
        "floorplan_pose": safe_serialize(floorplan_pose),
        "turn_mode": turn_mode,
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
    "select_turn_mode": select_turn_mode,
    "unav_navigation": unav_navigation,
}
