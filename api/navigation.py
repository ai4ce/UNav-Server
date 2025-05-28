from fastapi import APIRouter, Form
from typing import Dict, Any
import math
from server_state import nav, user_sessions

router = APIRouter()

@router.post("/navigate")
def navigate(
    user_id: str = Form(...),
    dest_place: str = Form(...),
    dest_building: str = Form(...),
    dest_floor: str = Form(...),
    dest_id: str = Form(...),
    unit: str = Form("feet")
) -> Dict[str, Any]:
    """
    Plan the navigation path and generate step-by-step instructions
    based on the user's last localization result.
    """
    # 1. Retrieve the user's most recent localization session from session storage
    session = user_sessions.get(user_id, None)
    if session is None or "floorplan_pose" not in session or "best_map_key" not in session:
        return {"error": "No localization data found for this user. Please perform localization first."}

    # 2. Extract starting information from the last localization result
    best_map_key = session["best_map_key"]
    floorplan_pose = session["floorplan_pose"]
    # Typically, best_map_key contains (place, building, floor, ...); unpack as needed
    start_place, start_building, start_floor = best_map_key.split("__")
    start_xy = floorplan_pose["xy"]
    theta = floorplan_pose["ang"]

    # 3. Call the navigation module to compute the shortest path and navigation results
    result = nav.find_path(
        start_place, start_building, start_floor, start_xy,
        dest_place, dest_building, dest_floor, dest_id
    )

    # 4. Generate human-readable navigation commands for the client
    from unav.navigator.commander import commands_from_result
    cmds = commands_from_result(
        nav,
        result,
        initial_heading=-math.degrees(theta),
        unit=unit
    )

    # 5. Return both the raw navigation result and the generated step commands
    return {
        "result": result,
        "cmds": cmds
    }
