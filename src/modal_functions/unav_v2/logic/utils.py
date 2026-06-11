from typing import Any, Dict, Optional


def run_safe_serialize(obj: Any) -> Any:
    """Helper function to safely serialize objects for JSON response"""
    import numpy as np

    def convert_obj(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, dict):
            return {k: convert_obj(v) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            return [convert_obj(item) for item in o]
        else:
            return o

    return convert_obj(obj)


def run_construct_mock_localization_output(
    x: float,
    y: float,
    angle: float,
    place: str,
    building: str,
    floor: str,
) -> dict:
    """
    Construct a mock localization output from user-provided coordinates.
    This allows skipping the actual localization phase when coordinates are known.
    """
    return {
        "floorplan_pose": {
            "xy": [x, y],
            "ang": angle
        },
        "best_map_key": (place, building, floor),
        "refinement_queue": {}
    }


# Legacy constant (the LightHouse 3_floor scale) kept only as a last-resort
# fallback so behavior is unchanged when no per-floor scale is available.
_LEGACY_FALLBACK_SCALE = 0.02205862195


def run_convert_navigation_to_trajectory(
    navigation_result: Dict[str, Any],
    scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convert navigation result format to trajectory output format.

    Args:
        navigation_result: Planner output dict.
        scale: Per-floor meters-per-pixel value (from the navigator's
            scale.json). Falls back to the legacy constant when missing
            or invalid.
    """
    result = navigation_result.get("result", {})
    cmds = navigation_result.get("cmds", [])
    best_map_key = navigation_result.get("best_map_key", [])
    floorplan_pose = navigation_result.get("floorplan_pose", {})

    place = best_map_key[0] if len(best_map_key) > 0 else ""
    building = best_map_key[1] if len(best_map_key) > 1 else ""
    floor = best_map_key[2] if len(best_map_key) > 2 else ""

    path_coords = result.get("path_coords", [])

    start_xy = floorplan_pose.get("xy", [])
    start_ang = floorplan_pose.get("ang")

    paths = []
    if start_xy and len(start_xy) >= 2:
        # `is not None` so a legitimate 0-degree heading is not dropped.
        if start_ang is not None:
            paths.append([start_xy[0], start_xy[1], start_ang])
        else:
            paths.append(start_xy)

    for coord in path_coords:
        if len(coord) >= 2:
            paths.append(coord)

    # The navigator defaults floors missing from scale.json to 1.0, and
    # historic scale.json files contain literal 0 placeholders; downstream
    # clients treat scale == 1.0 as "unscaled", so reject both and fall back.
    if not scale or scale <= 0 or scale == 1.0:
        scale = _LEGACY_FALLBACK_SCALE

    trajectory_data = {
        "trajectory": [
            {
                "0": {
                    "name": "destination",
                    "building": building,
                    "floor": floor,
                    "paths": paths,
                    "command": {
                        "instructions": cmds,
                        "are_instructions_generated": len(cmds) > 0,
                    },
                    "scale": scale,
                }
            },
            None,
        ],
        "scale": scale,
        "route_segments": navigation_result.get("route_segments", []),
    }

    return trajectory_data


def run_set_navigation_context(
    server: Any,
    user_id: str,
    dest_id: str,
    target_place: str,
    target_building: str,
    target_floor: str,
    unit: str = "meter",
    language: str = "en",
) -> Dict[str, Any]:
    """
    Set navigation context for a user session.
    """
    try:
        server.update_session(
            user_id,
            {
                "selected_dest_id": dest_id,
                "target_place": target_place,
                "target_building": target_building,
                "target_floor": target_floor,
                "unit": unit,
                "language": language,
            },
        )

        return {
            "status": "success",
            "message": "Navigation context set successfully",
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "type": type(e).__name__}
