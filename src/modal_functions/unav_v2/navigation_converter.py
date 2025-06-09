import json
from typing import Dict, List, Any, Optional


def convert_navigation_to_trajectory(
    navigation_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert navigation result format to trajectory output format.

    Args:
        navigation_result: Dictionary containing navigation result data

    Returns:
        Dictionary in trajectory output format
    """

    # Extract data from navigation result
    result = navigation_result.get("result", {})
    cmds = navigation_result.get("cmds", [])
    best_map_key = navigation_result.get("best_map_key", [])
    floorplan_pose = navigation_result.get("floorplan_pose", {})
    navigation_info = navigation_result.get("navigation_info", {})

    # Extract building, place, and floor information
    place = best_map_key[0] if len(best_map_key) > 0 else ""
    building = best_map_key[1] if len(best_map_key) > 1 else ""
    floor = best_map_key[2] if len(best_map_key) > 2 else ""

    # Get path coordinates from result
    path_coords = result.get("path_coords", [])

    # Add starting pose coordinates if available
    start_xy = floorplan_pose.get("xy", [])
    start_ang = floorplan_pose.get("ang", 0)

    # Create paths array - include start position with angle if available
    paths = []
    if start_xy and len(start_xy) >= 2:
        if start_ang:
            paths.append([start_xy[0], start_xy[1], start_ang])
        else:
            paths.append(start_xy)

    # Add all path coordinates
    for coord in path_coords:
        if len(coord) >= 2:
            paths.append(coord)

    # Calculate scale based on total cost and path distance (approximate)
    # This is an estimation - you may need to adjust based on your specific use case
    total_cost = result.get("total_cost", 0)
    scale = 0.02205862195  # Default scale, you might want to calculate this dynamically

    # Create trajectory structure
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
            None,  # This seems to be a placeholder in the original format
        ],
        "scale": scale,
    }

    return trajectory_data


def convert_and_save_trajectory(
    navigation_result: Dict[str, Any], output_file: str = "converted_trajectory.json"
) -> None:
    """
    Convert navigation result to trajectory format and save to JSON file.

    Args:
        navigation_result: Dictionary containing navigation result data
        output_file: Output filename for the trajectory JSON
    """
    trajectory_data = convert_navigation_to_trajectory(navigation_result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

    print(f"Trajectory data saved to {output_file}")


def load_and_convert_navigation_result(
    input_file: str = "navigation_result.json",
    output_file: str = "converted_trajectory.json",
) -> Dict[str, Any]:
    """
    Load navigation result from JSON file, convert to trajectory format, and optionally save.

    Args:
        input_file: Input filename for navigation result JSON
        output_file: Output filename for trajectory JSON (optional)

    Returns:
        Dictionary in trajectory output format
    """
    with open(input_file, "r", encoding="utf-8") as f:
        navigation_result = json.load(f)

    trajectory_data = convert_navigation_to_trajectory(navigation_result)

    if output_file:
        convert_and_save_trajectory(navigation_result, output_file)

    return trajectory_data


# Example usage
if __name__ == "__main__":
    # Example navigation result data (you can replace this with actual data)
    example_navigation_result = {
        "status": "success",
        "result": {
            "path_coords": [
                [560.3735533503349, 399.9288343348855],
                [458.252427184466, 292.53398058252435],
                [456.1149425287356, 186.19540229885058],
            ],
            "path_labels": ["(start)", "waypoint", "s1"],
            "path_keys": [
                "VIRT",
                ["New_York_City", "LightHouse", "3_floor", 0],
                ["New_York_City", "LightHouse", "3_floor", 42],
            ],
            "path_descriptions": ["", "", "staircase"],
            "total_cost": 254.5571538183059,
        },
        "cmds": [
            "You are currently in lobby on 3_floor of LightHouse, New_York_City.",
            "Sharp left to 7 o'clock",
            "Forward 11 feet",
            "Slight right to 1 o'clock",
            "Forward 8 feet and go through a door in 2 feet",
            "s1 on 12 o'clock ahead",
        ],
        "best_map_key": ["New_York_City", "LightHouse", "3_floor"],
        "floorplan_pose": {
            "xy": [560.3735533503349, 399.9288343348855],
            "ang": 6.635352211384816,
        },
        "navigation_info": {
            "start_location": "New_York_City/LightHouse/3_floor",
            "destination": "New_York_City/LightHouse/3_floor",
            "dest_id": "42",
            "unit": "feet",
            "language": "en",
        },
    }

    # Convert and display
    trajectory_result = convert_navigation_to_trajectory(example_navigation_result)
    print("Converted trajectory format:")
    print(json.dumps(trajectory_result, indent=2))

    # Save to file
    convert_and_save_trajectory(
        example_navigation_result, "example_converted_trajectory.json"
    )
