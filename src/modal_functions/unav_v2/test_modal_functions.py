import base64
import json
import os
import time

import modal

# Common parameters
BUILDING = "Langone"
PLACE = "New_York_University"
FLOOR = "17_floor"
DESTINATION_ID = "50"
SESSION_ID = "test_session_id_2"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "media", "vinay_sample.jpeg")


def _load_base64_image() -> str:
    with open(IMAGE_PATH, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    base64_encoded = _load_base64_image()

    # --- Destinations (CPU-only class, no GPU scheduling) ---
    DestServer = modal.Cls.from_name("Mast3r-UNav-Server", "DestinationsServer")
    dest_server = DestServer()

    print("Testing get_destinations_list...")
    start = time.time()
    result = dest_server.get_destinations_list.remote(
        floor=FLOOR,
        place=PLACE,
        building=BUILDING,
    )
    print(f"⏱️ get_destinations_list took {time.time() - start:.2f}s")
    print("Result:", result)

    # --- Localization + planner (GPU class) ---
    UnavServer = modal.Cls.from_name("Mast3r-UNav-Server", "UnavServer")
    unav_server = UnavServer()

    print("\n" + "=" * 50)
    print("Testing localize_user...")
    print("=" * 50)
    start = time.time()
    localize_result = unav_server.localize_user.remote(
        session_id=SESSION_ID,
        base_64_image=base64_encoded,
        place=PLACE,
        building=BUILDING,
        floor=FLOOR,
    )
    print(f"⏱️ localize_user took {time.time() - start:.2f}s")
    print("Localization Result:", localize_result)

    print("\n" + "=" * 50)
    print("Testing planner (full navigation)...")
    print("=" * 50)
    start = time.time()
    planner_result = unav_server.planner.remote(
        destination_id=DESTINATION_ID,
        base_64_image=base64_encoded,
        session_id=SESSION_ID,
        building=BUILDING,
        floor=FLOOR,
        place=PLACE,
        enable_multifloor=False,
    )
    print(f"⏱️ planner took {time.time() - start:.2f}s")
    if isinstance(planner_result, dict):
        print(f"⏱️ Planner timing dict: {planner_result.get('timing')}")
        print(f"⏱️ Upstream per-stage timings: {planner_result.get('timings')}")
        print(
            f"📍 total_inliers={planner_result.get('total_inliers')}, "
            f"floorplan_pose={planner_result.get('floorplan_pose')}"
        )
        # Snap-to-route debugging
        snapped_pose = planner_result.get('snapped_pose')
        floorplan_pose = planner_result.get('floorplan_pose')
        force_walkable = planner_result.get('force_walkable')
        print(f"\n🔍 [SNAP-TO-ROUTE DEBUG]")
        print(f"  force_walkable={force_walkable}")
        print(f"  floorplan_pose={floorplan_pose}")
        print(f"  snapped_pose={snapped_pose}")
        if snapped_pose and floorplan_pose:
            raw_xy = floorplan_pose.get('xy')
            snapped_xy = snapped_pose.get('xy')
            if raw_xy and snapped_xy:
                diff = ((raw_xy[0]-snapped_xy[0])**2 + (raw_xy[1]-snapped_xy[1])**2)**0.5
                print(f"  snap_diff={diff:.2f}px (0.0 means snap did nothing)")
                if diff == 0:
                    print(f"  ⚠️ Snap-to-route had no effect — route_network may be None/empty!")
    print("Planner Result:", planner_result)

    output_path = os.path.join(os.path.dirname(__file__), "planner_output.json")
    with open(output_path, "w") as f:
        json.dump(planner_result, f, indent=2)
    print(f"\nPlanner output saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
