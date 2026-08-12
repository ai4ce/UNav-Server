import os
import time

import modal


def main():
    # Common parameters
    BUILDING = "Langone"
    PLACE = "New_York_University"
    FLOOR = "16_floor"
    DESTINATION_ID = "50"
    SESSION_ID = "test_session_id_2"

    try:
        UnavServer = modal.Cls.from_name("Staging-Mast3r-unav-server", "DestinationsServer")
        unav_server = UnavServer()

        print("Testing get_destinations_list...")
        start = time.time()
        result = unav_server.get_destinations_list.remote(
            floor=FLOOR,
            place=PLACE,
            building=BUILDING,
            enable_multifloor=False,
        )
        print(f"⏱️ get_destinations_list took {time.time() - start:.2f}s")
        print("Result:", result)

        # print("\n" + "=" * 50)
        # print("Testing get_route_segments...")
        # print("=" * 50)
        # start = time.time()
        # route_segments_result = unav_server.get_route_segments.remote(
        #     place=PLACE,
        #     building=BUILDING,
        #     floor=FLOOR,
        # )
        # print(f"⏱️ get_route_segments took {time.time() - start:.2f}s")
        # print("Route Segments Result:", route_segments_result)

        # print("\n" + "=" * 50)
        # print("Testing localize_user...")
        # print("=" * 50)
        # start = time.time()
        # localize_result = unav_server.localize_user.remote(
        #     session_id=SESSION_ID,
        #     base_64_image=base64_encoded,
        #     place=PLACE,
        #     building=BUILDING,
        #     floor=FLOOR,
        # )
        # print(f"⏱️ localize_user took {time.time() - start:.2f}s")
        # print("Localization Result:", localize_result)

        # print("\n" + "=" * 50)
        # print("Testing planner (full navigation)...")
        # print("=" * 50)
        # start = time.time()
        # planner_result = unav_server.planner.remote(
        #     destination_id=DESTINATION_ID,
        #     base_64_image=base64_encoded,
        #     session_id=SESSION_ID,
        #     building=BUILDING,
        #     floor=FLOOR,
        #     place=PLACE,
        #     enable_multifloor=False,
        # )
        # print(f"⏱️ planner took {time.time() - start:.2f}s")
        # print("Planner Result:", planner_result)
        # if isinstance(planner_result, dict):
        #     print(f"⏱️ Planner timing dict: {planner_result.get('timing')}")
        #     print(f"⏱️ Upstream per-stage timings: {planner_result.get('timings')}")
        #     print(
        #         f"📍 total_inliers={planner_result.get('total_inliers')}, "
        #         f"floorplan_pose={planner_result.get('floorplan_pose')}"
        #     )

        # output_path = os.path.join(os.path.dirname(__file__), "planner_output.json")
        # with open(output_path, "w") as f:
        #     json.dump(planner_result, f, indent=2)
        # print(f"\nPlanner output saved to: {output_path}")

    except Exception as e:
        print(f"Error during Modal class lookup or execution: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
