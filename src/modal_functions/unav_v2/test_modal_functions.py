import base64
import json
import os

import cv2
import modal
import numpy as np


def main():
    # Common parameters
    BUILDING = "Langone"
    PLACE = "New_York_University"
    FLOOR = "17_floor"
    DESTINATION_ID = "50"
    SESSION_ID = "test_session_id_2"
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), "media", "vinay_sample.jpeg")

    try:
        UnavServer = modal.Cls.from_name("Staging-Mast3r-unav-server", "UnavServer")
        unav_server = UnavServer()
        full_image_path = IMAGE_PATH
        with open(full_image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode("utf-8")

        print("Testing get_destinations_list...")
        result = unav_server.get_destinations_list.remote(
            floor=FLOOR,
            place=PLACE,
            building=BUILDING,
        )
        print("Result:", result)

        print("\n" + "=" * 50)
        print("Testing get_route_segments...")
        print("=" * 50)
        route_segments_result = unav_server.get_route_segments.remote(
            place=PLACE,
            building=BUILDING,
            floor=FLOOR,
        )
        print("Route Segments Result:", route_segments_result)

        print("\n" + "=" * 50)
        print("Testing localize_user...")
        print("=" * 50)
        localize_result = unav_server.localize_user.remote(
            session_id=SESSION_ID,
            base_64_image=base64_encoded,
            place=PLACE,
            building=BUILDING,
            floor=FLOOR,
        )
        print("Localization Result:", localize_result)

        print("\n" + "=" * 50)
        print("Testing planner (full navigation)...")
        print("=" * 50)
        planner_result = unav_server.planner.remote(
            destination_id=DESTINATION_ID,
            base_64_image=base64_encoded,
            session_id=SESSION_ID,
            building=BUILDING,
            floor=FLOOR,
            place=PLACE,
            enable_multifloor=False,
        )
        print("Planner Result:", planner_result)

        output_path = os.path.join(os.path.dirname(__file__), "planner_output.json")
        with open(output_path, "w") as f:
            json.dump(planner_result, f, indent=2)
        print(f"\nPlanner output saved to: {output_path}")

        # print("\n" + "="*50)
        # print("Testing planner with user-provided coordinates (skip localization)...")
        # print("="*50)
        # # Test planner with user-provided coordinates
        # # These coordinates match the localization result from the previous test
        # planner_with_coords_result = unav_server.planner.remote(
        #     session_id=SESSION_ID + "_coords",
        #     base_64_image=None,  # Optional when using provided coordinates
        #     destination_id=DESTINATION_ID,
        #     place=PLACE,
        #     building=BUILDING,
        #     floor=FLOOR,
        #     should_use_user_provided_coordinate=True,
        #     x=2022.320618102614,
        #     y=439.39776200033907,
        #     angle=298.4154661831644,
        #     unit="meter",
        #     language="en"
        # )
        # print("Planner with Coordinates Result:", planner_with_coords_result)

    except Exception as e:
        print(f"Error during Modal class lookup or execution: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
