import base64
import os
import cv2
import numpy as np

import modal


def main():
    # Common parameters
    BUILDING = "Jubilee"
    PLACE = "Mahidol_University"
    FLOOR = "fl2"
    DESTINATION_ID = "32"
    SESSION_ID = "test_session_id_2"
    IMAGE_PATH = "media/sample_image_20.jpg"

    try:
        UnavServer = modal.Cls.lookup("unav-server-v21", "UnavServer")
        unav_server = UnavServer()
        current_directory = os.getcwd()
        full_image_path = os.path.join(current_directory, IMAGE_PATH)
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

        print("\n" + "="*50)
        print("Testing localize_user...")
        print("="*50)
        localize_result = unav_server.localize_user.remote(
            session_id=SESSION_ID,
            base_64_image=base64_encoded,
            place=PLACE,
            building=BUILDING,
            floor=FLOOR,
        )
        print("Localization Result:", localize_result)

        print("\n" + "="*50)
        print("Testing planner (full navigation)...")
        print("="*50)
        planner_result = unav_server.planner.remote(
            destination_id=DESTINATION_ID,
            base_64_image=base64_encoded,
            session_id=SESSION_ID,
            building=BUILDING,
            floor=FLOOR,
            place=PLACE,
        )
        print("Planner Result:", planner_result)

        print("\n" + "="*50)
        print("Testing generate_nav_instructions_from_coordinates...")
        print("="*50)
        # Create a mock localization result with provided coordinates
        mock_localization_result = {
            "status": "success",
            "floorplan_pose": {
                "xy": [2022.320618102614, 439.39776200033907],
                "ang": 298.4154661831644
            },
            "best_map_key": [PLACE, BUILDING, FLOOR],
            "location_info": {
                "place": PLACE,
                "building": BUILDING,
                "floor": FLOOR,
                "position": [2022.320618102614, 439.39776200033907],
                "heading": 298.4154661831644
            }
        }
        
        nav_instructions_result = unav_server.generate_nav_instructions_from_coordinates.remote(
            session_id=SESSION_ID,
            localization_result=mock_localization_result,
            dest_id=int(DESTINATION_ID),
            target_place=PLACE,
            target_building=BUILDING,
            target_floor=FLOOR,
            unit="meter",
            language="en"
        )
        print("Navigation Instructions Result:", nav_instructions_result)
        if nav_instructions_result.get("status") == "success":
            print("\nGenerated Instructions:")
            for idx, instruction in enumerate(nav_instructions_result.get("instructions", []), 1):
                print(f"  {idx}. {instruction}")
    except Exception as e:
        print(f"Error during Modal class lookup or execution: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
