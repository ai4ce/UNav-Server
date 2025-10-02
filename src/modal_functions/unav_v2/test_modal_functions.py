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

        print(
            unav_server.planner.remote(
                destination_id=DESTINATION_ID,
                base_64_image=base64_encoded,
                session_id=SESSION_ID,
                building=BUILDING,
                floor=FLOOR,
                place=PLACE,
            )
        )
    except Exception as e:
        print(f"Error during Modal class lookup or execution: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
