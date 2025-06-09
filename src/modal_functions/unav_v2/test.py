import base64
import os
import cv2
import numpy as np

import modal


def main():
    try:
        UnavServer = modal.Cls.lookup("unav-server-v2", "UnavServer")
        unav_server = UnavServer()
        current_directory = os.getcwd()
        full_image_path = os.path.join(
            current_directory, "sample_image_7.jpg"
        )
        destination_id = "42"
        floor = "3_floor"
        
        # Load image as BGR numpy array using OpenCV
        image_bgr = cv2.imread(full_image_path)
        
        if image_bgr is None:
            raise ValueError(f"Could not load image from {full_image_path}")

        print("Testing get_destinations_list...")
        result = unav_server.get_destinations_list.remote()
        print("Result:", result)

        print(
            unav_server.planner.remote(
                destination_id=destination_id,
                image=image_bgr,  # Pass BGR numpy array instead of base64
                session_id="test_session_id_2",
                building="LightHouse",
                floor=floor,
                place="New_York_City",
            )
        )
    except Exception as e:
        print(f"Error during Modal class lookup or execution: {e}")
        print("This might be because:")
        print("1. The Modal app 'unav-server-v2' is not deployed")
        print("2. There are import errors in the Modal environment")
        print("3. The required packages are not installed in the Modal environment")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
