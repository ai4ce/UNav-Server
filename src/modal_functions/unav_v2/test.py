import base64
import os
import cv2
import numpy as np

import modal


def main():
    try:
        UnavServer = modal.Cls.lookup("unav-server-v21", "UnavServer")
        unav_server = UnavServer()
        current_directory = os.getcwd()
        full_image_path = os.path.join(current_directory, "sample_image_7.jpg")
        destination_id = "44"
        floor = "3_floor"
        with open(full_image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode("utf-8")

        # print("Testing get_destinations_list...")
        # result = unav_server.get_destinations_list.remote(
        #     floor="fl2",
        #     place="Mahidol_University",
        #     building="Jubilee",
        # )
        # print("Result:", result)

        print(
            unav_server.planner.remote(
                destination_id=destination_id,
                base_64_image=base64_encoded,  # Pass BGR numpy array instead of base64
                session_id="test_session_id_2",
                building="LightHouse",
                floor=floor,
                place="New_York_City",
            )
        )
    except Exception as e:
        print(f"Error during Modal class lookup or execution: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
