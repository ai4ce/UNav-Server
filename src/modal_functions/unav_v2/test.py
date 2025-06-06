import base64
import os

import modal


def main():
    try:
        UnavServer = modal.Cls.lookup("unav-server-v2", "UnavServer")
        unav_server = UnavServer()
        current_directory = os.getcwd()
        full_image_path = os.path.join(
            current_directory, "sample_image_7.jpg"
        )
        destination_id = "03727"
        floor = "3_floor"
        with open(full_image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode("utf-8")

        print("Testing get_destinations...")
        result = unav_server.get_destinations.remote()
        print("Result:", result)

        print(
            unav_server.unav_navigation.remote(
                destination_id=destination_id,
                base_64_image=base64_encoded,
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
