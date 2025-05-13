import base64
import os

import modal


def main():
    UnavServer = modal.Cls.lookup("unav-server", "UnavServer")
    unav_server = UnavServer()
    current_directory = os.getcwd()
    full_image_path = os.path.join(
        current_directory, "modal_functions/misc/sample_image_7.jpg"
    )
    destination_id = "03727"
    floor = "3_floor"
    with open(full_image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_encoded = base64.b64encode(image_data).decode("utf-8")

    print(unav_server.get_destinations_list.remote(floor=floor))

    print(
        unav_server.planner.remote(
            destination_id=destination_id,
            base_64_image=base64_encoded,
            session_id="test_session_id_2",
            building="LightHouse",
            floor=floor,
            place="New_York_City",
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
