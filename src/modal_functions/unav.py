from modal import method, gpu, build, enter

from modal_config import app, unav_image, volume
from logger_utils import setup_logger


@app.cls(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
    gpu=gpu.Any(),
    enable_memory_snapshot=True,
    concurrency_limit=20,
    allow_concurrent_inputs=20,
    memory=32768,
)
class UnavServer:

    def find_use_true_feature(self, feature_dict):
        for key, value in feature_dict.items():
            if isinstance(value, dict) and value.get("use") is True:
                return key
        return None

    @build()
    @enter()
    def load_server(self):
        from server_manager import Server
        from modules.config.settings import load_config

        config = load_config("config.yaml")

        feature_global = config.get("feature", {}).get("global", {})
        feature = self.find_use_true_feature(feature_global)

        self.server = Server(logger=setup_logger(), config=config, feature=feature)

    @method()
    def get_destinations_list(self):

        response = self.server.get_destinations_list(
            building="LightHouse", floor="6_floor"
        )
        return response

    @method()
    def planner(
        self,
        session_id: str = "",
        destination_id: str = "",
        building: str = "",
        floor: str = "",
        place: str = "",
        base_64_image: str = None,
        get_floor_plan: bool = False,
    ):

        import json
        import time
        import base64
        import io
        from PIL import Image
        import numpy as np

        """
            Handle localization request by processing the provided image and returning the pose.
        """

        start_time = time.time()  # Start time for the entire function

        query_image_data = (
            base64.b64decode(base_64_image.split(",")[1])
            if "," in base_64_image
            else base64.b64decode(base_64_image)
        )
        query_image = Image.open(io.BytesIO(query_image_data)).convert("RGB")

        print("Query Image Converted from base64 to PIL Image")

        response = self.server.select_destination(
            session_id=session_id,
            place=place,
            building=building,
            floor=floor,
            destination_id=destination_id,
        )
        if response == None:
            print("Desintation Set to id: " + destination_id)
        else:
            print(response)

        # Measure time for handle_localization
        start_localization_time = time.time()
        image_np = np.array(query_image)
        pose = self.server.handle_localization(frame=image_np, session_id=session_id)
        end_localization_time = time.time()
        localization_time = end_localization_time - start_localization_time
        print(f"Localization Time: {localization_time:.2f} seconds")

        print("Pose: ", pose)

        # Measure time for handle_navigation
        start_navigation_time = time.time()
        trajectory = self.server.handle_navigation(session_id)
        end_navigation_time = time.time()
        navigation_time = end_navigation_time - start_navigation_time
        print(f"Navigation Time: {navigation_time:.2f} seconds")

        end_time = time.time()  # End time for the entire function
        elapsed_time = (
            end_time - start_time
        )  # Calculate elapsed time for the entire function

        print(
            f"Total Execution Time: {elapsed_time:.2f} seconds"
        )  # Print total elapsed time

        scale = self.server.config["location"]["scale"]

        if get_floor_plan:
            floorplan_base64 = pose["floorplan_base64"]
            return json.dumps(
                {
                    "trajectory": trajectory,
                    "scale": scale,
                    "floorplan_base64": floorplan_base64,
                }
            )  # return floor plan if requested

        return json.dumps({"trajectory": trajectory, "scale": scale})

    @method()
    def start_server(self):
        import json

        """
        Initializes and starts the serverless instance.
    
        This function helps in reducing the server response time for actual requests by pre-warming the server. 
        By starting the server in advance, it ensures that the server is ready to handle incoming requests immediately, 
        thus avoiding the latency associated with a cold start.
        """
        print("UNAV Container started...")

        response = {"status": "success", "message": "Server started."}
        return json.dumps(response)
