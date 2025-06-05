from modal import method, gpu, build, enter

from modal_config import app, unav_image, volume


@app.cls(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
    gpu=gpu.Any(),
    enable_memory_snapshot=True,
    concurrency_limit=20,
    allow_concurrent_inputs=20,
    memory=16152,
    container_idle_timeout=60,
)
class UnavServer:
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

    @method()
    def get_destinations(
        self,
        floor="6_floor",
        place="New_York_City",
        building="LightHouse",
    ):
        try:
            from unav.config import UNavConfig
            from unav.localizer.localizer import UNavLocalizer
            from unav.navigator.multifloor import FacilityNavigator

            DATA_ROOT = (
                "/root/UNav-IO/data"  # Point to the data subdirectory in the volume
            )
            FEATURE_MODEL = "DinoV2Salad"
            LOCAL_FEATURE_MODEL = "superpoint+lightglue"
            PLACES = {
                "New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"]}
            }

            # Initialize UNav configuration with all necessary parameters
            config = UNavConfig(
                data_final_root=DATA_ROOT,
                places=PLACES,  # Keep hierarchical structure for flexible access
                global_descriptor_model=FEATURE_MODEL,
                local_feature_model=LOCAL_FEATURE_MODEL,
            )

            print("✅ UNavConfig initialized successfully")

            # Extract specific sub-configs for localization and navigation modules
            localizor_config = config.localizer_config
            navigator_config = config.navigator_config

            print("✅ Config objects extracted successfully")

            # Initialize global singletons for UNav algorithm modules
            places = PLACES  # Global place/building/floor info

            print("🔍 Attempting to initialize UNavLocalizer...")
            localizer = UNavLocalizer(localizor_config)

            print("🔍 Attempting to load maps and features...")
            localizer.load_maps_and_features()  # Preload all maps and features for fast localization

            print("✅ UNavLocalizer initialized and loaded successfully")

            nav = FacilityNavigator(
                navigator_config
            )  # Initialize multi-floor navigator instance

            print("✅ All components initialized successfully")

            target_key = (place, building, floor)
            pf_target = nav.pf_map[target_key]

            destinations = [
                {"id": str(did), "name": pf_target.labels[did]}
                for did in pf_target.dest_ids
            ]

            return {
                "status": "success",
                "destinations": destinations,
            }

        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            return {"status": "error", "message": str(e), "type": type(e).__name__}
