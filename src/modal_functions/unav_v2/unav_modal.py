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
        place="NewYorkCity",
        building="LightHouse",
    ):
        try:
            import os
            import sys
            from config import DATA_ROOT, FEATURE_MODEL, LOCAL_FEATURE_MODEL, PLACES
            from unav.config import UNavConfig
            from unav.localizer.localizer import UNavLocalizer
            from unav.navigator.multifloor import FacilityNavigator
            from unav.navigator.commander import commands_from_result

            place_names = list(PLACES.keys())
            building_names = [b for p in PLACES.values() for b in p.keys()]
            floor_names = [f for p in PLACES.values() for b in p.values() for f in b]

            # Initialize UNav configuration with all necessary parameters
            config = UNavConfig(
                data_final_root=DATA_ROOT,
                places=PLACES,  # Keep hierarchical structure for flexible access
                global_descriptor_model=FEATURE_MODEL,
                local_feature_model=LOCAL_FEATURE_MODEL,
            )

            # Extract specific sub-configs for localization and navigation modules
            localizor_config = config.localizer_config
            navigator_config = config.navigator_config

            # Initialize global singletons for UNav algorithm modules
            places = PLACES  # Global place/building/floor info

            localizer = UNavLocalizer(localizor_config)
            localizer.load_maps_and_features()  # Preload all maps and features for fast localization

            nav = FacilityNavigator(
                navigator_config
            )  # Initialize multi-floor navigator instance

            commander = commands_from_result  # Navigation command generator function

        except Exception as e:
            import subprocess
            import traceback

            print(f"❌ Error during initialization: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            print("❌ Full traceback:")
            traceback.print_exc()

            # Check what packages are actually installed
            print("\n📦 Checking installed packages with pip freeze:")
            try:
                result = subprocess.run(
                    ["pip", "freeze"], capture_output=True, text=True
                )
                print("Installed packages:")
                for line in result.stdout.split("\n"):
                    if "unav" in line.lower() or "UNav" in line:
                        print(f"  🎯 {line}")
                    elif line.strip():
                        print(f"  {line}")
            except Exception as pip_error:
                print(f"❌ Failed to run pip freeze: {pip_error}")

            # Check sys.path
            import sys

            print(f"\n📁 Python path (sys.path):")
            for path in sys.path:
                print(f"  {path}")

            return {"status": "error", "message": str(e), "type": type(e).__name__}
