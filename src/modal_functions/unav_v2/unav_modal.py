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

            # Debug: Show current working directory and file structure
            print("🔍 DEBUG: Current working directory and file structure:")
            cwd = os.getcwd()
            print(f"Current working directory: {cwd}")

            print("\n📁 Contents of current directory:")
            for item in os.listdir(cwd):
                item_path = os.path.join(cwd, item)
                if os.path.isdir(item_path):
                    print(f"  📂 {item}/")
                    # Show first level of subdirectories
                    try:
                        subitems = os.listdir(item_path)[:10]  # Limit to first 10 items
                        for subitem in subitems:
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isdir(subitem_path):
                                print(f"    📂 {subitem}/")
                            else:
                                print(f"    📄 {subitem}")
                        if len(os.listdir(item_path)) > 10:
                            print(
                                f"    ... and {len(os.listdir(item_path)) - 10} more items"
                            )
                    except PermissionError:
                        print(f"    [Permission denied]")
                else:
                    print(f"  📄 {item}")

            # Check for common data directories
            common_paths = [
                "/root",
                "/root/UNav-IO",
                "/root/app",
                "/data",
                "/parameters",
            ]
            print("\n🔍 Checking common data paths:")
            for path in common_paths:
                if os.path.exists(path):
                    print(f"  ✅ {path} exists")
                    try:
                        contents = os.listdir(path)[:5]  # First 5 items
                        for item in contents:
                            print(f"    📄 {item}")
                        if len(os.listdir(path)) > 5:
                            print(f"    ... and {len(os.listdir(path)) - 5} more items")
                    except PermissionError:
                        print(f"    [Permission denied]")
                else:
                    print(f"  ❌ {path} does not exist")

            from unav.config import UNavConfig
            from unav.localizer.localizer import UNavLocalizer
            from unav.navigator.multifloor import FacilityNavigator
            from unav.navigator.commander import commands_from_result

            DATA_ROOT = (
                "/root/UNav-IO/data"  # Point to the data subdirectory in the volume
            )
            FEATURE_MODEL = "DinoV2Salad"
            LOCAL_FEATURE_MODEL = "superpoint+lightglue"
            PLACES = {
                "New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"]}
            }

            # Debug: Check if data paths exist
            print(f"\n🔍 Checking DATA_ROOT path: {DATA_ROOT}")
            if os.path.exists(DATA_ROOT):
                print(f"  ✅ {DATA_ROOT} exists")
                try:
                    contents = os.listdir(DATA_ROOT)
                    print(f"  📁 Contents of {DATA_ROOT}:")
                    for item in contents:
                        item_path = os.path.join(DATA_ROOT, item)
                        if os.path.isdir(item_path):
                            print(f"    📂 {item}/")
                            # Show contents of data directory specifically
                            if item == "data":
                                try:
                                    data_contents = os.listdir(item_path)
                                    print(f"      📁 Contents of data/:")
                                    for data_item in data_contents:
                                        data_item_path = os.path.join(
                                            item_path, data_item
                                        )
                                        if os.path.isdir(data_item_path):
                                            print(f"        📂 {data_item}/")
                                        else:
                                            print(f"        📄 {data_item}")
                                except:
                                    print(f"      ❌ Cannot read data directory")
                        else:
                            print(f"    📄 {item}")

                    # Check for expected map structure
                    expected_places = ["New_York_City", "NewYorkCity"]
                    for place in expected_places:
                        place_path = os.path.join(DATA_ROOT, place)
                        if os.path.exists(place_path):
                            print(f"  ✅ Found place directory: {place_path}")
                            try:
                                buildings = os.listdir(place_path)
                                for building in buildings:
                                    building_path = os.path.join(place_path, building)
                                    if os.path.isdir(building_path):
                                        print(f"    📂 Building: {building}")
                                        floors = os.listdir(building_path)
                                        for floor in floors:
                                            floor_path = os.path.join(
                                                building_path, floor
                                            )
                                            if os.path.isdir(floor_path):
                                                print(f"      📂 Floor: {floor}")
                                                # Check for required files
                                                boundaries_file = os.path.join(
                                                    floor_path, "boundaries.json"
                                                )
                                                colmap_dir = os.path.join(
                                                    floor_path, "colmap_map"
                                                )
                                                print(
                                                    f"        boundaries.json: {'✅' if os.path.exists(boundaries_file) else '❌'}"
                                                )
                                                print(
                                                    f"        colmap_map/: {'✅' if os.path.exists(colmap_dir) else '❌'}"
                                                )
                            except Exception as e:
                                print(f"    ❌ Error reading {place_path}: {e}")
                        else:
                            print(f"  ❌ Place directory not found: {place_path}")

                except Exception as e:
                    print(f"  ❌ Error listing {DATA_ROOT}: {e}")
            else:
                print(f"  ❌ {DATA_ROOT} does not exist")
                # Try alternative paths
                alternative_paths = [
                    "/root/UNav-IO/final",
                    "/root/UNav-IO/data",
                    "/data",
                    "data",
                ]
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        print(f"  🔄 Alternative path found: {alt_path}")
                        DATA_ROOT = alt_path
                        break

            # Look for parameters directory
            parameters_paths = [
                "/root/UNav-IO/parameters",
                "/root/UNav-IO/final/parameters",
                "/parameters",
                "parameters",
            ]
            print(f"\n🔍 Looking for parameters directory:")
            for param_path in parameters_paths:
                if os.path.exists(param_path):
                    print(f"  ✅ Found parameters at: {param_path}")
                    try:
                        contents = os.listdir(param_path)
                        print(f"    📁 Contents:")
                        for item in contents[:10]:  # Show first 10 items
                            print(f"      📂 {item}")
                    except Exception as e:
                        print(f"    ❌ Error listing: {e}")
                else:
                    print(f"  ❌ {param_path} not found")

            place_names = list(PLACES.keys())
            building_names = [b for p in PLACES.values() for b in p.keys()]
            floor_names = [f for p in PLACES.values() for b in p.values() for f in b]

            # Debug: Try to initialize UNav configuration
            print(f"\n🔍 Initializing UNavConfig with:")
            print(f"  DATA_ROOT: {DATA_ROOT}")
            print(f"  FEATURE_MODEL: {FEATURE_MODEL}")
            print(f"  LOCAL_FEATURE_MODEL: {LOCAL_FEATURE_MODEL}")
            print(f"  PLACES: {PLACES}")

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

            commander = commands_from_result  # Navigation command generator function

            print("✅ All components initialized successfully")

            return {
                "status": "success",
                "message": "UNav system initialized successfully",
            }

        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            return {"status": "error", "message": str(e), "type": type(e).__name__}
