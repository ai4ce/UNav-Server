from modal import method
from modal_config import app, unav_image, volume


@app.function(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
)
def test_unav_initialization():
    """Test UNav initialization with correct data paths"""
    try:
        import os

        # Verify data structure
        DATA_ROOT = "/root/UNav-IO/data"
        print(f"🔍 Testing with DATA_ROOT: {DATA_ROOT}")

        if not os.path.exists(DATA_ROOT):
            return {
                "status": "error",
                "message": f"DATA_ROOT {DATA_ROOT} does not exist",
            }

        # Check for expected structure
        expected_path = os.path.join(
            DATA_ROOT, "New_York_City", "LightHouse", "6_floor"
        )
        if not os.path.exists(expected_path):
            return {
                "status": "error",
                "message": f"Expected path {expected_path} does not exist",
            }

        # Check for required files
        boundaries_file = os.path.join(expected_path, "boundaries.json")
        colmap_dir = os.path.join(expected_path, "colmap_map")

        if not os.path.exists(boundaries_file):
            return {
                "status": "error",
                "message": f"boundaries.json not found at {boundaries_file}",
            }

        if not os.path.exists(colmap_dir):
            return {
                "status": "error",
                "message": f"colmap_map directory not found at {colmap_dir}",
            }

        # Test UNav imports
        from unav.config import UNavConfig
        from unav.localizer.localizer import UNavLocalizer
        from unav.navigator.multifloor import FacilityNavigator
        from unav.navigator.commander import commands_from_result

        print("✅ All imports successful")

        # Test UNav configuration
        FEATURE_MODEL = "DinoV2Salad"
        LOCAL_FEATURE_MODEL = "superpoint+lightglue"
        PLACES = {"New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"]}}

        config = UNavConfig(
            data_final_root=DATA_ROOT,
            parameters_root="/root/UNav-IO/parameters",  # Separate parameters path
            places=PLACES,
            global_descriptor_model=FEATURE_MODEL,
            local_feature_model=LOCAL_FEATURE_MODEL,
        )

        print("✅ UNavConfig initialized successfully")

        # Test localizer initialization
        localizor_config = config.localizer_config
        localizer = UNavLocalizer(localizor_config)

        print("✅ UNavLocalizer initialized successfully")

        # Test loading maps and features
        localizer.load_maps_and_features()

        print("✅ Maps and features loaded successfully")

        # Test navigator initialization
        navigator_config = config.navigator_config
        nav = FacilityNavigator(navigator_config)

        print("✅ FacilityNavigator initialized successfully")

        return {
            "status": "success",
            "message": "UNav system fully initialized and ready",
            "data_root": DATA_ROOT,
            "available_floors": ["3_floor", "4_floor", "6_floor"],
        }

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"❌ Error during UNav initialization: {e}")
        print(f"Full traceback:\n{error_trace}")
        return {
            "status": "error",
            "message": str(e),
            "type": type(e).__name__,
            "traceback": error_trace,
        }


@app.local_entrypoint()
def main():
    result = test_unav_initialization.remote()
    print(f"\n🎯 Test Result: {result}")
