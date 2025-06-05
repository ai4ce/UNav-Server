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
    @enter()
    def initialize_unav_system(self):
        """
        Initialize UNav system components during container startup.
        This runs once when the container starts, dramatically improving
        performance for subsequent method calls.
        """
        print("🚀 Initializing UNav system during container startup...")

        from unav.config import UNavConfig
        from unav.localizer.localizer import UNavLocalizer
        from unav.navigator.multifloor import FacilityNavigator

        # Configuration constants
        self.DATA_ROOT = "/root/UNav-IO/data"
        self.FEATURE_MODEL = "DinoV2Salad"
        self.LOCAL_FEATURE_MODEL = "superpoint+lightglue"
        self.PLACES = {
            "New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"]}
        }

        print("🔧 Initializing UNavConfig...")
        self.config = UNavConfig(
            data_final_root=self.DATA_ROOT,
            places=self.PLACES,
            global_descriptor_model=self.FEATURE_MODEL,
            local_feature_model=self.LOCAL_FEATURE_MODEL,
        )
        print("✅ UNavConfig initialized successfully")

        # Extract specific sub-configs for localization and navigation modules
        self.localizor_config = self.config.localizer_config
        self.navigator_config = self.config.navigator_config
        print("✅ Config objects extracted successfully")

        print("🤖 Initializing UNavLocalizer...")
        self.localizer = UNavLocalizer(self.localizor_config)

        print("📊 Loading maps and features...")
        self.localizer.load_maps_and_features()  # Preload all maps and features
        print("✅ UNavLocalizer initialized and maps/features loaded successfully")

        print("🧭 Initializing FacilityNavigator...")
        self.nav = FacilityNavigator(self.navigator_config)
        print("✅ FacilityNavigator initialized successfully")

        print("🎉 UNav system initialization complete! Ready for fast inference.")

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
        """
        Get destinations for a specific place, building, and floor.
        Uses pre-initialized UNav components for fast response.
        """
        try:
            print(f"🎯 Getting destinations for {place}/{building}/{floor}")

            # Use pre-initialized components from @enter method
            target_key = (place, building, floor)
            pf_target = self.nav.pf_map[target_key]

            destinations = [
                {"id": str(did), "name": pf_target.labels[did]}
                for did in pf_target.dest_ids
            ]

            print(f"✅ Found {len(destinations)} destinations")
            return {
                "status": "success",
                "destinations": destinations,
            }

        except Exception as e:
            print(f"❌ Error getting destinations: {e}")
            return {"status": "error", "message": str(e), "type": type(e).__name__}
