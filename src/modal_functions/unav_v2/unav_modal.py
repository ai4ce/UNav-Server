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
        from unav.navigator.commander import commands_from_result

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
        
        # Store commander function for navigation
        self.commander = commands_from_result

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
    
    @method()
    def unav_navigation(
        self,
        user_id: str,
        image,  # np.ndarray (BGR image)
        dest_id: str,
        target_place: str,
        target_building: str,
        target_floor: str,
        top_k: int = None,
        unit: str = "feet",
        language: str = "en",
        refinement_queue: dict = None,
    ):
        """
        Full localization and navigation pipeline.
        - Performs localization from query image.
        - Plans path to user-selected destination.
        - Generates human-readable navigation commands.

        Args:
            user_id: User identifier
            image: BGR image for localization
            dest_id: Destination ID
            target_place: Target place name
            target_building: Target building name
            target_floor: Target floor name
            top_k: Optional localization parameter
            unit: Unit for distances (default: "feet")
            language: Language for commands (default: "en")
            refinement_queue: Optional refinement queue for localization

        Returns:
            dict: {
                "result": dict (path info),
                "cmds": list(str) (step-by-step instructions),
                "best_map_key": tuple(str, str, str) (current floor),
                "floorplan_pose": dict (current pose)
            }
            or dict with "error" key on failure.
        """
        try:
            # Set default for mutable argument
            if refinement_queue is None:
                refinement_queue = {}
            
            query_img = image

            print(f"🧭 Starting navigation for user {user_id} to {target_place}/{target_building}/{target_floor}")

            # Perform localization using pre-initialized localizer
            output = self.localizer.localize(query_img, refinement_queue, top_k=top_k)
            if output is None or "floorplan_pose" not in output:
                return {"error": "Localization failed, no pose found."}

            floorplan_pose = output["floorplan_pose"]
            start_xy, start_heading = floorplan_pose["xy"], -floorplan_pose["ang"]
            source_key = output["best_map_key"]
            start_place, start_building, start_floor = source_key

            print(f"📍 Localized user at {start_place}/{start_building}/{start_floor}")

            # Plan navigation path to destination using pre-initialized navigator
            result = self.nav.find_path(
                start_place, start_building, start_floor, start_xy,
                target_place, target_building, target_floor, dest_id
            )

            # Generate spoken/navigation commands using pre-initialized commander
            cmds = self.commander(
                self.nav, result, initial_heading=start_heading, unit=unit, language=language
            )

            print(f"✅ Navigation path calculated with {len(cmds)} commands")

            # Return all relevant info - convert to JSON-serializable format
            return {
                "result": self._safe_serialize(result),
                "cmds": self._safe_serialize(cmds),
                "best_map_key": self._safe_serialize(source_key),
                "floorplan_pose": self._safe_serialize(floorplan_pose),
                "refinement_queue": self._safe_serialize(output["refinement_queue"]),
            }
            
        except Exception as e:
            print(f"❌ Error in navigation: {e}")
            return {"error": str(e), "type": type(e).__name__}
    
    def _safe_serialize(self, obj):
        """Helper method to safely serialize objects for JSON response"""
        import json
        import numpy as np
        
        def convert_obj(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            elif isinstance(o, np.integer):
                return int(o)
            elif isinstance(o, np.floating):
                return float(o)
            elif isinstance(o, dict):
                return {k: convert_obj(v) for k, v in o.items()}
            elif isinstance(o, (list, tuple)):
                return [convert_obj(item) for item in o]
            else:
                return o
        
        return convert_obj(obj)