from modal import method, gpu, build, enter
import json
import traceback
import numpy as np

from modal_config import app, unav_image, volume


@app.cls(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
    gpu=gpu.Any(),
    enable_memory_snapshot=True,
    concurrency_limit=20,
    allow_concurrent_inputs=20,
    memory=48152,
    container_idle_timeout=120,
)
class UnavServer:
    def __init__(self):
        # Initialize session storage for user contexts
        self.user_sessions = {}

    @build()
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

    def get_session(self, user_id: str) -> dict:
        """Get or create user session"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        return self.user_sessions[user_id]

    def update_session(self, user_id: str, updates: dict):
        """Update user session with new data"""
        session = self.get_session(user_id)
        session.update(updates)

    @method()
    def set_navigation_context(
        self,
        user_id: str,
        dest_id: str,
        target_place: str,
        target_building: str,
        target_floor: str,
        unit: str = "feet",
        language: str = "en",
    ):
        """Set navigation context for a user session"""
        try:
            self.update_session(
                user_id,
                {
                    "selected_dest_id": dest_id,
                    "target_place": target_place,
                    "target_building": target_building,
                    "target_floor": target_floor,
                    "unit": unit,
                    "language": language,
                },
            )

            return {
                "status": "success",
                "message": "Navigation context set successfully",
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "type": type(e).__name__}

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
        session_id: str,
        image,  # np.ndarray (BGR image)
        destination_id: str,
        place: str,
        building: str,
        floor: str,
        top_k: int = None,
        unit: str = "feet",
        language: str = "en",
        refinement_queue: dict = None,
    ):
        """
        Full localization and navigation pipeline.
        - Automatically sets navigation context in user session
        - Performs localization from query image
        - Updates user's current position and floor context
        - Plans path to user-selected destination
        - Generates human-readable navigation commands

        This method is self-contained and automatically manages the navigation context.
        No need to call set_navigation_context() separately.

        Args:
            user_id (str): Unique identifier for the user
            image (np.ndarray): BGR image for localization
            dest_id (str): Destination ID to navigate to
            target_place (str): Target place name
            target_building (str): Target building name
            target_floor (str): Target floor name
            top_k (int, optional): Number of top candidates for localization
            unit (str): Unit for distance measurements (default: "feet")
            language (str): Language for navigation commands (default: "en")
            refinement_queue (dict, optional): Refinement queue for localization

        Returns:
            dict: {
                "status": "success",
                "result": dict (path info),
                "cmds": list(str) (step-by-step instructions),
                "best_map_key": tuple(str, str, str) (current floor),
                "floorplan_pose": dict (current pose),
                "navigation_info": dict (summary information)
            }
            or dict with "status": "error", "error": str on failure.
        """
        try:
            dest_id = destination_id
            target_place = place
            target_building = building
            target_floor = floor
            user_id = session_id

            # Get user session
            session = self.get_session(user_id)

            # Use provided parameters or fallback to session values
            if not dest_id:
                dest_id = session.get("selected_dest_id")
            if not target_place:
                target_place = session.get("target_place")
            if not target_building:
                target_building = session.get("target_building")
            if not target_floor:
                target_floor = session.get("target_floor")
            if unit == "feet":
                unit = session.get("unit", "feet")
            if language == "en":
                language = session.get("language", "en")

            # Check for required navigation context
            if not all([dest_id, target_place, target_building, target_floor]):
                return {
                    "status": "error",
                    "error": "Incomplete navigation context. Please select a destination.",
                    "missing_fields": {
                        "dest_id": dest_id is None,
                        "target_place": target_place is None,
                        "target_building": target_building is None,
                        "target_floor": target_floor is None,
                    },
                }

            # Automatically set/update navigation context in session
            self.update_session(
                user_id,
                {
                    "selected_dest_id": dest_id,
                    "target_place": target_place,
                    "target_building": target_building,
                    "target_floor": target_floor,
                    "unit": unit,
                    "language": language,
                },
            )

            # Use provided refinement queue or get from session
            if refinement_queue is None:
                refinement_queue = session.get("refinement_queue") or {}

            # Perform localization
            output = self.localizer.localize(image, refinement_queue, top_k=top_k)
            
            if output is None or "floorplan_pose" not in output:
                return {
                    "status": "error",
                    "error": "Localization failed, no pose found.",
                }

            floorplan_pose = output["floorplan_pose"]
            start_xy, start_heading = floorplan_pose["xy"], -floorplan_pose["ang"]
            source_key = output["best_map_key"]
            start_place, start_building, start_floor = source_key

            # Validate start position
            if start_xy is None or len(start_xy) != 2:
                return {
                    "status": "error",
                    "error": "Invalid start position from localization.",
                }

            # Check if we're on the right floor for navigation
            if (start_place, start_building, start_floor) != (
                target_place,
                target_building,
                target_floor,
            ):
                # Multi-floor navigation will be handled by the navigator
                pass

            # Update user's current floor context and pose for real-time tracking
            self.update_session(
                user_id,
                {
                    "current_place": start_place,
                    "current_building": start_building,
                    "current_floor": start_floor,
                    "floorplan_pose": floorplan_pose,
                    "refinement_queue": output["refinement_queue"],
                },
            )

            # Convert dest_id to int if it's a string (common issue)
            try:
                dest_id_for_path = int(dest_id)
            except (ValueError, TypeError):
                dest_id_for_path = dest_id

            # Plan navigation path to destination
            result = self.nav.find_path(
                start_place,
                start_building,
                start_floor,
                start_xy,
                target_place,
                target_building,
                target_floor,
                dest_id_for_path,
            )

            if result is None:
                return {
                    "status": "error",
                    "error": "Path planning failed. Could not find route to destination.",
                }

            # Check if result contains an error
            if isinstance(result, dict) and "error" in result:
                return {
                    "status": "error",
                    "error": f"Path planning failed: {result['error']}",
                }

            # Generate spoken/navigation commands
            cmds = self.commander(
                self.nav,
                result,
                initial_heading=start_heading,
                unit=unit,
                language=language,
            )

            # Return all relevant info safely serialized for JSON
            return {
                "status": "success",
                "result": self._safe_serialize(result),
                "cmds": self._safe_serialize(cmds),
                "best_map_key": self._safe_serialize(source_key),
                "floorplan_pose": self._safe_serialize(floorplan_pose),
                "navigation_info": {
                    "start_location": f"{start_place}/{start_building}/{start_floor}",
                    "destination": f"{target_place}/{target_building}/{target_floor}",
                    "dest_id": dest_id,
                    "unit": unit,
                    "language": language,
                },
            }

        except Exception as e:
            # Log current state for debugging
            try:
                session = self.get_session(user_id)
            except:
                session = None

            import traceback
            traceback.print_exc()

            return {"status": "error", "error": str(e), "type": type(e).__name__}

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

    @method()
    def get_user_session(self, user_id: str):
        """Get current user session data"""
        try:
            session = self.get_session(user_id)
            return {"status": "success", "session": self._safe_serialize(session)}
        except Exception as e:
            return {"status": "error", "message": str(e), "type": type(e).__name__}

    @method()
    def clear_user_session(self, user_id: str):
        """Clear user session data"""
        try:
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            return {
                "status": "success",
                "message": f"Session cleared for user {user_id}",
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "type": type(e).__name__}

    @method()
    def unav_navigation_simple(self, inputs: dict):
        """
        Simplified navigation interface that matches the original function signature.

        Args:
            inputs (dict): {
                "user_id": str,
                "image": np.ndarray (BGR image),
                "top_k": Optional[int]
            }

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
            user_id = inputs["user_id"]
            image = inputs["image"]
            top_k = inputs.get("top_k", None)

            session = self.get_session(user_id)

            # Check for required navigation context
            dest_id = session.get("selected_dest_id")
            target_place = session.get("target_place")
            target_building = session.get("target_building")
            target_floor = session.get("target_floor")
            unit = session.get("unit", "feet")
            user_lang = session.get("language", "en")

            if not all([dest_id, target_place, target_building, target_floor]):
                return {
                    "error": "Incomplete navigation context. Please select a destination."
                }

            # Call the main navigation method
            result = self.unav_navigation(
                user_id=user_id,
                image=image,
                dest_id=dest_id,
                target_place=target_place,
                target_building=target_building,
                target_floor=target_floor,
                top_k=top_k,
                unit=unit,
                language=user_lang,
                refinement_queue=session.get("refinement_queue"),
            )

            # Return in the expected format
            if result.get("status") == "success":
                return {
                    "result": result["result"],
                    "cmds": result["cmds"],
                    "best_map_key": result["best_map_key"],
                    "floorplan_pose": result["floorplan_pose"],
                }
            else:
                return {"error": result.get("error", "Navigation failed")}

        except Exception as e:
            return {"error": str(e)}
