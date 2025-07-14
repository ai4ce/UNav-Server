from modal import method, gpu, build, enter
import json
import traceback
import numpy as np
import json
from typing import Dict, List, Any, Optional

from modal_config import app, unav_image, volume


@app.cls(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
    gpu=gpu.Any(),
    enable_memory_snapshot=True,
    concurrency_limit=20,
    allow_concurrent_inputs=20,
    memory=184320,  # Increased from 102400 MB to 202400 MB (200GB)
    container_idle_timeout=100,
)
class UnavServer:
    def __init__(self):
        # Initialize session storage for user contexts
        self.user_sessions = {}

    @enter(snap=True)
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
        unit: str = "meter",
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
    def get_destinations_list(
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
                "destinations": destinations,
            }

        except Exception as e:
            print(f"❌ Error getting destinations: {e}")
            return {"status": "error", "message": str(e), "type": type(e).__name__}

    @method()
    def planner(
        self,
        session_id: str,
        base_64_image,
        destination_id: str,
        place: str,
        building: str,
        floor: str,
        top_k: int = None,
        unit: str = "meter",
        language: str = "en",
        refinement_queue: dict = None,
    ):
        """
        Full localization and navigation pipeline with timing tracking.
        """
        import time

        # Start total timing
        start_time = time.time()
        timing_data = {}
        image = None

        # Validate and convert image input
        if base_64_image is None:
            return {
                "status": "error",
                "error": "No image provided. base_64_image parameter is required.",
                "timing": {"total": (time.time() - start_time) * 1000},
            }

        # Convert base64 string to BGR numpy array using OpenCV
        if isinstance(base_64_image, str):
            import base64
            import cv2

            try:
                # Fix base64 padding if needed
                base64_string = base_64_image
                print(f"Received base64 image string of length {len(base64_string)}")
                ## print the first 50 characers of bas64 string
                # print(f"{base64_string[0:50]}")
                # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
                if "," in base64_string:
                    base64_string = base64_string.split(",")[1]

                # Add padding if necessary
                missing_padding = len(base64_string) % 4
                if missing_padding:
                    base64_string += "=" * (4 - missing_padding)

                # Decode base64 string to bytes
                image_bytes = base64.b64decode(base64_string)

                # print(f"Image bytes {image_bytes}")
                # Convert bytes to numpy array
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                # Decode image using OpenCV (automatically in BGR format)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if image is None:
                    return {
                        "status": "error",
                        "error": "Failed to decode base64 image. Invalid image format.",
                        "timing": {"total": (time.time() - start_time) * 1000},
                    }
            except Exception as img_error:
                return {
                    "status": "error",
                    "error": f"Error processing base64 image: {str(img_error)}",
                    "timing": {"total": (time.time() - start_time) * 1000},
                }
        elif isinstance(base_64_image, np.ndarray):
            # If already a numpy array, use it directly (assume BGR format)
            image = base_64_image
        else:
            return {
                "status": "error",
                "error": f"Unsupported image format. Expected base64 string or numpy array, got {type(base_64_image)}",
                "timing": {"total": (time.time() - start_time) * 1000},
            }

        # --- GPU DEBUG INFO ---
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"[GPU DEBUG] torch.cuda.is_available(): {cuda_available}")
            if cuda_available:
                print(f"[GPU DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
                print(f"[GPU DEBUG] torch.cuda.current_device(): {torch.cuda.current_device()}")
                print(f"[GPU DEBUG] torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
            # import subprocess
            # nvidia_smi = subprocess.check_output(["nvidia-smi"]).decode()
            # print(f"[GPU DEBUG] nvidia-smi output:\n{nvidia_smi}")
        except Exception as gpu_debug_exc:
            print(f"[GPU DEBUG] Error printing GPU info: {gpu_debug_exc}")
        # --- END GPU DEBUG INFO ---

        try:
            # Step 1: Setup and session management
            setup_start = time.time()

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

            timing_data["setup"] = (time.time() - setup_start) * 1000  # Convert to ms
            print(f"⏱️ Setup: {timing_data['setup']:.2f}ms")

            # Step 2: Localization
            localization_start = time.time()

            # Perform localization
            output = self.localizer.localize(image, refinement_queue, top_k=top_k)

            timing_data["localization"] = (time.time() - localization_start) * 1000
            print(f"⏱️ Localization: {timing_data['localization']:.2f}ms")

            if output is None or "floorplan_pose" not in output:
                return {
                    "status": "error",
                    "error": "Localization failed, no pose found.",
                    "timing": timing_data,
                }

            # Step 3: Process localization results
            processing_start = time.time()

            floorplan_pose = output["floorplan_pose"]
            start_xy, start_heading = floorplan_pose["xy"], -floorplan_pose["ang"]
            source_key = output["best_map_key"]
            start_place, start_building, start_floor = source_key

            # Validate start position
            if start_xy is None or len(start_xy) != 2:
                return {
                    "status": "error",
                    "error": "Invalid start position from localization.",
                    "timing": timing_data,
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

            timing_data["processing"] = (time.time() - processing_start) * 1000
            print(f"⏱️ Processing: {timing_data['processing']:.2f}ms")

            # Step 4: Path planning
            path_planning_start = time.time()

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

            timing_data["path_planning"] = (time.time() - path_planning_start) * 1000
            print(f"⏱️ Path Planning: {timing_data['path_planning']:.2f}ms")

            if result is None:
                return {
                    "status": "error",
                    "error": "Path planning failed. Could not find route to destination.",
                    "timing": timing_data,
                }

            # Check if result contains an error
            if isinstance(result, dict) and "error" in result:
                return {
                    "status": "error",
                    "error": f"Path planning failed: {result['error']}",
                    "timing": timing_data,
                }

            # Step 5: Command generation
            command_generation_start = time.time()

            # Generate spoken/navigation commands
            cmds = self.commander(
                self.nav,
                result,
                initial_heading=start_heading,
                unit=unit,
                language=language,
            )

            timing_data["command_generation"] = (
                time.time() - command_generation_start
            ) * 1000
            print(f"⏱️ Command Generation: {timing_data['command_generation']:.2f}ms")

            # Step 6: Serialization
            serialization_start = time.time()

            serialized_result = self._safe_serialize(result)
            serialized_cmds = self._safe_serialize(cmds)
            serialized_source_key = self._safe_serialize(source_key)
            serialized_floorplan_pose = self._safe_serialize(floorplan_pose)

            timing_data["serialization"] = (time.time() - serialization_start) * 1000
            print(f"⏱️ Serialization: {timing_data['serialization']:.2f}ms")

            # Calculate total time
            timing_data["total"] = (time.time() - start_time) * 1000
            print(f"⏱️ Total Navigation Time: {timing_data['total']:.2f}ms")

            # Print summary
            print(f"📊 Timing Breakdown:")
            print(
                f"   Setup: {timing_data['setup']:.1f}ms ({timing_data['setup']/timing_data['total']*100:.1f}%)"
            )
            print(
                f"   Localization: {timing_data['localization']:.1f}ms ({timing_data['localization']/timing_data['total']*100:.1f}%)"
            )
            print(
                f"   Processing: {timing_data['processing']:.1f}ms ({timing_data['processing']/timing_data['total']*100:.1f}%)"
            )
            print(
                f"   Path Planning: {timing_data['path_planning']:.1f}ms ({timing_data['path_planning']/timing_data['total']*100:.1f}%)"
            )
            print(
                f"   Commands: {timing_data['command_generation']:.1f}ms ({timing_data['command_generation']/timing_data['total']*100:.1f}%)"
            )
            print(
                f"   Serialization: {timing_data['serialization']:.1f}ms ({timing_data['serialization']/timing_data['total']*100:.1f}%)"
            )

            # Return all relevant info safely serialized for JSON
            result = {
                "status": "success",
                "result": serialized_result,
                "cmds": serialized_cmds,
                "best_map_key": serialized_source_key,
                "floorplan_pose": serialized_floorplan_pose,
                "navigation_info": {
                    "start_location": f"{start_place}/{start_building}/{start_floor}",
                    "destination": f"{target_place}/{target_building}/{target_floor}",
                    "dest_id": dest_id,
                    "unit": unit,
                    "language": language,
                },
                "timing": timing_data,  # Include timing data in response
            }

            return self.convert_navigation_to_trajectory(result)

        except Exception as e:
            # Calculate partial timing data
            timing_data["total"] = (time.time() - start_time) * 1000

            # Log current state for debugging
            try:
                session = self.get_session(user_id)
            except:
                session = None

            import traceback

            traceback.print_exc()

            return {
                "status": "error",
                "error": str(e),
                "type": type(e).__name__,
                "timing": timing_data,
            }

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
            unit = session.get("unit", "meter")
            user_lang = session.get("language", "en")

            if not all([dest_id, target_place, target_building, target_floor]):
                return {
                    "error": "Incomplete navigation context. Please select a destination."
                }

            # Call the main navigation method
            result = self.planner(
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

    def convert_navigation_to_trajectory(
        self, navigation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert navigation result format to trajectory output format.

        Args:
            navigation_result: Dictionary containing navigation result data

        Returns:
            Dictionary in trajectory output format
        """

        # Extract data from navigation result
        result = navigation_result.get("result", {})
        cmds = navigation_result.get("cmds", [])
        best_map_key = navigation_result.get("best_map_key", [])
        floorplan_pose = navigation_result.get("floorplan_pose", {})
        navigation_info = navigation_result.get("navigation_info", {})

        # Extract building, place, and floor information
        place = best_map_key[0] if len(best_map_key) > 0 else ""
        building = best_map_key[1] if len(best_map_key) > 1 else ""
        floor = best_map_key[2] if len(best_map_key) > 2 else ""

        # Get path coordinates from result
        path_coords = result.get("path_coords", [])

        # Add starting pose coordinates if available
        start_xy = floorplan_pose.get("xy", [])
        start_ang = floorplan_pose.get("ang", 0)

        # Create paths array - include start position with angle if available
        paths = []
        if start_xy and len(start_xy) >= 2:
            if start_ang:
                paths.append([start_xy[0], start_xy[1], start_ang])
            else:
                paths.append(start_xy)

        # Add all path coordinates
        for coord in path_coords:
            if len(coord) >= 2:
                paths.append(coord)

        # Calculate scale based on total cost and path distance (approximate)
        # This is an estimation - you may need to adjust based on your specific use case
        total_cost = result.get("total_cost", 0)
        scale = (
            0.02205862195  # Default scale, you might want to calculate this dynamically
        )

        # Create trajectory structure
        trajectory_data = {
            "trajectory": [
                {
                    "0": {
                        "name": "destination",
                        "building": building,
                        "floor": floor,
                        "paths": paths,
                        "command": {
                            "instructions": cmds,
                            "are_instructions_generated": len(cmds) > 0,
                        },
                        "scale": scale,
                    }
                },
                None,  # This seems to be a placeholder in the original format
            ],
            "scale": scale,
        }

        return trajectory_data
