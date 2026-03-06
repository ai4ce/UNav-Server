from typing import Any, Dict

import numpy as np
from modal import method



def _safe_serialize(self, obj):
    """Helper method to safely serialize objects for JSON response"""

    def convert_obj(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, dict):
            return {k: convert_obj(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert_obj(item) for item in o]
        return o

    return convert_obj(obj)


@method()
def generate_nav_instructions_from_coordinates(
    self,
    session_id: str,
    localization_result: dict,
    dest_id: int,
    target_place: str,
    target_building: str,
    target_floor: str,
    unit: str = "meter",
    language: str = "en",
):
    """
    Generate navigation instructions from localized coordinates to a destination.

    This function takes the coordinates returned by localize_user and generates
    step-by-step navigation instructions to reach the specified destination.

    Args:
        session_id: Unique identifier for the user session
        localization_result: Result dictionary from localize_user containing:
            - floorplan_pose: Current position with xy coordinates and angle
            - best_map_key: Tuple of (place, building, floor)
        dest_id: Destination node ID in the navigation graph
        target_place: Target place name
        target_building: Target building name
        target_floor: Target floor name
        unit: Unit for distance measurements ("meter" or "foot")
        language: Language code for instructions (e.g., "en", "es", "fr")

    Returns:
        dict: Contains status, navigation instructions, path info, and timing
            - status: "success" or "error"
            - instructions: List of step-by-step navigation commands
            - path_info: Details about the route
            - timing: Performance metrics
    """
    import time

    start_time = time.time()
    timing_data = {}

    try:
        # Step 1: Validate input
        validation_start = time.time()

        if not localization_result or localization_result.get("status") != "success":
            return {
                "status": "error",
                "error": "Invalid localization result. Please provide a successful localization_result from localize_user.",
                "timing": {"total": (time.time() - start_time) * 1000},
            }

        floorplan_pose = localization_result.get("floorplan_pose")
        best_map_key = localization_result.get("best_map_key")

        if not floorplan_pose or not best_map_key:
            return {
                "status": "error",
                "error": "Missing floorplan_pose or best_map_key in localization result.",
                "timing": {"total": (time.time() - start_time) * 1000},
            }

        # Extract start position
        start_xy = floorplan_pose.get("xy")
        start_heading = -floorplan_pose.get("ang", 0)  # Negate angle for navigation

        if not start_xy or len(start_xy) != 2:
            return {
                "status": "error",
                "error": "Invalid start position in floorplan_pose.",
                "timing": {"total": (time.time() - start_time) * 1000},
            }

        # Extract start location
        if isinstance(best_map_key, (list, tuple)) and len(best_map_key) >= 3:
            start_place, start_building, start_floor = (
                best_map_key[0],
                best_map_key[1],
                best_map_key[2],
            )
        else:
            return {
                "status": "error",
                "error": "Invalid best_map_key format in localization result.",
                "timing": {"total": (time.time() - start_time) * 1000},
            }

        timing_data["validation"] = (time.time() - validation_start) * 1000

        # Step 2: Ensure GPU components and maps are ready
        setup_start = time.time()

        self.ensure_gpu_components_ready()
        self.ensure_maps_loaded(
            target_place, target_building, floor=target_floor, enable_multifloor=True
        )

        timing_data["setup"] = (time.time() - setup_start) * 1000

        # Step 3: Path planning
        path_planning_start = time.time()

        # Convert dest_id to int if necessary
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

        timing_data["path_planning"] = (time.time() - path_planning_start) * 1000

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

        # Step 4: Generate navigation instructions
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

        # Step 5: Serialize results
        serialization_start = time.time()

        serialized_result = self._safe_serialize(result)
        serialized_cmds = self._safe_serialize(cmds)
        serialized_source_key = self._safe_serialize(best_map_key)
        serialized_floorplan_pose = self._safe_serialize(floorplan_pose)

        timing_data["serialization"] = (time.time() - serialization_start) * 1000

        # Calculate total time
        timing_data["total"] = (time.time() - start_time) * 1000

        # Step 6: Update session with navigation context
        self.update_session(
            session_id,
            {
                "current_place": start_place,
                "current_building": start_building,
                "current_floor": start_floor,
                "target_place": target_place,
                "target_building": target_building,
                "target_floor": target_floor,
                "selected_dest_id": dest_id,
            },
        )

        # Return navigation instructions in same format as planner
        result_dict = {
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
            "timing": timing_data,
        }

        return self.convert_navigation_to_trajectory(result_dict)

    except Exception as e:
        timing_data["total"] = (time.time() - start_time) * 1000

        import traceback

        traceback.print_exc()

        return {
            "status": "error",
            "error": str(e),
            "type": type(e).__name__,
            "timing": timing_data,
        }


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
    _ = total_cost
    scale = 0.02205862195

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
            None,
        ],
        "scale": scale,
    }

    return trajectory_data



def run_vlm_on_image(self, image: np.ndarray) -> str:
    """
    Run VLM on the provided image to extract text using Gemini 2.5 Flash.

    Args:
        image (np.ndarray): BGR image array.

    Returns:
        str: Extracted text from the image.
    """
    # Create span for VLM extraction if tracer available
    if hasattr(self, "tracer") and self.tracer:
        with self.tracer.start_as_current_span("vlm_text_extraction_span"):
            try:
                # 1) Import required libraries
                from google import genai
                from google.genai import types
                import cv2
                import os

                # 2) Assign API key - get from environment variable (set by Modal Secret)
                GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
                if not GEMINI_API_KEY:
                    error_msg = "GEMINI_API_KEY environment variable not set. Please set it with your API key in Modal Secrets."
                    print(f"❌ {error_msg}")
                    return error_msg

                # Create client with API key
                client = genai.Client(api_key=GEMINI_API_KEY)

                # 3) Configure with Gemini 2.5 Flash model
                GEMINI_MODEL = "gemini-2.5-flash"

                # 4) Run VLM on the image
                # Convert BGR to RGB for proper processing
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Encode image to bytes (JPEG format)
                _, image_bytes = cv2.imencode(".jpg", image_rgb)
                image_bytes = image_bytes.tobytes()

                # Create the prompt for text extraction
                prompt = """Analyze this image and extract all visible text content.
            Please provide:
            1. All readable text, signs, labels, and written content
            2. Any numbers, codes, or identifiers visible
            3. Location descriptions or directional information if present

            Format the response as clear, readable text without extra formatting."""

                # Generate content using the model with proper SDK usage
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    ],
                )

                # Extract the text response
                extracted_text = response.text if response.text else "No text extracted"

                print(
                    f"✅ VLM extraction successful: {len(extracted_text)} characters extracted"
                )

                return extracted_text

            except ImportError as e:
                error_msg = f"Missing required library for VLM: {str(e)}. Please install: pip install google-genai"
                print(f"❌ {error_msg}")
                return error_msg
            except Exception as e:
                error_msg = f"VLM extraction failed: {str(e)}"
                print(f"❌ {error_msg}")
                return error_msg
    else:
        try:
            # 1) Import required libraries
            from google import genai
            from google.genai import types
            import cv2
            import os

            # 2) Assign API key - get from environment variable (set by Modal Secret)
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if not GEMINI_API_KEY:
                error_msg = "GEMINI_API_KEY environment variable not set. Please set it with your API key in Modal Secrets."
                print(f"❌ {error_msg}")
                return error_msg

            # Create client with API key
            client = genai.Client(api_key=GEMINI_API_KEY)

            # 3) Configure with Gemini 2.5 Flash model
            GEMINI_MODEL = "gemini-2.5-flash"

            # 4) Run VLM on the image
            # Convert BGR to RGB for proper processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Encode image to bytes (JPEG format)
            _, image_bytes = cv2.imencode(".jpg", image_rgb)
            image_bytes = image_bytes.tobytes()

            # Create the prompt for text extraction
            prompt = """Analyze this image and extract all visible text content.
            Please provide:
            1. All readable text, signs, labels, and written content
            2. Any numbers, codes, or identifiers visible
            3. Location descriptions or directional information if present

            Format the response as clear, readable text without extra formatting."""

            # Generate content using the model with proper SDK usage
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )

            # Extract the text response
            extracted_text = response.text if response.text else "No text extracted"

            print(f"✅ VLM extraction successful: {len(extracted_text)} characters extracted")

            return extracted_text

        except ImportError as e:
            error_msg = f"Missing required library for VLM: {str(e)}. Please install: pip install google-genai"
            print(f"❌ {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"VLM extraction failed: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
