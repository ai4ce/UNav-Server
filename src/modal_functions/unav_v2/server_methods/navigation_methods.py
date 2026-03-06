from modal import method

from ..utils.image_utils import decode_image_input

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
    is_vlm_extraction_enabled: bool = False,
    enable_multifloor: bool = False,
    should_use_user_provided_coordinate: bool = False,
    x: float = None,
    y: float = None,
    angle: float = None,
):
    """
    Full localization and navigation pipeline with timing tracking and middleware tracing.
    
    Args:
        session_id: Unique identifier for the user session
        base_64_image: Base64 encoded image or numpy array (optional if using provided coordinates)
        destination_id: Destination node ID
        place: Place name
        building: Building name
        floor: Floor name
        top_k: Number of top candidates for localization
        unit: Distance unit ("meter" or "feet")
        language: Language code for instructions
        refinement_queue: Queue for pose refinement
        is_vlm_extraction_enabled: Enable VLM text extraction fallback
        enable_multifloor: Enable multi-floor navigation
        should_use_user_provided_coordinate: If True, skip localization and use provided x, y, angle
        x: X coordinate on floor plan (required if should_use_user_provided_coordinate=True)
        y: Y coordinate on floor plan (required if should_use_user_provided_coordinate=True)
        angle: Heading angle in degrees (required if should_use_user_provided_coordinate=True)
    """
    import time
    import logging
    import uuid

    # Generate a unique ID for this specific call to planner
    call_id = str(uuid.uuid4())

    # Check tracing availability
    has_tracer = hasattr(self, "tracer") and self.tracer is not None
    print(
        f"📋 [PLANNER] Called with session_id={session_id}, call_id={call_id}, has_tracer={has_tracer}, tracer_type={type(getattr(self, 'tracer', None)).__name__}"
    )

    # Create parent span for the entire planner operation if tracer is available
    if has_tracer:
        print(f"📋 [PLANNER] Using TRACED execution path for call_id={call_id}")
        with self.tracer.start_as_current_span("planner_span") as parent_span:
            parent_span.set_attribute("unav.call_id", call_id)
            parent_span.set_attribute("unav.session_id", session_id)
            # Start total timing
            start_time = time.time()
            timing_data = {}
            image = None

            # Validate user-provided coordinates if enabled
            if should_use_user_provided_coordinate:
                if x is None or y is None or angle is None:
                    return {
                        "status": "error",
                        "error": "When should_use_user_provided_coordinate=True, x, y, and angle must all be provided.",
                        "timing": {"total": (time.time() - start_time) * 1000},
                    }
                print(f"📍 Using user-provided coordinates: x={x}, y={y}, angle={angle}°")
                # Image is optional when using provided coordinates
                if base_64_image is not None:
                    print("⚠️ Image provided but will be ignored since using provided coordinates")
            elif base_64_image is None:
                # Image is required when not using provided coordinates
                return {
                    "status": "error",
                    "error": "No image provided. base_64_image parameter is required when not using provided coordinates.",
                    "timing": {"total": (time.time() - start_time) * 1000},
                }

            if not should_use_user_provided_coordinate:
                image, decode_error = decode_image_input(base_64_image)
                if decode_error:
                    return {
                        "status": "error",
                        "error": decode_error,
                        "timing": {"total": (time.time() - start_time) * 1000},
                    }

            # --- GPU DEBUG INFO ---
            try:
                import torch

                cuda_available = torch.cuda.is_available()
                print(f"[GPU DEBUG] torch.cuda.is_available(): {cuda_available}")
                if cuda_available:
                    print(
                        f"[GPU DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}"
                    )
                    print(
                        f"[GPU DEBUG] torch.cuda.current_device(): {torch.cuda.current_device()}"
                    )
                    print(
                        f"[GPU DEBUG] torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}"
                    )
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

                timing_data["setup"] = (
                    time.time() - setup_start
                ) * 1000  # Convert to ms
                print(f"⏱️ Setup: {timing_data['setup']:.2f}ms")

                # Step 2: Localization (or skip if using provided coordinates)
                if should_use_user_provided_coordinate:
                    # Skip localization - construct mock output from provided coordinates
                    print("⏭️ Skipping localization - using provided coordinates")
                    localization_start = time.time()
                    
                    # Construct mock localization output
                    output = self._construct_mock_localization_output(
                        x=x,
                        y=y,
                        angle=angle,
                        place=target_place,
                        building=target_building,
                        floor=target_floor,
                    )
                    
                    timing_data["localization"] = (
                        time.time() - localization_start
                    ) * 1000
                    print(f"⏱️ Mock Localization: {timing_data['localization']:.2f}ms")
                else:
                    # Normal localization process
                    localization_start = time.time()

                    # Create child span for localization
                    with self.tracer.start_as_current_span(
                        "localization_span"
                    ) as localization_span:
                        # Ensure GPU components are ready (initializes localizer)
                        self.ensure_gpu_components_ready()

                        with self.tracer.start_as_current_span(
                            "load_maps_span"
                        ) as load_maps_span:
                            # Ensure maps are loaded for the target location
                            self.ensure_maps_loaded(
                                target_place,
                                target_building,
                                floor=target_floor,
                                enable_multifloor=enable_multifloor,
                            )

                        # Get the selective localizer for this building (all floors loaded)
                        map_key = (target_place, target_building)
                        localizer_to_use = self.selective_localizers.get(map_key)
                        if not localizer_to_use and target_floor:
                            floor_key = (target_place, target_building, target_floor)
                            localizer_to_use = self.selective_localizers.get(
                                floor_key, self.localizer
                            )
                        else:
                            localizer_to_use = localizer_to_use or self.localizer

                        # Localizer already patched in ensure_maps_loaded() or initialize_gpu_components()
                        # No need to patch again here to avoid double-wrapping spans

                        output = localizer_to_use.localize(
                            image, refinement_queue, top_k=top_k
                        )

                    timing_data["localization"] = (
                        time.time() - localization_start
                    ) * 1000
                    print(f"⏱️ Localization: {timing_data['localization']:.2f}ms")

                    if output is None or "floorplan_pose" not in output:
                        print("❌ Localization failed, no pose found.")

                        if is_vlm_extraction_enabled:
                            # Run VLM to extract text from image as fallback
                            try:
                                print(
                                    "🔄 Attempting VLM fallback for text extraction..."
                                )
                                extracted_text = self.run_vlm_on_image(image)

                                # Log the extracted text for debugging
                                print(
                                    f"📝 VLM extracted text: {extracted_text[:200]}..."
                                )

                                # You can add logic here to process the extracted text
                                # For example, search for room numbers, building names, etc.
                                # and use that information to provide approximate location or guidance

                                return {
                                    "status": "error",
                                    "error": "Localization failed, but VLM text extraction completed.",
                                    "extracted_text": extracted_text,
                                    "timing": timing_data,
                                    "fallback_info": "Text was extracted from the image but precise localization failed. Please try taking a clearer photo or move to a different location.",
                                }

                            except Exception as vlm_error:
                                print(f"❌ Error during VLM fallback: {vlm_error}")
                                return {
                                    "status": "error",
                                    "error": "Localization failed and VLM fallback also failed.",
                                    "vlm_error": str(vlm_error),
                                    "timing": timing_data,
                                }

                        return {
                            "status": "error",
                            "error": "Localization failed, no pose found.",
                            "error_code": "localization_failed",
                            "timing": timing_data,
                        }

                # Step 3: Process localization results
                processing_start = time.time()

                floorplan_pose = output["floorplan_pose"]
                start_xy, start_heading = (
                    floorplan_pose["xy"],
                    -floorplan_pose["ang"],
                )
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

                # Create child span for path planning
                with self.tracer.start_as_current_span(
                    "path_planning_span"
                ) as path_planning_span:
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

                timing_data["path_planning"] = (
                    time.time() - path_planning_start
                ) * 1000
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

                # Create child span for command generation
                with self.tracer.start_as_current_span(
                    "command_generation_span"
                ) as command_span:
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
                print(
                    f"⏱️ Command Generation: {timing_data['command_generation']:.2f}ms"
                )

                # Step 6: Serialization
                serialization_start = time.time()

                serialized_result = self._safe_serialize(result)
                serialized_cmds = self._safe_serialize(cmds)
                serialized_source_key = self._safe_serialize(source_key)
                serialized_floorplan_pose = self._safe_serialize(floorplan_pose)

                timing_data["serialization"] = (
                    time.time() - serialization_start
                ) * 1000
                print(f"⏱️ Serialization: {timing_data['serialization']:.2f}ms")

                # Calculate total time
                timing_data["total"] = (time.time() - start_time) * 1000
                print(f"⏱️ Total Navigation Time: {timing_data['total']:.2f}ms")

                # Log summary
                logging.info(
                    f"Navigation pipeline completed successfully. "
                    f"Total time: {timing_data['total']:.0f}ms, "
                    f"Localization: {timing_data['localization']:.0f}ms, "
                    f"Path planning: {timing_data['path_planning']:.0f}ms"
                )

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
    else:
        print("📋 [PLANNER] Using NON-TRACED execution path")
        pass

@method()
def localize_user(
    self,
    session_id: str,
    base_64_image,
    place: str,
    building: str,
    floor: str,
    top_k: int = None,
    refinement_queue: dict = None,
    enable_multifloor: bool = False,
):
    """
    Localize user position without navigation planning.
    Returns the user's current position and floor information.
    
    Args:
        session_id: Unique identifier for the user session
        base_64_image: Base64 encoded image or numpy array
        place: Target place name
        building: Target building name
        floor: Target floor name
        top_k: Number of top candidates to retrieve (optional)
        refinement_queue: Queue for pose refinement (optional)
        enable_multifloor: Whether to enable multi-floor localization
        
    Returns:
        dict: Contains status, floorplan_pose, best_map_key, and timing information
    """
    import time
    
    start_time = time.time()
    timing_data = {}
    
    image, decode_error = decode_image_input(base_64_image)
    if decode_error:
        return {
            "status": "error",
            "error": decode_error,
            "timing": {"total": (time.time() - start_time) * 1000},
        }
    
    try:
        # Setup and session management
        setup_start = time.time()
        
        session = self.get_session(session_id)
        
        # Use provided refinement queue or get from session
        if refinement_queue is None:
            refinement_queue = session.get("refinement_queue") or {}
        
        timing_data["setup"] = (time.time() - setup_start) * 1000
        
        # Localization
        localization_start = time.time()
        
        # Ensure GPU components ready
        self.ensure_gpu_components_ready()
        
        # Ensure maps are loaded
        self.ensure_maps_loaded(
            place, building, floor=floor,
            enable_multifloor=enable_multifloor
        )
        
        # Get the selective localizer
        map_key = (place, building)
        localizer_to_use = self.selective_localizers.get(map_key)
        if not localizer_to_use and floor:
            floor_key = (place, building, floor)
            localizer_to_use = self.selective_localizers.get(floor_key, self.localizer)
        else:
            localizer_to_use = localizer_to_use or self.localizer
        
        # Perform localization
        output = localizer_to_use.localize(image, refinement_queue, top_k=top_k)
        
        timing_data["localization"] = (time.time() - localization_start) * 1000
        
        # Check if localization succeeded
        if output is None or "floorplan_pose" not in output:
            return {
                "status": "error",
                "error": "Localization failed, no pose found.",
                "error_code": "localization_failed",
                "timing": timing_data,
            }
        
        # Process localization results
        processing_start = time.time()
        
        floorplan_pose = output["floorplan_pose"]
        source_key = output["best_map_key"]
        localized_place, localized_building, localized_floor = source_key
        
        # Update user's current floor context and pose
        self.update_session(
            session_id,
            {
                "current_place": localized_place,
                "current_building": localized_building,
                "current_floor": localized_floor,
                "floorplan_pose": floorplan_pose,
                "refinement_queue": output["refinement_queue"],
            },
        )
        
        timing_data["processing"] = (time.time() - processing_start) * 1000
        
        # Serialization
        serialization_start = time.time()
        
        serialized_source_key = self._safe_serialize(source_key)
        serialized_floorplan_pose = self._safe_serialize(floorplan_pose)
        
        timing_data["serialization"] = (time.time() - serialization_start) * 1000
        
        # Calculate total time
        timing_data["total"] = (time.time() - start_time) * 1000
        
        # Return localization results
        return {
            "status": "success",
            "floorplan_pose": serialized_floorplan_pose,
            "best_map_key": serialized_source_key,
            "location_info": {
                "place": localized_place,
                "building": localized_building,
                "floor": localized_floor,
                "position": serialized_floorplan_pose.get("xy"),
                "heading": serialized_floorplan_pose.get("ang"),
            },
            "timing": timing_data,
        }
        
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
