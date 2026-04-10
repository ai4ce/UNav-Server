"""
Navigation logic functions - called by endpoints in unav_modal.py
"""
import numpy as np
from typing import Dict, Any, Optional

from .utils import (
    run_safe_serialize,
    run_convert_navigation_to_trajectory,
    run_construct_mock_localization_output,
    run_set_navigation_context,
)
from .maps import run_ensure_maps_loaded
from .vlm import run_vlm_on_image


def run_planner(
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
    enable_multifloor: bool = True,
    should_use_user_provided_coordinate: bool = False,
    x: float = None,
    y: float = None,
    angle: float = None,
    turn_mode: str = "default",
) -> Dict[str, Any]:
    """Full localization and navigation pipeline logic."""
    import time
    import logging
    import uuid
    import base64
    import cv2

    from ..server_methods.helpers import (
        _get_queue_key_for_image_shape,
        _update_refinement_queue,
    )

    call_id = str(uuid.uuid4())
    has_tracer = hasattr(self, "tracer") and self.tracer is not None
    print(f"📋 [PLANNER] Called with session_id={session_id}, call_id={call_id}, has_tracer={has_tracer}")
    print(f"📋 [PLANNER] Params: destination_id={destination_id}, place={place}, building={building}, floor={floor}, top_k={top_k}, unit={unit}, language={language}, enable_multifloor={enable_multifloor}, should_use_user_provided_coordinate={should_use_user_provided_coordinate}")

    if has_tracer:
        print(f"📋 [PLANNER] Using TRACED execution path for call_id={call_id}")
        with self.tracer.start_as_current_span("planner_span") as parent_span:
            parent_span.set_attribute("unav.call_id", call_id)
            parent_span.set_attribute("unav.session_id", session_id)
            start_time = time.time()
            timing_data = {}
            image = None

            if should_use_user_provided_coordinate:
                if x is None or y is None or angle is None:
                    return {"status": "error", "error": "x, y, angle required", "timing": {"total": (time.time() - start_time) * 1000}}
                print(f"📍 Using user-provided coordinates: x={x}, y={y}, angle={angle}°")
                if base_64_image is not None:
                    print("⚠️ Image provided but will be ignored since using provided coordinates")
            elif base_64_image is None:
                return {"status": "error", "error": "No image provided", "timing": {"total": (time.time() - start_time) * 1000}}

            if not should_use_user_provided_coordinate:
                if isinstance(base_64_image, str):
                    try:
                        base64_string = base_64_image
                        if "," in base64_string:
                            base64_string = base64_string.split(",")[1]
                        missing_padding = len(base64_string) % 4
                        if missing_padding:
                            base64_string += "=" * (4 - missing_padding)
                        image_bytes = base64.b64decode(base64_string)
                        print(f"Received base64 image string of length {len(base64_string)}")
                        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                        if image is None:
                            return {"status": "error", "error": "Failed to decode image", "timing": {"total": (time.time() - start_time) * 1000}}
                        print(f"📷 [PLANNER] Image metadata: shape={image.shape}, dtype={image.dtype}")
                    except Exception as img_error:
                        return {"status": "error", "error": f"Error: {str(img_error)}", "timing": {"total": (time.time() - start_time) * 1000}}
                elif isinstance(base_64_image, np.ndarray):
                    image = base_64_image
                    print(f"📷 [PLANNER] Image metadata: shape={image.shape}, dtype={image.dtype}")
                else:
                    return {"status": "error", "error": f"Unsupported: {type(base_64_image)}", "timing": {"total": (time.time() - start_time) * 1000}}

            try:
                import torch
                cuda_available = torch.cuda.is_available()
                print(f"[GPU DEBUG] torch.cuda.is_available(): {cuda_available}")
                if cuda_available:
                    print(f"[GPU DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
                    print(f"[GPU DEBUG] torch.cuda.current_device(): {torch.cuda.current_device()}")
                    print(f"[GPU DEBUG] torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

                setup_start = time.time()
                dest_id = destination_id
                target_place = place
                target_building = building
                target_floor = floor
                user_id = session_id
                session = self.get_session(user_id)

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

                if not all([dest_id, target_place, target_building, target_floor]):
                    return {"status": "error", "error": "Incomplete navigation context"}

                self.update_session(user_id, {"selected_dest_id": dest_id, "target_place": target_place, "target_building": target_building, "target_floor": target_floor, "unit": unit, "language": language})

                if refinement_queue is None:
                    session_refinement_queue = session.get("refinement_queue") or {}
                    if image is not None and hasattr(image, 'shape'):
                        queue_key = image.shape[:2]
                        if queue_key in session_refinement_queue:
                            refinement_queue = session_refinement_queue[queue_key]
                        else:
                            refinement_queue = {}
                    else:
                        refinement_queue = {}

                timing_data["setup"] = (time.time() - setup_start) * 1000
                print(f"⏱️ Setup: {timing_data['setup']:.2f}ms")

                if should_use_user_provided_coordinate:
                    print("⏭️ Skipping localization - using provided coordinates")
                    localization_start = time.time()
                    output = run_construct_mock_localization_output(x=x, y=y, angle=angle, place=target_place, building=target_building, floor=target_floor)
                    timing_data["localization"] = (time.time() - localization_start) * 1000
                    print(f"⏱️ Mock Localization: {timing_data['localization']:.2f}ms")
                else:
                    localization_start = time.time()
                    with self.tracer.start_as_current_span("localization_span"):
                        self.ensure_gpu_components_ready()
                        with self.tracer.start_as_current_span("load_maps_span"):
                            run_ensure_maps_loaded(
                                server=self,
                                place=target_place,
                                building=target_building,
                                floor=target_floor,
                                enable_multifloor=enable_multifloor,
                            )

                        map_key = (target_place, target_building)
                        localizer_to_use = self.selective_localizers.get(map_key)
                        if not localizer_to_use and target_floor:
                            floor_key = (target_place, target_building, target_floor)
                            localizer_to_use = self.selective_localizers.get(floor_key, self.localizer)
                        else:
                            localizer_to_use = localizer_to_use or self.localizer

                        queue_key = _get_queue_key_for_image_shape(image.shape)
                        is_cold_start = len(refinement_queue) == 0
                        print(f"🔍 Cold start: {is_cold_start}, refinement_queue size: {len(refinement_queue)}")
                        local_extractor = getattr(localizer_to_use, "local_extractor", None)
                        local_matcher = getattr(localizer_to_use, "local_matcher", None)
                        global_extractor = getattr(localizer_to_use, "global_extractor", None)
                        local_feature_model = getattr(
                            getattr(localizer_to_use, "config", None),
                            "local_feature_model",
                            getattr(self, "LOCAL_FEATURE_MODEL", "unknown"),
                        )
                        print(
                            "🧪 [LOCALIZER DEBUG] "
                            f"local_feature_model={local_feature_model}, "
                            f"local_extractor={type(local_extractor).__name__}(callable={callable(local_extractor)}), "
                            f"local_matcher={type(local_matcher).__name__}(callable={callable(local_matcher)}), "
                            f"global_extractor={type(global_extractor).__name__}(callable={callable(global_extractor)})"
                        )

                        if is_cold_start:
                            print("⏭️ Cold-start bootstrap disabled; running single-pass localization.")
                        output = localizer_to_use.localize(image, refinement_queue, top_k=top_k)
                        output["bootstrap_mode"] = "none"

                        output["map_scope"] = "building_level_multifloor" if enable_multifloor else "floor_locked"
                        output["queue_key"] = queue_key

                timing_data["localization"] = (time.time() - localization_start) * 1000
                print(f"⏱️ Localization: {timing_data['localization']:.2f}ms")
                print(f"📍 Localization result: floorplan_pose={output.get('floorplan_pose')}, map_key={output.get('best_map_key')}, map_scope={output.get('map_scope')}")

                if output is None or "floorplan_pose" not in output:
                    print("❌ Localization failed, no pose found.")
                    local_feature_model_name = getattr(self, "LOCAL_FEATURE_MODEL", "unknown")
                    if "localizer_to_use" in locals():
                        local_feature_model_name = getattr(
                            getattr(localizer_to_use, "config", None),
                            "local_feature_model",
                            local_feature_model_name,
                        )
                    failed_results = []
                    if isinstance(output, dict):
                        raw_results = output.get("results") or []
                        failed_results = [r for r in raw_results if isinstance(r, dict)]
                    max_inliers = max((r.get("inliers", 0) for r in failed_results), default=0)
                    print(
                        "❌ [PLANNER RESULT] "
                        f"status=error, "
                        f"stage={(output or {}).get('stage') if isinstance(output, dict) else 'no_output'}, "
                        f"reason={(output or {}).get('reason') if isinstance(output, dict) else 'no_output'}, "
                        f"best_map_key={(output or {}).get('best_map_key') if isinstance(output, dict) else None}, "
                        f"local_feature_model={local_feature_model_name}, "
                        f"top_candidates_count={len((output or {}).get('top_candidates', []) if isinstance(output, dict) else [])}, "
                        f"results_count={len(failed_results)}, "
                        f"max_inliers={max_inliers}"
                    )
                    if is_vlm_extraction_enabled:
                        try:
                            extracted_text = run_vlm_on_image(server=self, image=image)
                            return {"status": "error", "error": "Localization failed", "extracted_text": extracted_text, "timing": timing_data}
                        except Exception as vlm_error:
                            print(f"❌ Error during VLM fallback: {vlm_error}")
                            return {"status": "error", "error": "VLM failed", "vlm_error": str(vlm_error), "timing": timing_data}
                    return {"status": "error", "error": "Localization failed", "timing": timing_data}

                processing_start = time.time()
                floorplan_pose = output["floorplan_pose"]
                start_xy, start_heading = floorplan_pose["xy"], -floorplan_pose["ang"]
                source_key = output["best_map_key"]
                start_place, start_building, start_floor = source_key

                if image is not None and hasattr(image, 'shape'):
                    queue_key = image.shape[:2]
                    current_session_queue = session.get("refinement_queue") or {}
                    current_session_queue[queue_key] = output.get("refinement_queue", {})
                else:
                    current_session_queue = {}
                self.update_session(user_id, {"current_place": start_place, "current_building": start_building, "current_floor": start_floor, "floorplan_pose": floorplan_pose, "refinement_queue": current_session_queue})

                try:
                    dest_id_for_path = int(dest_id)
                except (ValueError, TypeError):
                    dest_id_for_path = dest_id

                timing_data["processing"] = (time.time() - processing_start) * 1000
                print(f"⏱️ Processing: {timing_data['processing']:.2f}ms")

                path_planning_start = time.time()
                with self.tracer.start_as_current_span("path_planning_span"):
                    result = self.nav.find_path(start_place, start_building, start_floor, start_xy, target_place, target_building, target_floor, dest_id_for_path)

                timing_data["path_planning"] = (time.time() - path_planning_start) * 1000
                print(f"⏱️ Path Planning: {timing_data['path_planning']:.2f}ms")

                if result is None or (isinstance(result, dict) and "error" in result):
                    return {"status": "error", "error": "Path planning failed", "timing": timing_data}

                command_generation_start = time.time()
                with self.tracer.start_as_current_span("command_generation_span"):
                    cmds = self.commander(
                        self.nav,
                        result,
                        initial_heading=start_heading,
                        unit=unit,
                        language=language,
                        turn_mode=turn_mode,
                    )

                timing_data["command_generation"] = (time.time() - command_generation_start) * 1000

                serialization_start = time.time()
                serialized_result = run_safe_serialize(result)
                serialized_cmds = run_safe_serialize(cmds)
                serialized_source_key = run_safe_serialize(source_key)
                serialized_floorplan_pose = run_safe_serialize(floorplan_pose)
                timing_data["serialization"] = (time.time() - serialization_start) * 1000
                print(f"⏱️ Serialization: {timing_data['serialization']:.2f}ms")

                timing_data["total"] = (time.time() - start_time) * 1000
                print(f"⏱️ Total Navigation Time: {timing_data['total']:.2f}ms")

                local_feature_model_name = getattr(self, "LOCAL_FEATURE_MODEL", "unknown")
                if "localizer_to_use" in locals():
                    local_feature_model_name = getattr(
                        getattr(localizer_to_use, "config", None),
                        "local_feature_model",
                        local_feature_model_name,
                    )
                localization_results = output.get("results") or []

                result = {
                    "status": "success",
                    "result": serialized_result,
                    "cmds": serialized_cmds,
                    "best_map_key": serialized_source_key,
                    "floorplan_pose": serialized_floorplan_pose,
                    "turn_mode": turn_mode,
                    "total_inliers": sum(
                        r.get("inliers", 0)
                        for r in localization_results
                        if isinstance(r, dict)
                    ),
                    "per_candidate_inliers": run_safe_serialize(
                        [
                            {
                                "ref": r.get("ref_image_name"),
                                "score": r.get("score"),
                                "inliers": r.get("inliers", 0),
                            }
                            for r in localization_results
                            if isinstance(r, dict)
                        ]
                    ),
                    "timings": output.get("timings"),
                    "top_candidates": run_safe_serialize(output.get("top_candidates")),
                    "local_feature_model": local_feature_model_name,
                    "navigation_info": {"start_location": f"{start_place}/{start_building}/{start_floor}", "destination": f"{target_place}/{target_building}/{target_floor}", "dest_id": dest_id, "unit": unit, "language": language},
                    "timing": timing_data,
                    "debug_info": {"map_scope": output.get("map_scope", "unknown"), "bootstrap_mode": output.get("bootstrap_mode", "none"), "bootstrap_passes": output.get("bootstrap_passes"), "queue_key": output.get("queue_key", "unknown"), "n_frames": output.get("n_frames"), "top_candidates_count": len(output.get("top_candidates", []))},
                }
                print(
                    "✅ [PLANNER RESULT] "
                    f"status={result.get('status')}, "
                    f"best_map_key={result.get('best_map_key')}, "
                    f"floorplan_pose={result.get('floorplan_pose')}, "
                    f"local_feature_model={result.get('local_feature_model')}, "
                    f"total_inliers={result.get('total_inliers')}, "
                    f"top_candidates_count={len(result.get('top_candidates') or [])}"
                )

                return run_convert_navigation_to_trajectory(result)

            except Exception as e:
                timing_data["total"] = (time.time() - start_time) * 1000
                print(f"❌ Error in planner: {str(e)}")
                import traceback
                traceback.print_exc()
                return {"status": "error", "error": str(e), "type": type(e).__name__, "timing": timing_data}
    else:
        print("📋 [PLANNER] Using NON-TRACED execution path")
        pass


def run_localize_user(
    self,
    session_id: str,
    base_64_image,
    place: str,
    building: str,
    floor: str,
    top_k: int = None,
    refinement_queue: dict = None,
    enable_multifloor: bool = True,
) -> Dict[str, Any]:
    """Localize user position without navigation planning."""
    import time
    import cv2
    import base64

    from ..server_methods.helpers import _get_queue_key_for_image_shape

    print(f"📋 [LOCALIZE_USER] Called with session_id={session_id}")
    print(f"📋 [LOCALIZE_USER] Params: place={place}, building={building}, floor={floor}, top_k={top_k}, enable_multifloor={enable_multifloor}")
    start_time = time.time()

    if base_64_image is None:
        return {"status": "error", "error": "No image provided", "timing": {"total": (time.time() - start_time) * 1000}}

    if isinstance(base_64_image, str):
        try:
            base64_string = base_64_image
            if "," in base64_string:
                base64_string = base64_string.split(",")[1]
            missing_padding = len(base64_string) % 4
            if missing_padding:
                base64_string += "=" * (4 - missing_padding)
            image_bytes = base64.b64decode(base64_string)
            print(f"Received base64 image string of length {len(base64_string)}")
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                return {"status": "error", "error": "Failed to decode image", "timing": {"total": (time.time() - start_time) * 1000}}
            print(f"📷 [LOCALIZE_USER] Image metadata: shape={image.shape}, dtype={image.dtype}")
        except Exception as img_error:
            return {"status": "error", "error": f"Error: {str(img_error)}", "timing": {"total": (time.time() - start_time) * 1000}}
    elif isinstance(base_64_image, np.ndarray):
        image = base_64_image
        print(f"📷 [LOCALIZE_USER] Image metadata: shape={image.shape}, dtype={image.dtype}")
    else:
        return {"status": "error", "error": f"Unsupported: {type(base_64_image)}", "timing": {"total": (time.time() - start_time) * 1000}}

    try:
        has_tracer = hasattr(self, "tracer") and self.tracer is not None

        if has_tracer:
            with self.tracer.start_as_current_span("localize_user_span") as span:
                span.set_attribute("unav.session_id", session_id)
                self.ensure_gpu_components_ready()
                run_ensure_maps_loaded(
                    server=self,
                    place=place,
                    building=building,
                    floor=floor,
                    enable_multifloor=enable_multifloor,
                )

                map_key = (place, building)
                localizer = self.selective_localizers.get(map_key)
                if not localizer and floor:
                    localizer = self.selective_localizers.get((place, building, floor), self.localizer)
                else:
                    localizer = localizer or self.localizer

                queue_key = _get_queue_key_for_image_shape(image.shape)

                if refinement_queue is None:
                    session = self.get_session(session_id)
                    session_refinement_queue = session.get("refinement_queue") or {}
                    queue_key_tuple = image.shape[:2]
                    if queue_key_tuple in session_refinement_queue:
                        refinement_queue = session_refinement_queue[queue_key_tuple]
                    else:
                        refinement_queue = {}

                is_cold_start = len(refinement_queue) == 0
                print(f"🔍 Cold start: {is_cold_start}, refinement_queue size: {len(refinement_queue)}")

                if is_cold_start:
                    bootstrap_outputs = []
                    empty_queue = refinement_queue.copy()
                    for bootstrap_pass in range(2):
                        print(f"🔄 Bootstrap pass {bootstrap_pass + 1}/2...")
                        bootstrap_output = localizer.localize(image, empty_queue, top_k=top_k)
                        if bootstrap_output and bootstrap_output.get("success"):
                            bootstrap_outputs.append(bootstrap_output)
                            best_map_key = bootstrap_output.get("best_map_key")
                            print(f"   ✅ Pass {bootstrap_pass + 1}: best_map_key={best_map_key}")

                    if len(bootstrap_outputs) >= 2:
                        xy_sum = [0.0, 0.0]
                        ang_sum = 0.0
                        for bo in bootstrap_outputs:
                            fp = bo.get("floorplan_pose", {})
                            xy = fp.get("xy", [0, 0])
                            xy_sum[0] += xy[0]
                            xy_sum[1] += xy[1]
                            ang_sum += fp.get("ang", 0)
                        avg_xy = [xy_sum[0] / len(bootstrap_outputs), xy_sum[1] / len(bootstrap_outputs)]
                        avg_ang = ang_sum / len(bootstrap_outputs)
                        output = bootstrap_outputs[-1].copy()
                        output["floorplan_pose"] = {"xy": avg_xy, "ang": avg_ang}
                        output["bootstrap_mode"] = "mean_all_passes"
                    elif bootstrap_outputs:
                        output = bootstrap_outputs[-1]
                        output["bootstrap_mode"] = "single_pass"
                    else:
                        output = localizer.localize(image, refinement_queue, top_k=top_k)
                        output["bootstrap_mode"] = "none"
                else:
                    output = localizer.localize(image, refinement_queue, top_k=top_k)
                    output["bootstrap_mode"] = "none"

                output["map_scope"] = "building_level_multifloor" if enable_multifloor else "floor_locked"
                output["queue_key"] = queue_key
        else:
            self.ensure_gpu_components_ready()
            run_ensure_maps_loaded(
                server=self,
                place=place,
                building=building,
                floor=floor,
                enable_multifloor=enable_multifloor,
            )

            map_key = (place, building)
            localizer = self.selective_localizers.get(map_key)
            if not localizer and floor:
                localizer = self.selective_localizers.get((place, building, floor), self.localizer)
            else:
                localizer = localizer or self.localizer

            output = localizer.localize(image, refinement_queue or {}, top_k=top_k)
            output["bootstrap_mode"] = "none"
            output["map_scope"] = "building_level_multifloor" if enable_multifloor else "floor_locked"

        if output is None or "floorplan_pose" not in output:
            return {"status": "error", "error": "Localization failed", "timing": {"total": (time.time() - start_time) * 1000}}

        floorplan_pose = output["floorplan_pose"]
        best_map_key = output["best_map_key"]

        queue_key_tuple = image.shape[:2]
        current_session_queue = session.get("refinement_queue") or {}
        current_session_queue[queue_key_tuple] = output.get("refinement_queue", {})
        self.update_session(session_id, {"current_place": best_map_key[0], "current_building": best_map_key[1], "current_floor": best_map_key[2], "floorplan_pose": floorplan_pose, "refinement_queue": current_session_queue})

        timing_data = {"total": (time.time() - start_time) * 1000}
        print(f"⏱️ Localization total: {timing_data['total']:.2f}ms")
        print(f"📍 Result: floorplan_pose={floorplan_pose}, best_map_key={best_map_key}")

        return {
            "status": "success",
            "floorplan_pose": run_safe_serialize(floorplan_pose),
            "best_map_key": run_safe_serialize(best_map_key),
            "timing": timing_data,
            "debug_info": {"map_scope": output.get("map_scope"), "bootstrap_mode": output.get("bootstrap_mode"), "queue_key": output.get("queue_key")},
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e), "type": type(e).__name__, "timing": {"total": (time.time() - start_time) * 1000}}
