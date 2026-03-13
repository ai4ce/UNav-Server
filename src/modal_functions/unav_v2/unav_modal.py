from modal import method, gpu, enter
import json
import traceback
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional

from .deploy_config import get_scaledown_window, get_gpu_config, get_memory_mb
from .modal_config import app, unav_image, volume, gemini_secret, middleware_secret
from .destinations_service import get_destinations_list_impl
from .server_methods.helpers import (
    _get_queue_key_for_image_shape,
    _get_refinement_queue_for_map,
    _update_refinement_queue,
)
from .logic import (
    run_planner,
    run_localize_user,
    run_init_middleware,
    run_init_cpu_components,
    run_init_gpu_components,
    run_monkey_patch_localizer_methods,
    run_monkey_patch_pose_refinement,
    run_monkey_patch_feature_extractors,
    run_monkey_patch_matching_and_ransac,
)


@app.cls(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
    gpu=get_gpu_config(),
    enable_memory_snapshot=False,
    memory=get_memory_mb(),
    timeout=600,
    scaledown_window=get_scaledown_window(),
    secrets=[gemini_secret, middleware_secret],
)
class UnavServer:
    # Initialize session storage for user contexts
    user_sessions: Dict[str, Any] = {}
    tracer: Optional[Any] = None
    _middleware_init_pending: bool = False

    def _gpu_available(self) -> bool:
        """Delegated to logic.init"""
        return self._check_gpu_available()

    def _configure_middleware_tracing(self):
        """Delegated to logic.init"""
        self._setup_middleware_tracing()

    @enter(snap=False)
    def initialize_middleware(self):
        """Delegated to logic.init"""
        run_init_middleware(self)

    @enter(snap=False)
    def initialize_cpu_components(self):
        """Delegated to logic.init"""
        run_init_cpu_components(self)

    @enter(snap=False)
    def initialize_gpu_components(self):
        """Delegated to logic.init"""
        run_init_gpu_components(self)

    def _monkey_patch_localizer_methods(self, localizer, method_names=None):
        """Delegated to logic.init"""
        run_monkey_patch_localizer_methods(self, localizer, method_names)

    def _monkey_patch_pose_refinement(self):
        """Delegated to logic.init"""
        run_monkey_patch_pose_refinement(self)

    def _monkey_patch_feature_extractors(self):
        """Delegated to logic.init"""
        run_monkey_patch_feature_extractors(self)

    def _monkey_patch_matching_and_ransac(self):
        """Delegated to logic.init"""
        run_monkey_patch_matching_and_ransac(self)

    def _check_gpu_available(self):
        return True

    def _setup_middleware_tracing(self):
        pass

    # Placeholder methods - actual implementations in logic/init.py
    def get_places(self, target_place=None, target_building=None, target_floor=None, enable_multifloor=False):
        return {}

    def _get_fallback_places(self):
        return {"New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"]}}

    def get_places(self, target_place=None, target_building=None, target_floor=None, enable_multifloor=False):
        """Get places - actual implementation below"""
        return {}
        superpoint_patched = False
        import_paths_to_try = [
            ("unav.core.feature.local_extractor", "Superpoint"),
            ("unav.core.third_party.SuperPoint_SuperGlue.base_model", "Superpoint"),
        ]

        for module_path, class_name in import_paths_to_try:
            if superpoint_patched:
                break
            try:
                import importlib

                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    Superpoint = getattr(module, class_name)
                    print(
                        f"🔍 [DEBUG] Found {class_name} in {module_path}: {Superpoint}"
                    )
                    print(
                        f"🔍 [DEBUG] Superpoint has extract_local_features: {hasattr(Superpoint, 'extract_local_features')}"
                    )
                    print(
                        f"🔍 [DEBUG] Superpoint already patched: {getattr(Superpoint, '__mw_patched__', False)}"
                    )

                    if not getattr(Superpoint, "__mw_patched__", False):
                        original_extract_local = Superpoint.extract_local_features

                        @functools.wraps(original_extract_local)
                        def traced_extract_local(self, image0):
                            # Simply wrap the original method - no code rewriting
                            with tracer.start_as_current_span(
                                "unav.local_extractor.extract_local_features"
                            ):
                                return original_extract_local(self, image0)

                        Superpoint.extract_local_features = traced_extract_local
                        Superpoint.__mw_patched__ = True
                        superpoint_patched = True
                        print(
                            f"🔧 ✅ Patched Superpoint.extract_local_features from {module_path}"
                        )
                    else:
                        print("⚠️ Superpoint already patched, skipping")
                        superpoint_patched = True
                else:
                    print(f"🔍 [DEBUG] {class_name} not found in {module_path}")
            except ImportError as e:
                print(f"🔍 [DEBUG] Could not import {module_path}: {e}")
            except Exception as e:
                print(f"⚠️ Error patching Superpoint from {module_path}: {e}")
                import traceback

                traceback.print_exc()

        if not superpoint_patched:
            print("⚠️ Could not find Superpoint class to patch in any known location")

        # Patch LightGlue._forward to trace internal matching operations
        self._monkey_patch_lightglue(tracer)

    def _monkey_patch_lightglue(self, tracer):
        """
        Patch LightGlue's internal modules to add granular tracing.
        We patch the forward methods of:
        - Transformer (self-attention)
        - CrossTransformer (cross-attention)
        - MatchAssignment (match scoring)
        - LearnableFourierPositionalEncoding (position encoding)
        And wrap the main _forward method.
        """
        import functools

        lightglue_patched = False
        import_paths_to_try = [
            "unav.core.third_party.LightGlue.lightglue",
            "unav.core.feature.lightglue",
        ]

        for module_path in import_paths_to_try:
            if lightglue_patched:
                break
            try:
                import importlib

                module = importlib.import_module(module_path)

                # Check if LightGlue class exists
                if not hasattr(module, "LightGlue"):
                    print(f"🔍 [DEBUG] LightGlue not found in {module_path}")
                    continue

                LightGlue = getattr(module, "LightGlue")
                print(f"🔍 [DEBUG] Found LightGlue in {module_path}")

                if getattr(LightGlue, "__mw_patched__", False):
                    print("⚠️ LightGlue already patched, skipping")
                    lightglue_patched = True
                    continue

                # 1. Patch Transformer.forward (self-attention)
                if hasattr(module, "Transformer"):
                    Transformer = getattr(module, "Transformer")
                    if not getattr(Transformer, "__mw_patched__", False):
                        original_transformer_forward = Transformer.forward

                        @functools.wraps(original_transformer_forward)
                        def traced_transformer_forward(
                            self, x0, x1, encoding0=None, encoding1=None
                        ):
                            with tracer.start_as_current_span(
                                "unav.local_matcher.self_attention"
                            ):
                                return original_transformer_forward(
                                    self, x0, x1, encoding0, encoding1
                                )

                        Transformer.forward = traced_transformer_forward
                        Transformer.__mw_patched__ = True
                        print("🔧 ✅ Patched Transformer.forward (self-attention)")

                # 2. Patch CrossTransformer.forward (cross-attention)
                if hasattr(module, "CrossTransformer"):
                    CrossTransformer = getattr(module, "CrossTransformer")
                    if not getattr(CrossTransformer, "__mw_patched__", False):
                        original_cross_forward = CrossTransformer.forward

                        @functools.wraps(original_cross_forward)
                        def traced_cross_forward(self, x0, x1):
                            with tracer.start_as_current_span(
                                "unav.local_matcher.cross_attention"
                            ):
                                return original_cross_forward(self, x0, x1)

                        CrossTransformer.forward = traced_cross_forward
                        CrossTransformer.__mw_patched__ = True
                        print(
                            "🔧 ✅ Patched CrossTransformer.forward (cross-attention)"
                        )

                # 3. Patch MatchAssignment.forward (match scoring)
                if hasattr(module, "MatchAssignment"):
                    MatchAssignment = getattr(module, "MatchAssignment")
                    if not getattr(MatchAssignment, "__mw_patched__", False):
                        original_match_forward = MatchAssignment.forward

                        @functools.wraps(original_match_forward)
                        def traced_match_forward(self, desc0, desc1):
                            with tracer.start_as_current_span(
                                "unav.local_matcher.match_assignment"
                            ):
                                return original_match_forward(self, desc0, desc1)

                        MatchAssignment.forward = traced_match_forward
                        MatchAssignment.__mw_patched__ = True
                        print("🔧 ✅ Patched MatchAssignment.forward")

                # 4. Patch LearnableFourierPositionalEncoding.forward (position encoding)
                if hasattr(module, "LearnableFourierPositionalEncoding"):
                    PosEnc = getattr(module, "LearnableFourierPositionalEncoding")
                    if not getattr(PosEnc, "__mw_patched__", False):
                        original_posenc_forward = PosEnc.forward

                        @functools.wraps(original_posenc_forward)
                        def traced_posenc_forward(self, x):
                            with tracer.start_as_current_span(
                                "unav.local_matcher.position_encoding"
                            ):
                                return original_posenc_forward(self, x)

                        PosEnc.forward = traced_posenc_forward
                        PosEnc.__mw_patched__ = True
                        print(
                            "🔧 ✅ Patched LearnableFourierPositionalEncoding.forward"
                        )

                # 5. Patch filter_matches function
                if hasattr(module, "filter_matches"):
                    original_filter = getattr(module, "filter_matches")
                    if not getattr(original_filter, "__mw_patched__", False):

                        @functools.wraps(original_filter)
                        def traced_filter_matches(scores, th):
                            with tracer.start_as_current_span(
                                "unav.local_matcher.filter_matches"
                            ):
                                return original_filter(scores, th)

                        traced_filter_matches.__mw_patched__ = True
                        setattr(module, "filter_matches", traced_filter_matches)
                        print("🔧 ✅ Patched filter_matches")

                # 6. Wrap the main LightGlue._forward method
                original_lightglue_forward = LightGlue._forward

                @functools.wraps(original_lightglue_forward)
                def traced_lightglue_forward(self, data):
                    with tracer.start_as_current_span("unav.local_matcher._forward"):
                        return original_lightglue_forward(self, data)

                LightGlue._forward = traced_lightglue_forward
                LightGlue.__mw_patched__ = True
                lightglue_patched = True
                print(f"🔧 ✅ Patched LightGlue._forward from {module_path}")

            except ImportError as e:
                print(f"🔍 [DEBUG] Could not import {module_path}: {e}")
            except Exception as e:
                print(f"⚠️ Error patching LightGlue from {module_path}: {e}")
                import traceback

                traceback.print_exc()

        if not lightglue_patched:
            print("⚠️ Could not find LightGlue class to patch in any known location")

    def _monkey_patch_matching_and_ransac(self):
        """
        Patch the child functions of batch_local_matching_and_ransac.
        The function is in unav.localizer.tools.matcher, and internally calls
        match_query_to_database and ransac_filter from unav.core.feature_filter.
        """
        if not hasattr(self, "tracer") or not self.tracer:
            return

        import functools

        tracer = self.tracer

        # Patch unav.core.feature_filter functions at MODULE level
        # These are called by the matcher module's batch_local_matching_and_ransac
        try:
            from unav.core import feature_filter

            print(f"🔍 [DEBUG] Attempting to patch unav.core.feature_filter")
            print(f"🔍 [DEBUG] feature_filter module: {feature_filter}")
            print(
                f"🔍 [DEBUG] match_query_to_database: {hasattr(feature_filter, 'match_query_to_database')}"
            )
            print(
                f"🔍 [DEBUG] ransac_filter: {hasattr(feature_filter, 'ransac_filter')}"
            )

            # Check if already patched
            if not getattr(feature_filter, "__mw_patched__", False):
                # Patch match_query_to_database
                if hasattr(feature_filter, "match_query_to_database"):
                    original_match = feature_filter.match_query_to_database

                    @functools.wraps(original_match)
                    def traced_match(*args, **kwargs):
                        print("🔍 [TRACE] ✅ Entering match_query_to_database")
                        with tracer.start_as_current_span(
                            "unav.match_query_to_database"
                        ):
                            result = original_match(*args, **kwargs)
                            print(
                                f"🔍 [TRACE] ✅ Exiting match_query_to_database, returned {len(result[0]) if result[0] else 0} matches"
                            )
                            return result

                    feature_filter.match_query_to_database = traced_match
                    print("🔧 ✅ Patched match_query_to_database")
                else:
                    print("⚠️ match_query_to_database not found in feature_filter")

                # Patch ransac_filter
                if hasattr(feature_filter, "ransac_filter"):
                    original_ransac = feature_filter.ransac_filter

                    @functools.wraps(original_ransac)
                    def traced_ransac(*args, **kwargs):
                        print("🔍 [TRACE] ✅ Entering ransac_filter")
                        with tracer.start_as_current_span("unav.ransac_filter"):
                            result = original_ransac(*args, **kwargs)
                            print("🔍 [TRACE] ✅ Exiting ransac_filter")
                            return result

                    feature_filter.ransac_filter = traced_ransac
                    print("🔧 ✅ Patched ransac_filter")
                else:
                    print("⚠️ ransac_filter not found in feature_filter")

                feature_filter.__mw_patched__ = True
                print("🔧 Patched feature_filter module successfully")
            else:
                print("⚠️ feature_filter already patched, skipping")
        except ImportError as e:
            print(f"⚠️ Could not import unav.core.feature_filter: {e}")
            import traceback

            traceback.print_exc()
        except Exception as e:
            print(f"⚠️ Failed to patch feature_filter: {e}")
            import traceback

            traceback.print_exc()

    def get_places(
        self,
        target_place: Optional[str] = None,
        target_building: Optional[str] = None,
        target_floor: Optional[str] = None,
        enable_multifloor: bool = False,
    ):
        """Get available places configuration, optionally filtering to a specific floor"""
        try:
            print("📁 Fetching places from data directory...")

            # Define folders to skip at all levels
            SKIP_FOLDERS = {
                "features",
                "colmap_map",
                ".ipynb_checkpoints",
                "parameters",
            }

            def should_skip_folder(folder_name):
                """Check if folder should be skipped based on name patterns"""
                return (
                    folder_name in SKIP_FOLDERS
                    or "_old" in folder_name.lower()
                    or folder_name.endswith("_old")
                )

            places: Dict[str, Dict[str, List[str]]] = {}

            data_root = getattr(self, "DATA_ROOT", "/root/UNav-IO/data")

            # Get all place directories (depth 1 under data/)
            if os.path.exists(data_root):
                for place_name in os.listdir(data_root):
                    place_path = os.path.join(data_root, place_name)
                    if os.path.isdir(place_path) and not should_skip_folder(place_name):
                        if target_place and place_name != target_place:
                            continue
                        places[place_name] = {}
                        print(f"  ✓ Found place: {place_name}")

                        # Get buildings for this place (depth 2)
                        for building_name in os.listdir(place_path):
                            building_path = os.path.join(place_path, building_name)
                            if os.path.isdir(building_path) and not should_skip_folder(
                                building_name
                            ):
                                if target_building and building_name != target_building:
                                    continue
                                floors = []

                                # Get floors for this building (depth 3)
                                for floor_name in os.listdir(building_path):
                                    floor_path = os.path.join(building_path, floor_name)
                                    if os.path.isdir(
                                        floor_path
                                    ) and not should_skip_folder(floor_name):
                                        # Skip specific problematic floor
                                        if (
                                            place_name == "New_York_City"
                                            and building_name == "LOH"
                                            and floor_name == "9_floor"
                                        ):
                                            print(
                                                f"    ⚠️ Skipping {building_name}/{floor_name}: explicitly excluded"
                                            )
                                            continue
                                        
                                        if (
                                            not enable_multifloor
                                            and target_floor
                                            and floor_name != target_floor
                                        ):
                                            continue
                                        
                                        # Validate that required navigation files exist
                                        boundaries_file = os.path.join(floor_path, "boundaries.json")
                                        if not os.path.exists(boundaries_file):
                                            print(
                                                f"    ⚠️ Skipping {building_name}/{floor_name}: missing boundaries.json"
                                            )
                                            continue
                                        
                                        floors.append(floor_name)

                                if floors:  # Only add building if it has floors
                                    places[place_name][building_name] = floors
                                    print(
                                        f"    ✓ Building: {building_name} with floors: {floors}"
                                    )

                # Remove places that have no buildings
                places = {k: v for k, v in places.items() if v}

                if not enable_multifloor and target_floor:
                    # Ensure we only return the specific floor requested
                    for p_name in list(places.keys()):
                        buildings = places[p_name]
                        for b_name in list(buildings.keys()):
                            filtered_floors = [
                                f_name
                                for f_name in buildings[b_name]
                                if f_name == target_floor
                            ]
                            if filtered_floors:
                                buildings[b_name] = filtered_floors
                            else:
                                del buildings[b_name]
                        if not buildings:
                            del places[p_name]

                print(f"✅ Found {len(places)} places with buildings and floors")
                return places
            else:
                print(f"⚠️ Data root {data_root} does not exist, using fallback")
                return self._get_fallback_places()

        except Exception as e:
            print(f"❌ Error fetching places: {e}, using fallback")
            return self._get_fallback_places()

    def _get_fallback_places(self):
        """Fallback hardcoded places configuration"""
        return {
            "New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"]},
            "New_York_University": {"Langone": ["15_floor", "16_floor", "17_floor"]},
            "Mahidol_University": {"Jubilee": ["fl1", "fl2", "fl3"]},
        }

    def ensure_maps_loaded(
        self,
        place: str,
        building: str = None,
        floor: str = None,
        enable_multifloor: bool = False,
    ):
        """
        Ensure that maps for a specific place/building are loaded.
        When building is specified, loads all floors for that building.
        Creates selective localizer instances for true lazy loading.
        """
        if building:
            if enable_multifloor or not floor:
                map_key = (place, building)
            else:
                map_key = (place, building, floor)
        else:
            map_key = place

        if map_key in self.maps_loaded:
            return  # Already loaded

        print(f"🔄 [Phase 4] Creating selective localizer for: {map_key}")

        # Create selective places config with only the requested location
        if building:
            selective_places = self.get_places(
                target_place=place,
                target_building=building,
                target_floor=floor,
                enable_multifloor=enable_multifloor,
            )
        else:
            selective_places = self.get_places(target_place=place)

        if not selective_places:
            print(
                "⚠️ No matching places found for selective load; skipping localizer creation"
            )
            return

        # Create selective config and localizer
        from unav.config import UNavConfig

        selective_config = UNavConfig(
            data_final_root=self.DATA_ROOT,
            places=selective_places,
            global_descriptor_model=self.FEATURE_MODEL,
            local_feature_model=self.LOCAL_FEATURE_MODEL,
        )

        from unav.localizer.localizer import UNavLocalizer
        import time

        selective_localizer = UNavLocalizer(selective_config.localizer_config)

        # # Optionally patch the selective localizer too
        if hasattr(self, "tracer") and self.tracer:
            try:
                self._monkey_patch_localizer_methods(selective_localizer)
                # Note: feature extractors are already patched at module level, no need to patch per instance
            except Exception as e:
                print(f"⚠️ Failed to patch selective localizer: {e}")

        # Load maps and features with tracing if available
        if hasattr(self, "tracer") and self.tracer:
            with self.tracer.start_as_current_span(
                "load_maps_and_features_span"
            ) as load_span:
                load_span.add_event("Starting map and feature loading")
                load_span.set_attribute("map_key", str(map_key))
                load_span.set_attribute("selective_places", str(selective_places))

                start_load_time = time.time()
                selective_localizer.load_maps_and_features()
                load_duration = time.time() - start_load_time

                load_span.set_attribute("load_duration_seconds", load_duration)
                load_span.add_event("Map and feature loading completed")
        else:
            print(f"⏱️ Starting load_maps_and_features for: {map_key}")
            start_load_time = time.time()
            selective_localizer.load_maps_and_features()
            load_duration = time.time() - start_load_time
            print(f"⏱️ Completed load_maps_and_features in {load_duration:.2f} seconds")

        # Cache the selective localizer
        self.selective_localizers[map_key] = selective_localizer
        self.maps_loaded.add(map_key)
        print(f"✅ Selective localizer created and maps loaded for: {map_key}")

    def ensure_gpu_components_ready(self):
        """
        Ensure GPU components are initialized before processing requests.
        This is called by methods that need the localizer.
        """
        if not hasattr(self, "gpu_components_initialized"):
            print("🔧 GPU components not ready, initializing now...")
            self.initialize_gpu_components()

    def get_session(self, user_id: str) -> dict:
        """Get or create user session"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        return self.user_sessions[user_id]

    def update_session(self, user_id: str, updates: dict):
        """Update user session with new data"""
        session = self.get_session(user_id)
        session.update(updates)

    def _construct_mock_localization_output(
        self,
        x: float,
        y: float,
        angle: float,
        place: str,
        building: str,
        floor: str,
    ) -> dict:
        """
        Construct a mock localization output from user-provided coordinates.
        This allows skipping the actual localization phase when coordinates are known.
        
        Args:
            x: X coordinate on the floor plan
            y: Y coordinate on the floor plan
            angle: Heading angle (direction user is facing)
            place: Place name
            building: Building name
            floor: Floor name
            
        Returns:
            dict: Mock localization output matching the structure from localizer.localize()
        """
        return {
            "floorplan_pose": {
                "xy": [x, y],
                "ang": angle
            },
            "best_map_key": (place, building, floor),
            "refinement_queue": {}  # Empty since we're not doing actual localization
        }

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
        import logging

        """
        Initializes and starts the serverless instance.
    
        This function helps in reducing the server response time for actual requests by pre-warming the server. 
        By starting the server in advance, it ensures that the server is ready to handle incoming requests immediately, 
        thus avoiding the latency associated with a cold start.
        """
        print("UNAV Container started...")
        logging.info("UNav server initialization requested")

        # Create span for server start if tracer available
        if hasattr(self, "tracer") and self.tracer:
            with self.tracer.start_as_current_span("start_server_span") as span:
                response = {"status": "success", "message": "Server started."}
                logging.info("Server warmup completed successfully")
                return json.dumps(response)
        else:
            response = {"status": "success", "message": "Server started."}
            return json.dumps(response)

    @method()
    def get_destinations_list(
        self,
        floor="6_floor",
        place="New_York_City",
        building="LightHouse",
        enable_multifloor: bool = False,
    ):
        """
        Get destinations for a specific place, building, and floor.
        Loads places on demand for fast startup.
        """
        return get_destinations_list_impl(
            server=self,
            floor=floor,
            place=place,
            building=building,
            enable_multifloor=enable_multifloor,
        )

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
        enable_multifloor: bool = True,
        should_use_user_provided_coordinate: bool = False,
        x: float = None,
        y: float = None,
        angle: float = None,
    ):
        """Full localization and navigation pipeline."""
        return run_planner(
            self,
            session_id=session_id,
            base_64_image=base_64_image,
            destination_id=destination_id,
            place=place,
            building=building,
            floor=floor,
            top_k=top_k,
            unit=unit,
            language=language,
            refinement_queue=refinement_queue,
            is_vlm_extraction_enabled=is_vlm_extraction_enabled,
            enable_multifloor=enable_multifloor,
            should_use_user_provided_coordinate=should_use_user_provided_coordinate,
            x=x,
            y=y,
            angle=angle,
        )

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
        enable_multifloor: bool = True,
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
        import cv2
        import base64
        
        print(
            f"📋 [LOCALIZE_USER] Called with session_id={session_id}, "
            f"place={place}, building={building}, floor={floor}, "
            f"enable_multifloor={enable_multifloor}, top_k={top_k}"
        )
        
        start_time = time.time()
        timing_data = {}
        
        # Validate and convert image input
        if base_64_image is None:
            return {
                "status": "error",
                "error": "No image provided. base_64_image parameter is required.",
                "timing": {"total": (time.time() - start_time) * 1000},
            }
        
        # Convert base64 string to BGR numpy array
        if isinstance(base_64_image, str):
            try:
                base64_string = base_64_image
                
                # Remove data URL prefix if present
                if "," in base64_string:
                    base64_string = base64_string.split(",")[1]
                
                # Add padding if necessary
                missing_padding = len(base64_string) % 4
                if missing_padding:
                    base64_string += "=" * (4 - missing_padding)
                
                # Decode and convert to image
                image_bytes = base64.b64decode(base64_string)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
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
            image = base_64_image
        else:
            return {
                "status": "error",
                "error": f"Unsupported image format. Expected base64 string or numpy array, got {type(base_64_image)}",
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
            
            queue_key = _get_queue_key_for_image_shape(image.shape)
            is_cold_start = len(refinement_queue) == 0

            if is_cold_start:
                print(f"🔄 Cold-start detected, running stabilization passes...")
                bootstrap_outputs = []
                empty_queue = refinement_queue.copy()

                for bootstrap_pass in range(2):
                    bootstrap_output = localizer_to_use.localize(
                        image, empty_queue, top_k=top_k
                    )
                    if bootstrap_output and bootstrap_output.get("success"):
                        bootstrap_outputs.append(bootstrap_output)
                        best_map_key = bootstrap_output.get("best_map_key")
                        fp = bootstrap_output.get("floorplan_pose", {})
                        xy = fp.get("xy", [0, 0])
                        ang = fp.get("ang", 0)
                        print(f"  📍 Pass {bootstrap_pass + 1}: xy={xy}, ang={ang:.2f}°, map={best_map_key}")
                        new_queue = bootstrap_output.get("refinement_queue", {})
                        if best_map_key and new_queue:
                            empty_queue = _update_refinement_queue(
                                empty_queue, best_map_key, queue_key,
                                new_queue.get(best_map_key, {}).get(queue_key, {"pairs": [], "initial_poses": [], "pps": []})
                            )

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
                    print(f"  ➡️  Averaged {len(bootstrap_outputs)} passes: xy={avg_xy}, ang={avg_ang:.2f}°")
                    output = bootstrap_outputs[-1].copy()
                    output["floorplan_pose"] = {
                        "xy": avg_xy,
                        "ang": avg_ang
                    }
                    output["bootstrap_mode"] = "mean_all_passes"
                    output["bootstrap_passes"] = len(bootstrap_outputs)
                    print(f"✅ Cold-start stabilization complete")
                elif bootstrap_outputs:
                    output = bootstrap_outputs[-1]
                    output["bootstrap_mode"] = "single_pass"
                    print(f"⚠️ Only 1 pass succeeded, using single pass result")
                else:
                    output = localizer_to_use.localize(
                        image, refinement_queue, top_k=top_k
                    )
                    output["bootstrap_mode"] = "none"
                    print(f"❌ All stabilization passes failed, using fallback")
            else:
                output = localizer_to_use.localize(image, refinement_queue, top_k=top_k)
                output["bootstrap_mode"] = "none"

            output["map_scope"] = "building_level_multifloor" if enable_multifloor else "floor_locked"
            output["queue_key"] = queue_key
            
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
                "debug_info": {
                    "map_scope": output.get("map_scope", "unknown"),
                    "bootstrap_mode": output.get("bootstrap_mode", "none"),
                    "bootstrap_passes": output.get("bootstrap_passes"),
                    "queue_key": output.get("queue_key", "unknown"),
                    "n_frames": output.get("n_frames"),
                    "top_candidates_count": len(output.get("top_candidates", [])),
                },
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
                start_place, start_building, start_floor = best_map_key[0], best_map_key[1], best_map_key[2]
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
                target_place, target_building, floor=target_floor,
                enable_multifloor=True
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
            
            timing_data["command_generation"] = (time.time() - command_generation_start) * 1000
            
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
            with self.tracer.start_as_current_span(
                "vlm_text_extraction_span"
            ) as vlm_span:
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
                            types.Part.from_bytes(
                                data=image_bytes, mime_type="image/jpeg"
                            ),
                        ],
                    )

                    # Extract the text response
                    extracted_text = (
                        response.text if response.text else "No text extracted"
                    )

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
