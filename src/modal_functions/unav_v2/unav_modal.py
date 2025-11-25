from modal import method, gpu, enter
import json
import traceback
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional

from modal_config import app, unav_image, volume, gemini_secret, middleware_secret


@app.cls(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
    gpu=["T4"],
    enable_memory_snapshot=False,
    memory=73728,
    secrets=[gemini_secret, middleware_secret],
)
class UnavServer:
    # Initialize session storage for user contexts
    user_sessions: Dict[str, Any] = {}
    tracer: Optional[Any] = None
    _middleware_init_pending: bool = False

    def _gpu_available(self) -> bool:
        """Utility to detect whether CUDA GPUs are currently accessible."""
        try:
            import torch

            available = torch.cuda.is_available()
            print(f"[GPU CHECK] torch.cuda.is_available(): {available}")
            return available
        except Exception as exc:
            print(f"[GPU CHECK] Unable to determine GPU availability: {exc}")
            return False

    def _configure_middleware_tracing(self):
        print("🔧 [CONFIGURE] Starting Middleware.io configuration...")
        from middleware import mw_tracker, MWOptions
        from opentelemetry import trace
        import os

        api_key = os.environ.get("MW_API_KEY")
        target = os.environ.get("MW_TARGET")

        if not api_key or not target:
            print(
                "⚠️ Warning: MW_API_KEY and MW_TARGET not set. Skipping middleware initialization."
            )
            self.tracer = None
            return

        try:
            mw_tracker(
                MWOptions(
                    access_token=api_key,
                    target=target,
                    service_name="UNav-Server",
                    console_exporter=False,
                    log_level="INFO",
                    collect_profiling=True,
                    collect_traces=True,
                    collect_metrics=True,
                )
            )

            self.tracer = trace.get_tracer(__name__)
            print("✅ Middleware.io initialized successfully")
            print(f"✅ [CONFIGURE] Tracer created: {type(self.tracer).__name__}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize Middleware.io: {e}")
            self.tracer = None
            print(f"⚠️ [CONFIGURE] Tracer set to None due to error")

    @enter(snap=False)
    def initialize_middleware(self):
        """
        Initialize Middleware.io tracking for profiling and telemetry.
        """
        print("🔧 [Phase 0] Initializing Middleware.io...")
        print(
            f"🔧 [Phase 0] Current tracer state: {getattr(self, 'tracer', 'NOT_SET')}"
        )
        print(
            f"🔧 [Phase 0] Middleware init pending: {getattr(self, '_middleware_init_pending', 'NOT_SET')}"
        )

        if not self._gpu_available():
            print(
                "⏸️ GPU not yet available; deferring Middleware.io initialization until a GPU-backed container is ready."
            )
            self._middleware_init_pending = True
            self.tracer = None
            print(f"🔧 [Phase 0] Set _middleware_init_pending=True, tracer=None")
            return

        self._middleware_init_pending = False
        print("✅ GPU available; proceeding with Middleware.io initialization")
        self._configure_middleware_tracing()

    @enter(snap=False)
    def initialize_cpu_components(self):
        """
        Initialize CPU-only components that can be safely snapshotted.
        This includes configuration, data loading, and navigation setup.
        """
        print("🚀 [Phase 1] Initializing CPU components for snapshotting...")

        from unav.config import UNavConfig
        from unav.navigator.multifloor import FacilityNavigator
        from unav.navigator.commander import commands_from_result

        # Configuration constants
        self.DATA_ROOT = "/root/UNav-IO/data"
        self.FEATURE_MODEL = "DinoV2Salad"
        self.LOCAL_FEATURE_MODEL = "superpoint+lightglue"
        self.PLACES = self.get_places()  # Load all places but defer map loading

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

        print("🧭 Initializing FacilityNavigator (CPU-only)...")
        self.nav = FacilityNavigator(self.navigator_config)
        print("✅ FacilityNavigator initialized successfully")

        # Store commander function for navigation
        self.commander = commands_from_result

        # Initialize loaded places tracking (all places are now in config, but maps not loaded)
        self.maps_loaded = set()

        # Cache for selective localizers (key: (place, building, floor) -> localizer instance)
        self.selective_localizers = {}

        # Set flag to indicate CPU components are ready
        self.cpu_components_initialized = True
        print("📸 CPU components ready for snapshotting!")

    @enter(snap=False)
    def initialize_gpu_components(self):
        """
        Initialize GPU-dependent components that cannot be snapshotted.
        This must run after snapshot restoration on GPU-enabled containers.
        """
        print("🚀 [Phase 2] Initializing GPU components after snapshot restoration...")

        # --- GPU DEBUG INFO ---
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            print(f"[GPU DEBUG] torch.cuda.is_available(): {cuda_available}")

            if not cuda_available:
                print(
                    "[GPU ERROR] CUDA not available! This will cause model loading to fail."
                )
                print(
                    "[GPU ERROR] Modal should have allocated a GPU. Raising exception to trigger retry..."
                )
                raise RuntimeError(
                    "GPU not available when required. Modal will retry with GPU allocation."
                )

            print(f"[GPU DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
            print(
                f"[GPU DEBUG] torch.cuda.current_device(): {torch.cuda.current_device()}"
            )
            print(
                f"[GPU DEBUG] torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}"
            )
        except Exception as gpu_debug_exc:
            print(f"[GPU DEBUG] Error printing GPU info: {gpu_debug_exc}")
            # If it's our intentional GPU check failure, re-raise it
            if "GPU not available when required" in str(gpu_debug_exc):
                raise
        # --- END GPU DEBUG INFO ---

        # Ensure CPU components are initialized
        if not hasattr(self, "cpu_components_initialized"):
            print("⚠️ CPU components not initialized, initializing now...")
            self.initialize_cpu_components()

        from unav.localizer.localizer import UNavLocalizer

        print("🤖 Initializing UNavLocalizer (GPU-dependent)...")
        self.localizer = UNavLocalizer(self.localizor_config)

        # Add fine-grained tracing to internal localizer steps without editing the package
        #(monkey-patch key methods if available)
        try:
            self._monkey_patch_localizer_methods(self.localizer)
            self._monkey_patch_pose_refinement()
            #self._monkey_patch_matching_and_ransac()
            self._monkey_patch_feature_extractors()
        except Exception as e:
            print(f"⚠️ Failed to monkey-patch UNavLocalizer methods: {e}")

        # Skip loading all maps/features at startup - will load on demand
        print("✅ UNavLocalizer initialized (maps will load on demand)")

        # Set flag to indicate full system is ready
        self.gpu_components_initialized = True
        print("🎉 Full UNav system initialization complete! Ready for fast inference.")
        print(
            f"🎉 [Phase 2] Checking for deferred middleware init: _middleware_init_pending={getattr(self, '_middleware_init_pending', 'NOT_SET')}"
        )

        if getattr(self, "_middleware_init_pending", False):
            print(
                "🔁 GPU acquired; completing deferred Middleware.io initialization..."
            )
            print(
                f"🔁 [Phase 2] Before deferred init: tracer={getattr(self, 'tracer', 'NOT_SET')}"
            )
            self._middleware_init_pending = False
            self._configure_middleware_tracing()
            # Re-apply patches with the new tracer
            try:
                self._monkey_patch_localizer_methods(self.localizer)
                self._monkey_patch_pose_refinement()
                self._monkey_patch_matching_and_ransac()
                self._monkey_patch_feature_extractors()
            except Exception as e:
                print(f"⚠️ Failed to re-patch after deferred init: {e}")
            print(
                f"🔁 [Phase 2] After deferred init: tracer={getattr(self, 'tracer', 'NOT_SET')}, _middleware_init_pending={getattr(self, '_middleware_init_pending', 'NOT_SET')}"
            )
        else:
            print("✅ [Phase 2] No deferred middleware initialization needed")

    def _monkey_patch_localizer_methods(
        self, localizer, method_names: Optional[list] = None
    ):
        """
        Add spans to a set of internal UNavLocalizer methods by monkey-patching them.

        Args:
            localizer: the UNavLocalizer instance to patch
            method_names: optional list of method names to patch; if None, a conservative
                default list is used and we also try to discover other candidates.
        """
        import os

        if not hasattr(self, "tracer") or not self.tracer:
            return

        import functools
        import inspect
        import asyncio

        tracer = self.tracer

        # Target the ACTUAL UNavLocalizer methods from the localize() pipeline
        # Main pipeline methods
        default_candidates = [
            "extract_query_features",
            "vpr_retrieve",
            "get_candidates_data",
            "batch_local_matching_and_ransac",
            "multi_frame_pose_refine",
            "transform_pose_to_floorplan",
        ]

        # Additional internal components that are called by the pipeline methods
        # These will show as child spans under their parent methods
        internal_components = [
            "global_extractor",  # Called by extract_query_features
            "local_extractor",  # Called by extract_query_features
            "local_matcher",  # Called by batch_local_matching_and_ransac
        ]

        # Allow overriding the names via env var, e.g. MW_UNAV_TRACE_METHODS=extract_query_features,vpr_retrieve
        override = os.getenv("MW_UNAV_TRACE_METHODS")
        if override:
            method_names = [m.strip() for m in override.split(",") if m.strip()]
        else:
            # Combine both pipeline methods and internal components
            method_names = method_names or (default_candidates + internal_components)

        def _wrap(orig, name):
            # don't double-wrap
            if getattr(orig, "__mw_wrapped__", False):
                return orig

            if inspect.iscoroutinefunction(orig):

                async def _async_wrapper(*args, **kwargs):
                    with tracer.start_as_current_span(f"unav.{name}") as span:
                        try:
                            return await orig(*args, **kwargs)
                        except Exception as exc:
                            span.record_exception(exc)
                            raise

                _async_wrapper.__mw_wrapped__ = True
                return functools.wraps(orig)(_async_wrapper)

            else:

                def _sync_wrapper(*args, **kwargs):
                    with tracer.start_as_current_span(f"unav.{name}") as span:
                        try:
                            return orig(*args, **kwargs)
                        except Exception as exc:
                            span.record_exception(exc)
                            raise

                _sync_wrapper.__mw_wrapped__ = True
                return functools.wraps(orig)(_sync_wrapper)

        patched = []

        # Patch all methods explicitly listed (these are the exact UNavLocalizer methods)
        for mname in method_names:
            if hasattr(localizer, mname):
                try:
                    orig = getattr(localizer, mname)
                    wrapped = _wrap(orig, mname)
                    setattr(localizer, mname, wrapped)
                    patched.append(mname)
                except Exception as e:
                    print(f"⚠️ Failed to patch {mname}: {e}")
                    continue

        if patched:
            print(f"🔧 Patched localizer methods for tracing: {patched}")
        else:
            print(
                f"⚠️ Warning: No methods were patched. Available methods: {[m for m in dir(localizer) if not m.startswith('_')]}"
            )

    def _monkey_patch_pose_refinement(self):
        """
        Patch the child libraries (poselib, pyimplicitdist) that refine_pose_from_queue calls.
        This is cleaner than rewriting the entire function and traces the actual bottlenecks.
        """
        if not hasattr(self, "tracer") or not self.tracer:
            return

        import functools

        tracer = self.tracer

        # Patch poselib functions
        try:
            import poselib

            # Check if already patched
            if not getattr(poselib, "__mw_patched__", False):
                original_estimate = poselib.estimate_1D_radial_absolute_pose

                @functools.wraps(original_estimate)
                def traced_estimate(*args, **kwargs):
                    with tracer.start_as_current_span(
                        "unav.poselib.estimate_1D_radial"
                    ):
                        return original_estimate(*args, **kwargs)

                poselib.estimate_1D_radial_absolute_pose = traced_estimate
                poselib.__mw_patched__ = True
                print("🔧 Patched poselib.estimate_1D_radial_absolute_pose")
        except Exception as e:
            print(f"⚠️ Failed to patch poselib: {e}")

        # Patch pyimplicitdist functions
        try:
            import pyimplicitdist

            # Check if already patched
            if not getattr(pyimplicitdist, "__mw_patched__", False):
                # Patch pose_refinement_1D_radial
                original_refine_1d = pyimplicitdist.pose_refinement_1D_radial

                @functools.wraps(original_refine_1d)
                def traced_refine_1d(*args, **kwargs):
                    with tracer.start_as_current_span(
                        "unav.pyimplicitdist.pose_refinement_1D_radial"
                    ):
                        return original_refine_1d(*args, **kwargs)

                pyimplicitdist.pose_refinement_1D_radial = traced_refine_1d

                # Patch build_cost_matrix_multi
                original_build_cm = pyimplicitdist.build_cost_matrix_multi

                @functools.wraps(original_build_cm)
                def traced_build_cm(*args, **kwargs):
                    with tracer.start_as_current_span(
                        "unav.pyimplicitdist.build_cost_matrix_multi"
                    ):
                        return original_build_cm(*args, **kwargs)

                pyimplicitdist.build_cost_matrix_multi = traced_build_cm

                # Patch pose_refinement_multi
                original_refine_multi = pyimplicitdist.pose_refinement_multi

                @functools.wraps(original_refine_multi)
                def traced_refine_multi(*args, **kwargs):
                    with tracer.start_as_current_span(
                        "unav.pyimplicitdist.pose_refinement_multi"
                    ):
                        return original_refine_multi(*args, **kwargs)

                pyimplicitdist.pose_refinement_multi = traced_refine_multi

                pyimplicitdist.__mw_patched__ = True
                print(
                    "🔧 Patched pyimplicitdist functions (pose_refinement_1D_radial, build_cost_matrix_multi, pose_refinement_multi)"
                )
        except Exception as e:
            print(f"⚠️ Failed to patch pyimplicitdist: {e}")

    def _monkey_patch_feature_extractors(self):
        """
        Patch the feature extraction pipeline to add granular tracing.
        Patches:
        1. extract_query_features function to trace preprocessing/postprocessing
        2. GlobalExtractors.__call__ to trace model inference
        3. Superpoint.extract_local_features to trace local extraction
        """
        if not hasattr(self, "tracer") or not self.tracer:
            return

        import functools

        tracer = self.tracer

        # Patch the extract_query_features function from unav.localizer.tools.feature_extractor
        try:
            from unav.localizer.tools import feature_extractor

            if not getattr(feature_extractor, "__mw_patched__", False):
                original_extract = feature_extractor.extract_query_features

                @functools.wraps(original_extract)
                def traced_extract_query_features(
                    query_img,
                    global_extractor,
                    local_extractor,
                    global_model_name,
                    device,
                ):
                    # Simply wrap the original function - no code rewriting
                    with tracer.start_as_current_span("unav.extract_query_features"):
                        return original_extract(
                            query_img,
                            global_extractor,
                            local_extractor,
                            global_model_name,
                            device,
                        )

                feature_extractor.extract_query_features = traced_extract_query_features
                feature_extractor.__mw_patched__ = True
                print("🔧 Patched extract_query_features with tracing wrapper")
        except Exception as e:
            print(f"⚠️ Failed to patch extract_query_features: {e}")
            import traceback

            traceback.print_exc()

        # Patch GlobalExtractors.__call__ to trace the actual model forward pass
        try:
            from unav.core.feature.Global_Extractors import GlobalExtractors

            if not getattr(GlobalExtractors, "__mw_patched__", False):
                original_call = GlobalExtractors.__call__

                @functools.wraps(original_call)
                def traced_global_call(self, request_model, images):
                    with tracer.start_as_current_span(
                        f"unav.global_extractor.{request_model}.model_forward"
                    ):
                        result = original_call(self, request_model, images)
                    return result

                GlobalExtractors.__call__ = traced_global_call
                GlobalExtractors.__mw_patched__ = True
                print(
                    "🔧 Patched GlobalExtractors.__call__ for model inference tracing"
                )
        except Exception as e:
            print(f"⚠️ Failed to patch GlobalExtractors: {e}")
            import traceback

            traceback.print_exc()

        # Patch Superpoint.extract_local_features to trace preprocessing and model inference
        # Try multiple import paths since the module structure may vary
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
                                        if (
                                            not enable_multifloor
                                            and target_floor
                                            and floor_name != target_floor
                                        ):
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
        # Create span for getting destinations if tracer available
        if hasattr(self, "tracer") and self.tracer:
            with self.tracer.start_as_current_span(
                "get_destinations_list_span"
            ) as span:
                try:
                    print(
                        f"🎯 [Phase 3] Getting destinations for {place}/{building}/{floor}"
                    )

                    with self.tracer.start_as_current_span(
                        "ensure_maps_loaded"
                    ) as load_span:
                        # Ensure maps are loaded for this location (load all floors for the building)
                        self.ensure_maps_loaded(
                            place,
                            building,
                            floor=floor,
                            enable_multifloor=enable_multifloor,
                        )

                    # Use components with the loaded place
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
                    return {
                        "status": "error",
                        "message": str(e),
                        "type": type(e).__name__,
                    }
        else:
            try:
                print(
                    f"🎯 [Phase 3] Getting destinations for {place}/{building}/{floor}"
                )

                # Ensure maps are loaded for this location (load all floors for the building)
                self.ensure_maps_loaded(
                    place,
                    building,
                    floor=floor,
                    enable_multifloor=enable_multifloor,
                )

                # Use components with the loaded place
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
        is_vlm_extraction_enabled: bool = False,
        enable_multifloor: bool = False,
    ):
        """
        Full localization and navigation pipeline with timing tracking and middleware tracing.
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
                        print(
                            f"Received base64 image string of length {len(base64_string)}"
                        )
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

                    # Step 2: Localization
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
