"""
Initialization and setup methods for UnavServer.
These are called during container startup.
"""
from modal import enter

from .places import run_get_places


def run_init_middleware(self):
    """Initialize Middleware.io tracking for profiling and telemetry."""
    print("🔧 [Phase 0] Initializing Middleware.io...")
    print(f"🔧 [Phase 0] Current tracer state: {getattr(self, 'tracer', 'NOT_SET')}")
    print(f"🔧 [Phase 0] Middleware init pending: {getattr(self, '_middleware_init_pending', 'NOT_SET')}")

    if not _gpu_available():
        print("⏸️ GPU not yet available; deferring Middleware.io initialization...")
        self._middleware_init_pending = True
        self.tracer = None
        return

    self._middleware_init_pending = False
    print("✅ GPU available; proceeding with Middleware.io initialization")
    _configure_middleware_tracing(self)


def run_init_cpu_components(self):
    """Initialize CPU-only components that can be safely snapshotted."""
    print("🚀 [Phase 1] Initializing CPU components for snapshotting...")

    from unav.config import UNavConfig
    from unav.navigator.multifloor import FacilityNavigator
    from unav.navigator.commander import commands_from_result

    self.DATA_ROOT = "/root/UNav-IO/data"
    self.FEATURE_MODEL = "DinoV2Salad"
    self.LOCAL_FEATURE_MODEL = "superpoint+lightglue"
    self.PLACES = run_get_places(self)

    print("🔧 Initializing UNavConfig...")
    self.config = UNavConfig(
        data_final_root=self.DATA_ROOT,
        places=self.PLACES,
        global_descriptor_model=self.FEATURE_MODEL,
        local_feature_model=self.LOCAL_FEATURE_MODEL,
    )
    print("✅ UNavConfig initialized successfully")

    self.localizor_config = self.config.localizer_config
    self.navigator_config = self.config.navigator_config
    print("✅ Config objects extracted successfully")

    print("🧭 Initializing FacilityNavigator (CPU-only)...")
    self.nav = FacilityNavigator(self.navigator_config)
    print("✅ FacilityNavigator initialized successfully")

    self.commander = commands_from_result
    self.maps_loaded = set()
    self.selective_localizers = {}
    self.cpu_components_initialized = True
    print("📸 CPU components ready for snapshotting!")


def run_init_gpu_components(self):
    """Initialize GPU-dependent components that cannot be snapshotted."""
    print("🚀 [Phase 2] Initializing GPU components after snapshot restoration...")

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"[GPU DEBUG] torch.cuda.is_available(): {cuda_available}")

        if not cuda_available:
            print("[GPU ERROR] CUDA not available! Raising exception...")
            raise RuntimeError("GPU not available when required.")

        print(f"[GPU DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"[GPU DEBUG] torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"[GPU DEBUG] torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    except Exception as gpu_debug_exc:
        print(f"[GPU DEBUG] Error printing GPU info: {gpu_debug_exc}")
        if "GPU not available when required" in str(gpu_debug_exc):
            raise

    if not hasattr(self, "cpu_components_initialized"):
        print("⚠️ CPU components not initialized, initializing now...")
        run_init_cpu_components(self)

    from unav.localizer.localizer import UNavLocalizer

    print("🤖 Initializing UNavLocalizer (GPU-dependent)...")
    self.localizer = UNavLocalizer(self.localizor_config)

    try:
        self._monkey_patch_localizer_methods(self.localizer)
        self._monkey_patch_pose_refinement()
        self._monkey_patch_feature_extractors()
    except Exception as e:
        print(f"⚠️ Failed to monkey-patch UNavLocalizer methods: {e}")

    print("✅ UNavLocalizer initialized (maps will load on demand)")

    # Pre-load maps for common places to include in memory snapshot
    print("🗺️ Pre-loading maps for memory snapshot...")
    from .maps import run_ensure_maps_loaded
    
    # Load all floors for common places
    PRELOAD_PLACES = [
        ("New_York_City", "LightHouse", None),  # All floors
        ("New_York_University", "Langone", None),  # All floors
    ]
    
    for place, building, floor in PRELOAD_PLACES:
        if floor:
            print(f"   📦 Pre-loading: {place} / {building} / {floor}")
        else:
            print(f"   📦 Pre-loading: {place} / {building} (all floors)")
        run_ensure_maps_loaded(self, place=place, building=building, floor=floor)
    
    print("✅ Maps pre-loaded for snapshot")

    self.gpu_components_initialized = True
    print("🎉 Full UNav system initialization complete! Ready for fast inference.")
    print(f"🎉 [Phase 2] Checking for deferred middleware init: _middleware_init_pending={getattr(self, '_middleware_init_pending', 'NOT_SET')}")

    if getattr(self, "_middleware_init_pending", False):
        print("🔁 GPU acquired; completing deferred Middleware.io initialization...")
        self._middleware_init_pending = False
        _configure_middleware_tracing(self)
        try:
            self._monkey_patch_localizer_methods(self.localizer)
            self._monkey_patch_pose_refinement()
            self._monkey_patch_matching_and_ransac()
            self._monkey_patch_feature_extractors()
        except Exception as e:
            print(f"⚠️ Failed to re-patch after deferred init: {e}")
    else:
        print("✅ [Phase 2] No deferred middleware initialization needed")


def _gpu_available() -> bool:
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
    """Configure Middleware.io tracing."""
    print("🔧 [CONFIGURE] Starting Middleware.io configuration...")
    from middleware import mw_tracker, MWOptions
    from opentelemetry import trace
    import os

    api_key = os.environ.get("MW_API_KEY")
    target = os.environ.get("MW_TARGET")

    if not api_key or not target:
        print("⚠️ Warning: MW_API_KEY and MW_TARGET not set. Skipping middleware initialization.")
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


def run_monkey_patch_localizer_methods(self, localizer, method_names=None):
    """Add spans to internal UNavLocalizer methods by monkey-patching."""
    import os
    import functools
    import inspect

    if not hasattr(self, "tracer") or not self.tracer:
        return

    tracer = self.tracer
    default_candidates = [
        "extract_query_features",
        "vpr_retrieve",
        "get_candidates_data",
        "batch_local_matching_and_ransac",
        "multi_frame_pose_refine",
        "transform_pose_to_floorplan",
    ]
    internal_components = ["global_extractor", "local_extractor", "local_matcher"]

    override = os.getenv("MW_UNAV_TRACE_METHODS")
    if override:
        method_names = [m.strip() for m in override.split(",") if m.strip()]
    else:
        method_names = method_names or (default_candidates + internal_components)

    def _wrap(orig, name):
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


def run_monkey_patch_pose_refinement(self):
    """Patch pose refinement libraries."""
    if not hasattr(self, "tracer") or not self.tracer:
        return

    tracer = self.tracer
    import functools

    try:
        import poselib
        if not getattr(poselib, "__mw_patched__", False):
            original_estimate = poselib.estimate_1D_radial_absolute_pose

            @functools.wraps(original_estimate)
            def traced_estimate(*args, **kwargs):
                with tracer.start_as_current_span("unav.poselib.estimate_1D_radial"):
                    return original_estimate(*args, **kwargs)

            poselib.estimate_1D_radial_absolute_pose = traced_estimate
            poselib.__mw_patched__ = True
            print("🔧 Patched poselib.estimate_1D_radial_absolute_pose")
    except Exception as e:
        print(f"⚠️ Failed to patch poselib: {e}")

    try:
        import pyimplicitdist
        if not getattr(pyimplicitdist, "__mw_patched__", False):
            original_refine_1d = pyimplicitdist.pose_refinement_1D_radial

            @functools.wraps(original_refine_1d)
            def traced_refine_1d(*args, **kwargs):
                with tracer.start_as_current_span("unav.pyimplicitdist.pose_refinement_1D_radial"):
                    return original_refine_1d(*args, **kwargs)

            pyimplicitdist.pose_refinement_1D_radial = traced_refine_1d

            original_build_cm = pyimplicitdist.build_cost_matrix_multi

            @functools.wraps(original_build_cm)
            def traced_build_cm(*args, **kwargs):
                with tracer.start_as_current_span("unav.pyimplicitdist.build_cost_matrix_multi"):
                    return original_build_cm(*args, **kwargs)

            pyimplicitdist.build_cost_matrix_multi = traced_build_cm

            original_refine_multi = pyimplicitdist.pose_refinement_multi

            @functools.wraps(original_refine_multi)
            def traced_refine_multi(*args, **kwargs):
                with tracer.start_as_current_span("unav.pyimplicitdist.pose_refinement_multi"):
                    return original_refine_multi(*args, **kwargs)

            pyimplicitdist.pose_refinement_multi = traced_refine_multi
            pyimplicitdist.__mw_patched__ = True
            print("🔧 Patched pyimplicitdist functions")
    except Exception as e:
        print(f"⚠️ Failed to patch pyimplicitdist: {e}")


def run_monkey_patch_feature_extractors(self):
    """Patch feature extraction pipeline."""
    if not hasattr(self, "tracer") or not self.tracer:
        return

    tracer = self.tracer
    import functools

    try:
        from unav.localizer.tools import feature_extractor

        if not getattr(feature_extractor, "__mw_patched__", False):
            original_extract = feature_extractor.extract_query_features

            @functools.wraps(original_extract)
            def traced_extract_query_features(query_img, global_extractor, local_extractor, global_model_name, device):
                with tracer.start_as_current_span("unav.extract_query_features"):
                    return original_extract(query_img, global_extractor, local_extractor, global_model_name, device)

            feature_extractor.extract_query_features = traced_extract_query_features
            feature_extractor.__mw_patched__ = True
            print("🔧 Patched extract_query_features with tracing wrapper")
    except Exception as e:
        print(f"⚠️ Failed to patch extract_query_features: {e}")

    try:
        from unav.core.feature.Global_Extractors import GlobalExtractors

        if not getattr(GlobalExtractors, "__mw_patched__", False):
            original_call = GlobalExtractors.__call__

            @functools.wraps(original_call)
            def traced_global_call(self, request_model, images):
                with tracer.start_as_current_span(f"unav.global_extractor.{request_model}.model_forward"):
                    return original_call(self, request_model, images)

            GlobalExtractors.__call__ = traced_global_call
            GlobalExtractors.__mw_patched__ = True
            print("🔧 Patched GlobalExtractors.__call__ for model inference tracing")
    except Exception as e:
        print(f"⚠️ Failed to patch GlobalExtractors: {e}")


def run_monkey_patch_matching_and_ransac(self):
    """Patch matching and RANSAC functions."""
    if not hasattr(self, "tracer") or not self.tracer:
        return

    tracer = self.tracer
    import functools

    try:
        from unav.core import feature_filter

        if not getattr(feature_filter, "__mw_patched__", False):
            if hasattr(feature_filter, "match_query_to_database"):
                original_match = feature_filter.match_query_to_database

                @functools.wraps(original_match)
                def traced_match(*args, **kwargs):
                    with tracer.start_as_current_span("unav.match_query_to_database"):
                        return original_match(*args, **kwargs)

                feature_filter.match_query_to_database = traced_match
                print("🔧 Patched match_query_to_database")

            if hasattr(feature_filter, "ransac_filter"):
                original_ransac = feature_filter.ransac_filter

                @functools.wraps(original_ransac)
                def traced_ransac(*args, **kwargs):
                    with tracer.start_as_current_span("unav.ransac_filter"):
                        return original_ransac(*args, **kwargs)

                feature_filter.ransac_filter = traced_ransac
                print("🔧 Patched ransac_filter")

            feature_filter.__mw_patched__ = True
            print("🔧 Patched feature_filter module successfully")
    except Exception as e:
        print(f"⚠️ Failed to patch feature_filter: {e}")
