import os

from modal import enter


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
    from middleware import MWOptions, mw_tracker
    from opentelemetry import trace

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
        print("⚠️ [CONFIGURE] Tracer set to None due to error")


@enter(snap=False)
def initialize_middleware(self):
    """Initialize Middleware.io tracking for profiling and telemetry."""
    print("🔧 [Phase 0] Initializing Middleware.io...")
    print(f"🔧 [Phase 0] Current tracer state: {getattr(self, 'tracer', 'NOT_SET')}")
    print(
        "🔧 [Phase 0] Middleware init pending: "
        f"{getattr(self, '_middleware_init_pending', 'NOT_SET')}"
    )

    if not self._gpu_available():
        print(
            "⏸️ GPU not yet available; deferring Middleware.io initialization until a GPU-backed container is ready."
        )
        self._middleware_init_pending = True
        self.tracer = None
        print("🔧 [Phase 0] Set _middleware_init_pending=True, tracer=None")
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
    from unav.navigator.commander import commands_from_result
    from unav.navigator.multifloor import FacilityNavigator

    self.DATA_ROOT = "/root/UNav-IO/data"
    self.FEATURE_MODEL = "DinoV2Salad"
    self.LOCAL_FEATURE_MODEL = "superpoint+lightglue"
    self.PLACES = self.get_places()

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


@enter(snap=False)
def initialize_gpu_components(self):
    """
    Initialize GPU-dependent components that cannot be snapshotted.
    This must run after snapshot restoration on GPU-enabled containers.
    """
    print("🚀 [Phase 2] Initializing GPU components after snapshot restoration...")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"[GPU DEBUG] torch.cuda.is_available(): {cuda_available}")

        if not cuda_available:
            print("[GPU ERROR] CUDA not available! This will cause model loading to fail.")
            print(
                "[GPU ERROR] Modal should have allocated a GPU. Raising exception to trigger retry..."
            )
            raise RuntimeError(
                "GPU not available when required. Modal will retry with GPU allocation."
            )

        print(f"[GPU DEBUG] torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"[GPU DEBUG] torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(
            f"[GPU DEBUG] torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}"
        )
    except Exception as gpu_debug_exc:
        print(f"[GPU DEBUG] Error printing GPU info: {gpu_debug_exc}")
        if "GPU not available when required" in str(gpu_debug_exc):
            raise

    if not hasattr(self, "cpu_components_initialized"):
        print("⚠️ CPU components not initialized, initializing now...")
        self.initialize_cpu_components()

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

    self.gpu_components_initialized = True
    print("🎉 Full UNav system initialization complete! Ready for fast inference.")
    print(
        "🎉 [Phase 2] Checking for deferred middleware init: "
        f"_middleware_init_pending={getattr(self, '_middleware_init_pending', 'NOT_SET')}"
    )

    if getattr(self, "_middleware_init_pending", False):
        print("🔁 GPU acquired; completing deferred Middleware.io initialization...")
        print(
            "🔁 [Phase 2] Before deferred init: "
            f"tracer={getattr(self, 'tracer', 'NOT_SET')}"
        )
        self._middleware_init_pending = False
        self._configure_middleware_tracing()
        try:
            self._monkey_patch_localizer_methods(self.localizer)
            self._monkey_patch_pose_refinement()
            self._monkey_patch_matching_and_ransac()
            self._monkey_patch_feature_extractors()
        except Exception as e:
            print(f"⚠️ Failed to re-patch after deferred init: {e}")
        print(
            "🔁 [Phase 2] After deferred init: "
            f"tracer={getattr(self, 'tracer', 'NOT_SET')}, "
            f"_middleware_init_pending={getattr(self, '_middleware_init_pending', 'NOT_SET')}"
        )
    else:
        print("✅ [Phase 2] No deferred middleware initialization needed")
