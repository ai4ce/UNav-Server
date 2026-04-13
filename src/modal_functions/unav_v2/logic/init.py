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
    print(
        f"🔧 [Phase 0] Middleware init pending: {getattr(self, '_middleware_init_pending', 'NOT_SET')}"
    )

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
    self.LOCAL_FEATURE_MODEL = "mast3r"
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


def _debug_print_localizer_config(localizer):
    print("🔍 [CONFIG DEBUG] Localizer configuration:")
    config = localizer.config
    try:
        loc_cfg = (
            config.localization_config if hasattr(config, "localization_config") else {}
        )
        feat_cfg = (
            config.feature_extraction_config
            if hasattr(config, "feature_extraction_config")
            else {}
        )
        local_ext_cfg = (
            feat_cfg.get("local_extractor_config", {})
            if isinstance(feat_cfg, dict)
            else {}
        )
        global_desc_cfg = (
            feat_cfg.get("global_descriptor_config", {})
            if isinstance(feat_cfg, dict)
            else {}
        )

        print(f"   localization_config: {loc_cfg}")
        print(
            f"   local_feature_model: {getattr(config, 'local_feature_model', 'N/A')}"
        )
        print(
            f"   global_descriptor_model: {getattr(config, 'global_descriptor_model', 'N/A')}"
        )
        print(
            f"   local_extractor_config keys: {list(local_ext_cfg.keys()) if isinstance(local_ext_cfg, dict) else 'N/A'}"
        )

        mast3r_cfg = local_ext_cfg.get(
            "mast3r",
            local_ext_cfg.get("MASt3R", local_ext_cfg.get("mast3r_matching", {})),
        )
        if mast3r_cfg:
            print(f"   mast3r config: {mast3r_cfg}")
        else:
            print("   mast3r config: not found in local_extractor_config")

        print(f"   global_descriptor_config: {global_desc_cfg}")
        print(f"   topk: {loc_cfg.get('topk', 'N/A')}")
        print(f"   min_inliers: {loc_cfg.get('min_inliers', 'N/A')}")
        print(
            f"   feature_score_threshold: {loc_cfg.get('feature_score_threshold', 'N/A')}"
        )
    except Exception as e:
        print(f"⚠️ [CONFIG DEBUG] Error reading config: {e}")


def _override_mast3r_config(localizer):
    config = localizer.config
    overridden = []

    try:
        loc_cfg = (
            config.localization_config if hasattr(config, "localization_config") else {}
        )
        feat_cfg = (
            config.feature_extraction_config
            if hasattr(config, "feature_extraction_config")
            else {}
        )
        local_ext_cfg = (
            feat_cfg.get("local_extractor_config", {})
            if isinstance(feat_cfg, dict)
            else {}
        )

        mast3r_cfg = local_ext_cfg.get(
            "mast3r",
            local_ext_cfg.get("MASt3R", local_ext_cfg.get("mast3r_matching", {})),
        )
        if isinstance(mast3r_cfg, dict):
            if "max_nn_dist" in mast3r_cfg:
                old_val = mast3r_cfg["max_nn_dist"]
                mast3r_cfg["max_nn_dist"] = 100.0
                print(f"🔧 [CONFIG OVERRIDE] max_nn_dist: {old_val} -> 100.0")
                overridden.append("max_nn_dist")
            if "max_candidates" in mast3r_cfg:
                old_val = mast3r_cfg["max_candidates"]
                mast3r_cfg["max_candidates"] = 50
                print(f"🔧 [CONFIG OVERRIDE] max_candidates: {old_val} -> 50")
                overridden.append("max_candidates")
            if "subsample" in mast3r_cfg:
                old_val = mast3r_cfg["subsample"]
                if old_val != 1:
                    mast3r_cfg["subsample"] = 1
                    print(
                        f"🔧 [CONFIG OVERRIDE] subsample: {old_val} -> 1 (use all dense matches)"
                    )
                    overridden.append("subsample")

        if hasattr(config, "localization_config") and isinstance(
            config.localization_config, dict
        ):
            loc = config.localization_config
            if loc.get("topk", 5) < 50:
                old_val = loc.get("topk", 5)
                loc["topk"] = 50
                print(f"🔧 [CONFIG OVERRIDE] topk: {old_val} -> 50")
                overridden.append("topk")
            if loc.get("max_candidates", 10) < 50:
                old_val = loc.get("max_candidates", 10)
                loc["max_candidates"] = 50
                print(
                    f"🔧 [CONFIG OVERRIDE] max_candidates (localization): {old_val} -> 50"
                )
                overridden.append("max_candidates")
            if loc.get("min_inliers", 50) > 5:
                old_val = loc.get("min_inliers", 50)
                loc["min_inliers"] = 5
                print(f"🔧 [CONFIG OVERRIDE] min_inliers: {old_val} -> 5")
                overridden.append("min_inliers")
            if (
                "early_stop_inliers" not in loc
                or loc.get("early_stop_inliers", 0) < 200
            ):
                old_val = loc.get("early_stop_inliers", "not set")
                loc["early_stop_inliers"] = 200
                print(f"🔧 [CONFIG OVERRIDE] early_stop_inliers: {old_val} -> 200")
                overridden.append("early_stop_inliers")
            if "feature_score_threshold" in loc:
                old_val = loc["feature_score_threshold"]
                loc["feature_score_threshold"] = 0.01
                print(
                    f"🔧 [CONFIG OVERRIDE] feature_score_threshold: {old_val} -> 0.01"
                )
                overridden.append("feature_score_threshold")

        if overridden:
            print(f"✅ [CONFIG OVERRIDE] Overridden: {overridden}")
        else:
            print(
                "ℹ️ [CONFIG OVERRIDE] No overrides needed (config keys not found or already optimal)"
            )
    except Exception as e:
        print(f"⚠️ [CONFIG OVERRIDE] Error: {e}")


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
        print(
            f"[GPU DEBUG] torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}"
        )
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
    _apply_mast3r_extraction_fallback(self, self.localizer)

    _debug_print_localizer_config(self.localizer)
    _override_mast3r_config(self.localizer)

    try:
        _patch_mast3r_matching_debug()
    except Exception as e:
        print(f"⚠️ Failed to patch MASt3R matching debug: {e}")

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
        f"🎉 [Phase 2] Checking for deferred middleware init: _middleware_init_pending={getattr(self, '_middleware_init_pending', 'NOT_SET')}"
    )

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


def _apply_mast3r_extraction_fallback(server, localizer):
    """
    Patch localizer.extract_query_features for MASt3R compatibility.

    Some UNav/MASt3R combinations expose a callable local_extractor that
    internally fails with "'NoneType' object is not callable". In that case,
    retry using local_matcher as the local-feature callable.
    """
    import functools

    if getattr(localizer, "__mast3r_fallback_patched__", False):
        return

    original_extract = getattr(localizer, "extract_query_features", None)
    if not callable(original_extract):
        return

    @functools.wraps(original_extract)
    def _patched_extract_query_features(query_img):
        try:
            return original_extract(query_img)
        except TypeError as exc:
            error_message = str(exc)
            local_model = str(
                getattr(
                    getattr(localizer, "config", None),
                    "local_feature_model",
                    getattr(server, "LOCAL_FEATURE_MODEL", ""),
                )
            ).lower()
            if local_model != "mast3r":
                raise
            if "NoneType" not in error_message or "callable" not in error_message:
                raise

            print(
                "⚠️ MASt3R extract_query_features failed with NoneType callable; "
                "retrying with local_matcher fallback."
            )
            local_matcher = getattr(localizer, "local_matcher", None)
            if not callable(local_matcher):
                print("❌ local_matcher fallback unavailable or not callable.")
                raise

            from unav.localizer.tools.feature_extractor import extract_query_features

            return extract_query_features(
                query_img,
                localizer.global_extractor,
                local_matcher,
                localizer.config.global_descriptor_model,
                localizer.device,
            )

    localizer.extract_query_features = _patched_extract_query_features
    localizer.__mast3r_fallback_patched__ = True
    print("🔧 Applied MASt3R extract_query_features fallback patch")


def _add_localizer_debug_logging(localizer):
    """Add debug logging to localizer methods to diagnose MASt3R matching failures."""
    import functools

    if getattr(localizer, "__debug_logging_patched__", False):
        return

    if hasattr(localizer, "vpr_retrieve") and callable(localizer.vpr_retrieve):
        orig_vpr = localizer.vpr_retrieve

        @functools.wraps(orig_vpr)
        def debug_vpr_retrieve(global_feat, top_k=None, **kwargs):
            config_topk = 50
            cfg = getattr(localizer, "config", None)
            if cfg and hasattr(cfg, "localization_config"):
                config_topk = cfg.localization_config.get("topk", 50)
            effective_topk = (
                max(config_topk, top_k if top_k else 0) if top_k else config_topk
            )
            if top_k and top_k < config_topk:
                print(
                    f"🔧 [VPR OVERRIDE] top_k {top_k} -> {effective_topk} (config_topk={config_topk})"
                )
            print(
                f"🔍 [VPR DEBUG] vpr_retrieve called with top_k={top_k}, effective_topk={effective_topk}"
            )
            results = orig_vpr(global_feat, top_k=effective_topk, **kwargs)
            print(f"🔍 [VPR DEBUG] vpr_retrieve returned {len(results)} candidates")
            for i, item in enumerate(results):
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    print(
                        f"   candidate[{i}]: map_key={item[0]}, img={item[1]}, score={item[2]:.6f}"
                    )
            return results

        localizer.vpr_retrieve = debug_vpr_retrieve

    if hasattr(localizer, "batch_local_matching_and_ransac") and callable(
        localizer.batch_local_matching_and_ransac
    ):
        orig_batch = localizer.batch_local_matching_and_ransac

        @functools.wraps(orig_batch)
        def debug_batch_lmr(local_feat_dict, candidates_data, **kwargs):
            print(f"🔍 [MATCH DEBUG] batch_local_matching_and_ransac called")
            if local_feat_dict is None:
                print(f"   local_feat_dict is None (MASt3R fallback path)")
            elif isinstance(local_feat_dict, dict):
                for k, v in list(local_feat_dict.items())[:5]:
                    if hasattr(v, "shape"):
                        print(
                            f"   local_feat_dict[{k}]: shape={v.shape}, dtype={getattr(v, 'dtype', 'N/A')}"
                        )
            if isinstance(candidates_data, dict):
                print(f"   candidates_data: {len(candidates_data)} candidates")
                for i, (k, v) in enumerate(list(candidates_data.items())[:3]):
                    if isinstance(v, dict):
                        print(f"   candidate[{k}]: keys={list(v.keys())[:8]}")
            else:
                print(f"   candidates_data: type={type(candidates_data).__name__}")
            real_min_inliers = kwargs.get("min_inliers", "default")
            real_feat_thresh = kwargs.get("feature_score_threshold", "default")
            print(
                f"   kwargs: min_inliers={real_min_inliers}, feature_score_threshold={real_feat_thresh}"
            )
            result = orig_batch(local_feat_dict, candidates_data, **kwargs)
            best_map_key, pnp_pairs, results = result
            print(
                f"🔍 [MATCH DEBUG] result: best_map_key={best_map_key}, results={len(results) if results else 0}"
            )
            if results:
                for i, r in enumerate(results[:5]):
                    if isinstance(r, dict):
                        print(
                            f"   result[{i}]: inliers={r.get('inliers', 'N/A')}, score={r.get('score', 'N/A')}, ref={r.get('ref_image_name', 'N/A')}"
                        )
            else:
                print(f"   No results - all candidates filtered by RANSAC/matching")
            return result

        localizer.batch_local_matching_and_ransac = debug_batch_lmr

    if hasattr(localizer, "extract_query_features") and callable(
        localizer.extract_query_features
    ):
        orig_extract = localizer.extract_query_features

        @functools.wraps(orig_extract)
        def debug_extract(query_img, **kwargs):
            result = orig_extract(query_img, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                global_feat, local_feat_dict = result
                print(
                    f"🔍 [EXTRACT DEBUG] extract_query_features: global_feat type={type(global_feat).__name__}"
                )
                if hasattr(global_feat, "shape"):
                    print(f"   global_feat shape={global_feat.shape}")
                if isinstance(local_feat_dict, dict):
                    print(
                        f"   local_feat_dict keys={list(local_feat_dict.keys())[:10]}"
                    )
                    for k, v in list(local_feat_dict.items())[:5]:
                        if hasattr(v, "shape"):
                            print(
                                f"   local_feat_dict[{k}]: shape={v.shape}, dtype={getattr(v, 'dtype', 'N/A')}"
                            )
                elif local_feat_dict is None:
                    print(f"   local_feat_dict is None (will use fallback path)")
                else:
                    print(f"   local_feat_dict type={type(local_feat_dict).__name__}")
            return result

        localizer.extract_query_features = debug_extract

    localizer.__debug_logging_patched__ = True
    print("🔧 Patched localizer methods for debug logging")


def _patch_matcher_module(module):
    """Patch functions inside unav.localizer.tools.matcher."""
    import functools

    if getattr(module, "__mast3r_debug_patched__", False):
        print("ℹ️ [MATCHER DEBUG] Already patched")
        return

    avail = [a for a in dir(module) if not a.startswith("_")]
    print(f"🔍 [MATCHER DEBUG] Available in matcher module: {avail}")

    target_funcs = [
        "batch_local_matching_and_ransac",
        "mast3r_matching_and_pnp",
        "match_query_to_database",
        "ransac_filter",
        "mast3r_retrieve_matches_and_pnp",
    ]

    FORCE_MIN_INLIERS = 5
    FORCE_EARLY_STOP = 200

    for fname in target_funcs:
        if hasattr(module, fname):
            orig = getattr(module, fname)
            if not callable(orig):
                continue

            @functools.wraps(orig)
            def make_wrapper(f, name):
                def wrapper(*args, **kwargs):
                    if name == "mast3r_matching_and_pnp":
                        kwargs["min_inliers"] = FORCE_MIN_INLIERS
                        kwargs["early_stop_inliers"] = FORCE_EARLY_STOP
                        print(
                            f"🔧 [MATCHER.{name}] FORCED min_inliers={FORCE_MIN_INLIERS}, early_stop_inliers={FORCE_EARLY_STOP}"
                        )
                    print(
                        f"🔍 [MATCHER.{name}] called — args={len(args)}, kwargs={list(kwargs.keys())}"
                    )
                    for i, a in enumerate(args):
                        t = type(a).__name__
                        print(
                            f"   arg[{i}]: {t}"
                            + (
                                f", shape={a.shape}"
                                if hasattr(a, "shape")
                                else (
                                    f", len={len(a)}" if hasattr(a, "__len__") else ""
                                )
                            )
                        )
                    for k, v in kwargs.items():
                        print(
                            f"   {k}={type(v).__name__}"
                            + (
                                f", shape={v.shape}"
                                if hasattr(v, "shape")
                                else (
                                    f", val={v}"
                                    if isinstance(v, (int, float, str))
                                    else ""
                                )
                            )
                        )
                    result = f(*args, **kwargs)
                    if isinstance(result, tuple) and len(result) == 3:
                        bmk, pairs, res = result
                        pairs_info = "N/A"
                        if isinstance(pairs, dict):
                            pair_shapes = {}
                            for k, v in pairs.items():
                                if hasattr(v, "shape"):
                                    pair_shapes[k] = f"shape={v.shape}"
                                elif hasattr(v, "__len__"):
                                    pair_shapes[k] = f"len={len(v)}"
                                else:
                                    pair_shapes[k] = type(v).__name__
                            pairs_info = str(pair_shapes)
                        print(
                            f"🔍 [MATCHER.{name}] result: best={bmk}, pairs={pairs_info}, results={len(res) if res else 0}"
                        )
                        if res and len(res) > 0:
                            for ri, ritem in enumerate(res[:5]):
                                print(f"   result[{ri}]: {ritem}")
                    elif isinstance(result, dict):
                        print(
                            f"🔍 [MATCHER.{name}] result dict keys: {list(result.keys())[:10]}"
                        )
                    elif result is None:
                        print(f"🔍 [MATCHER.{name}] result: None")
                    else:
                        print(f"🔍 [MATCHER.{name}] result: {type(result).__name__}")
                    return result

                return wrapper

            wrapped = make_wrapper(orig, fname)
            setattr(module, fname, wrapped)
            print(f"🔧 [MATCHER DEBUG] Patched {fname}")

    module.__mast3r_debug_patched__ = True


def _patch_local_extractor_module(module):
    """Patch MASt3RExtractor and related functions in local_extractor."""
    import functools

    if getattr(module, "__mast3r_debug_patched__", False):
        print("ℹ️ [LOCAL_EXT DEBUG] Already patched")
        return

    avail = [a for a in dir(module) if not a.startswith("_")]
    print(f"🔍 [LOCAL_EXT DEBUG] Available: {avail}")

    for attr_name in avail:
        attr = getattr(module, attr_name)
        if callable(attr) and (
            "mast3r" in attr_name.lower()
            or "matcher" in attr_name.lower()
            or "extractor" in attr_name.lower()
        ):

            @functools.wraps(attr)
            def make_ext_wrapper(f, name):
                def wrapper(*args, **kwargs):
                    print(f"🔍 [LOCAL_EXT.{name}] called")
                    result = f(*args, **kwargs)
                    print(
                        f"🔍 [LOCAL_EXT.{name}] result type: {type(result).__name__}"
                        + (
                            f", shape={result.shape}"
                            if hasattr(result, "shape")
                            else ""
                        )
                    )
                    return result

                return wrapper

            wrapped = make_ext_wrapper(attr, attr_name)
            setattr(module, attr_name, wrapped)
            print(f"🔧 [LOCAL_EXT DEBUG] Patched {attr_name}")

    module.__mast3r_debug_patched__ = True


def _patch_feature_extractor_module(module):
    """Patch feature_extractor module functions."""
    import functools

    if getattr(module, "__mast3r_debug_patched__", False):
        return

    avail = [a for a in dir(module) if not a.startswith("_")]
    print(f"🔍 [FEAT_EXT DEBUG] Available: {avail}")

    for attr_name in avail:
        attr = getattr(module, attr_name)
        if callable(attr):

            @functools.wraps(attr)
            def make_fe_wrapper(f, name):
                def wrapper(*args, **kwargs):
                    print(f"🔍 [FEAT_EXT.{name}] called")
                    result = f(*args, **kwargs)
                    if isinstance(result, tuple):
                        print(f"🔍 [FEAT_EXT.{name}] returned tuple len={len(result)}")
                        for i, r in enumerate(result):
                            print(
                                f"   [{i}]: {type(r).__name__}"
                                + (
                                    f", shape={r.shape}"
                                    if hasattr(r, "shape")
                                    else (
                                        f", is dict={isinstance(r, dict)}"
                                        if isinstance(r, dict)
                                        else ""
                                    )
                                )
                            )
                            if isinstance(r, dict) and len(r) <= 3:
                                for k, v in list(r.items())[:3]:
                                    print(
                                        f"      [{k}]: {type(v).__name__}"
                                        + (
                                            f", shape={v.shape}"
                                            if hasattr(v, "shape")
                                            else ""
                                        )
                                    )
                    return result

                return wrapper

            wrapped = make_fe_wrapper(attr, attr_name)
            setattr(module, attr_name, wrapped)
            print(f"🔧 [FEAT_EXT DEBUG] Patched {attr_name}")

    module.__mast3r_debug_patched__ = True


def _patch_mast3r_matching_debug():
    """Monkey-patch the unav matching module to log MASt3R matching details."""
    import functools

    tried_imports = []

    try:
        from unav.localizer.tools import matcher as matcher_module

        tried_imports.append("unav.localizer.tools.matcher")
        _patch_matcher_module(matcher_module)
    except ImportError as e:
        print(f"⚠️ [MASt3R DEBUG] Could not import matcher: {e}")
        tried_imports.append(f"FAILED: {e}")

    try:
        from unav.core.feature import local_extractor as local_ext_module

        tried_imports.append("unav.core.feature.local_extractor")
        _patch_local_extractor_module(local_ext_module)
    except ImportError as e:
        print(f"⚠️ [MASt3R DEBUG] Could not import local_extractor: {e}")
        tried_imports.append(f"FAILED: {e}")

    try:
        from unav.localizer.tools import feature_extractor as feat_ext_module

        tried_imports.append("unav.localizer.tools.feature_extractor")
        _patch_feature_extractor_module(feat_ext_module)
    except ImportError as e:
        print(f"⚠️ [MASt3R DEBUG] Could not import feature_extractor: {e}")
        tried_imports.append(f"FAILED: {e}")

    print(f"🔍 [MASt3R DEBUG] Import attempts: {tried_imports}")


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

    override = os.getenv("MW_UNAV_TRACE_METHODS")
    if override:
        method_names = [m.strip() for m in override.split(",") if m.strip()]
    else:
        method_names = method_names or default_candidates

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
                if not callable(orig):
                    print(f"ℹ️ Skipping non-callable localizer attribute: {mname}")
                    continue
                wrapped = _wrap(orig, mname)
                setattr(localizer, mname, wrapped)
                patched.append(mname)
            except Exception as e:
                print(f"⚠️ Failed to patch {mname}: {e}")
                continue

    if patched:
        print(f"🔧 Patched localizer methods for tracing: {patched}")

    _add_localizer_debug_logging(localizer)


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
                with tracer.start_as_current_span(
                    "unav.pyimplicitdist.pose_refinement_1D_radial"
                ):
                    return original_refine_1d(*args, **kwargs)

            pyimplicitdist.pose_refinement_1D_radial = traced_refine_1d

            original_build_cm = pyimplicitdist.build_cost_matrix_multi

            @functools.wraps(original_build_cm)
            def traced_build_cm(*args, **kwargs):
                with tracer.start_as_current_span(
                    "unav.pyimplicitdist.build_cost_matrix_multi"
                ):
                    return original_build_cm(*args, **kwargs)

            pyimplicitdist.build_cost_matrix_multi = traced_build_cm

            original_refine_multi = pyimplicitdist.pose_refinement_multi

            @functools.wraps(original_refine_multi)
            def traced_refine_multi(*args, **kwargs):
                with tracer.start_as_current_span(
                    "unav.pyimplicitdist.pose_refinement_multi"
                ):
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
            def traced_extract_query_features(
                query_img, global_extractor, local_extractor, global_model_name, device
            ):
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

    try:
        from unav.core.feature.Global_Extractors import GlobalExtractors

        if not getattr(GlobalExtractors, "__mw_patched__", False):
            original_call = GlobalExtractors.__call__

            @functools.wraps(original_call)
            def traced_global_call(self, request_model, images):
                with tracer.start_as_current_span(
                    f"unav.global_extractor.{request_model}.model_forward"
                ):
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
