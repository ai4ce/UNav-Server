from typing import Optional


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
                with tracer.start_as_current_span("unav.poselib.estimate_1D_radial"):
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
            print("🔧 Patched GlobalExtractors.__call__ for model inference tracing")
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
                print(f"🔍 [DEBUG] Found {class_name} in {module_path}: {Superpoint}")
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
                    print("🔧 ✅ Patched CrossTransformer.forward (cross-attention)")

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
                    print("🔧 ✅ Patched LearnableFourierPositionalEncoding.forward")

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

        print("🔍 [DEBUG] Attempting to patch unav.core.feature_filter")
        print(f"🔍 [DEBUG] feature_filter module: {feature_filter}")
        print(
            f"🔍 [DEBUG] match_query_to_database: {hasattr(feature_filter, 'match_query_to_database')}"
        )
        print(f"🔍 [DEBUG] ransac_filter: {hasattr(feature_filter, 'ransac_filter')}")

        # Check if already patched
        if not getattr(feature_filter, "__mw_patched__", False):
            # Patch match_query_to_database
            if hasattr(feature_filter, "match_query_to_database"):
                original_match = feature_filter.match_query_to_database

                @functools.wraps(original_match)
                def traced_match(*args, **kwargs):
                    print("🔍 [TRACE] ✅ Entering match_query_to_database")
                    with tracer.start_as_current_span("unav.match_query_to_database"):
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
