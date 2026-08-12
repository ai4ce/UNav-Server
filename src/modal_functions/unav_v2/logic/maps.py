from typing import Any, Dict, Optional, Set

import os
import functools
import time

from .places import run_get_places


def run_ensure_maps_loaded(
    server: Any,
    place: str,
    building: Optional[str] = None,
    floor: Optional[str] = None,
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

    if map_key in server.maps_loaded:
        return

    print(f"🔄 [Phase 4] Creating selective localizer for: {map_key}")

    if building:
        selective_places = run_get_places(
            server,
            target_place=place,
            target_building=building,
            target_floor=floor,
            enable_multifloor=enable_multifloor,
        )
    else:
        selective_places = run_get_places(server, target_place=place)

    if not selective_places:
        print(
            "⚠️ No matching places found for selective load; skipping localizer creation"
        )
        return

    from unav.config import UNavConfig

    selective_config = UNavConfig(
        data_final_root=server.DATA_ROOT,
        places=selective_places,
        global_descriptor_model=server.FEATURE_MODEL,
        local_feature_model=server.LOCAL_FEATURE_MODEL,
    )

    from unav.localizer.localizer import UNavLocalizer
    import time

    try:
        _install_upstream_instrumentation(UNavLocalizer)
    except Exception as e:
        print(f"⚠️ Failed to install upstream instrumentation: {e}")

    try:
        _install_matcher_instrumentation()
    except Exception as e:
        print(f"⚠️ Failed to install matcher instrumentation: {e}")

    try:
        if getattr(server, "LOCAL_FEATURE_MODEL", "") == "mast3r":
            _install_fast_colmap_loader()
    except Exception as e:
        print(f"⚠️ Failed to install fast COLMAP loader: {e}")

    selective_localizer = UNavLocalizer(selective_config.localizer_config)
    try:
        from .init import _apply_mast3r_extraction_fallback

        _apply_mast3r_extraction_fallback(server, selective_localizer)
    except Exception as e:
        print(f"⚠️ Failed to apply MASt3R fallback on selective localizer: {e}")

    try:
        _apply_mast3r_tuning(selective_localizer)
    except Exception as e:
        print(f"⚠️ Failed to apply MASt3R tuning on selective localizer: {e}")

    if hasattr(server, "tracer") and server.tracer:
        try:
            server._monkey_patch_localizer_methods(selective_localizer)
        except Exception as e:
            print(f"⚠️ Failed to patch selective localizer: {e}")

    if hasattr(server, "tracer") and server.tracer:
        with server.tracer.start_as_current_span(
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

    server.selective_localizers[map_key] = selective_localizer
    server.maps_loaded.add(map_key)
    print(f"✅ Selective localizer created and maps loaded for: {map_key}")


def _apply_mast3r_tuning(localizer):
    """
    Apply MASt3R inference knobs (input resolution) to a localizer's MASt3RExtractor.

    The upstream extractor reads mast3r_size from its config dict on every
    match (unav/core/feature/local_extractor.py); this repo cannot touch the
    deployed unav package, so we override the instance config here instead.
    """
    if getattr(localizer, "__mast3r_tuned__", False):
        return
    local_matcher = getattr(localizer, "local_matcher", None)
    if local_matcher is None:
        return
    config = getattr(local_matcher, "config", None)
    if not isinstance(config, dict):
        return
    try:
        size = int(os.getenv("UNAV_MAST3R_SIZE", "384"))
    except (TypeError, ValueError):
        size = 384
    config["mast3r_size"] = min(max(size, 224), 512)
    localizer.__mast3r_tuned__ = True
    print(f"🔧 Applied MASt3R tuning: mast3r_size={config['mast3r_size']}")


def _install_fast_colmap_loader():
    """
    Patch upstream load_colmap_model to skip the heavy points3D.bin parse.

    MASt3R relpose localization only needs per-image camera poses (qvec/tvec)
    from images.bin. The full read_model() also parses points3D.bin (millions
    of 3D points, ~18s per floor in logs) which the relpose path never uses.
    Only valid while the matcher is mast3r_relpose_localization.
    """
    import functools
    from unav.localizer.tools import io as _io_mod
    if getattr(_io_mod.load_colmap_model, "_unav_fast_colmap", False):
        return

    @functools.wraps(_io_mod.load_colmap_model)
    def fast_load_colmap_model(model_dir, ext=".bin"):
        from unav.core.colmap.read_write_model import read_images_binary
        images_path = os.path.join(model_dir, "images" + ext)
        if not os.path.exists(images_path):
            return {}
        images = read_images_binary(images_path)
        frames_by_name = {}
        for img in images.values():
            frames_by_name[img.name] = {
                "qvec": img.qvec,
                "tvec": img.tvec,
                "points2D_xy": None,
                "points3D_xyz": [],
                "points3D_id": None,
            }
        return frames_by_name

    fast_load_colmap_model._unav_fast_colmap = True
    _io_mod.load_colmap_model = fast_load_colmap_model
    # UNavLocalizer._ensure_colmap_model calls load_colmap_model via a direct
    # `from unav.localizer.tools.io import load_colmap_model` name binding, so
    # the localizer module's own reference must be patched as well.
    try:
        from unav.localizer import localizer as _localizer_mod
        _localizer_mod.load_colmap_model = fast_load_colmap_model
    except Exception as e:
        print(f"⚠️ Could not patch localizer.load_colmap_model reference: {e}")
    print("🔧 Installed fast COLMAP loader (images.bin only, no points3D)")


def _install_upstream_instrumentation(UNavLocalizer):
    """Wrap upstream UNavLocalizer.localize and batch_local_matching_and_ransac to log runtime behavior."""
    import functools
    if getattr(UNavLocalizer.localize, "_unav_traced", False):
        return
    orig_localize = UNavLocalizer.localize
    orig_match = UNavLocalizer.batch_local_matching_and_ransac
    print(
        f"🧪 [INSTRUMENT] Class={UNavLocalizer.__module__}.{UNavLocalizer.__name__}, "
        f"orig_localize={orig_localize}, orig_match={orig_match}",
        flush=True,
    )

    @functools.wraps(orig_localize)
    def traced_localize(self, query_img, refinement_queue, top_k=None, **kwargs):
        print(
            f"🧪 [UPSTREAM LOCALIZE] shape={getattr(query_img, 'shape', None)}, top_k={top_k}, "
            f"self_class={type(self).__module__}.{type(self).__name__}, "
            f"use_mast3r={getattr(self, 'use_mast3r', 'N/A')}, "
            f"local_matcher={type(self.local_matcher).__name__}",
            flush=True,
        )
        if top_k is None:
            try:
                top_k = int(os.getenv("UNAV_VPR_TOP_K", "10")) or None
            except (TypeError, ValueError):
                top_k = None
        result = orig_localize(self, query_img, refinement_queue, top_k=top_k, **kwargs)
        print(
            f"🧪 [UPSTREAM LOCALIZE DONE] success={result.get('success')}, "
            f"stage={result.get('stage')}, reason={str(result.get('reason', ''))[:80]}, "
            f"top_candidates={len(result.get('top_candidates', []) or [])}",
            flush=True,
        )
        return result

    @functools.wraps(orig_match)
    def traced_match(self, local_feat_dict, candidates_data, query_img_path=None, pp=None):
        print(
            f"🧪 [UPSTREAM MATCH] use_mast3r={getattr(self, 'use_mast3r', 'N/A')}, "
            f"candidates={len(candidates_data)}, query_img_path={query_img_path}, "
            f"query_exists={os.path.exists(query_img_path) if query_img_path else False}",
            flush=True,
        )
        # Call original with overridden data_roots for the MASt3R dispatcher path
        if getattr(self, 'use_mast3r', False):
            mast3r_data_roots = ("/root/UNav-IO/mnt/data/UNav-IO/temp", "/root/UNav-IO/data")
            from unav.localizer.tools import matcher as _matcher
            # mast3r_relpose_localization computes the pose directly from
            # MASt3R relative geometry (Procrustes) — no COLMAP 3D point cloud
            # and no PnP refinement needed. It references a module-level `pp`
            # that upstream never defines (NameError); defaulting it to None
            # falls back to the image-center principal point, as intended.
            if not hasattr(_matcher, "pp"):
                _matcher.pp = None
            result = _matcher.mast3r_relpose_localization(
                query_img_path=query_img_path,
                candidates_data=candidates_data,
                mast3r_matcher=self.local_matcher,
                colmap_models=self.all_colmap_models,
                transform_matrices=self.transform_matrices,
                min_inliers=10,
                max_candidates=int(os.getenv("UNAV_MAST3R_CANDIDATES", "5")),
                data_roots=mast3r_data_roots,
            )
        else:
            result = orig_match(self, local_feat_dict, candidates_data, query_img_path=query_img_path, pp=pp)
        if result is None:
            print("🧪 [UPSTREAM MATCH DONE] result=None", flush=True)
            return None
        best_map_key, pnp_pairs, results = result
        print(
            f"🧪 [UPSTREAM MATCH DONE] best_map_key={best_map_key}, "
            f"results_count={len(results) if results else 0}, "
            f"pnp_pairs_type={type(pnp_pairs).__name__}",
            flush=True,
        )
        return result

    traced_localize._unav_traced = True
    traced_match._unav_traced = True
    UNavLocalizer.localize = traced_localize
    UNavLocalizer.batch_local_matching_and_ransac = traced_match
    print("🧪 [INSTRUMENT] Wrapped upstream UNavLocalizer.localize and batch_local_matching_and_ransac", flush=True)


def _install_matcher_instrumentation():
    """Wrap mast3r_matching_and_pnp to log DB resolution and per-candidate match results."""
    import functools
    from unav.localizer.tools import matcher as _matcher_mod
    if getattr(_matcher_mod.mast3r_matching_and_pnp, "_unav_traced", False):
        return
    original = _matcher_mod.mast3r_matching_and_pnp

    @functools.wraps(original)
    def traced(query_img_path, candidates_data, mast3r_matcher, colmap_models,
               max_nn_dist=20.0, min_inliers=6, max_candidates=10,
               early_stop_inliers=50, data_roots=None):
        from unav.localizer.tools.matcher import _resolve_db_image_path
        if data_roots is None:
            from unav.config import UNavConfig
            data_roots = ("/root/UNav-IO/mnt/data/UNav-IO/temp", "/root/UNav-IO/data")
        drs = tuple(data_roots) if data_roots else _matcher_mod.DEFAULT_DB_IMAGE_ROOTS
        ref_img_names = list(candidates_data.keys())[:max_candidates]
        db_paths, db_names = [], []
        skipped = []
        for name in ref_img_names:
            cand = candidates_data[name]
            place, building, floor = cand["map_key"]
            p = _resolve_db_image_path(drs, place, building, floor, name)
            if p is None:
                skipped.append(name)
                continue
            db_paths.append(p)
            db_names.append(name)
        sample_cand = next(iter(candidates_data.values()), None) if candidates_data else None
        sample_map_key = sample_cand.get("map_key") if sample_cand else None
        print(
            f"🧪 [MAST3R INSIDE] query={query_img_path}, "
            f"ref_names={len(ref_img_names)}, db_paths_resolved={len(db_paths)}, "
            f"skipped={len(skipped)} sample_skipped={skipped[:2]}, "
            f"data_roots={drs}, sample_map_key={sample_map_key}, "
            f"sample_expected_path={os.path.join(drs[0], *sample_map_key, 'perspectives', ref_img_names[0]) if sample_map_key and drs and ref_img_names else 'N/A'}",
            flush=True,
        )
        if not db_paths:
            print("🧪 [MAST3R INSIDE] No DB paths resolved — returning empty result.", flush=True)
            return None, {"image_points": [], "object_points": []}, []
        result = original(
            query_img_path=query_img_path,
            candidates_data=candidates_data,
            mast3r_matcher=mast3r_matcher,
            colmap_models=colmap_models,
            max_nn_dist=max_nn_dist,
            min_inliers=min_inliers,
            max_candidates=max_candidates,
            early_stop_inliers=early_stop_inliers,
            data_roots=data_roots,
        )
        return result

    traced._unav_traced = True
    _matcher_mod.mast3r_matching_and_pnp = traced
    print("🧪 [INSTRUMENT] Wrapped mast3r_matching_and_pnp", flush=True)
