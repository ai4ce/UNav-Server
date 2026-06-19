import os
import time
import torch
import numpy as np

from typing import Dict, Any

# Feature/model abstraction imports
from unav.core.feature.Global_Extractors import GlobalExtractors
from unav.core.feature.local_extractor import Local_extractor

# Utility tools for I/O and matching
from unav.localizer.tools.io import load_colmap_model, load_global_features, load_local_features
from unav.localizer.tools.feature_extractor import extract_query_features
from unav.localizer.tools.retriever import (
    search_vpr_topk_candidates,
    fetch_candidates_data
)
from unav.localizer.tools.matcher import batch_local_matching_and_ransac
from unav.localizer.tools.pnp import (
    refine_pose_from_queue,
    transform_pose_to_floorplan,
)


def _wrap_mast3r_matcher():
    """Monkey-patch mast3r_matching_and_pnp at import time to log DB resolution and per-candidate match results."""
    import functools
    from unav.localizer.tools import matcher as _matcher_mod
    if getattr(_matcher_mod.mast3r_matching_and_pnp, "_unav_traced", False):
        return
    original = _matcher_mod.mast3r_matching_and_pnp

    @functools.wraps(original)
    def traced(query_img_path, candidates_data, mast3r_matcher, colmap_models,
               max_nn_dist=20.0, min_inliers=6, max_candidates=10,
               early_stop_inliers=50, pp=None, data_roots=None):
        from unav.localizer.tools.matcher import _resolve_db_image_path
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
        print(
            f"🧪 [MAST3R INSIDE] query={query_img_path}, "
            f"ref_names={len(ref_img_names)}, db_paths_resolved={len(db_paths)}, "
            f"skipped={len(skipped)} sample_skipped={skipped[:2]}",
            flush=True,
        )
        if not db_paths:
            print("🧪 [MAST3R INSIDE] No DB paths resolved — returning empty result.", flush=True)
            return None, {"image_points": [], "object_points": []}, []

        if hasattr(mast3r_matcher, 'match_batch') and len(db_paths) > 1:
            batch_results = mast3r_matcher.match_batch(query_img_path, db_paths)
        else:
            batch_results = [mast3r_matcher.match_pair(query_img_path, p) for p in db_paths]

        none_count = 0
        small_count = 0
        sizes = []
        for r in batch_results:
            if r is None:
                none_count += 1
                continue
            q, db, conf = r
            if q is None or db is None:
                none_count += 1
                continue
            n = len(q)
            sizes.append(n)
            if n < min_inliers:
                small_count += 1
        print(
            f"🧪 [MAST3R INSIDE] match results: total={len(batch_results)}, "
            f"none_or_invalid={none_count}, below_min_inliers={small_count}, "
            f"sizes_sample={sizes[:5]}",
            flush=True,
        )

        result = original(
            query_img_path=query_img_path,
            candidates_data=candidates_data,
            mast3r_matcher=mast3r_matcher,
            colmap_models=colmap_models,
            max_nn_dist=max_nn_dist,
            min_inliers=min_inliers,
            max_candidates=max_candidates,
            early_stop_inliers=early_stop_inliers,
            pp=pp,
            data_roots=data_roots,
        )
        return result

    traced._unav_traced = True
    _matcher_mod.mast3r_matching_and_pnp = traced


_wrap_mast3r_matcher()


def _wrap_upstream_localizer():
    """Instrument the upstream UNavLocalizer (the one actually used at runtime)."""
    import functools
    from unav.localizer import localizer as _upstream
    if getattr(_upstream.UNavLocalizer.localize, "_unav_traced", False):
        return
    orig_localize = _upstream.UNavLocalizer.localize
    orig_match = _upstream.UNavLocalizer.batch_local_matching_and_ransac

    @functools.wraps(orig_localize)
    def traced_localize(self, query_img, refinement_queue, top_k=None, **kwargs):
        print(
            f"🧪 [UPSTREAM LOCALIZE] shape={query_img.shape}, top_k={top_k}, "
            f"use_mast3r={getattr(self, 'use_mast3r', 'N/A')}, "
            f"local_matcher={type(self.local_matcher).__name__}",
            flush=True,
        )
        result = orig_localize(self, query_img, refinement_queue, top_k=top_k, **kwargs)
        print(
            f"🧪 [UPSTREAM LOCALIZE DONE] success={result.get('success')}, "
            f"stage={result.get('stage')}, reason={result.get('reason', '')[:80]}, "
            f"top_candidates={len(result.get('top_candidates', []) or [])}",
            flush=True,
        )
        return result

    @functools.wraps(orig_match)
    def traced_match(self, local_feat_dict, candidates_data, query_img_path=None):
        print(
            f"🧪 [UPSTREAM MATCH] use_mast3r={getattr(self, 'use_mast3r', 'N/A')}, "
            f"candidates={len(candidates_data)}, query_img_path={query_img_path}, "
            f"query_exists={os.path.exists(query_img_path) if query_img_path else False}",
            flush=True,
        )
        result = orig_match(self, local_feat_dict, candidates_data, query_img_path=query_img_path)
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
    _upstream.UNavLocalizer.localize = traced_localize
    _upstream.UNavLocalizer.batch_local_matching_and_ransac = traced_match


_wrap_upstream_localizer()


class UNavLocalizer:
    """
    Thin compat shim — the real UNavLocalizer is `unav.localizer.localizer.UNavLocalizer`,
    imported and used at runtime by `logic/maps.py`. This class is here only so
    that any direct import of `from .localizer import UNavLocalizer` still works.
    """

    def __init__(self, config):
        from unav.localizer.localizer import UNavLocalizer as _Real
        self._impl = _Real(config)

    def __getattr__(self, name):
        return getattr(self._impl, name)

