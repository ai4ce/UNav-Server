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
    """Monkey-patch mast3r_matching_and_pnp to log DB resolution and per-candidate match results."""
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

class UNavLocalizer:
    """
    UNavLocalizer: Unified Visual Place Recognition and Pose Estimation for UNav System

    - Responsible for managing all models, maps, and feature data for large-scale visual localization.
    - All heavy data loading is separated from initialization for efficiency and scalability.
    - Modular design supports multi-building, multi-floor, and multi-map environments.
    """

    def __init__(self, config):
        """
        Initialize the localizer with system configuration.
        Only sets up model pointers; heavy map/feature data are loaded on demand.

        Args:
            config: Configuration object containing all system parameters and paths.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_extractor = None
        self.global_extractor = None
        self.local_matcher = None
        self._init_models()

        # Data containers for loaded map/models/features
        self.all_colmap_models = {}      # {place__building__floor: frames_by_name}
        self.all_global_features = {}    # {place__building__floor: (features, names)}
        self.global_feat_paths = {}      # {place__building__floor: h5_path}
        self.local_feat_paths = {}       # {place__building__floor: h5_path}
        self.transform_matrices = {}     # {place__building__floor: np.ndarray or None}

    def _init_models(self):
        """
        Initialize local and global feature extraction models (but not map/features).
        """
        feat_cfg = self.config.feature_extraction_config
        print(
            f"[INFO] Initializing models: "
            f"Local -> {self.config.local_feature_model} | "
            f"Global -> {self.config.global_descriptor_model}"
        )
        self.local_extractor = Local_extractor(feat_cfg["local_extractor_config"]).extractor()
        self.local_matcher = Local_extractor(feat_cfg["local_extractor_config"]).matcher().to(self.device)
        self.global_extractor = GlobalExtractors(
            feat_cfg["parameters_root"],
            {self.config.global_descriptor_model: feat_cfg["global_descriptor_config"]},
            data_parallel=False
        )
        self.global_extractor.set_train(False)

    def load_maps_and_features(self):
        """
        Load all COLMAP models, features, and transformation matrices for all regions.
        Should be called after __init__, or whenever maps are updated.
        """
        for place, bld_dict in self.config.places.items():
            for building, floors in bld_dict.items():
                for floor in floors:
                    key = (place, building, floor)
                    feature_dir = os.path.join(self.config.data_final_root, place, building, floor, "features")
                    self.global_feat_paths[key] = os.path.join(feature_dir, f"global_features_{self.config.global_descriptor_model}.h5")
                    self.local_feat_paths[key] = os.path.join(feature_dir, "local_features.h5")
                    model_dir = os.path.join(self.config.data_final_root, place, building, floor, "colmap_map")
                    transform_path = os.path.join(self.config.data_final_root, place, building, floor, "transform_matrix.npy")
                    # Load COLMAP model
                    try:
                        frames_by_name = load_colmap_model(model_dir, ext=".bin")
                        self.all_colmap_models[key] = frames_by_name
                        print(f"[✓] Loaded COLMAP model for {key}: {len(frames_by_name)} frames")
                    except Exception as e:
                        print(f"[WARNING] Could not load COLMAP model for {key}: {e}")
                    # Load global features
                    h5_path = self.global_feat_paths[key]
                    if os.path.exists(h5_path):
                        try:
                            feats, names = load_global_features(h5_path)
                            self.all_global_features[key] = (feats, names)
                            print(f"[✓] Loaded global features for {key}: {len(names)} images")
                        except Exception as e:
                            print(f"[WARNING] Could not load global features for {key}: {e}")
                    # Load transformation matrix if present
                    if os.path.exists(transform_path):
                        try:
                            matrix = np.load(transform_path)
                            self.transform_matrices[key] = matrix
                            print(f"[✓] Loaded transform matrix for {key}: shape={matrix.shape}")
                        except Exception as e:
                            print(f"[WARNING] Could not load transform matrix for {key}: {e}")
                            self.transform_matrices[key] = None
                    else:
                        self.transform_matrices[key] = None
        print("[INFO] All map and feature loading complete.")

    def extract_query_features(self, query_img: np.ndarray):
        """
        Extract global and local features from the query image using the loaded models.

        Args:
            query_img (np.ndarray): Query image (H, W, 3)

        Returns:
            Tuple of (global_feature, local_feature_dict)
        """
        return extract_query_features(
            query_img,
            self.global_extractor,
            self.local_extractor,
            self.config.global_descriptor_model,
            self.device
        )

    def vpr_retrieve(self, global_feat, top_k=None):
        """
        Run visual place recognition retrieval to get top-K candidates.

        Args:
            global_feat: Query image global feature.
            top_k (int, optional): Number of top matches to return.

        Returns:
            List of (map_key, img_name, score) tuples.
        """
        topk = top_k or self.config.localization_config.get("topk", 5)
        return search_vpr_topk_candidates(
            query_feature=global_feat,
            all_map_features=self.all_global_features,
            top_k=topk,
            device=str(self.device)
        )

    def get_candidates_data(self, top_candidates):
        """
        Load all local features and COLMAP metadata for the VPR candidate set.

        Args:
            top_candidates: List of (map_key, img_name, score) tuples.

        Returns:
            Dict mapping image name to data needed for local matching.
        """
        return fetch_candidates_data(
            self.all_colmap_models,
            self.local_feat_paths,
            top_candidates,
            load_local_features
        )

    def batch_local_matching_and_ransac(self, local_feat_dict, candidates_data):
        """
        Perform local matching and geometric verification in batch.

        Args:
            local_feat_dict: Query local features dict.
            candidates_data: Dict of reference image data.

        Returns:
            best_map_key (str): Map region with most inliers.
            pnp_pairs (dict): All correspondences for pose estimation.
            results (list): Per-candidate match info.
        """
        local_model = getattr(self.config, "local_feature_model", None)
        if local_model == "mast3r":
            from unav.localizer.tools.matcher import mast3r_matching_and_pnp
            print(
                f"🧪 [MAST3R DISPATCH] Dispatching to mast3r_matching_and_pnp, "
                f"candidates={len(candidates_data)}, matcher={type(self.local_matcher).__name__}",
                flush=True,
            )
            data_roots = [
                getattr(self.config, "data_temp_root", None),
                getattr(self.config, "data_final_root", None),
            ]
            data_roots = [r for r in data_roots if r]
            query_img_path = getattr(self, "_current_query_img_path", None)
            print(
                f"🧪 [MAST3R INPUTS] query_img_path={query_img_path}, "
                f"query_exists={os.path.exists(query_img_path) if query_img_path else False}, "
                f"data_roots={data_roots}",
                flush=True,
            )

            from unav.localizer.tools.matcher import _resolve_db_image_path
            sample_name = next(iter(candidates_data.keys()), None) if candidates_data else None
            sample_path = None
            if sample_name and candidates_data:
                cand = candidates_data[sample_name]
                place, building, floor = cand["map_key"]
                sample_path = _resolve_db_image_path(data_roots, place, building, floor, sample_name)
            print(
                f"🧪 [MAST3R DB RESOLVE] sample_candidate={sample_name}, "
                f"resolved_path={sample_path}, exists={os.path.exists(sample_path) if sample_path else False}",
                flush=True,
            )

            result = mast3r_matching_and_pnp(
                query_img_path=query_img_path,
                candidates_data=candidates_data,
                mast3r_matcher=self.local_matcher,
                colmap_models=self.all_colmap_models,
                max_nn_dist=self.config.feature_extraction_config.get("local_extractor_config", {}).get("mast3r", {}).get("max_nn_dist", 20.0),
                min_inliers=self.config.localization_config.get("min_inliers", 6),
                max_candidates=10,
                early_stop_inliers=80,
                data_roots=data_roots,
            )
            best_map_key, pnp_pairs, results = result
            print(
                f"🧪 [MAST3R RESULT] best_map_key={best_map_key}, "
                f"results_count={len(results) if results else 0}, "
                f"pnp_pairs_keys={list(pnp_pairs.keys()) if isinstance(pnp_pairs, dict) else type(pnp_pairs).__name__}",
                flush=True,
            )
            return result

        print(
            f"🧪 [SUPERPOINT DISPATCH] Dispatching to batch_local_matching_and_ransac, "
            f"candidates={len(candidates_data)}, matcher={type(self.local_matcher).__name__}",
            flush=True,
        )
        return batch_local_matching_and_ransac(
            local_feat_dict,
            candidates_data,
            matcher=self.local_matcher,
            feature_score_threshold=self.config.localization_config.get("feature_score_threshold", 0.09),
            min_inliers=self.config.localization_config.get("min_inliers", 50),
            device=self.device
        )

    def multi_frame_pose_refine(self, pnp_pairs, img_shape, refinement_queue):
        """
        Multi-frame pose refinement (PnP filtering + queue update).

        Args:
            pnp_pairs: Dict of 2D-3D correspondences for this region.
            img_shape: Shape of query image.
            refinement_queue: History queue for pose refinement.

        Returns:
            Dict containing pose, queue, success, etc.
        """
        return refine_pose_from_queue(
            current_pairs=pnp_pairs,
            current_img_shape=img_shape,
            refinement_queue=refinement_queue,
            max_history=self.config.localization_config.get("max_history", 5)
        )

    def transform_pose_to_floorplan(self, qvec, tvec, transform_matrix):
        """
        Transform a 6-DoF COLMAP pose to floorplan (2D+theta) coordinates.

        Args:
            qvec (np.ndarray): Quaternion (w, x, y, z)
            tvec (np.ndarray): Translation (x, y, z)
            transform_matrix (np.ndarray): Floorplan transform

        Returns:
            Dict with position and angle on the floorplan, or None if unavailable.
        """
        return transform_pose_to_floorplan(qvec, tvec, transform_matrix)

    def localize(
        self,
        query_img: np.ndarray,
        refinement_queue: dict,
        top_k: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Full end-to-end localization pipeline.

        Args:
            query_img (np.ndarray): Input image (H, W, 3)
            refinement_queue (dict): Dict tracking pose history for each map_key
            top_k (int, optional): Number of VPR candidates (default from config)

        Returns:
            Dict with keys: success, qvec, tvec, floorplan_pose, results, top_candidates,
            n_frames, refinement_queue, best_map_key, localization_time, etc.
            On failure, always includes: success=False, reason, stage, timings
        """
        # Start total localization timer
        start_time = time.time()
        timings = {}
        t0 = start_time
        print(f"🧪 [LOCALIZE ENTRY] query_img.shape={query_img.shape}, top_k={top_k}", flush=True)

        # 1. Extract features from query image
        try:
            global_feat, local_feat_dict = self.extract_query_features(query_img)
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during feature extraction: {e}",
                "stage": "extract_query_features",
                "timings": timings
            }
        t1 = time.time()
        timings['extract_query_features'] = t1 - t0
        t0 = t1
        print(f"🧪 [STEP 1 DONE] extract_query_features={t1 - start_time:.3f}s", flush=True)

        # 2. VPR: retrieve top candidates
        try:
            top_candidates = self.vpr_retrieve(global_feat, top_k=top_k)
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during VPR retrieval: {e}",
                "stage": "vpr_retrieve",
                "timings": timings
            }
        t1 = time.time()
        timings['vpr_retrieve'] = t1 - t0
        t0 = t1
        print(f"🧪 [STEP 2 DONE] vpr_retrieve={t1 - start_time:.3f}s, n_candidates={len(top_candidates) if top_candidates else 0}", flush=True)

        if not top_candidates:
            return {
                "success": False,
                "reason": "VPR failed (no candidates found).",
                "stage": "vpr_retrieve",
                "timings": timings
            }

        # 3. Gather map/model/feature data for all candidates
        try:
            candidates_data = self.get_candidates_data(top_candidates)
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during candidates data gathering: {e}",
                "stage": "get_candidates_data",
                "top_candidates": top_candidates,
                "timings": timings
            }
        t1 = time.time()
        timings['get_candidates_data'] = t1 - t0
        t0 = t1

        if not candidates_data:
            return {
                "success": False,
                "reason": "No candidate data found.",
                "stage": "get_candidates_data",
                "top_candidates": top_candidates,
                "timings": timings
            }

        # 4. Local matching + RANSAC, grouped by region/map_key
        try:
            local_model = getattr(self.config, "local_feature_model", None)
            candidate_names = list(candidates_data.keys()) if candidates_data else []
            print(
                f"🧪 [LOCAL MATCH] local_feature_model={local_model}, "
                f"matcher_type={type(self.local_matcher).__name__}, "
                f"candidates={len(candidate_names)}",
                flush=True,
            )
            if local_model == "mast3r":
                from unav.localizer.tools.matcher import _resolve_db_image_path
                data_roots = [
                    getattr(self.config, "data_temp_root", None),
                    getattr(self.config, "data_final_root", None),
                ]
                data_roots = [r for r in data_roots if r]
                sample_name = candidate_names[0] if candidate_names else None
                sample_path = (
                    _resolve_db_image_path(
                        data_roots,
                        top_candidates[0][0][0],
                        top_candidates[0][0][1],
                        top_candidates[0][0][2],
                        sample_name,
                    )
                    if sample_name and top_candidates
                    else None
                )
                print(
                    f"🧪 [MAST3R DB LOOKUP] data_roots={data_roots}, "
                    f"sample_candidate={sample_name}, resolved_path={sample_path}, "
                    f"exists={os.path.exists(sample_path) if sample_path else False}",
                    flush=True,
                )
        except Exception as e:
            print(f"⚠️ [LOCAL MATCH DEBUG] logging failed: {e}", flush=True)

        try:
            best_map_key, pnp_pairs, results = self.batch_local_matching_and_ransac(local_feat_dict, candidates_data)
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during local matching & RANSAC: {e}",
                "stage": "batch_local_matching_and_ransac",
                "top_candidates": top_candidates,
                "timings": timings
            }
        t1 = time.time()
        timings['batch_local_matching_and_ransac'] = t1 - t0
        t0 = t1

        if best_map_key is None or not results:
            return {
                "success": False,
                "reason": "No candidates passed local matching + RANSAC.",
                "stage": "batch_local_matching_and_ransac",
                "top_candidates": top_candidates,
                "timings": timings
            }

        # 5. Pose refinement (multi-frame queue) for this region only
        map_queue = refinement_queue.get(best_map_key, {
            "pairs": [], "initial_poses": [], "pps": []
        })
        try:
            refine_result = self.multi_frame_pose_refine(pnp_pairs, query_img.shape, map_queue)
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during multi-frame pose refinement: {e}",
                "stage": "multi_frame_pose_refine",
                "top_candidates": top_candidates,
                "best_map_key": best_map_key,
                "timings": timings
            }
        t1 = time.time()
        timings['multi_frame_pose_refine'] = t1 - t0
        t0 = t1

        if not refine_result["success"]:
            return {
                "success": False,
                "reason": refine_result.get("reason", "Pose refinement failed."),
                "stage": "multi_frame_pose_refine",
                "top_candidates": top_candidates,
                "best_map_key": best_map_key,
                "timings": timings
            }

        # 6. Transform output pose to floorplan coordinates if possible
        colmap_pose = {"qvec": refine_result.get("qvec"), "tvec": refine_result.get("tvec")}
        transform_matrix = self.transform_matrices.get(best_map_key, None)
        try:
            floorplan_pose = (
                transform_pose_to_floorplan(colmap_pose["qvec"], colmap_pose["tvec"], transform_matrix)
                if (colmap_pose["tvec"] is not None and transform_matrix is not None)
                else None
            )
        except Exception as e:
            return {
                "success": False,
                "reason": f"Exception during pose transformation: {e}",
                "stage": "transform_pose_to_floorplan",
                "top_candidates": top_candidates,
                "best_map_key": best_map_key,
                "timings": timings
            }
        t1 = time.time()
        timings['transform_pose_to_floorplan'] = t1 - t0
        t0 = t1

        # 7. Update refinement queue for just this map region
        updated_queue = refinement_queue.copy()
        updated_queue[best_map_key] = refine_result["new_refinement_queue"]

        # 8. Output structured result
        localization_time = time.time() - start_time
        timings['total'] = localization_time

        output = {
            "success": True,
            "qvec": refine_result.get("qvec"),
            "tvec": refine_result.get("tvec"),
            "floorplan_pose": floorplan_pose,
            "results": results,
            "top_candidates": top_candidates,
            "n_frames": refine_result.get("n_frames"),
            "refinement_queue": updated_queue,
            "best_map_key": best_map_key,
            "timings": timings
        }
        return output