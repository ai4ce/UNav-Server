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