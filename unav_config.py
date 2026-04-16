import os
import cv2
import yaml
from typing import Dict, List, Any, Optional

class UNavConfig:
    """
    Unified configuration container for the UNav system.
    Centralizes the management of mapping, localization, and navigation module configurations.
    """
    def __init__(
        self,
        data_temp_root: Optional[str] = None,
        data_final_root: str = "/mnt/data/UNav-IO/final",
        places: Dict[str, Dict[str, List[str]]] = {"New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"],"OtherBuilding": ["1_floor"]}},
        mapping_place: str = "New_York_City",
        mapping_building: str = "LightHouse",
        mapping_floor: str = "3_floor",
        global_descriptor_model: str = "DinoV2Salad",
        local_feature_model: str = "superpoint+lightglue",
        mapping_mode: str = "whole"
    ) -> None:
        """
        Initialize unified configuration for the entire UNav system.

        Args:
            data_temp_root (str): Path for temporary/intermediate files.
            data_final_root (str): Path for final output/results.
            places (Dict[str, Dict[str, List[str]]]): Supported places.
            mapping_place (str): Default mapping place.
            mapping_building (str): Default mapping building.
            mapping_floor (str): Default mapping floor.
            global_descriptor_model (str): Global descriptor model name.
            local_feature_model (str): Local feature extractor name.
        """
        self.data_final_dir: str = os.path.join(
            data_final_root, mapping_place, mapping_building, mapping_floor
        )
        # Prepare floorplan JSON dictionary for navigation
        building_jsons = {}
        for place, bld_dict in places.items():
            building_jsons[place] = {}
            for building, floors in bld_dict.items():
                building_jsons[place][building] = {}
                for floor in floors:
                    building_jsons[place][building][floor] = os.path.join(
                        data_final_root, place, building, floor, "boundaries.json"
                    )
        scale_file = os.path.join(data_final_root, "scale.json")
        
        # Only create mapping_config if data_temp_root is provided
        self.mapping_config = None
        if data_temp_root:
            self.mapping_config = UNavMappingConfig(
                data_temp_root=data_temp_root,
                data_final_root=data_final_root,
                place=mapping_place,
                building=mapping_building,
                floor=mapping_floor,
                global_descriptor_model=global_descriptor_model,
                local_feature_model=local_feature_model,
                mapping_mode=mapping_mode
            )
        self.localizer_config = UNavLocalizationConfig(
            data_final_root=data_final_root,
            places=places,
            global_descriptor_model=global_descriptor_model,
            local_feature_model=local_feature_model
        )
        self.navigator_config = UNavNavigationConfig(
            building_jsons=building_jsons,
            scale_file=scale_file
        )

    def to_dict(self) -> dict:
        """
        Export all configuration blocks as a nested dictionary.

        Returns:
            dict: A dictionary representing the full configuration.
        """
        result = {"mapping_config": self.mapping_config.to_dict()}
        if self.localizer_config is not None:
            result["localizer_config"] = self.localizer_config.to_dict()
        if self.navigator_config is not None:
            result["navigator_config"] = self.navigator_config.to_dict()
        return result

# -------------------------------- Mapping Config --------------------------------

class UNavMappingConfig:
    """
    Configuration class for the UNav Mapping pipeline.
    Supports flexible model selection and unifies data/model/output path management.
    """

    # ----------- Supported Models/Extractors -----------
    SUPPORTED_FRAME_EXTRACTORS: Dict[str, Dict[str, Any]] = {
        "default": {
            "frame_interval": 1,        # Extract every frame
            "img_ext": "jpg",           # Image file extension
            "resize": None,             # None or (width, height)
        }
    }
    SUPPORTED_GLOBAL_MODELS: Dict[str, Dict[str, Any]] = {
        "MixVPR": {
            "ckpt_path": 'parameters/MixVPR/ckpts/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt',
            "pt_img_size": [320, 320],
            "cuda": True,
            "model_resize": [320, 320]
        },
        "CricaVPR": {
            "ckpt_path": 'parameters/CricaVPR/ckpts/CricaVPR_clean.pth',
            "cuda": True,
            "model_resize": [320, 320]
        },
        "DinoV2Salad": {
            "ckpt_path": 'parameters/DinoV2Salad/ckpts/dino_salad.ckpt',
            "max_image_size": 1024,
            "num_channels": 384,
            "cuda": True,
            "model_resize": [224, 224]
        },
        "NetVlad": {
            "ckpt_path": 'parameters/netvlad/paper/checkpoints',
            "arch": 'vgg16',
            "num_clusters": 64,
            "pooling": 'netvlad',
            "vladv2": False,
            "nocuda": False,
            "model_resize": [480, 640]
        },
        "AnyLoc": {
            "model_type": 'dinov2_vitg14',
            "ckpt_path": 'None',
            "max_image_size": 1024,
            "desc_layer": 31,
            "desc_facet": 'value',
            "num_clusters": 32,
            "domain": 'indoor',
            "cache_dir": 'parameters/AnyLoc/demo/cache',
            "cuda": True,
            "model_resize": [224, 224]
        }
    }
    SUPPORTED_LOCAL_EXTRACTORS: Dict[str, Dict[str, Any]] = {
        "superpoint+lightglue": {
            "detector_name": "superpoint",
            "nms_radius": 4,
            "max_keypoints": 4096,
            "matcher_name": "lightglue",
            "match_conf": {
                "width_confidence": -1,
                "depth_confidence": -1
            }
        },
        "mast3r": {
            "model_name": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
            "mast3r_size": 512,
            "max_nn_dist": 30.0,
            "max_matches": 2000,
            "subsample": 8,
        }
    }

    def __init__(
        self,
        data_temp_root: str = "/mnt/data/UNav-IO/temp",
        data_final_root: str = "/mnt/data/UNav-IO/final",
        place: str = "New_York_City",
        building: str = "LightHouse",
        floor: str = "3_floor",
        global_descriptor_model: str = "DinoV2Salad",
        local_feature_model: str = "superpoint+lightglue",
        frame_extractor_name: str = "default",
        frame_extractor_overrides: Dict[str, Any] = None,
        mapping_mode: str = "segment"
    ) -> None:
        """
        Initialize the UNav mapping config.
        """
        self.data_temp_root: str = data_temp_root
        self.data_final_root: str = data_final_root
        self.place: str = place
        self.building: str = building
        self.floor: str = floor
        self.mapping_mode = mapping_mode
        self.global_descriptor_model: str = global_descriptor_model
        self.local_feature_model: str = local_feature_model
        self.data_temp_dir: str = os.path.join(data_temp_root, place, building, floor)
        self.data_final_dir: str = os.path.join(data_final_root, place, building, floor)
        self.frame_extractor_name: str = frame_extractor_name
        self.frame_extractor_config: Dict[str, Any] = self._init_frame_extractor_config(frame_extractor_overrides)
        self.slam_config: Dict[str, Any] = self._init_slam_config()
        self.aligner_config: Dict[str, Any] = self._init_aligner_config()
        self.slicer_config: Dict[str, Any] = self._init_slicing_config()
        self.feature_extraction_config: Dict[str, Any] = self._init_feature_extraction_config()
        self.matcher_config: Dict[str, Any] = self._init_matching_config()
        self.colmap_config: Dict[str, Any] = self._init_colmap_config()
        if not self.mapping_mode == "segment":
            self._generate_stella_vslam_yaml()

    def to_dict(self) -> Dict[str, Any]:
        """Export config as a nested dictionary."""
        return {
            "data_temp_root": self.data_temp_root,
            "data_final_root": self.data_final_root,
            "place": self.place,
            "building": self.building,
            "floor": self.floor,
            "global_descriptor_model": self.global_descriptor_model,
            "local_feature_model": self.local_feature_model,
            "frame_extractor_config": self.frame_extractor_config,
            "slam_config": self.slam_config,
            "aligner_config": self.aligner_config,
            "slicer_config": self.slicer_config,
            "feature_extraction_config": self.feature_extraction_config,
            "matcher_config": self.matcher_config,
            "colmap_config": self.colmap_config
        }

    def __repr__(self) -> str:
        return (f"<UNavMappingConfig {self.place}/{self.building}/{self.floor} "
                f"G: {self.global_descriptor_model} L: {self.local_feature_model}>")

    # ----------- Internal Init Functions -----------
    
    # ----------- Frame Extractor Config -----------
    def _init_frame_extractor_config(self, overrides: Dict[str, Any] = None) -> dict:
        """
        Config for perspective frame extraction from videos.
        """
        video_folder = os.path.join(self.data_temp_root, self.place, self.building, self.floor)
        perspective_folder = os.path.join(video_folder, "perspectives")
        # Select extractor profile
        extractor = self.SUPPORTED_FRAME_EXTRACTORS.get(self.frame_extractor_name, {}).copy()
        if overrides:
            extractor.update(overrides)
        # Define input/output dirs based on current config
        extractor.update({
            "input_folder": video_folder,
            "output_folder": perspective_folder,
        })
        return extractor
    
    # ----------- SLAM Config -----------
    def _init_slam_config(self) -> dict:
        """
        Initialize config for stella_vslam_dense. Container paths and host paths both managed.
        """
        container_data_root = "/data"
        host_data_root = self.data_temp_root

        output_base = os.path.join(container_data_root, self.place, self.building, self.floor, "stella_vslam_dense")
        return {
            "container_name": f"vslam_{self.floor}",
            "gpu_id": 0,
            "viewer": False,
            "vocab_path": os.path.join(container_data_root, "orb_vocab.fbow"),
            "config_yaml": os.path.join(container_data_root, "equirectangular.yaml"),
            "video_path": os.path.join(container_data_root, self.place, self.building, self.floor, f"{self.floor}.mp4"),
            "output_dir": output_base,
            "eval_log_dir": os.path.join(output_base, "eval_logs"),
            "map_db_out": os.path.join(output_base, "final_map.msg"),
            "pc_out": os.path.join(output_base, "output_cloud.ply"),
            "kf_out": os.path.join(output_base, "keyframes"),
            "host_data_root": host_data_root,
            "container_data_root": container_data_root,
            "host_eval_log_dir": os.path.join(self.data_temp_dir, "stella_vslam_dense", "eval_logs"),
            "host_keyframe_dir": os.path.join(self.data_temp_dir, "stella_vslam_dense", "keyframes")
        }

    # ----------- Aligner Config -----------
    def _init_aligner_config(self) -> dict:
        """
        Config for aligning SLAM point cloud and floorplan.
        """
        temp_dir = os.path.join(self.data_temp_dir, "aligner")
        return {
            "temp_dir": temp_dir,
            "final_dir": self.data_final_dir,
            "scale_file": os.path.join(self.data_final_root, 'scale.json'),
            "map_db_out": os.path.join(self.data_temp_dir, "stella_vslam_dense", "final_map.msg"),
            "floorplan_path": os.path.join(self.data_final_dir, 'floorplan.png')
        }

    # ----------- Slicing (Perspective Image) Config -----------
    def _init_slicing_config(self) -> dict:
        """
        Config for slicing equirectangular keyframes into perspective images.
        """
        return {
            "input_keyframe_dir": os.path.join(self.data_temp_dir, "stella_vslam_dense", "keyframes"),
            "trajectory_file": os.path.join(self.data_temp_dir, "stella_vslam_dense", "eval_logs", "keyframe_trajectory.txt"),
            "output_perspective_dir": os.path.join(self.data_temp_dir, "perspectives"),
            "rotate_along_local_y_axis": False,
            "num_perspectives": 18,
            "fov": 90,
            "pitch": 0
        }

    # ----------- Feature Extraction Config -----------
    def _init_feature_extraction_config(self) -> dict:
        """
        Config for local and global feature extraction, including all model configs.
        """
        if self.global_descriptor_model not in self.SUPPORTED_GLOBAL_MODELS:
            raise ValueError(f"Unsupported global descriptor model: {self.global_descriptor_model}")
        if self.local_feature_model not in self.SUPPORTED_LOCAL_EXTRACTORS:
            raise ValueError(f"Unsupported local feature extractor: {self.local_feature_model}")

        feature_dir = os.path.join(self.data_final_dir, "features")

        return {
            "parameters_root": self.data_final_root,
            "input_perspective_dir": os.path.join(self.data_temp_dir, "perspectives"),
            "output_feature_dir": feature_dir,
            "local_feature_model": self.local_feature_model,
            "local_extractor_config": {
                self.local_feature_model: self.SUPPORTED_LOCAL_EXTRACTORS[self.local_feature_model]
            },
            "global_descriptor_model": self.global_descriptor_model,
            "global_descriptor_config": self.SUPPORTED_GLOBAL_MODELS[self.global_descriptor_model],
            "local_feat_save_path": os.path.join(feature_dir, "local_features.h5"),
            "global_feat_save_path": os.path.join(feature_dir, f"global_features_{self.global_descriptor_model}.h5")
        }

    # ----------- Feature Matching Config -----------
    def _init_matching_config(self) -> dict:
        """
        Config for feature matching and geometric verification.
        """
        feature_dir = os.path.join(self.data_final_dir, "features")
        sfm_dir = os.path.join(self.data_temp_dir, "colmap_sfm")
        return {
            "feature_dir": feature_dir,
            "colmap_local_feature_file": os.path.join(sfm_dir, "local_features.txt"),
            "colmap_match_file": os.path.join(sfm_dir, "matches.txt"),
            "top_k_matches": 50,
            "min_keypoints": 50,
            "feature_score_threshold": 0.09,
            "gv_threshold_pos": 0.005,
            "gv_threshold_angle_deg": 10.0,
        }

    # ----------- COLMAP Config -----------
    def _init_colmap_config(self) -> dict:
        """
        Config for COLMAP triangulation, input/output paths.
        """
        colmap_read_root = os.path.join(self.data_temp_dir, "colmap_sfm")
        sparse_dir = os.path.join(colmap_read_root, "sparse", "0")
        colmap_output_dir = os.path.join(self.data_final_dir, "colmap_map")
        return {
            "sparse_dir": sparse_dir,
            "colmap_output_dir": colmap_output_dir,
            "database_path": os.path.join(colmap_read_root, "database.db"),
            "camera_file": os.path.join(sparse_dir, "cameras.txt"),
            "image_file": os.path.join(sparse_dir, "images.txt"),
            "pairs_txt": os.path.join(colmap_read_root, "pairs.txt"),
            "match_file": os.path.join(colmap_read_root, "matches.h5"),
        }

    # ----------- YAML Generation (for SLAM) -----------
    def _generate_stella_vslam_yaml(self) -> None:
        """
        Generate equirectangular.yaml for stella_vslam_dense based on video properties.
        """
        video_path = os.path.join(self.data_temp_root, self.place, self.building, self.floor, f"{self.floor}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        yaml_content = {
            'Camera': {
                'name': 'Insta360',
                'setup': 'monocular',
                'model': 'equirectangular',
                'fps': round(fps, 2),
                'cols': cols,
                'rows': rows,
                'color_order': 'BGR'
            },
            'Preprocessing': {
                'min_size': 800,
                'mask_rectangles': [
                    [0.0, 1.0, 0.0, 0.1],
                    [0.0, 1.0, 0.84, 1.0],
                    [0.0, 0.2, 0.7, 1.0],
                    [0.8, 1.0, 0.7, 1.0]
                ]
            },
            'Feature': {
                'name': 'default ORB feature extraction setting',
                'scale_factor': 1.2,
                'num_levels': 8,
                'ini_fast_threshold': 20,
                'min_fast_threshold': 7
            },
            'Mapping': {
                'keyframe_insert_interval': 7,
                'baseline_dist_thr_ratio': 0.02,
                'redundant_obs_ratio_thr': 0.95,
            },
            'LoopDetector': {
                'enabled': True,
                'reject_by_graph_distance': True,
                'min_distance_on_graph': 50
            },
            'SocketPublisher': {
                'image_quality': 80
            },
            'PatchMatch': {
                'enabled': True,
                'cols': 640,
                'rows': 320,
                'min_patch_std_dev': 0,
                'patch_size': 7,
                'patchmatch_iterations': 4,
                'min_score': 0.1,
                'min_consistent_views': 3,
                'depthmap_queue_size': 5,
                'depthmap_same_depth_threshold': 0.08,
                'min_views': 1,
                'pointcloud_queue_size': 4,
                'pointcloud_same_depth_threshold': 0.08,
                'min_stereo_score': 0
            }
        }
        yaml_output_path = os.path.join(self.data_temp_root, "equirectangular.yaml")
        with open(yaml_output_path, 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"[✓] YAML written to: {yaml_output_path}.")

# -------------------------------- Localization Config --------------------------------

class UNavLocalizationConfig:
    """
    Unified configuration class for the UNav localization module.
    """
    SUPPORTED_GLOBAL_MODELS: Dict[str, Dict[str, Any]] = {
        "MixVPR": {
            "ckpt_path": 'parameters/MixVPR/ckpts/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt',
            "pt_img_size": [320, 320],
            "cuda": True,
            "model_resize": [320, 320]
        },
        "CricaVPR": {
            "ckpt_path": 'parameters/CricaVPR/ckpts/CricaVPR_clean.pth',
            "cuda": True,
            "model_resize": [320, 320]
        },
        "DinoV2Salad": {
            "ckpt_path": 'parameters/DinoV2Salad/ckpts/dino_salad.ckpt',
            "max_image_size": 1024,
            "num_channels": 384,
            "cuda": True,
            "model_resize": [224, 224]
        },
        "NetVlad": {
            "ckpt_path": 'parameters/netvlad/paper/checkpoints',
            "arch": 'vgg16',
            "num_clusters": 64,
            "pooling": 'netvlad',
            "vladv2": False,
            "nocuda": False,
            "model_resize": [480, 640]
        },
        "AnyLoc": {
            "model_type": 'dinov2_vitg14',
            "ckpt_path": 'None',
            "max_image_size": 1024,
            "desc_layer": 31,
            "desc_facet": 'value',
            "num_clusters": 32,
            "domain": 'indoor',
            "cache_dir": 'parameters/AnyLoc/demo/cache',
            "cuda": True,
            "model_resize": [224, 224]
        }
    }
    SUPPORTED_LOCAL_EXTRACTORS: Dict[str, Dict[str, Any]] = {
        "superpoint+lightglue": {
            "detector_name": "superpoint",
            "nms_radius": 4,
            "max_keypoints": 4096,
            "matcher_name": "lightglue",
            "match_conf": {
                "width_confidence": -1,
                "depth_confidence": -1
            }
        },
        "mast3r": {
            "model_name": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
            "mast3r_size": 512,
            "max_nn_dist": 30.0,
            "max_matches": 2000,
            "subsample": 8,
        }
    }
    def __init__(
        self,
        data_final_root: str = "/mnt/data/UNav-IO/final",
        places: Dict[str, Dict[str, List[str]]] = {"New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"],"OtherBuilding": ["1_floor"]}},
        global_descriptor_model: str = "DinoV2Salad",
        local_feature_model: str = "superpoint+lightglue"
    ) -> None:
        self.data_final_root: str = data_final_root
        self.places: Dict[str, Dict[str, List[str]]] = places
        self.global_descriptor_model: str = global_descriptor_model
        self.local_feature_model: str = local_feature_model
        self.feature_extraction_config: Dict[str, Any] = self._init_feature_extraction_config()
        self.localization_config: Dict[str, Any] = self._init_localizer_config()

    def _init_feature_extraction_config(self) -> dict:
        """
        Config for feature extraction including model configs.
        """
        if self.global_descriptor_model not in self.SUPPORTED_GLOBAL_MODELS:
            raise ValueError(f"Unsupported global descriptor model: {self.global_descriptor_model}")
        if self.local_feature_model not in self.SUPPORTED_LOCAL_EXTRACTORS:
            raise ValueError(f"Unsupported local feature extractor: {self.local_feature_model}")
        return {
            "parameters_root": self.data_final_root,
            "local_feature_model": self.local_feature_model,
            "local_extractor_config": {
                self.local_feature_model: self.SUPPORTED_LOCAL_EXTRACTORS[self.local_feature_model]
            },
            "global_descriptor_config": self.SUPPORTED_GLOBAL_MODELS[self.global_descriptor_model]
        }

    def _init_localizer_config(self) -> dict:
        """
        Config for localization hyperparameters.
        """
        return {
            "topk": 50,
            "min_keypoints": 10,
            "feature_score_threshold": 0.09
        }
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_extraction_config": self.feature_extraction_config,
            "localization_config": self.localization_config,
        }

# -------------------------------- Floor Map Config --------------------------------

class UNavFloorMapConfig:
    """
    Configuration class for the Floor Map Generation pipeline.
    Generates floor point cloud and 2D floor map from equirectangular keyframes using DA3 + SAM3.
    """

    def __init__(
        self,
        data_temp_root: str = "/mnt/data/UNav-IO/temp",
        data_final_root: str = "/mnt/data/UNav-IO/final",
        place: str = "New_York_City",
        building: str = "LightHouse",
        floor: str = "3_floor",
        num_images: int = 10,
        yaw_angles: List[int] = None,
        pitch_angles: List[int] = None,
        fov: float = 90.0,
        conf_thresh: float = 0.5,
        resolution: float = 0.02,
    ) -> None:
        """
        Initialize the Floor Depth Analyzer config.

        Args:
            data_temp_root: Root directory for temporary data
            data_final_root: Root directory for final output
            place: Place name
            building: Building name
            floor: Floor name
            num_images: Number of keyframes to process
            yaw_angles: List of yaw angles for slicing (default: 8 directions)
            pitch_angles: List of pitch angles for slicing (default: [0, -20])
            fov: Field of view for perspective slices
            conf_thresh: Depth confidence threshold
            resolution: Floor map resolution (m/pixel)
        """
        self.data_temp_root = data_temp_root
        self.data_final_root = data_final_root
        self.place = place
        self.building = building
        self.floor = floor

        # Processing parameters
        self.num_images = num_images
        self.yaw_angles = yaw_angles or [0, 45, 90, 135, 180, 225, 270, 315]
        self.pitch_angles = pitch_angles or [0, -20]
        self.fov = fov
        self.conf_thresh = conf_thresh
        self.resolution = resolution

        # Derived paths
        self.data_temp_dir = os.path.join(data_temp_root, place, building, floor)
        self.data_final_dir = os.path.join(data_final_root, place, building, floor)

        # Input paths (from SLAM output)
        self.keyframe_dir = os.path.join(
            self.data_temp_dir, "stella_vslam_dense", "keyframes"
        )
        self.trajectory_file = os.path.join(
            self.data_temp_dir, "stella_vslam_dense", "eval_logs", "keyframe_trajectory.txt"
        )

        # Output paths
        self.output_dir = os.path.join(self.data_temp_dir, "floor_map")
        self.floor_pointcloud_glb = os.path.join(self.output_dir, "floor_pointcloud.glb")
        self.floor_points_npy = os.path.join(self.output_dir, "floor_points.npy")
        self.floor_map_dir = os.path.join(self.output_dir, "floor_map")

    def to_dict(self) -> Dict[str, Any]:
        """Export config as a dictionary."""
        return {
            "data_temp_root": self.data_temp_root,
            "data_final_root": self.data_final_root,
            "place": self.place,
            "building": self.building,
            "floor": self.floor,
            "num_images": self.num_images,
            "yaw_angles": self.yaw_angles,
            "pitch_angles": self.pitch_angles,
            "fov": self.fov,
            "conf_thresh": self.conf_thresh,
            "resolution": self.resolution,
            "keyframe_dir": self.keyframe_dir,
            "trajectory_file": self.trajectory_file,
            "output_dir": self.output_dir,
        }

    def __repr__(self) -> str:
        return (f"<UNavFloorMapConfig {self.place}/{self.building}/{self.floor} "
                f"num_images={self.num_images}>")


# -------------------------------- Navigation Config --------------------------------

class UNavNavigationConfig:
    """
    Unified configuration class for the UNav navigation module.
    """
    def __init__(self, building_jsons: Dict[str, Dict[str, Dict[str, str]]], scale_file: Optional[str] = None):
        """
        Args:
            building_jsons (Dict): Hierarchical structure {place: {building: {floor: boundary_json_path}}}
            scale_file (str): Optional scale file for navigation (meters/pixel, etc.)
        """
        self.building_jsons: Dict[str, Dict[str, Dict[str, str]]] = building_jsons
        self.scale_file: Optional[str] = scale_file

    def to_dict(self) -> Dict[str, Any]:
        return {
            "building_jsons": self.building_jsons,
            "scale_file": self.scale_file
        }