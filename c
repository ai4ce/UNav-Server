[1mdiff --git a/.gitignore b/.gitignore[m
[1mdeleted file mode 100644[m
[1mindex 441aad5..0000000[m
[1m--- a/.gitignore[m
[1m+++ /dev/null[m
[36m@@ -1,145 +0,0 @@[m
[31m-# Byte-compiled / optimized / DLL files[m
[31m-__pycache__/[m
[31m-*.py[cod][m
[31m-*$py.class[m
[31m-[m
[31m-# C extensions[m
[31m-*.so[m
[31m-[m
[31m-# Distribution / packaging[m
[31m-.Python[m
[31m-build/[m
[31m-develop-eggs/[m
[31m-dist/[m
[31m-downloads/[m
[31m-eggs/[m
[31m-.eggs/[m
[31m-lib/[m
[31m-lib64/[m
[31m-parts/[m
[31m-sdist/[m
[31m-var/[m
[31m-wheels/[m
[31m-*.egg-info/[m
[31m-.installed.cfg[m
[31m-*.egg[m
[31m-[m
[31m-# PyInstaller[m
[31m-#  Usually these files are written by a python script from a template[m
[31m-#  before PyInstaller builds the exe, so as to inject date/other infos into it.[m
[31m-*.manifest[m
[31m-*.spec[m
[31m-[m
[31m-# Installer logs[m
[31m-pip-log.txt[m
[31m-pip-delete-this-directory.txt[m
[31m-[m
[31m-# Unit test / coverage reports[m
[31m-htmlcov/[m
[31m-.tox/[m
[31m-.coverage[m
[31m-.coverage.*[m
[31m-.cache[m
[31m-nosetests.xml[m
[31m-coverage.xml[m
[31m-*.cover[m
[31m-.hypothesis/[m
[31m-.pytest_cache/[m
[31m-[m
[31m-# Translations[m
[31m-*.mo[m
[31m-*.pot[m
[31m-[m
[31m-# Django stuff:[m
[31m-*.log[m
[31m-local_settings.py[m
[31m-db.sqlite3[m
[31m-db.sqlite3-journal[m
[31m-[m
[31m-# Flask stuff:[m
[31m-instance/[m
[31m-.webassets-cache[m
[31m-[m
[31m-# Scrapy stuff:[m
[31m-.scrapy[m
[31m-[m
[31m-# Sphinx documentation[m
[31m-docs/_build/[m
[31m-[m
[31m-# PyBuilder[m
[31m-target/[m
[31m-[m
[31m-# Jupyter Notebook[m
[31m-.ipynb_checkpoints[m
[31m-[m
[31m-# IPython[m
[31m-profile_default/[m
[31m-ipython_config.py[m
[31m-[m
[31m-# pyenv[m
[31m-.python-version[m
[31m-[m
[31m-# pipenv[m
[31m-#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.[m
[31m-#   However, in case of collaboration, if having platform-specific dependencies or dependencies[m
[31m-#   having no cross-platform support, pipenv may install dependencies that don't work, or not[m
[31m-#   install all needed dependencies.[m
[31m-#Pipfile.lock[m
[31m-[m
[31m-# PEP 582; used by e.g. github.com/David-OConnor/pyflow[m
[31m-__pypackages__/[m
[31m-[m
[31m-# Celery stuff[m
[31m-celerybeat-schedule[m
[31m-celerybeat.pid[m
[31m-[m
[31m-# SageMath parsed files[m
[31m-*.sage.py[m
[31m-[m
[31m-# Environments[m
[31m-.env[m
[31m-.venv[m
[31m-env/[m
[31m-venv/[m
[31m-ENV/[m
[31m-env.bak/[m
[31m-venv.bak/[m
[31m-work/[m
[31m-[m
[31m-# Spyder project settings[m
[31m-.spyderproject[m
[31m-.spyproject[m
[31m-[m
[31m-# Rope project settings[m
[31m-.ropeproject[m
[31m-[m
[31m-# mkdocs documentation[m
[31m-/site[m
[31m-[m
[31m-# mypy[m
[31m-.mypy_cache/[m
[31m-.dmypy.json[m
[31m-dmypy.json[m
[31m-[m
[31m-# Pyre type checker[m
[31m-.pyre/[m
[31m-[m
[31m-# pytype static type analyzer[m
[31m-.pytype/[m
[31m-[m
[31m-# Cython debug symbols[m
[31m-cython_debug/[m
[31m-[m
[31m-# CMake[m
[31m-CMakeFiles/[m
[31m-CMakeCache.txt[m
[31m-CMakeScripts/[m
[31m-CTestTestfile.cmake[m
[31m-cmake_install.cmake[m
[31m-install_manifest.txt[m
[31m-*.cmake[m
[31m-*.cmake.in[m
[31m-*.db[m
[31m-[m
[31m-#test files[m
[31m-test1.py[m
[1mdiff --git a/requirements.txt b/requirements.txt[m
[1mindex 661750d..e69de29 100644[m
[1m--- a/requirements.txt[m
[1m+++ b/requirements.txt[m
[36m@@ -1,20 +0,0 @@[m
[31m-Flask==2.2.5[m
[31m-Flask-Mail==0.9.1[m
[31m-Flask-SocketIO==5.1.1[m
[31m-Flask-SQLAlchemy==2.5.1[m
[31m-SQLAlchemy==1.4.47[m
[31m-PyYAML==6.0[m
[31m-Pillow==9.0.1[m
[31m-numpy==1.21.6[m
[31m-opencv-python==4.6.0.66[m
[31m-ipywidgets==7.7.2[m
[31m-matplotlib==3.5.3[m
[31m-h5py==3.7.0[m
[31m-torch==1.13.1[m
[31m-kornia==0.7.3[m
[31m-kornia_rs==0.1.5[m
[31m-torchvision==0.14.1[m
[31m-scikit-learn==1.3.2[m
[31m-pytorch-lightning==2.2.5[m
[31m-pytorch-metric-learning==2.6.0[m
[31m-modal==0.64.102[m
[1mdiff --git a/src/UNav_core/src/feature/global_extractor.py b/src/UNav_core/src/feature/global_extractor.py[m
[1mindex c454cfa..408489b 100644[m
[1m--- a/src/UNav_core/src/feature/global_extractor.py[m
[1m+++ b/src/UNav_core/src/feature/global_extractor.py[m
[36m@@ -6,31 +6,23 @@[m [mimport torch[m
 [m
 class Global_Extractors():[m
     def __init__(self, config):[m
[31m-        self.root = config['IO_root'][m
[32m+[m[32m        self.root=config['IO_root'][m
         self.extractor = config['feature']['global'][m
[31m-        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set device[m
 [m
     def netvlad(self, content):[m
[31m-        model = NetVladFeatureExtractor(join(self.root, content['ckpt_path']), [m
[31m-                                        arch=content['arch'],[m
[31m-                                        num_clusters=content['num_clusters'],[m
[31m-                                        pooling=content['pooling'], [m
[31m-                                        vladv2=content['vladv2'], [m
[31m-                                        nocuda=content['nocuda'])[m
[31m-        if hasattr(model, 'model'):  # Check if it has a model attribute[m
[31m-            model.model = model.model.to(self.device)  # Move model to the GPU[m
[31m-         # Move NetVlad model to GPU if available[m
[31m-        return model[m
[31m-[m
[31m-    def mixvpr(self, content):[m
[32m+[m[32m        return NetVladFeatureExtractor(join(self.root,content['ckpt_path']), arch=content['arch'],[m
[32m+[m[32m         num_clusters=content['num_clusters'],[m
[32m+[m[32m         pooling=content['pooling'], vladv2=content['vladv2'], nocuda=content['nocuda'])[m
[32m+[m[41m    [m
[32m+[m[32m    def mixvpr(self,content):[m
         model = VPRModel(backbone_arch=content["backbone_arch"], [m
[31m-                         layers_to_crop=[content['layers_to_crop']],[m
[31m-                         agg_arch=content['agg_arch'],[m
[31m-                         agg_config=content['agg_config'],[m
[31m-                         )[m
[31m-        state_dict = torch.load(content["ckpt_path"], map_location=self.device)  # Load model state to the same device[m
[32m+[m[32m                        layers_to_crop=[content['layers_to_crop']],[m
[32m+[m[32m                        agg_arch=content['agg_arch'],[m
[32m+[m[32m                        agg_config=content['agg_config'],[m
[32m+[m[32m                        )[m
[32m+[m
[32m+[m[32m        state_dict = torch.load(content["ckpt_path"])[m
         model.load_state_dict(state_dict)[m
[31m-        model = model.to(self.device)  # Move MixVPR model to GPU if available[m
         model.eval()[m
         return model[m
 [m
[36m@@ -43,4 +35,4 @@[m [mclass Global_Extractors():[m
             if extractor == 'vlad':[m
                 pass[m
             if extractor == 'bovw':[m
[31m-                pass[m
[32m+[m[32m                pass[m
\ No newline at end of file[m
[1mdiff --git a/src/UNav_core/src/track/hierarchical_localization.py b/src/UNav_core/src/track/hierarchical_localization.py[m
[1mindex 44d8ecc..51da7c2 100644[m
[1m--- a/src/UNav_core/src/track/hierarchical_localization.py[m
[1m+++ b/src/UNav_core/src/track/hierarchical_localization.py[m
[36m@@ -56,33 +56,25 @@[m [mclass Coarse_Locator:[m
         return torch.tensor(descriptors, dtype=torch.float32).to(self.device), segment_ids[m
 [m
     def coarse_vpr(self, image):[m
[31m-            """[m
[31m-            Perform coarse visual place recognition.[m
[31m-            :param image: The query image for which to find the place.[m
[31m-            :return: Top-k matches and a boolean indicating if the corresponding segment is found.[m
[31m-            """[m
[31m-            print("Starting coarse_vpr function")[m
[31m-            [m
[31m-            # Extract global descriptor from the query image[m
[31m-            query_desc = self.global_extractor(image).to(self.device)[m
[31m-            print(f"Extracted query descriptor: {query_desc.shape}")[m
[31m-    [m
[31m-            # Compute similarity between the query descriptor and database descriptors[m
[31m-            sim = torch.einsum('id,jd->ij', query_desc, self.global_descriptors)[m
[31m-            print(f"Computed similarity matrix: {sim.shape}")[m
[31m-            [m
[31m-            topk_indices = torch.topk(sim, self.config['retrieval_num'], dim=1).indices.cpu().numpy()[m
[31m-            print(f"Top-k indices: {topk_indices}")[m
[31m-    [m
[31m-            # Retrieve the corresponding segment IDs for the top-k matches[m
[31m-            topk_segments = self.segment_ids[topk_indices[0]][m
[31m-            print(f"Top-k segments: {topk_segments}")[m
[31m-            [m
[31m-            # Analyze top-k results[m
[31m-            segment, success = self.analyze_topk_results(topk_segments)[m
[31m-            print(f"Analyzed top-k results: segment={segment}, success={success}")[m
[31m-            [m
[31m-            return topk_segments, segment, success[m
[32m+[m[32m        """[m
[32m+[m[32m        Perform coarse visual place recognition.[m
[32m+[m[32m        :param image: The query image for which to find the place.[m
[32m+[m[32m        :return: Top-k matches and a boolean indicating if the corresponding segment is found.[m
[32m+[m[32m        """[m
[32m+[m[32m        # Extract global descriptor from the query image[m
[32m+[m[32m        query_desc = self.global_extractor(image).to(self.device)[m
[32m+[m
[32m+[m[32m        # Compute similarity between the query descriptor and database descriptors[m
[32m+[m[32m        sim = torch.einsum('id,jd->ij', query_desc, self.global_descriptors)[m
[32m+[m[32m        topk_indices = torch.topk(sim, self.config['retrieval_num'], dim=1).indices.cpu().numpy()[m
[32m+[m
[32m+[m[32m        # Retrieve the corresponding segment IDs for the top-k matches[m
[32m+[m[32m        topk_segments = self.segment_ids[topk_indices[0]][m
[32m+[m[41m        [m
[32m+[m[32m        # Analyze top-k results[m
[32m+[m[32m        segment, success = self.analyze_topk_results(topk_segments)[m
[32m+[m[41m        [m
[32m+[m[32m        return topk_segments, segment, success[m
     [m
     def get_topk_segments(self, topk_indices):[m
         """[m
[36m@@ -101,9 +93,6 @@[m [mclass Coarse_Locator:[m
         :param topk_segments: List of segment IDs corresponding to the top-k matches.[m
         :return: The most likely segment and a boolean indicating if localization succeeded.[m
         """[m
[31m-        print("Starting analyze_topk_results function")[m
[31m-        print(f"Top-k segments: {topk_segments}")[m
[31m-    [m
         segment_counts = {}[m
         [m
         # First, count occurrences of each segment in topk_segments[m
[36m@@ -113,11 +102,9 @@[m [mclass Coarse_Locator:[m
             else:[m
                 segment_counts[segment] = 1[m
         [m
[31m-        print(f"Segment counts: {segment_counts}")[m
[31m-        [m
         # Initialize a dictionary to accumulate counts for segments and their neighbors[m
         segment_wt_neighbor_counts = {}[m
[31m-    [m
[32m+[m
         # Accumulate counts including neighbor segments[m
         for segment, count in segment_counts.items():[m
             # Start with the count of the segment itself[m
[36m@@ -132,13 +119,12 @@[m [mclass Coarse_Locator:[m
             # Record the accumulated count for the segment[m
             segment_wt_neighbor_counts[segment] = total_count[m
         [m
[31m-        print(f"Segment with neighbor counts: {segment_wt_neighbor_counts}")[m
[31m-        [m
         # Determine the segment with the highest total count[m
         most_likely_segment = max(segment_wt_neighbor_counts, key=segment_wt_neighbor_counts.get)[m
         success = (segment_wt_neighbor_counts[most_likely_segment] / len(topk_segments)) >= 0.1[m
         [m
         return most_likely_segment, success[m
[32m+[m
     [m
     def get_segment_id(self, index):[m
         """[m
[36m@@ -239,20 +225,15 @@[m [mclass Hloc():[m
         with torch.inference_mode():  # Use torch.no_grad during inference[m
             image_np = np.array(image)[m
             feats0 = self.local_feature_extractor(image_np)[m
[31m-            image_np = np.array(image)[m
[31m-            feats0 = self.local_feature_extractor(image_np)[m
         pts0_list,pts1_list,lms_list=[],[],[][m
         max_len=0[m
         [m
[31m-        valid_db_frame_name = [][m
[31m-        [m
         valid_db_frame_name = [][m
         for i in topk[0]:[m
             pts0,pts1,lms=self.local_feature_matcher.lightglue(i, feats0)[m
             [m
             feat_inliner_size=pts0.shape[0][m
             if feat_inliner_size>self.thre:[m
[31m-                valid_db_frame_name.append(self.db_name[i])[m
                 valid_db_frame_name.append(self.db_name[i])[m
                 pts0_list.append(pts0)[m
                 pts1_list.append(pts1)[m
[36m@@ -263,7 +244,6 @@[m [mclass Hloc():[m
         del self.query_desc, feats0[m
         torch.cuda.empty_cache()[m
         return valid_db_frame_name, pts0_list,pts1_list,lms_list,max_len[m
[31m-        return valid_db_frame_name, pts0_list,pts1_list,lms_list,max_len[m
     [m
     def feature_matching_superglue(self,image,topk):[m
         """[m
[36m@@ -340,48 +320,33 @@[m [mclass Hloc():[m
         #     return None, torch.tensor([]), None[m
 [m
 [m
[31m-    def pnp(self, image, feature2D, landmark3D):[m
[32m+[m[32m    def pnp(self,image,feature2D,landmark3D):[m
         """[m
         Start Perspective-n-points:[m
             Estimate the current location using implicit distortion model[m
         """[m
[31m-        print("Starting pnp function")[m
[31m-        if feature2D.size()[0] > 0:[m
[31m-            print("Feature2D size is greater than 0")[m
[31m-[m
[31m-            if not isinstance(image, np.ndarray):[m
[31m-                image = np.array(image)[m
[31m-            [m
[32m+[m[32m        if feature2D.size()[0]>0:[m
             height, width, _ = image.shape[m
[31m-            print(f"Image shape: height={height}, width={width}")[m
[31m-            feature2D, landmark3D = feature2D.cpu().numpy(), landmark3D.cpu().numpy()[m
[31m-            print("Converted feature2D and landmark3D to numpy arrays")[m
[32m+[m[32m            feature2D, landmark3D=feature2D.cpu().numpy(),landmark3D.cpu().numpy()[m
             out, p2d_inlier, p3d_inlier = coarse_pose(feature2D, landmark3D, np.array([width / 2, height / 2]))[m
[31m-            print(f"Coarse pose output: {out}")[m
             self.list_2d.append(p2d_inlier)[m
             self.list_3d.append(p3d_inlier)[m
             self.initial_poses.append(out['pose'])[m
             self.pps.append(out['pp'])[m
[31m-            print("Appended inliers and initial pose to lists")[m
             if len(self.list_2d) > self.config['implicit_num']:[m
[31m-                print("List sizes exceeded implicit_num, popping oldest elements")[m
                 self.list_2d.pop(0)[m
                 self.list_3d.pop(0)[m
                 self.initial_poses.pop(0)[m
                 self.pps.pop(0)[m
[31m-            pose = pose_multi_refine(self.list_2d, self.list_3d, self.initial_poses, self.pps, self.rot_base, self.T)[m
[31m-            print(f"Refined pose: {pose}")[m
[31m-    [m
[31m-            # Reset reload num[m
[32m+[m[32m            pose = pose_multi_refine(self.list_2d, self.list_3d, self.initial_poses, self.pps,self.rot_base,self.T)[m
[32m+[m
[32m+[m[32m            #reset reload num[m
             self.current_reload_num = 0[m
[31m-            print("Reset current_reload_num to 0")[m
         else:[m
[31m-            pose = None[m
[32m+[m[32m            pose =None[m
             self.logger.warning("!!!Cannot localize at this point, please take some steps or turn around!!!")[m
[31m-            print("Feature2D size is 0, cannot localize")[m
[31m-        print("Returning pose")[m
         return pose[m
[31m-    [m
[32m+[m
     def _determine_next_segment(self, candidates):[m
         candidate_histogram = {}[m
         max_counts = 0[m
[1mdiff --git a/src/modal_functions/config.yaml b/src/modal_functions/config.yaml[m
[1mdeleted file mode 100644[m
[1mindex 9992ff6..0000000[m
[1m--- a/src/modal_functions/config.yaml[m
[1m+++ /dev/null[m
[36m@@ -1,43 +0,0 @@[m
[31m-server:[m
[31m-  host: "0.0.0.0"[m
[31m-  port: 5000[m
[31m-[m
[31m-location:[m
[31m-  place: New_York_City[m
[31m-  building: LightHouse[m
[31m-  floor: "6_floor"[m
[31m-  scale: 0.01098358101[m
[31m-[m
[31m-IO_root: "/root/UNav-IO"[m
[31m-[m
[31m-devices: "cuda:0"[m
[31m-[m
[31m-hloc:[m
[31m-  retrieval_num: 50[m
[31m-  implicit_num: 1[m
[31m-  ransac_thre: 10[m
[31m-  # match_type: 'nvs'[m
[31m-  match_type: "lightglue"[m
[31m-  batch_mode: true[m
[31m-  load_all_maps: False[m
[31m-  map_loading_keyframes_reload: 0[m
[31m-[m
[31m-feature:[m
[31m-  global:[m
[31m-    netvlad:[m
[31m-      ckpt_path: "parameters/paper"[m
[31m-      arch: "vgg16"[m
[31m-      vladv2: true[m
[31m-      nocuda: false[m
[31m-      num_clusters: 64[m
[31m-      pooling: "netvlad"[m
[31m-[m
[31m-  local:[m
[31m-    superpoint+lightglue:[m
[31m-      detector_name: superpoint[m
[31m-      nms_radius: 4[m
[31m-      max_keypoints: 4096[m
[31m-      matcher_name: lightglue[m
[31m-      match_conf:[m
[31m-        width_confidence: -1[m
[31m-        depth_confidence: -1[m
[1mdiff --git a/src/modal_functions/logger_utils.py b/src/modal_functions/logger_utils.py[m
[1mdeleted file mode 100644[m
[1mindex 7130edf..0000000[m
[1m--- a/src/modal_functions/logger_utils.py[m
[1m+++ /dev/null[m
[36m@@ -1,15 +0,0 @@[m
[31m-import logging[m
[31m-[m
[31m-[m
[31m-def setup_logger(name="server_logger", level=logging.DEBUG):[m
[31m-    logger = logging.getLogger(name)[m
[31m-    logger.setLevel(level)[m
[31m-[m
[31m-    handler = logging.StreamHandler()[m
[31m-    formatter = logging.Formatter([m
[31m-        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"[m
[31m-    )[m
[31m-    handler.setFormatter(formatter)[m
[31m-[m
[31m-    logger.addHandler(handler)[m
[31m-    return logger[m
[1mdiff --git a/src/modal_functions/misc/sample.png b/src/modal_functions/misc/sample.png[m
[1mdeleted file mode 100644[m
[1mindex f2cbe38..0000000[m
Binary files a/src/modal_functions/misc/sample.png and /dev/null differ
[1mdiff --git a/src/modal_functions/misc/sample2.png b/src/modal_functions/misc/sample2.png[m
[1mdeleted file mode 100644[m
[1mindex d374275..0000000[m
Binary files a/src/modal_functions/misc/sample2.png and /dev/null differ
[1mdiff --git a/src/modal_functions/misc/sample3.png b/src/modal_functions/misc/sample3.png[m
[1mdeleted file mode 100644[m
[1mindex aa3e5a8..0000000[m
Binary files a/src/modal_functions/misc/sample3.png and /dev/null differ
[1mdiff --git a/src/modal_functions/modal_config.py b/src/modal_functions/modal_config.py[m
[1mdeleted file mode 100644[m
[1mindex 6215848..0000000[m
[1m--- a/src/modal_functions/modal_config.py[m
[1m+++ /dev/null[m
[36m@@ -1,68 +0,0 @@[m
[31m-from modal import App, Image, Mount, NetworkFileSystem, Volume[m
[31m-from pathlib import Path[m
[31m-[m
[31m-volume = Volume.from_name("Visiondata")[m
[31m-[m
[31m-MODEL_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"[m
[31m-LIGHTGLUE_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"[m
[31m-[m
[31m-# Get the current file's directory[m
[31m-current_dir = Path(__file__).resolve().parent[m
[31m-[m
[31m-# Construct the path to the src directory[m
[31m-local_dir = current_dir / ".."[m
[31m-[m
[31m-[m
[31m-def download_torch_hub_weights():[m
[31m-    import torch[m
[31m-    model_weights = torch.hub.load_state_dict_from_url(MODEL_URL, progress=True)[m
[31m-    torch.save(model_weights, "vgg16_weights.pth")[m
[31m-[m
[31m-    lightglue_weights = torch.hub.load_state_dict_from_url(LIGHTGLUE_URL, progress=True)[m
[31m-    torch.save(lightglue_weights,"superpoint_lightglue_v0-1_arxiv-pth")[m
[31m-[m
[31m-[m
[31m-app = App([m
[31m-    name="unav-server",[m
[31m-    mounts=[[m
[31m-        Mount.from_local_dir(local_dir.resolve(), remote_path="/root"),[m
[31m-        Mount.from_local_file([m
[31m-            "modal_functions/config.yaml", remote_path="/root/config.yaml"[m
[31m-        ),[m
[31m-    ],[m
[31m-)[m
[31m-[m
[31m-unav_image = ([m
[31m-    Image.debian_slim(python_version="3.8")[m
[31m-    .run_commands([m
[31m-        "apt-get update",[m
[31m-        "apt-get install -y cmake git libgl1-mesa-glx libceres-dev libsuitesparse-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev",[m
[31m-    )[m
[31m-    .run_commands("git clone https://gitlab.com/libeigen/eigen.git eigen")[m
[31m-    .workdir("/eigen")[m
[31m-    .run_commands([m
[31m-        "git checkout 3.4",[m
[31m-        "mkdir build",[m
[31m-    )[m
[31m-    .workdir("/eigen/build")[m
[31m-    .run_commands([m
[31m-        "cmake ..",[m
[31m-        "make",[m
[31m-        "make install",[m
[31m-    )[m
[31m-    .workdir("/")[m
[31m-    .run_commands([m
[31m-        "git clone https://github.com/cvg/implicit_dist.git implicit_dist",[m
[31m-    )[m
[31m-    .workdir("/implicit_dist")[m
[31m-    .run_commands([m
[31m-        "ls",[m
[31m-        "python3 -m venv .venv",[m
[31m-        ". .venv/bin/activate",[m
[31m-        "pip install .",[m
[31m-        "pip freeze",[m
[31m-    )[m
[31m-    .pip_install_from_requirements("modal_functions/modal_requirements.txt")[m
[31m-    .workdir("/root")[m
[31m-    .run_function(download_torch_hub_weights)[m
[31m-)[m
[1mdiff --git a/src/modal_functions/modal_requirements.txt b/src/modal_functions/modal_requirements.txt[m
[1mdeleted file mode 100644[m
[1mindex 769a6e0..0000000[m
[1m--- a/src/modal_functions/modal_requirements.txt[m
[1m+++ /dev/null[m
[36m@@ -1,22 +0,0 @@[m
[31m-Pillow==9.0.1[m
[31m-numpy==1.21.6[m
[31m-unav==0.1.40[m
[31m-prettytable==3.11.0[m
[31m-timm==1.0.9[m
[31m-einops==0.8.0[m
[31m-ipywidgets==7.7.2[m
[31m-matplotlib==3.5.3[m
[31m-h5py==3.7.0[m
[31m-torch==1.13.1[m
[31m-kornia==0.7.3[m
[31m-kornia_rs==0.1.5[m
[31m-torchvision==0.14.1[m
[31m-scikit-learn==1.3.2[m
[31m-pytorch-lightning==2.2.5[m
[31m-pytorch-metric-learning==2.6.0[m
[31m-poselib[m
[31m-Flask==2.2.5[m
[31m-Flask-Mail==0.9.1[m
[31m-Flask-SocketIO==5.1.1[m
[31m-Flask-SQLAlchemy==2.5.1[m
[31m-SQLAlchemy==1.4.47[m
\ No newline at end of file[m
[1mdiff --git a/src/modal_functions/test_modal_functions.py b/src/modal_functions/test_modal_functions.py[m
[1mdeleted file mode 100644[m
[1mindex aeb029e..0000000[m
[1m--- a/src/modal_functions/test_modal_functions.py[m
[1m+++ /dev/null[m
[36m@@ -1,35 +0,0 @@[m
[31m-import base64[m
[31m-import os[m
[31m-[m
[31m-import modal[m
[31m-[m
[31m-[m
[31m-def main():[m
[31m-    UnavServer = modal.Cls.lookup("unav-server", "UnavServer")[m
[31m-    unav_server = UnavServer()[m
[31m-    current_directory = os.getcwd()[m
[31m-    full_image_path = os.path.join([m
[31m-        current_directory, "modal_functions/misc/sample3.png"[m
[31m-    )[m
[31m-    destination_id = "07993"[m
[31m-    with open(full_image_path, "rb") as image_file:[m
[31m-        image_data = image_file.read()[m
[31m-        base64_encoded = base64.b64encode(image_data).decode("utf-8")[m
[31m-    [m
[31m-    print([m
[31m-        unav_server.planner.remote([m
[31m-            destination_id=destination_id,[m
[31m-            base_64_image=base64_encoded,[m
[31m-            session_id="test_session_id_2",[m
[31m-            building="LightHouse",[m
[31m-            floor="6_floor",[m
[31m-            place="New_York_City",[m
[31m-        )[m
[31m-    )[m
[31m-[m
[31m-[m
[31m-if __name__ == "__main__":[m
[31m-    try:[m
[31m-        main()[m
[31m-    except Exception as e:[m
[31m-        print(f"An error occurred: {e}")[m
[1mdiff --git a/src/modal_functions/unav.py b/src/modal_functions/unav.py[m
[1mdeleted file mode 100644[m
[1mindex 9bdfb80..0000000[m
[1m--- a/src/modal_functions/unav.py[m
[1m+++ /dev/null[m
[36m@@ -1,122 +0,0 @@[m
[31m-from modal import method, gpu, build, enter[m
[31m-[m
[31m-from modal_config import app, unav_image, volume[m
[31m-from logger_utils import setup_logger[m
[31m-[m
[31m-[m
[31m-@app.cls([m
[31m-    image=unav_image,[m
[31m-    volumes={"/root/UNav-IO": volume},[m
[31m-    gpu=gpu.Any(),[m
[31m-    enable_memory_snapshot=True,[m
[31m-    concurrency_limit=20,[m
[31m-    allow_concurrent_inputs=20,[m
[31m-)[m
[31m-class UnavServer:[m
[31m-[m
[31m-    @build()[m
[31m-    @enter()[m
[31m-    def load_server(self):[m
[31m-        from server_manager import Server[m
[31m-        from modules.config.settings import load_config[m
[31m-[m
[31m-        config = load_config("config.yaml")[m
[31m-[m
[31m-        self.server = Server(logger=setup_logger(), config=config)[m
[31m-[m
[31m-    @method()[m
[31m-    def get_destinations_list(self):[m
[31m-[m
[31m-        response = self.server.get_destinations_list([m
[31m-            building="LightHouse", floor="6_floor"[m
[31m-        )[m
[31m-        return response[m
[31m-[m
[31m-    @method()[m
[31m-    def planner([m
[31m-        self,[m
[31m-        session_id: str = "",[m
[31m-        destination_id: str = "",[m
[31m-        building: str = "",[m
[31m-        floor: str = "",[m
[31m-        place: str = "",[m
[31m-        base_64_image: str = None,[m
[31m-    ):[m
[31m-[m
[31m-        import json[m
[31m-        import time[m
[31m-        import base64[m
[31m-        import io[m
[31m-        from PIL import Image[m
[31m-[m
[31m-        """[m
[31m-            Handle localization request by processing the provided image and returning the pose.[m
[31m-        """[m
[31m-[m
[31m-        start_time = time.time()  # Start time for the entire function[m
[31m-[m
[31m-        query_image_data = ([m
[31m-            base64.b64decode(base_64_image.split(",")[1])[m
[31m-            if "," in base_64_image[m
[31m-            else base64.b64decode(base_64_image)[m
[31m-        )[m
[31m-        query_image = Image.open(io.BytesIO(query_image_data)).convert("RGB")[m
[31m-[m
[31m-        print("Query Image Converted from base64 to PIL Image")[m
[31m-[m
[31m-        response = self.server.select_destination([m
[31m-            session_id=session_id,[m
[31m-            place=place,[m
[31m-            building=building,[m
[31m-            floor=floor,[m
[31m-            destination_id=destination_id,[m
[31m-        )[m
[31m-        if response == None:[m
[31m-            print("Desintation Set to id: " + destination_id)[m
[31m-        else:[m
[31m-            print(response)[m
[31m-[m
[31m-        # Measure time for handle_localization[m
[31m-        start_localization_time = time.time()[m
[31m-        pose = self.server.handle_localization(frame=query_image, session_id=session_id)[m
[31m-        end_localization_time = time.time()[m
[31m-        localization_time = end_localization_time - start_localization_time[m
[31m-        print(f"Localization Time: {localization_time:.2f} seconds")[m
[31m-[m
[31m-        print("Pose: ", pose)[m
[31m-[m
[31m-        # Measure time for handle_navigation[m
[31m-        start_navigation_time = time.time()[m
[31m-        trajectory = self.server.handle_navigation(session_id)[m
[31m-        end_navigation_time = time.time()[m
[31m-        navigation_time = end_navigation_time - start_navigation_time[m
[31m-        print(f"Navigation Time: {navigation_time:.2f} seconds")[m
[31m-[m
[31m-        end_time = time.time()  # End time for the entire function[m
[31m-        elapsed_time = ([m
[31m-            end_time - start_time[m
[31m-        )  # Calculate elapsed time for the entire function[m
[31m-[m
[31m-        print([m
[31m-            f"Total Execution Time: {elapsed_time:.2f} seconds"[m
[31m-        )  # Print total elapsed time[m
[31m-[m
[31m-        scale = self.server.config["location"]["scale"][m
[31m-[m
[31m-        return json.dumps({"trajectory": trajectory, "scale": scale})[m
[31m-[m
[31m-    @method()[m
[31m-    def start_server(self):[m
[31m-        import json[m
[31m-[m
[31m-        """[m
[31m-        Initializes and starts the serverless instance.[m
[31m-    [m
[31m-        This function helps in reducing the server response time for actual requests by pre-warming the server. [m
[31m-        By starting the server in advance, it ensures that the server is ready to handle incoming requests immediately, [m
[31m-        thus avoiding the latency associated with a cold start.[m
[31m-        """[m
[31m-        print("Server Started...")[m
[31m-[m
[31m-        response = {"status": "success", "message": "Server started."}[m
[31m-        return json.dumps(response)[m
[1mdiff --git a/src/modal_functions/volume/modalvolumedata_setup.py b/src/modal_functions/volume/modalvolumedata_setup.py[m
[1mdeleted file mode 100644[m
[1mindex a978dda..0000000[m
[1m--- a/src/modal_functions/volume/modalvolumedata_setup.py[m
[1m+++ /dev/null[m
[36m@@ -1,108 +0,0 @@[m
[31m-import modal[m
[31m-import gdown[m
[31m-import os[m
[31m-import shutil [m
[31m-import logging[m
[31m-import yaml[m
[31m-import zipfile[m
[31m-from dotenv import load_dotenv[m
[31m-import boto3[m
[31m-[m
[31m-logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')[m
[31m-[m
[31m-[m
[31m-# Reference the requirements-modal.txt for installing dependencies[m
[31m-image = modal.Image.debian_slim().pip_install_from_requirements("modal_functions/volumesetup_requirements.txt")[m
[31m-[m
[31m-volume = modal.Volume.from_name("Visiondata", create_if_missing=True)[m
[31m-[m
[31m-app = modal.App("DataSetup", image=image, mounts=[modal.Mount.from_local_file(".env")])[m
[31m-logging.info('created s3_client')[m
[31m-load_dotenv()[m
[31m-s3_client = boto3.client([m
[31m-    's3',[m
[31m-    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),[m
[31m-    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')[m
[31m-)[m
[31m-[m
[31m-bucket_name = 'vis4ion'[m
[31m-[m
[31m-files = {[m
[31m-    "demo_query.png": "17MzPE9TyKiNsi6G59rqLCMMd40cIK3bU",[m
[31m-    "destination.json": "1sIzFujoumSsVlZqlwwO20l96ZziORP-w",[m
[31m-    "hloc.yaml": "15JYLqU9Y56keMrg9ZfxwfbkbL6_haYpx",[m
[31m-    "MapConnection_Graph.pkl": "199xZSc9jSajiCqzDW_AzhuqOp_YS41fZ",[m
[31m-}[m
[31m-[m
[31m-@app.function(volumes={"/files": volume})   ## to create necesasry directories [m
[31m-def create_directories():  [m
[31m-    os.makedirs(os.path.join("/files", "data", "New_York_City", "LightHouse"), exist_ok=True)[m
[31m-    logging.info('created path : /files/data/New_York_City/LightHouse')[m
[31m-    os.makedirs(os.path.join("/files", "configs"), exist_ok=True)[m
[31m-    logging.info('created path : /files/Configs')[m
[31m-[m
[31m-@app.function(volumes={"/files": volume}, timeout=86400)[m
[31m-def checkAndDownload_file_from_remoteStorage():   ## download the data to the respective locations[m
[31m-    for filename, file_id in files.items():[m
[31m-        logging.info(f"Processing {filename}")[m
[31m-        if filename == "destination.json":[m
[31m-            download_path = os.path.join('/files', 'data', 'destination.json')[m
[31m-        elif filename == "hloc.yaml":[m
[31m-            download_path = os.path.join("/files", "configs", "hloc.yaml")[m
[31m-        elif filename == "MapConnection_Graph.pkl":[m
[31m-            download_path = os.path.join("/files", "data", "New_York_City", "MapConnection_Graph.pkl")[m
[31m-        else:[m
[31m-            download_path = os.path.join("/files", filename)[m
[31m-[m
[31m-        if not os.path.exists(download_path):[m
[31m-            gdown.download(f'https://drive.google.com/uc?id={file_id}', download_path, quiet=False)[m
[31m-            logging.info(f"Downloaded {download_path}")[m
[31m-        else:[m
[31m-            logging.info(f"{download_path} already exists. Skipping download.")[m
[31m-        [m
[31m-    with open(os.path.join("/files", "configs", "hloc.yaml"), 'r') as file:[m
[31m-        config = yaml.safe_load(file)[m
[31m-    config['IO_root'] = "/root/UNav-IO"[m
[31m-[m
[31m-    with open(os.path.join("/files", "configs", "hloc.yaml"), 'w') as file:[m
[31m-        yaml.safe_dump(config, file)[m
[31m-[m
[31m-    #downloading from s3 to modal volumes [m
[31m-    modal_directory = "/files/data"[m
[31m-     # List objects in the S3 bucket[m
[31m-    logging.info(f"Listing objects in the S3 bucket: {bucket_name}")[m
[31m-    response = s3_client.list_objects_v2(Bucket=bucket_name)[m
[31m-[m
[31m-    if 'Contents' not in response:[m
[31m-        logging.info(f"No objects found in S3 bucket {bucket_name}.")[m
[31m-        return[m
[31m-    [m
[31m-    # Download each object from S3[m
[31m-    for obj in response['Contents']:[m
[31m-        s3_key = obj['Key'][m
[31m-        file_path = os.path.join(modal_directory, s3_key)[m
[31m-[m
[31m-        # Check if the file already exists in the modal volume[m
[31m-        if not os.path.exists(file_path):[m
[31m-            # Create directories for nested keys if necessary[m
[31m-            os.makedirs(os.path.dirname(file_path), exist_ok=True)[m
[31m-[m
[31m-            # Log the downloading process[m
[31m-            logging.info(f"Downloading {s3_key} to {file_path}")[m
[31m-[m
[31m-            # Download the file from S3 to the modal volume[m
[31m-            s3_client.download_file(bucket_name, s3_key, file_path)[m
[31m-        else:[m
[31m-            logging.info(f"{file_path} already exists, skipping download.")[m
[31m-    [m
[31m-    logging.info(f"All files from S3 bucket {bucket_name} have been downloaded to {modal_directory}")[m
[31m-[m
[31m-[m
[31m-    logging.info("All files downloaded successfully from google drive and s3 bucket.")[m
[31m-[m
[31m-[m
[31m-[m
[31m-if __name__ == "__main__":[m
[31m-    with app.run():[m
[31m-        create_directories.remote()[m
[31m-        checkAndDownload_file_from_remoteStorage.remote()[m
[1mdiff --git a/src/modal_functions/volume/volumedata_setup.py b/src/modal_functions/volume/volumedata_setup.py[m
[1mdeleted file mode 100644[m
[1mindex 3e2a540..0000000[m
[1m--- a/src/modal_functions/volume/volumedata_setup.py[m
[1m+++ /dev/null[m
[36m@@ -1,103 +0,0 @@[m
[31m-import os[m
[31m-import shutil [m
[31m-import logging[m
[31m-import yaml[m
[31m-import zipfile[m
[31m-from dotenv import load_dotenv[m
[31m-import boto3[m
[31m-import gdown[m
[31m-[m
[31m-# Logging configuration[m
[31m-logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')[m
[31m-[m
[31m-class DataManager:[m
[31m-    def __init__(self, s3_bucket_name, env_file=".env"):[m
[31m-        # Load environment variables[m
[31m-        load_dotenv(env_file)[m
[31m-[m
[31m-        # AWS credentials setup[m
[31m-        self.s3_client = boto3.client([m
[31m-            's3',[m
[31m-            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),[m
[31m-            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')[m
[31m-        )[m
[31m-[m
[31m-        self.bucket_name = s3_bucket_name[m
[31m-        self.remotefilesIds = {[m
[31m-            "demo_query.png": "17MzPE9TyKiNsi6G59rqLCMMd40cIK3bU",[m
[31m-            "destination.json": "1sIzFujoumSsVlZqlwwO20l96ZziORP-w",[m
[31m-            "hloc.yaml": "15JYLqU9Y56keMrg9ZfxwfbkbL6_haYpx",[m
[31m-            "MapConnection_Graph.pkl": "199xZSc9jSajiCqzDW_AzhuqOp_YS41fZ",[m
[31m-        }[m
[31m-[m
[31m-    def create_directories(self, base_path="/files"):[m
[31m-        """Create necessary directories."""[m
[31m-        os.makedirs(os.path.join(base_path, "data", "New_York_City", "LightHouse"), exist_ok=True)[m
[31m-        logging.info('Created path: /files/data/New_York_City/LightHouse')[m
[31m-[m
[31m-        os.makedirs(os.path.join(base_path, "configs"), exist_ok=True)[m
[31m-        logging.info('Created path: /files/configs')[m
[31m-[m
[31m-    def download_files_from_google_drive(self, base_path="/files"):[m
[31m-        """Download files from Google Drive."""[m
[31m-        for filename, file_id in self.remotefilesIds.items():[m
[31m-            logging.info(f"Processing {filename}")[m
[31m-            download_path = self.get_download_path(filename, base_path)[m
[31m-[m
[31m-            if not os.path.exists(download_path):[m
[31m-                gdown.download(f'https://drive.google.com/uc?id={file_id}', download_path, quiet=False)[m
[31m-                logging.info(f"Downloaded {download_path}")[m
[31m-            else:[m
[31m-                logging.info(f"{download_path} already exists. Skipping download.")[m
[31m-[m
[31m-    def get_download_path(self, filename, base_path):[m
[31m-        """Generate the download path based on the file type."""[m
[31m-        if filename == "destination.json":[m
[31m-            return os.path.join(base_path, 'data', 'destination.json')[m
[31m-        elif filename == "hloc.yaml":[m
[31m-            return os.path.join(base_path, "configs", "hloc.yaml")[m
[31m-        elif filename == "MapConnection_Graph.pkl":[m
[31m-            return os.path.join(base_path, "data", "New_York_City", "MapConnection_Graph.pkl")[m
[31m-        else:[m
[31m-            return os.path.join(base_path, filename)[m
[31m-[m
[31m-    def modify_hloc_yaml(self, base_path="/files"):[m
[31m-        """Modify hloc.yaml configuration."""[m
[31m-        with open(os.path.join(base_path, "configs", "hloc.yaml"), 'r') as file:[m
[31m-            config = yaml.safe_load(file)[m
[31m-        config['IO_root'] = "/root/UNav-IO"[m
[31m-[m
[31m-        with open(os.path.join(base_path, "configs", "hloc.yaml"), 'w') as file:[m
[31m-            yaml.safe_dump(config, file)[m
[31m-[m
[31m-    def download_files_from_s3(self, modal_directory="/files/data"):[m
[31m-        """Download all files from the specified S3 bucket."""[m
[31m-        logging.info(f"Listing objects in the S3 bucket: {self.bucket_name}")[m
[31m-        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)[m
[31m-[m
[31m-        if 'Contents' not in response:[m
[31m-            logging.info(f"No objects found in S3 bucket {self.bucket_name}.")[m
[31m-            return[m
[31m-        [m
[31m-        # Download each object from S3[m
[31m-        for obj in response['Contents']:[m
[31m-            s3_key = obj['Key'][m
[31m-            file_path = os.path.join(modal_directory, s3_key)[m
[31m-[m
[31m-            # Create directories for nested keys if necessary[m
[31m-            os.makedirs(os.path.dirname(file_path), exist_ok=True)[m
[31m-[m
[31m-            logging.info(f"Downloading {s3_key} to {file_path}")[m
[31m-            [m
[31m-            # Download the file[m
[31m-            self.s3_client.download_file(self.bucket_name, s3_key, file_path)[m
[31m-[m
[31m-        logging.info(f"All files from S3 bucket {self.bucket_name} have been downloaded to {modal_directory}")[m
[31m-[m
[31m-    def run(self, base_path="/files"):[m
[31m-        """Main function to create directories, download files from Google Drive and S3."""[m
[31m-        self.create_directories(base_path)[m
[31m-        self.download_files_from_google_drive(base_path)[m
[31m-        self.modify_hloc_yaml(base_path)[m
[31m-        self.download_files_from_s3(os.path.join(base_path, "data"))[m
[31m-        logging.info("All files downloaded successfully from Google Drive and S3 bucket.")[m
[1mdiff --git a/src/modal_functions/volume/volumesetup_requirements.txt b/src/modal_functions/volume/volumesetup_requirements.txt[m
[1mdeleted file mode 100644[m
[1mindex 298ced8..0000000[m
[1m--- a/src/modal_functions/volume/volumesetup_requirements.txt[m
[1m+++ /dev/null[m
[36m@@ -1,4 +0,0 @@[m
[31m-gdown[m
[31m-PyYAML[m
[31m-boto3[m
[31m-python-dotenv[m
[1mdiff --git a/src/modules/routes/frame_routes.py b/src/modules/routes/frame_routes.py[m
[1mindex 06b16d9..351576f 100644[m
[1m--- a/src/modules/routes/frame_routes.py[m
[1m+++ b/src/modules/routes/frame_routes.py[m
[36m@@ -46,20 +46,9 @@[m [mdef register_frame_routes(app, server, socketio):[m
             # Resize the image[m
             resized_image = frame.resize((new_width, new_height))[m
 [m
[31m-            image_np = np.array(resized_image)[m
[31m-[m
[31m-            original_width, original_height = frame.size[m
[31m-[m
[31m-            new_width = 640[m
[31m-            new_height = int((new_width / original_width) * original_height)[m
[31m-[m
[31m-            # Resize the image[m
[31m-            resized_image = frame.resize((new_width, new_height))[m
[31m-[m
             image_np = np.array(resized_image)[m
             [m
             if frame is not None:[m
[31m-                client_frames[session_id] = image_np[m
                 client_frames[session_id] = image_np[m
                 response_data = {'status': 'frame received'}[m
 [m
[36m@@ -76,7 +65,6 @@[m [mdef register_frame_routes(app, server, socketio):[m
 [m
                 buffered = io.BytesIO()[m
                 resized_image.save(buffered, format="JPEG")[m
[31m-                resized_image.save(buffered, format="JPEG")[m
                 new_frame_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')[m
                 response_data['floorplan_base64'] = pose_update_info.get('floorplan_base64')[m
                 socketio.emit('camera_frame', {'session_id': session_id, 'frame': new_frame_base64})[m
[36m@@ -107,8 +95,6 @@[m [mdef register_frame_routes(app, server, socketio):[m
         else:[m
             return jsonify({'error': 'No frame available for this client'}), 404[m
 [m
[31m-    @app.route('/get_image/<id>/<imageName>', methods=['POST'])[m
[31m-    def get_image(id, imageName):[m
     @app.route('/get_image/<id>/<imageName>', methods=['POST'])[m
     def get_image(id, imageName):[m
         """[m
[36m@@ -117,10 +103,6 @@[m [mdef register_frame_routes(app, server, socketio):[m
         data = request.json[m
         session_id = data.get('username')[m
     [m
[31m-        image_path = os.path.join(server.root, 'logs', server.config['location']['place'], server.config['location']['building'], server.config['location']['floor'], id, 'images', imageName)[m
[31m-        data = request.json[m
[31m-        session_id = data.get('username')[m
[31m-    [m
         image_path = os.path.join(server.root, 'logs', server.config['location']['place'], server.config['location']['building'], server.config['location']['floor'], id, 'images', imageName)[m
         if os.path.exists(image_path):[m
             return send_file(image_path, mimetype='image/png')[m
[1mdiff --git a/src/server_manager.py b/src/server_manager.py[m
[1mindex bf5d144..901530b 100644[m
[1m--- a/src/server_manager.py[m
[1m+++ b/src/server_manager.py[m
[36m@@ -31,8 +31,6 @@[m [mclass Server(DataHandler):[m
 [m
         self.load_all_maps = config['hloc']['load_all_maps'][m
             [m
[31m-        self.load_all_maps = config['hloc']['load_all_maps'][m
[31m-            [m
         self.coarse_locator = Coarse_Locator(config=self.config)[m
         self.refine_locator = localization(self.coarse_locator, config=self.config, logger=self.logger)[m
         [m
[36m@@ -42,7 +40,6 @@[m [mclass Server(DataHandler):[m
         self.localization_states = {}[m
         self.destination_states = {}[m
             [m
[31m-            [m
         with open(os.path.join(self.root, 'data', 'scale.json'), 'r') as f:[m
             self.scale_data = json.load(f)[m
 [m
[36m@@ -61,21 +58,9 @@[m [mclass Server(DataHandler):[m
 [m
         # image_rgb = resized_image.convert("RGB")[m
         [m
[31m-        # self.image_np = np.array(image_rgb)[m
[31m-        # original_width, original_height = image.size[m
[31m-[m
[31m-        # new_width = 640[m
[31m-        # new_height = int((new_width / original_width) * original_height)[m
[31m-[m
[31m-        # # Resize the image[m
[31m-        # resized_image = image.resize((new_width, new_height))[m
[31m-[m
[31m-        # image_rgb = resized_image.convert("RGB")[m
[31m-        [m
         # self.image_np = np.array(image_rgb)[m
         ############################################# test data #################################################[m
         [m
[31m-        [m
     def update_config(self, new_config):[m
         # Merge the new configuration with the existing one[m
         [m
[36m@@ -131,7 +116,6 @@[m [mclass Server(DataHandler):[m
             'floorplan': floorplan_base64,[m
         }[m
 [m
[31m-[m
     def get_destinations_list(self, building, floor):[m
         # Load destination data[m
         destinations = self.all_buildings_data.get(building,{}).get(floor,{}).get('destinations',{})[m
[36m@@ -165,7 +149,6 @@[m [mclass Server(DataHandler):[m
         images = {id: os.listdir(os.path.join(base_path, id, 'images')) for id in ids if os.path.isdir(os.path.join(base_path, id, 'images'))}[m
         return images[m
 [m
[31m-[m
     def _split_id(self, segment_id):[m
         # Load the current segment and its neighbors[m
         parts = segment_id.split('_')[m
[36m@@ -173,11 +156,9 @@[m [mclass Server(DataHandler):[m
         floor = parts[1] + '_' + parts[2]  # Extract floor name (e.g., '6_floor')[m
         return building, floor[m
 [m
[31m-[m
     def _update_next_step(self):[m
         pass[m
 [m
[31m-[m
     def handle_localization(self, session_id, frame):[m
         """[m
         Handles the localization process for a given session and frame.[m
[36m@@ -332,7 +313,6 @@[m [mclass Server(DataHandler):[m
         [m
         return pose_update_info[m
 [m
[31m-[m
     def handle_navigation(self, session_id):[m
         if session_id not in self.destination_states:[m
             self.logger.error("Selected destination ID is not set.")[m
