# from UNav_core.src.feature.global_extractor import Global_Extractors
from UNav_core.src.feature.Global_Extractors import GlobalExtractors
from UNav_core.src.feature.local_extractor import Local_extractor
from UNav_core.src.feature.local_matcher import Local_matcher
from UNav_core.src.third_party.torchSIFT.src.torchsift.ransac.ransac import ransac
from UNav_core.src.track.implicit_distortion_model import coarse_pose,pose_multi_refine
import torch
import torchvision.transforms as transforms
import numpy as np
from os.path import join
from time import time
import h5py
import pickle
from skimage.transform import resize

def read_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the pickle file: {e}")
        return None
    
class Coarse_Locator:
    def __init__(self, feature, config):
        """
        Initializes the VPR class.
        :param root: Path to the directory containing the combined global_features.h5 file.
        :param config: Configuration dictionary containing parameters for VPR.
        """
        self.device = config['devices']
        self.feature = feature
        self.global_extractor = GlobalExtractors(config['IO_root'], config['feature']['global'])
        self.global_extractor.set_float32()
        self.global_extractor.set_train(False)
        self.model_name = list(self.global_extractor.models)[0]

        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),  # Adjust based on model's expected input size
            transforms.ToTensor(),
            # Add normalization if required by your model
            # transforms.Normalize(mean=[...], std=[...]),
        ])
        
        # self.global_extractor = Global_Extractors(config).get()
        self.config = config['hloc']

        # Load global descriptors and segment IDs
        self.global_descriptors, self.segment_ids = self.load_global_features(join(config['IO_root'],'data', config['location']['place']))
        
        # Load connection graph
        file_path = join(config['IO_root'],'data', config['location']['place'], 'MapConnnection_Graph.pkl')
        connection_graph = read_pickle_file(file_path)
        self.connection_graph = connection_graph
        
    def load_global_features(self, root):
        """
        Loads the global descriptors and segment IDs from the global_features.h5 file.
        :param root: Path to the directory containing the global_features.h5 file.
        :return: Numpy arrays of global descriptors and their corresponding segment IDs.
        """
        # global_features_path = join(root, f'global_features_{self.feature}.h5')
        # with h5py.File(global_features_path, 'r') as f:
        #     descriptors = f['descriptors'][:]
        #     segment_ids = f['segments'][:].astype(str)
        
        # return torch.tensor(descriptors, dtype=torch.float32).to(self.device), segment_ids
        return None, None

    def coarse_vpr(self, image):
        """
        Perform coarse visual place recognition.
        :param image: The query image for which to find the place.
        :return: Top-k matches and a boolean indicating if the corresponding segment is found.
        """
        # Extract global descriptor from the query image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with torch.autocast("cuda", torch.float32):
                outputs = self.global_extractor(self.model_name, image_tensor)
        _, query_desc = outputs
        
        # query_desc = self.global_extractor(image).to(self.device)

        # Compute similarity between the query descriptor and database descriptors
        sim = torch.einsum('id,jd->ij', query_desc, self.global_descriptors)
        topk_indices = torch.topk(sim, self.config['retrieval_num'], dim=1).indices.cpu().numpy()

        # Retrieve the corresponding segment IDs for the top-k matches
        topk_segments = self.segment_ids[topk_indices[0]]
        
        # Analyze top-k results
        segment, success = self.analyze_topk_results(topk_segments)
        
        return topk_segments, segment, success
    
    def get_topk_segments(self, topk_indices):
        """
        Retrieve the corresponding segments for the top-k indices.
        :param topk_indices: Top-k indices from the coarse VPR step.
        :return: List of segment IDs corresponding to the top-k indices.
        """
        # Assuming that each descriptor in the database has a corresponding segment ID
        # which can be retrieved (this mapping should be maintained outside of this class)
        topk_segments = [self.get_segment_id(index) for index in topk_indices[0]]
        return topk_segments
    
    def analyze_topk_results(self, topk_segments):
        """
        Analyze the top-k segments to determine if a majority belong to the same segment or its neighbors.
        :param topk_segments: List of segment IDs corresponding to the top-k matches.
        :return: The most likely segment and a boolean indicating if localization succeeded.
        """
        segment_counts = {}
        
        # First, count occurrences of each segment in topk_segments
        for segment in topk_segments:
            if segment in segment_counts:
                segment_counts[segment] += 1
            else:
                segment_counts[segment] = 1
        
        # Initialize a dictionary to accumulate counts for segments and their neighbors
        segment_wt_neighbor_counts = {}

        # Accumulate counts including neighbor segments
        for segment, count in segment_counts.items():
            # Start with the count of the segment itself
            total_count = count
            
            # Add counts of neighbors if they exist in the topk_segments
            if segment in self.connection_graph:
                for neighbor in self.connection_graph[segment]:
                    if neighbor in segment_counts:
                        total_count += segment_counts[neighbor]
            
            # Record the accumulated count for the segment
            segment_wt_neighbor_counts[segment] = total_count
        
        # Determine the segment with the highest total count
        most_likely_segment = max(segment_wt_neighbor_counts, key=segment_wt_neighbor_counts.get)
        success = (segment_wt_neighbor_counts[most_likely_segment] / len(topk_segments)) >= 0.1
        
        return most_likely_segment, success

    def get_segment_id(self, index):
        """
        This method should map a database index to its corresponding segment ID.
        :param index: Index in the global descriptor database.
        :return: Corresponding segment ID.
        """
        # Placeholder for actual implementation
        # The mapping from database indices to segment IDs needs to be managed outside this class
        # and passed in or accessed here.
        segment_id_mapping = {}  # This would be defined elsewhere
        return segment_id_mapping.get(index, None)

    def get_global_extractor(self):
        return self.global_extractor
    
class Hloc():
    device='cuda' if torch.cuda.is_available() else "cpu"
    def __init__(self, coarse_locator, config, logger):
        # loading config setting
        self.config=config['hloc']
        self.batch_mode=self.config['batch_mode']
        self.thre=self.config['ransac_thre']
        self.match_type=self.config['match_type']
        self.feature_configs=config['feature']

        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),  # Adjust based on model's expected input size
            transforms.ToTensor(),
            # Add normalization if required by your model
            # transforms.Normalize(mean=[...], std=[...]),
        ])
        
        self.global_extractor = coarse_locator.get_global_extractor()
        self.model_name = list(self.global_extractor.models)[0]

        local_feature = Local_extractor(self.feature_configs['local'])
        self.local_feature_extractor = local_feature.extractor()
        
        self.logger = logger

    def form_global_descriptor_tensor(self):
        """
        Iterate over self.map_data['perspective_frames'] and form a global descriptor tensor.
        """
        global_descriptors = []
        self.db_name = []

        # Iterate over each frame in perspective_frames
        for frame_name, frame_data in self.map_data['perspective_frames'].items():
            # Extract the global_descriptor for this frame
            global_descriptor = frame_data['global_descriptor']
            
            # Append frame name
            self.db_name.append(frame_name)
            
            # Append to the list of global descriptors
            global_descriptors.append(torch.tensor(global_descriptor, dtype=torch.float32))

        # Stack all global descriptors into a single tensor
        global_descriptor_tensor = torch.stack([descriptor.squeeze(0) for descriptor in global_descriptors], dim=0)

        return global_descriptor_tensor

    def update_maps(self, map_data):
        # loading map data
        self.map_data=map_data
        
        self.rot_base=map_data['rot_base']
        self.T=map_data['T']
        self.db_global_descriptors=self.form_global_descriptor_tensor()
        
        self.local_feature_matcher= Local_matcher(self.db_name, self.map_data['perspective_frames'], threshold = self.thre, **self.feature_configs)
        
        # load process default data
        self.list_2d, self.list_3d, self.initial_poses, self.pps = [], [], [], []
        self.last_time=time()
        
    def global_retrieval(self, image):
        # # Extract the global descriptor from the query image
        # self.query_desc = self.global_extractor(image)
        
        # Extract global descriptor from the query image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with torch.autocast("cuda", torch.float32):
                outputs = self.global_extractor(self.model_name, image_tensor)
        _, self.query_desc = outputs
        
        sim = torch.einsum('id,jd->ij', self.query_desc, self.db_global_descriptors.to(self.device))
        topk = torch.topk(sim, self.config['retrieval_num'], dim=1).indices.cpu().numpy()

        return topk

    def feature_matching_lightglue_batch(self,image,topk):
        """
        Local Feature Matching:
            Match the local features between query image and retrieved database images
        """
        with torch.inference_mode():  # Use torch.no_grad during inference
            image_np = np.array(image)
            feats0 = self.local_feature_extractor(image_np)
            valid_db_frame_name, pts0_list,pts1_list,lms_list,max_len=self.local_feature_matcher.lightglue_batch(self, topk[0], feats0, image_np)

        return valid_db_frame_name, pts0_list,pts1_list,lms_list,max_len

    def feature_matching_lightglue(self,image,topk):
        """
        Local Feature Matching:
            Match the local features between query image and retrieved database images
        """
        with torch.inference_mode():  # Use torch.no_grad during inference
            image_np = np.array(image)
            feats0 = self.local_feature_extractor(image_np)
        pts0_list,pts1_list,lms_list=[],[],[]
        max_len=0
        
        valid_db_frame_name = []
        for i in topk[0]:
            pts0,pts1,lms=self.local_feature_matcher.lightglue(i, feats0)
            
            feat_inliner_size=pts0.shape[0]
            if feat_inliner_size>self.thre:
                valid_db_frame_name.append(self.db_name[i])
                pts0_list.append(pts0)
                pts1_list.append(pts1)
                lms_list.append(lms)
                if feat_inliner_size>max_len:
                    max_len=feat_inliner_size
            del pts0,pts1,lms
        del self.query_desc, feats0
        torch.cuda.empty_cache()
        return valid_db_frame_name, pts0_list,pts1_list,lms_list,max_len
    
    def feature_matching_superglue(self,image,topk):
        """
        Local Feature Matching:
            Match the local features between query image and retrieved database images
        """
        with torch.inference_mode():  # Use torch.no_grad during inference
            feats0 = self.local_feature_extractor(image)
        pts0_list,pts1_list,lms_list=[],[],[]
        max_len=0
        for i in topk[0]:
            pts0,pts1,lms=self.local_feature_matcher.superglue(i, feats0)
            feat_inliner_size=pts0.shape[0]
            if feat_inliner_size>self.thre:
                pts0_list.append(pts0)
                pts1_list.append(pts1)
                lms_list.append(lms)
                if feat_inliner_size>max_len:
                    max_len=feat_inliner_size
        del self.query_desc, feats0
        torch.cuda.empty_cache()
        return pts0_list,pts1_list,lms_list,max_len

    def geometric_verification(self, valid_db_frame_name, pts0_list, pts1_list, lms_list, max_len):
        """
        Geometric verification:
            Apply geometric verification between query and database images
        """
        batch_size = len(pts0_list)
        pts0 = torch.empty((batch_size, max_len, 2), dtype=float)
        pts1 = torch.empty((batch_size, max_len, 2), dtype=float)
        lms = torch.empty((batch_size, max_len, 3), dtype=float)
        mask = torch.zeros((batch_size, max_len, max_len))

        for i in range(batch_size):
            inliner_size = len(pts0_list[i])
            pts0[i, :inliner_size, :] = torch.from_numpy(pts0_list[i])
            pts1[i, :inliner_size, :] = torch.from_numpy(pts1_list[i])
            lms[i, :inliner_size, :] = torch.from_numpy(lms_list[i])
            mask[i, :inliner_size, :inliner_size] = torch.ones((inliner_size, inliner_size))
            
        pts0, pts1, lms, mask = pts0.to(self.device), pts1.to(self.device), lms.to(self.device), mask.to(self.device)
        
        # try:
        _, inliners, _ = ransac(pts0, pts1, mask)
        diag_masks = torch.diagonal(inliners, dim1=-2, dim2=-1)

        # Calculate sizes for each item in the batch
        sizes = diag_masks.sum(-1)

        # Filtering based on the threshold
        valid_indices = sizes > self.thre

        # Apply thresholding
        diag_masks = diag_masks[valid_indices]
        final_candidates = [name for name, valid in zip(valid_db_frame_name, valid_indices.cpu().numpy()) if valid]
        pts0 = pts0[valid_indices]
        lms = lms[valid_indices]

        # Masking pts0, pts1, and lms
        masked_pts0 = [pts0[i][diag_masks[i]] for i in range(pts0.size(0))]
        masked_lms = [lms[i][diag_masks[i]] for i in range(lms.size(0))]

        del pts0, pts1, lms, mask
        torch.cuda.empty_cache()
        
        if len(masked_pts0) > 0:
            return final_candidates, torch.cat(masked_pts0), torch.cat(masked_lms)
        else:
            return [], torch.tensor([]), None
        # except:
        #     del pts0, pts1, lms, mask
        #     torch.cuda.empty_cache()
        #     return None, torch.tensor([]), None


    def pnp(self,image,feature2D,landmark3D):
        """
        Start Perspective-n-points:
            Estimate the current location using implicit distortion model
        """
        if feature2D.size()[0]>0:
            height, width, _ = image.shape
            feature2D, landmark3D=feature2D.cpu().numpy(),landmark3D.cpu().numpy()
            out, p2d_inlier, p3d_inlier = coarse_pose(feature2D, landmark3D, np.array([width / 2, height / 2]))
            self.list_2d.append(p2d_inlier)
            self.list_3d.append(p3d_inlier)
            self.initial_poses.append(out['pose'])
            self.pps.append(out['pp'])
            if len(self.list_2d) > self.config['implicit_num']:
                self.list_2d.pop(0)
                self.list_3d.pop(0)
                self.initial_poses.pop(0)
                self.pps.pop(0)
            pose = pose_multi_refine(self.list_2d, self.list_3d, self.initial_poses, self.pps,self.rot_base,self.T)

            #reset reload num
            self.current_reload_num = 0
        else:
            pose =None
            self.logger.warning("!!!Cannot localize at this point, please take some steps or turn around!!!")
        return pose

    def _determine_next_segment(self, candidates):
        candidate_histogram = {}
        max_counts = 0
        next_segment_id = None
        for candidate in candidates:
            segment_id = self.map_data['perspective_frames'][candidate].get('segment_id')
            if segment_id in candidate_histogram:
                candidate_histogram[segment_id] += 1
            else:
                candidate_histogram[segment_id] = 1
            if candidate_histogram[segment_id] > max_counts:
                max_counts = candidate_histogram[segment_id]
                next_segment_id = segment_id
        
        return next_segment_id
            
    def get_location(self, image):
        self.logger.debug("Start image retrieval")

        topk=self.global_retrieval(image)
        
        valid_db_frame_name = []
        next_segment_id = None
        
        if self.match_type=='superglue':
            self.logger.debug("Matching local feature")
            pts0_list,pts1_list,lms_list,max_matched_num=self.feature_matching_superglue(image,topk)
            self.logger.debug("Start geometric verification")
            feature2D,landmark3D=self.geometric_verification(pts0_list, pts1_list, lms_list, max_matched_num)

        elif self.match_type=='lightglue':
            self.logger.debug("Matching local feature")
            if self.batch_mode:
                valid_db_frame_name, pts0_list,pts1_list,lms_list,max_matched_num=self.feature_matching_lightglue_batch(image,topk)
            else:
                valid_db_frame_name, pts0_list,pts1_list,lms_list,max_matched_num=self.feature_matching_lightglue(image,topk)
            
            self.logger.debug("Start geometric verification")
            if len(pts0_list)>0:
                final_candidates, feature2D,landmark3D=self.geometric_verification(valid_db_frame_name, pts0_list, pts1_list, lms_list, max_matched_num)
                if len(final_candidates)>0:
                    print("matched images:", final_candidates)
                    next_segment_id = self._determine_next_segment(final_candidates)
                else:
                    return None, None
            else:
                return None, None

        self.logger.debug("Estimate the camera pose using PnP algorithm")
        pose=self.pnp(image,feature2D,landmark3D)
        return pose, next_segment_id