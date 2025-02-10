from UNav_core.src.third_party.local_feature.SuperPoint_SuperGlue.base_model import dynamic_load
from UNav_core.src.third_party.local_feature.SuperPoint_SuperGlue import extractors,matchers
from UNav_core.src.third_party.local_feature.LightGlue.lightglue import LightGlue
import numpy as np
import torch
import cv2

class Superpoint():
    def __init__(self,device,conf):
        Model_sp = dynamic_load(extractors, conf['detector_name'])
        self.local_feature_extractor=Model_sp({'name':conf['detector_name'],'nms_radius':conf['nms_radius'],'max_keypoints':conf['max_keypoints']}).eval().to(device)
        self.device=device

    def prepare_data(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image = image[None]
        data = torch.from_numpy(image / 255.).unsqueeze(0)
        return data

    def extract_local_features(self, image0):
        data0 = self.prepare_data(image0)
        pred0 = self.local_feature_extractor(data0.to(self.device))
        del data0
        torch.cuda.empty_cache()
        pred0 = {k: v[0].cpu().detach().numpy() for k, v in pred0.items()}
        if 'keypoints' in pred0:
            pred0['keypoints'] = (pred0['keypoints'] + .5) - .5
        pred0.update({'image_size': np.array([image0.shape[1], image0.shape[0]])})
        return pred0

class Local_extractor():
    def __init__(self,configs):
        self.configs=configs
        self.device='cuda' if torch.cuda.is_available() else 'cpu'

    def lightglue(self,conf):
        Model_lg=LightGlue(pretrained='superpoint', **conf['match_conf']).eval()
        return Model_lg
    
    def superglue(self,conf):
        Model_sg = dynamic_load(matchers, conf['matcher_name'])
        return Model_sg({'name':conf['matcher_name'],'weights':conf['weights'],'sinkhorn_iterations':conf['sinkhorn_iterations']}).eval()

    def extractor(self):
        for name,content in self.configs.items():
            if name=='superpoint+superglue':
                superpoint=Superpoint(self.device,self.configs['superpoint+superglue'])
                return superpoint.extract_local_features
            elif name=='superpoint+lightglue':
                superpoint=Superpoint(self.device,self.configs['superpoint+lightglue'])
                return superpoint.extract_local_features
            elif name=='sift':
                pass

            elif name=='surf':
                pass

    def matcher(self):
        for name,content in self.configs.items():
            if name == 'superpoint+superglue':
                return self.superglue(self.configs['superpoint+superglue'])
            elif name == 'superpoint+lightglue':
                return self.lightglue(self.configs['superpoint+lightglue'])
            elif name == 'sift':
                pass
            elif name == 'surf':
                pass