from UNav_core.src.third_party.global_feature.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
from UNav_core.src.third_party.global_feature.mixVPR_main.main import VPRModel
from os.path import join
import torch


class Global_Extractors():
    def __init__(self, config):
        self.root = config['IO_root']
        self.extractor = config['feature']['global']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set device

    def netvlad(self, content):
        model = NetVladFeatureExtractor(join(self.root, content['ckpt_path']), 
                                        arch=content['arch'],
                                        num_clusters=content['num_clusters'],
                                        pooling=content['pooling'], 
                                        vladv2=content['vladv2'], 
                                        nocuda=content['nocuda'])
        if hasattr(model, 'model'):  # Check if it has a model attribute
            model.model = model.model.to(self.device)  # Move model to the GPU
         # Move NetVlad model to GPU if available
        return model

    def mixvpr(self, content):
        model = VPRModel(backbone_arch=content["backbone_arch"], 
                         layers_to_crop=[content['layers_to_crop']],
                         agg_arch=content['agg_arch'],
                         agg_config=content['agg_config'],
                         )
        state_dict = torch.load(content["ckpt_path"], map_location=self.device)  # Load model state to the same device
        model.load_state_dict(state_dict)
        model = model.to(self.device)  # Move MixVPR model to GPU if available
        model.eval()
        return model

    def get(self):
        for extractor, content in self.extractor.items():
            if extractor == 'netvlad':
                return self.netvlad(content).feature
            if extractor == 'mixvpr':
                return self.mixvpr(content)
            if extractor == 'vlad':
                pass
            if extractor == 'bovw':
                pass
