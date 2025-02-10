from os.path import join
from UNav_core.src.feature.third_party.pytorch_NetVlad.Feature_Extractor import NetVladFeatureExtractor
from UNav_core.src.feature.third_party.MixVPR.feature_extract import MixVPRFeatureExtractor
from UNav_core.src.feature.third_party.AnyLoc.feature_extract import VLADDinoV2FeatureExtractor
from UNav_core.src.feature.third_party.salad.feature_extract import DINOV2SaladFeatureExtractor
from UNav_core.src.feature.third_party.CricaVPR.feature_extract import CricaVPRFeatureExtractor


class GlobalExtractors:

    model_classes = {"MixVPR": MixVPRFeatureExtractor, "AnyLoc": VLADDinoV2FeatureExtractor, "DinoV2Salad": DINOV2SaladFeatureExtractor,
                     "CricaVPR": CricaVPRFeatureExtractor}

    def __init__(self, root, g_extr_conf, pipeline=False, data_parallel=False):
        """
        Creates a container for a list of vpr methods for inference.
        - root: prefix path for model parameters (same as in test_trained_model.py)
        - g_extr_conf: global extractor configurations; corresponding key under test_trained_model.yaml and trainer.yaml
        - pipeline: controls whether to use a trained model (assumes "model_best.pth") or the pretrained verison ("checkpoint.pth")
        """
        self.models_objs = {}
        for model_name, model_configs in {name: conf for name, conf in g_extr_conf.items() if conf["use"]}.items():
            if model_name == "NetVlad":
                # Special setup for netvlad
                netvlad_model = NetVladFeatureExtractor(join(root, model_configs["ckpt_path"]), type="pipeline" if pipeline else None,
                    arch=model_configs['arch'], num_clusters=model_configs['num_clusters'], pooling=model_configs['pooling'],
                    vladv2=model_configs['vladv2'], nocuda=model_configs['nocuda'])
                self.models_objs["NetVlad"] = netvlad_model
                if data_parallel: self.models_objs["NetVlad"].set_parallel()
            elif model_name in GlobalExtractors.model_classes:
                self.models_objs[model_name] = GlobalExtractors.model_classes[model_name](root, model_configs, pipeline)
                if data_parallel: self.models_objs[model_name].set_parallel()
            else:
                print(f"GlobalExtractors doesn't have {model_name}'s implementation, skipped.")

    def __call__(self, request_model, images):
        """
        Evaluate input images, assumed to be batched tensors already normalized, on the requested vpr model
        - request_model: string name of the desired vpr model
        """
        if request_model not in self.models_objs:
            print(f"GlobalExtractors wasn't initialized with {request_model}, skipped. options: {self.models}")
        else:
            return self.models_objs[request_model](images)
    
    def set_train(self, is_train):
        """
        All models are created in eval mode. This method explicitly sets the mode of all models.
        """
        for model in self.models_objs:
            self.models_objs[model].set_train(is_train)
    
    def torch_compile(self, **compile_args):
        """
        Apply torch.compile with all given keyword arguments to all models
        """
        for model in self.models_objs:
            self.models_objs[model].torch_compile(**compile_args)
    
    def set_float32(self):
        """
        Change all model's precision to torch.float32
        """
        for model in self.models_objs:
            self.models_objs[model].set_float32()
    
    def save_state(self, model, save_path, new_state):
        """
        Save a new epoch number and recall score to 
        """
        self.models_objs[model].save_state(save_path, new_state)

    @property
    def models(self):
        return list(self.models_objs.keys())

    def last_epoch(self, model):
        return self.models_objs[model].last_epoch

    def best_score(self, model):
        return self.models_objs[model].best_score

    def model_parameters(self, model):
        return self.models_objs[model].parameters

    def feature_length(self, model):
        return self.models_objs[model].feature_length