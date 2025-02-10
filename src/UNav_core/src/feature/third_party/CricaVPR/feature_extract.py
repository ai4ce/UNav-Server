from os.path import join
from types import SimpleNamespace
import torch
from torchvision import transforms
from .network import CricaVPRNet
from .util import resume_model


class CricaVPRFeatureExtractor:

    def __init__(self, root, content, pipeline=False):
       self.device = "cuda" if content["cuda"] else "cpu"
       self.saved_state, self.model = self.load_model(root, content, pipeline)
    
    def load_model(self, root, content, pipeline):
        model = CricaVPRNet()
        model = model.eval().to(self.device)
        ckpt_path = join(root, content["ckpt_path"])
        if pipeline:
            saved_state = torch.load(join(ckpt_path, "model_best.pth"), map_location=self.device)
            # Remove module prefix from state dict
            state_dict_keys = list(saved_state["state_dict"].keys())
            for state_key in state_dict_keys:
                if state_key.startswith("module"):
                    new_key = state_key.removeprefix("module.")
                    saved_state["state_dict"][new_key] = saved_state["state_dict"][state_key]
                    del saved_state["state_dict"][state_key]
            model.load_state_dict(saved_state["state_dict"])
        else:
            ckpt_info = SimpleNamespace(resume=ckpt_path, device=self.device)
            model = resume_model(ckpt_info, model)
            saved_state = {"epoch": 0, "best_score": 0}
        print(f"CricaVPR loaded from {ckpt_path} successfully!")
        return saved_state, model

    def __call__(self, images):
        # Input images must be square; implemented "hard_resize" strategy from datasets_ws.py
        new_side_len = min(images.shape[-1], images.shape[-2]) // 14 * 14    # dimensions must be divisible by 14
        resized_imgs = transforms.CenterCrop((new_side_len, new_side_len))(images)
        encodings, descriptors = self.model(resized_imgs)
        return encodings, descriptors
    
    def set_train(self, is_train):
        self.model.train(is_train)
    
    def torch_compile(self, **compile_args):
        self.model = torch.compile(self.model, **compile_args)
    
    def set_parallel(self):
        self.model = torch.nn.DataParallel(self.model)

    def set_float32(self):
        self.model.to(torch.float32)
    
    def save_state(self, save_path, new_state):
        new_state["state_dict"] = self.model.state_dict()
        torch.save(new_state, save_path)
    
    @property
    def last_epoch(self): return self.saved_state["epoch"]

    @property
    def best_score(self): return self.saved_state["best_score"]

    @property
    def parameters(self): return self.model.parameters()
    
    @property
    def feature_length(self):
        return 14 * 768    # patch_size * embed_dim