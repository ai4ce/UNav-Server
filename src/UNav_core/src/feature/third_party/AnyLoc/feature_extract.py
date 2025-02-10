import os
import torch
from torchvision import transforms
from torchvision.transforms import functional
from .utilities import DinoV2ExtractFeatures, VLAD


class VLADDinoV2FeatureExtractor:

    def __init__(self, root, content, pipeline=False):
        self.max_image_size = content["max_image_size"]
        self.device = "cuda" if content["cuda"] else "cpu"

        if content["ckpt_path"] == "None":
            weights_source = content["model_type"]
            print(f"AnyLoc extractor ignores pipeline option (={pipeline}) because ckpt_path is None")
            self.saved_state = {"epoch": 0, "best_score": 0}
        else:
            weights_source = os.path.join(content["ckpt_path"], "model_best.pth") if pipeline else content["ckpt_path"]
            self.saved_state = torch.load(weights_source, map_location=self.device)
            if self.saved_state.keys() == {"epoch", "best_score", "state_dict"}:
                # Remove module prefix from state dict
                state_dict_keys = list(self.saved_state["state_dict"].keys())
                for state_key in state_dict_keys:
                    if state_key.startswith("module"):
                        new_key = state_key.removeprefix("module.")
                        self.saved_state["state_dict"][new_key] = self.saved_state["state_dict"][state_key]
                        del self.saved_state["state_dict"][state_key]
                weights_source = os.path.join(content["ckpt_path"], "dino_model_temp.pth")
                torch.save(self.saved_state["state_dict"], weights_source)
            else:
                self.saved_state = {"epoch": 0, "best_score": 0}
        # DinoV2 & VLAD settings
        self.dinov2_extractor = DinoV2ExtractFeatures(weights_source, content["desc_layer"], content["desc_facet"], device=self.device)
        if os.path.exists(os.path.join(content["ckpt_path"], "dino_model_temp.pth")): os.remove(os.path.join(content["ckpt_path"], "dino_model_temp.pth"))
        self.saved_state["state_dict"] = self.dinov2_extractor.dino_model.state_dict()

        vocab_dir = os.path.join(content["model_type"], f"l{content['desc_layer']}_{content['desc_facet']}_c{content['num_clusters']}")
        c_centers_file = os.path.join(root, content["cache_dir"], "vocabulary", vocab_dir, content["domain"], "c_centers.pt")
        c_centers = torch.load(c_centers_file)
        assert c_centers.shape[0] == content["num_clusters"], f"Wrong number of clusters! (from file {c_centers.shape[0]} != from config {content['num_clusters']})"
        self.vlad = VLAD(content["num_clusters"], desc_dim=None, cache_dir=os.path.dirname(c_centers_file))
        self.vlad.fit(None)

    def __call__(self, images):
        scaled_imgs = self.downscale(images)
        b, c, h, w = scaled_imgs.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        scaled_imgs = transforms.CenterCrop((h_new, w_new))(scaled_imgs)
        # Extract descriptor
        features = self.dinov2_extractor(scaled_imgs) # [num_imgs, num_patches, desc_dim]
        vlad_feats = [self.vlad.generate(features[i].cpu().squeeze()) for i in range(features.shape[0])] # VLAD:  [agg_dim]
        features_map = features.reshape((b, h_new // 14, w_new // 14, features.shape[-1])).permute(0, 3, 1, 2)
        return features_map, torch.stack(vlad_feats).to(self.device) # shape: [num_imgs, agg_dim]
    
    def downscale(self, images):
        if max(images.shape[-2:]) > self.max_image_size:
            b, c, h, w = images.shape
            # Maintain aspect ratio
            if h == max(images.shape[-2:]):
                w = int(w * self.max_image_size / h)
                h = self.max_image_size
            else:
                h = int(h * self.max_image_size / w)
                w = self.max_image_size
            return functional.resize(images, (h, w), interpolation=functional.InterpolationMode.BICUBIC)
        return images

    def set_train(self, is_train):
        self.dinov2_extractor.dino_model.train(is_train)
    
    def torch_compile(self, **compile_args):
        self.dinov2_extractor.dino_model = torch.compile(self.dinov2_extractor.dino_model, **compile_args)
    
    def set_parallel(self):
        self.dinov2_extractor.set_parallel()
    
    def set_float32(self):
        self.dinov2_extractor.dino_model.to(torch.float32)
    
    def save_state(self, save_path, new_state):
        new_state["state_dict"] = self.dinov2_extractor.dino_model.state_dict()
        torch.save(new_state, save_path)

    @property
    def last_epoch(self): return self.saved_state["epoch"]

    @property
    def best_score(self): return self.saved_state["best_score"]

    @property
    def parameters(self): return self.dinov2_extractor.dino_model.parameters()

    @property
    def feature_length(self): return self.vlad.desc_dim * self.vlad.num_clusters
