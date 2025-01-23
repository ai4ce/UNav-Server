import os
from os.path import join
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import yaml
from AnyLoc.feature_extract import VLADDinoV2FeatureExtractor


with open("../../configs/trainer_pitts250_for_running_mix.yaml", 'r') as f:
    config = yaml.safe_load(f)

model_configs = config["vpr"]["global_extractor"]["AnyLoc"]
extr = VLADDinoV2FeatureExtractor(config["root"], model_configs)
extr.set_float32()
extr.set_parallel()

input_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

images_dir = "/mnt/data/VPR4LQQ/logs/tum_lsi/2015-08-16_15.34.11"
for image in tqdm(os.listdir(images_dir)):
    image = Image.open(join(images_dir, image))
    images = input_transform(image).unsqueeze(0).to("cuda")
    extr(images)
    image.close()
