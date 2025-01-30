from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import Flatten
import h5py
from pathlib import Path
from PIL import Image
from types import SimpleNamespace

import numpy as np
import netvlad

parser = argparse.ArgumentParser(description='pytorch-NetVlad')
parser.add_argument('--nocuda', action='store_true', help='Dont use cuda')
parser.add_argument('--ckpt_path', type=str, default='vgg16_netvlad_checkpoint', help='Path to load checkpoint from, for resuming training or testing.')
parser.add_argument('--arch', type=str, default='vgg16', 
        help='basenetwork to use', choices=['vgg16', 'alexnet'])
parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
parser.add_argument('--pooling', type=str, default='netvlad', help='type of pooling to use',
        choices=['netvlad', 'max', 'avg'])
parser.add_argument('--num_clusters', type=int, default=64, help='Number of NetVlad clusters. Default=64')
parser.add_argument('--image_dir', type=Path)
parser.add_argument('--output', type=Path, default='output',help='Output folder for featues')

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

class NetVladFeatureExtractor:
    def __init__ (self, ckpt_path, arch='vgg16', num_clusters=64, pooling='netvlad', vladv2=False, nocuda=False, input_transform=input_transform()):
        self.input_transform = input_transform
        
        flag_file = join(ckpt_path, 'checkpoints', 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = json.load(f)
                stored_num_clusters = stored_flags.get('num_clusters')
                if stored_num_clusters is not None:
                    num_clusters = stored_num_clusters
                    print(f'restore num_clusters to : {num_clusters}')
                stored_pooling = stored_flags.get('pooling')
                if stored_pooling is not None:
                    pooling = stored_pooling
                    print(f'restore pooling to : {pooling}')

        cuda = not nocuda
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --nocuda")

        self.device = torch.device("cuda" if cuda else "cpu")

        print('===> Building model')

        if arch.lower() == 'alexnet':
            encoder_dim = 256
            encoder = models.alexnet(pretrained=True)
            # capture only features and remove last relu and maxpool
            layers = list(encoder.features.children())[:-2]

            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

        elif arch.lower() == 'vgg16':
            encoder_dim = 512
            encoder = models.vgg16(pretrained=True)
            # capture only feature part and remove last relu and maxpool
            layers = list(encoder.features.children())[:-2]

            # if using pretrained then only train conv5_1, conv5_2, and conv5_3
            for l in layers[:-5]: 
                for p in l.parameters():
                    p.requires_grad = False
                    
        # TODO 2: add a new arch option called "edvr"

        encoder = nn.Sequential(*layers)
        self.model = nn.Module() 
        self.model.add_module('encoder', encoder)

        if pooling.lower() == 'netvlad':
            net_vlad = netvlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=vladv2)
            self.model.add_module('pool', net_vlad)
        elif pooling.lower() == 'max':
            global_pool = nn.AdaptiveMaxPool2d((1,1))
            self.model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        elif pooling.lower() == 'avg':
            global_pool = nn.AdaptiveAvgPool2d((1,1))
            self.model.add_module('pool', nn.Sequential(*[global_pool, Flatten(), L2Norm()]))
        else:
            raise ValueError('Unknown pooling type: ' + pooling)
    
        resume_ckpt = join(ckpt_path, 'checkpoints', 'checkpoint.pth.tar')

        if isfile(resume_ckpt):
            print("=> loading checkpoint '{}'".format(resume_ckpt))
            checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            best_metric = checkpoint['best_score']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model = self.model.eval().to(self.device)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_ckpt))
        
    def feature(self, image):
        if self.input_transform:
            image = self.input_transform(image)
            #batch size 1
            image = torch.stack([image])
        
        with torch.no_grad():
            input = image.to(self.device)
            # TODO 3: Maybe need to reformat input so it can be fed into the new encoder
            image_encoding = self.model.encoder(input)  # going through edvr
            # TODO 4: Maybe need to reformat image_encoding so it can be fed into pool
            vlad_encoding = self.model.pool(image_encoding) 

            return vlad_encoding.detach().cpu().numpy()

if __name__ == "__main__":
    args = parser.parse_args()

    globs=['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    image_paths = []
    for g in globs:
        image_paths += list(Path(args.image_dir).glob('**/'+g))
    if len(image_paths) == 0:
        raise ValueError(f'Could not find any image in root: {root}.')
    image_paths = sorted(list(set(image_paths)))
    image_paths = [i.relative_to(args.image_dir) for i in image_paths]
    
    global_feature_path=Path(args.output, 'global_features.h5')
    global_feature_path.parent.mkdir(exist_ok=True, parents=True)
    global_feature_file = h5py.File(str(global_feature_path), 'w')

    extractor = NetVladFeatureExtractor(args.ckpt_path, arch=args.arch, num_clusters=args.num_clusters, pooling=args.pooling, vladv2=args.vladv2, nocuda=args.nocuda)
    
    cnt = 0
    for im_file in image_paths:
        image = Image.open(str(args.image_dir / im_file))
        feature = extractor.feature(image)[0]
        #print(f'shape {feature.shape}')
        grp = global_feature_file.create_group(str(im_file))
        grp.create_dataset('global_descriptor', data=feature)
        cnt += 1
        if cnt % 200 == 0 :
            print('{} images processed'.format(cnt))

    global_feature_file.close()

