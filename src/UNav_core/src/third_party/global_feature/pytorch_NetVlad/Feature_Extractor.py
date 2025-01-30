# import sys
from os.path import join,exists, isfile
# sys.path.append(join(sys.path[0],'third_party','pytorch_NetVlad'))
from .netvlad import NetVLAD
import torchvision.transforms as transforms
import json
import torch
import torchvision.models as models
import torch.nn as nn

def input_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class NetVladFeatureExtractor:
    def __init__(self, ckpt_path, arch='vgg16', num_clusters=64, pooling='netvlad', vladv2=False, nocuda=False,
                 input_transform=input_transform()):
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

        encoder = nn.Sequential(*layers)
        self.model = nn.Module()
        self.model.add_module('encoder', encoder)

        if pooling.lower() == 'netvlad':
            net_vlad = NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=vladv2)
            self.model.add_module('pool', net_vlad)
        else:
            raise ValueError('Unknown pooling type: ' + pooling)

        resume_ckpt = join(ckpt_path, 'checkpoints', 'checkpoint.pth.tar')
        print(resume_ckpt)

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
            image = self.input_transform(image).unsqueeze(0)

        with torch.no_grad():
            input = image.to(self.device)
            image_encoding = self.model.encoder(input)
            vlad_encoding = self.model.pool(image_encoding)

            del input
            torch.cuda.empty_cache()
            return vlad_encoding