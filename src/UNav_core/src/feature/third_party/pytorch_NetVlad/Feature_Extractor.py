from os.path import join,exists, isfile
from . import netvlad
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
    def __init__(self, ckpt_path, type=None, arch='vgg16', num_clusters=64, pooling='netvlad', vladv2=False, nocuda=False,
                 input_transform=input_transform()):
        self.input_transform = input_transform
        self.num_clusters = num_clusters

        flag_file = join(ckpt_path, 'flags.json')
        if exists(flag_file):
            with open(flag_file, 'r') as f:
                stored_flags = json.load(f)
                stored_num_clusters = stored_flags.get('num_clusters')
                if stored_num_clusters is not None:
                    self.num_clusters = stored_num_clusters
                    print(f'restore num_clusters to : {self.num_clusters}')
                stored_pooling = stored_flags.get('pooling')
                if stored_pooling is not None:
                    pooling = stored_pooling
                    print(f'restore pooling to : {pooling}')

        cuda = not nocuda
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run with --nocuda")

        self.device = torch.device("cuda" if cuda else "cpu")
        self.is_parallel = False

        print('===> Building model')

        if arch.lower() == 'alexnet':
            self.encoder_dim = 256
            encoder = models.alexnet(pretrained=True)
            # capture only features and remove last relu and maxpool
            layers = list(encoder.features.children())[:-2]

            # if using pretrained only train conv5
            for l in layers[:-1]:
                for p in l.parameters():
                    p.requires_grad = False

        elif arch.lower() == 'vgg16':
            self.encoder_dim = 512
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
            net_vlad = netvlad.NetVLAD(num_clusters=self.num_clusters, dim=self.encoder_dim, vladv2=vladv2)
            self.model.add_module('pool', net_vlad)
        else:
            raise ValueError('Unknown pooling type: ' + pooling)

        if type=='pipeline':
            resume_ckpt = join(ckpt_path,'model_best.pth')
        else:
            resume_ckpt = join(ckpt_path,'checkpoint.pth')

        if isfile(resume_ckpt):
            print("=> loading model weights '{}'".format(resume_ckpt))
            self.checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
            best_metric = self.checkpoint['best_score']
            state_dict=self.checkpoint['state_dict']
            # Remove module prefix from state dict
            state_dict_keys = list(state_dict.keys())
            for state_key in state_dict_keys:
                if state_key.startswith("module"):
                    new_key = state_key.removeprefix("module.")
                    state_dict[new_key] = state_dict[state_key]
                    del state_dict[state_key]
            state_dict={k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            self.model = self.model.eval().to(self.device)
            print("=> loaded weights '{}' (epoch {})"
                  .format(resume_ckpt, self.checkpoint['epoch']))
        else:
            print("=> no model weights found at '{}'".format(resume_ckpt))
            exit()

    def __call__(self, images):
        image_encoding = self.model.module.encoder(images) if self.is_parallel else self.model.encoder(images)
        vlad_encoding = self.model.module.pool(image_encoding) if self.is_parallel else self.model.pool(image_encoding)
        return image_encoding, vlad_encoding
    
    def set_train(self, is_train):
        self.model.train(is_train)
    
    def torch_compile(self, **compile_args):
        self.model = torch.compile(self.model, **compile_args)
    
    def set_parallel(self):
        self.model = torch.nn.DataParallel(self.model)
        self.is_parallel = True
    
    def set_float32(self):
        self.model.to(torch.float32)
    
    def save_state(self, save_path, new_state):
        new_statestate_dict = self.model.state_dict()
        torch.save(new_state, save_path)

    @property
    def last_epoch(self): return self.checkpoint["epoch"]

    @property
    def best_score(self): return self.checkpoint["best_score"]

    @property
    def parameters(self): return self.model.parameters()

    @property
    def feature_length(self): return self.encoder_dim * self.num_clusters
