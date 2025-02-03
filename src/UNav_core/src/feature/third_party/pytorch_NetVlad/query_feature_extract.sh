#!/bin/bash

name=GS010021
query_name=GA_01

localization_path=/home/endeleze/Desktop/WeNev/Localization
ckpt_path=$localization_path/pytorch_NetVlad/Out/$name/log/Aug27_00-19-26_vgg16_netvlad
image_dir=/home/endeleze/Desktop/WeNev/Mapping/data/src_images/$query_name
output=$localization_path/pytorch_NetVlad/Out/$name/query
script=$localization_path/pytorch_NetVlad/feature_extract.py


source ~/anaconda3/etc/profile.d/conda.sh

conda activate open3d

python $script --ckpt_path $ckpt_path --image_dir $image_dir --output $output
