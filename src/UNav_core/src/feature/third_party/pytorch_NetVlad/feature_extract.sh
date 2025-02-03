#!/bin/bash

name=NYU_15thfloor

localization_path=/home/endeleze/Desktop/WeNev/Localization
ckpt_path=$localization_path/pytorch_NetVlad/netvlad_parameters/log/paper
#image_dir=$localization_path/pytorch_NetVlad/data/$name/database
#output=$localization_path/pytorch_NetVlad/data/$name
image_dir=/home/endeleze/Desktop/WeNev/Mapping/data/src_images/$name
output=/home/endeleze/Desktop/WeNev/Mapping/data/maps/$name
script=$localization_path/pytorch_NetVlad/feature_extract.py


source ~/anaconda3/etc/profile.d/conda.sh

conda activate open3d

python $script --ckpt_path $ckpt_path --image_dir $image_dir --output $output
