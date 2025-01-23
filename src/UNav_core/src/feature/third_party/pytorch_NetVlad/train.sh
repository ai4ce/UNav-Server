#!/bin/bash

name=GS010021

localization_path=/home/endeleze/Desktop/WeNev/Localization
dataPath=$localization_path/pytorch_NetVlad/data/$name
script=$localization_path/pytorch_NetVlad/main.py
outf=$dataPath/Out

source ~/anaconda3/etc/profile.d/conda.sh

conda activate open3d

python $script --dataPath $dataPath --runsPath $outf/log --cachePath $outf/cache --mode cluster
python $script --dataPath $dataPath --runsPath $outf/log --cachePath $outf/cache --mode train
