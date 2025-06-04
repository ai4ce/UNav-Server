#!/bin/bash

# 用户设置数据根路径（只需改这里）
DATA_ROOT=/mnt/data/UNav-IO/data

# 镜像名
IMAGE_NAME=unav-server

# 宿主机端口（可修改）
HOST_PORT=5001
CONTAINER_PORT=5001

echo "Launching $IMAGE_NAME"
echo "Host data: $DATA_ROOT"
echo "Mounting to container: /data"

docker run --gpus all -it \
  -p ${HOST_PORT}:${CONTAINER_PORT} \
  -v ${DATA_ROOT}:/data \
  ${IMAGE_NAME}
