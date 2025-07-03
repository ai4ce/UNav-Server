#!/bin/bash

# Path to the local data directory to be mounted inside the container
DATA_ROOT="/mnt/data/UNav-IO/data"

# Docker image name for the UNav server
IMAGE_NAME="unav-server"

# Host and container port mapping
HOST_PORT=5001
CONTAINER_PORT=5001

echo "Launching Docker container: ${IMAGE_NAME}"
echo "Mounting local data directory: ${DATA_ROOT} -> /data (in container)"
echo "Exposing port: ${HOST_PORT} -> ${CONTAINER_PORT}"

docker run --gpus device=1 --rm -it \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v ./config.py:/workspace/config.py \
  -v "${DATA_ROOT}:/data" \
  "${IMAGE_NAME}"
