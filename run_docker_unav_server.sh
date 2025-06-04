#!/bin/bash

# Path to the local data directory to be mounted inside the container
DATA_ROOT="/mnt/d/unav/data"

# Docker image name for the UNav server
IMAGE_NAME="unav-server"

# Host and container port mapping
HOST_PORT=5001
CONTAINER_PORT=5001

echo "Launching Docker container: ${IMAGE_NAME}"
echo "Mounting local data directory: ${DATA_ROOT} -> /data (in container)"
echo "Exposing port: ${HOST_PORT} -> ${CONTAINER_PORT}"

docker run --gpus all -it \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${DATA_ROOT}:/data" \
  "${IMAGE_NAME}"
