#!/bin/bash

# Path to the local data directory to be mounted inside the container
DATA_ROOT="/mnt/data/UNav-IO/data"

# Docker image name for the UNav server
IMAGE_NAME="unav-server"

# Host and container port mapping
HOST_PORT=5001
CONTAINER_PORT=5001

# Absolute path of this script's directory — so `api/`, `main.py`, `core/`
# etc. can be bind-mounted into /workspace inside the container. This lets
# us ship backend code changes (e.g. api/trial_api.py for the research
# TrialRecorder endpoint) without rebuilding the Docker image.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Launching Docker container: ${IMAGE_NAME}"
echo "Mounting local data directory: ${DATA_ROOT} -> /data (in container)"
echo "Mounting backend source: ${SCRIPT_DIR}/{api,main.py,core,db,models} -> /workspace/..."
echo "Exposing port: ${HOST_PORT} -> ${CONTAINER_PORT}"

ENV_FILE="${SCRIPT_DIR}/.env.local"
ENV_FILE_ARGS=()
if [ -f "${ENV_FILE}" ]; then
  ENV_FILE_ARGS+=( --env-file "${ENV_FILE}" )
else
  echo "Warning: ${ENV_FILE} not found; LLM providers will only use container defaults." >&2
fi

docker run --gpus device=1 --rm -it \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v "${SCRIPT_DIR}/config.py:/workspace/config.py" \
  -v "${SCRIPT_DIR}/api:/workspace/api" \
  -v "${SCRIPT_DIR}/main.py:/workspace/main.py" \
  -v "${SCRIPT_DIR}/core:/workspace/core" \
  -v "${SCRIPT_DIR}/db:/workspace/db" \
  -v "${SCRIPT_DIR}/models:/workspace/models" \
  -v "${DATA_ROOT}:/data" \
  -v "/mnt/data/UNav-IO/temp:/mnt/data/UNav-IO/temp:ro" \
  -v "/home/unav/Desktop/unav/unav:/opt/conda/envs/unav/lib/python3.10/site-packages/unav" \
  -v "/home/unav/Desktop/mast3r:/workspace/mast3r" \
  -v "/home/unav/.cache/huggingface:/root/.cache/huggingface" \
  -v "/home/unav/.cache/torch:/root/.cache/torch" \
  "${ENV_FILE_ARGS[@]}" \
  -e PYTHONPATH=/workspace/mast3r \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${IMAGE_NAME}" \
  bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate unav && pip install 'bcrypt<4.0.0' -q && cd /workspace && PYTHONPATH=/workspace/mast3r uvicorn main:app --host 0.0.0.0 --port ${CONTAINER_PORT}"
