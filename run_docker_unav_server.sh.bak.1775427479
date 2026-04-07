#!/bin/bash
set -a
[ -f .env.local ] && source ./.env.local
set +a

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

ENV_ARGS=()
[ -n "${OPENAI_API_KEY:-}" ] && ENV_ARGS+=( -e OPENAI_API_KEY="$OPENAI_API_KEY" )
[ -n "${OPENAI_MODEL:-}" ] && ENV_ARGS+=( -e OPENAI_MODEL="$OPENAI_MODEL" )
[ -n "${GEMINI_API_KEY:-}" ] && ENV_ARGS+=( -e GEMINI_API_KEY="$GEMINI_API_KEY" )
[ -n "${GEMINI_MODEL:-}" ] && ENV_ARGS+=( -e GEMINI_MODEL="$GEMINI_MODEL" )
[ -n "${LLM_PROVIDER:-}" ] && ENV_ARGS+=( -e LLM_PROVIDER="$LLM_PROVIDER" )
[ -n "${LLM_FALLBACK_PROVIDER:-}" ] && ENV_ARGS+=( -e LLM_FALLBACK_PROVIDER="$LLM_FALLBACK_PROVIDER" )
[ -n "${OPENAI_API_URL:-}" ] && ENV_ARGS+=( -e OPENAI_API_URL="$OPENAI_API_URL" )
[ -n "${GEMINI_API_URL:-}" ] && ENV_ARGS+=( -e GEMINI_API_URL="$GEMINI_API_URL" )

docker run --gpus device=1 --rm -it \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -v ./config.py:/workspace/config.py \
  -v "${DATA_ROOT}:/data" \
  "${ENV_ARGS[@]}" \
  "${IMAGE_NAME}"
