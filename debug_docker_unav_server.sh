#!/usr/bin/env bash
# File: debug_docker_unav_server.sh
# Purpose: Launch the UNav server inside Docker with VS Code attach debugging (debugpy).
# Author: You :)
# Usage: bash debug_docker_unav_server.sh
# Optional env overrides:
#   START_WITH=uvicorn|script     # Default: script
#   GPU_DEVICES="device=1"        # Default: device=1
#   HOST_PORT=5001                # Default: 5001
#   DEBUG_HOST_PORT=5678          # Default: 5678
#   IMAGE_NAME="unav-server"      # Default: unav-server
#   DATA_ROOT="/mnt/data/UNav-IO/data"  # Default path to host data dir

set -Eeuo pipefail

#####################################
# ---- User-configurable section ----
#####################################
IMAGE_NAME="${IMAGE_NAME:-unav-server}"

# App and debug ports
HOST_PORT="${HOST_PORT:-5001}"
CONTAINER_PORT="${CONTAINER_PORT:-5001}"
DEBUG_HOST_PORT="${DEBUG_HOST_PORT:-5678}"
DEBUG_CONTAINER_PORT="${DEBUG_CONTAINER_PORT:-5678}"

# GPU settings (e.g., "all" or "device=0" or "device=1")
GPU_DEVICES="${GPU_DEVICES:-device=1}"

# Where your dataset lives on the host; will be mounted to /data in container
DATA_ROOT="${DATA_ROOT:-/mnt/data/UNav-IO/data}"

# Project root on the host (mounted to /workspace in the container)
REPO_ROOT="$(pwd)"

# Conda env inside the image
CONDA_ENV_NAME="${CONDA_ENV_NAME:-unav}"
PYBIN="/opt/conda/envs/${CONDA_ENV_NAME}/bin/python"

# How to start the app inside the container:
# - "uvicorn": run uvicorn with `main:app`
# - "script" : run `python main.py`
START_WITH="${START_WITH:-script}"

#####################################
# ---- Derived/internal settings ----
#####################################
APP_CMD=""
case "${START_WITH}" in
  uvicorn)
    # If main.py defines `app = FastAPI()`, use this:
    APP_CMD="${PYBIN} -m uvicorn main:app --host 0.0.0.0 --port ${CONTAINER_PORT}"
    ;;
  script)
    # If main.py has a `if __name__ == '__main__'` entry, use this:
    APP_CMD="${PYBIN} main.py"
    ;;
  *)
    echo "[ERROR] Unknown START_WITH='${START_WITH}'. Use 'uvicorn' or 'script'." >&2
    exit 1
    ;;
esac

# Final wrapped command with debugpy
RUN_CMD="${PYBIN} -m debugpy --listen 0.0.0.0:${DEBUG_CONTAINER_PORT} --wait-for-client ${APP_CMD#${PYBIN} }"

#####################################
# ---- Display info ----
#####################################
echo "Launching Docker container: ${IMAGE_NAME}"
echo "Mounting source:  ${REPO_ROOT}  -> /workspace"
echo "Mounting data:    ${DATA_ROOT}  -> /data"
echo "Expose ports:     ${HOST_PORT}:${CONTAINER_PORT} (app), ${DEBUG_HOST_PORT}:${DEBUG_CONTAINER_PORT} (debugpy)"
echo "Conda env:        ${CONDA_ENV_NAME}"
echo "Python binary:    ${PYBIN}"
echo "Start mode:       ${START_WITH}"
echo "Run command:      ${RUN_CMD}"

#####################################
# ---- Run container ----
#####################################
docker run --gpus "${GPU_DEVICES}" --rm -it \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -p "${DEBUG_HOST_PORT}:${DEBUG_CONTAINER_PORT}" \
  -v "${REPO_ROOT}:/workspace" \
  -v "${DATA_ROOT}:/data" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash -lc "
    set -Eeuo pipefail
    echo '[INFO] Python used:' ${PYBIN}
    # Ensure debugpy is available in the SAME interpreter we will run
    ${PYBIN} - <<'PY'
import sys
print('[INFO] sys.executable:', sys.executable)
try:
    import debugpy
    print('[INFO] debugpy OK:', debugpy.__version__)
except ImportError:
    print('[WARN] debugpy not found in this env. Installing now...', flush=True)
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'debugpy'])
    import debugpy
    print('[INFO] debugpy installed:', debugpy.__version__)
PY
    echo '[INFO] Waiting for VS Code to attach on 0.0.0.0:${DEBUG_CONTAINER_PORT} ...'
    ${RUN_CMD}
  "
