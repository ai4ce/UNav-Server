"""Modal v2 deployment for UNav Server using existing Dockerfile.

Builds the Modal image from the project Dockerfile and exposes the
existing uvicorn FastAPI server via Modal's web server proxy.

Usage:
    modal run modal_unav_v2.py          # Run locally
    modal deploy modal_unav_v2.py       # Deploy to Modal cloud
"""
import subprocess
import sys

import modal
from modal import Image, App, Volume

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GPU_MODEL = "T4"              # "T4", "A100", "H100", etc.
DATA_VOLUME_NAME = "unav-data"
CONTAINER_PORT = 5001

# ---------------------------------------------------------------------------
# Image – build from the existing Dockerfile
# ---------------------------------------------------------------------------
image = Image.from_dockerfile(
    "Dockerfile",
    context_dir=".",
).add_local_file("config.py", remote_path="/workspace/config.py")

# ---------------------------------------------------------------------------
# Volume for persistent data
# ---------------------------------------------------------------------------
data_volume = Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)

# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------
app = App("anbang-unav-server-v2")


@app.function(
    image=image,
    gpu=GPU_MODEL,
    volumes={"/data": data_volume},
    timeout=3600,
    scaledown_window=600,
)
@modal.web_server(CONTAINER_PORT)
def fastapi_app():
    # Test-mode hotfix: install Modal runtime deps at container start to avoid image rebuilds.
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "aiohttp",
            "cbor2",
            "certifi",
            "click~=8.1",
            "grpclib>=0.4.7,<0.4.10",
            "protobuf>=3.19,<7.0,!=4.24.0",
            "rich>=12",
            "synchronicity~=0.11.1",
            "toml",
            "typer>=0.9",
            "types-certifi",
            "types-toml",
            "watchfiles",
            "typing_extensions~=4.6",
        ]
    )
    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"],
        cwd="/workspace",
    )


if __name__ == "__main__":
    app.deploy()
