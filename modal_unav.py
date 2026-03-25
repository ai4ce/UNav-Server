"""Modal deployment for UNav Server.

Usage:
    modal run modal_unav.py          # Run locally
    modal deploy modal_unav.py       # Deploy to Modal cloud
"""
from modal import Image, App, Mount, asgi_app, Volume

IMAGE_MODEL = "T4"  # or "A100", "H100", "H200" depending on needs

pip_requirements = """
einops==0.8.1
faiss-gpu-cu12==1.11.0
fast-pytorch-kmeans==0.2.2
flask==3.0.3
h5py==3.13.0
jupyter-core==5.8.1
jupyterlab-widgets==3.0.15
PyJWT>=2.0.0
kornia==0.8.1
lazy-loader==0.4
matplotlib==3.10.3
opencv-python==4.11.0.86
pandas==2.2.3
passlib==1.7.4
pillow==11.2.1
poselib==2.0.4
pytorch-lightning==2.5.1.post0
pytorch-metric-learning==2.8.1
scikit-image==0.25.2
scikit-learn==1.6.1
scipy==1.15.3
shapely==2.1.1
sqlalchemy==2.0.41
timm==1.0.15
tqdm==4.67.1
transformers==4.52.4
wandb==0.20.0
fastapi
uvicorn[standard]
pydantic
""".strip()

image = (
    Image.from_registry("nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04")
    .apt_install("wget", "git", "vim", "cmake", "libeigen3-dev", "libceres-dev", "build-essential", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands(
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(pip_requirements)
    .pip_install("open3d==0.19.0")
    .pip_install("xformers==0.0.30")
    .pip_install("git+https://github.com/cvg/implicit_dist.git")
    .pip_install("git+https://github.com/endeleze/UNav.git")
    .workdir("/workspace")
)

data_volume = Volume.from_name("unav-data", create_if_missing=True)

source_mount = Mount.from_local_dir(
    ".",
    remote_path="/workspace",
    condition=lambda path: path not in [
        ".git", "environment.yml", "Dockerfile", 
        "run_docker_unav_server.sh", ".dockerignore",
        "modal_unav.py", "README.md", ".env"
    ]
)

app = App("unav-server")


@app.function(
    image=image,
    mounts=[source_mount],
    gpu=IMAGE_MODEL,
    volumes={"/data": data_volume},
    timeout=3600,
    container_idle_timeout=600,
    allow_concurrent_inputs=100,
)
@asgi_app()
def fastapi_app():
    from fastapi import FastAPI
    from contextlib import asynccontextmanager
    import asyncio

    from api.user_api import router as user_router
    from api.task_api import router as task_router
    from core.unav_state import cleanup_sessions

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cleanup_task = asyncio.create_task(cleanup_sessions())
        yield
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    fastapi_app_instance = FastAPI(
        lifespan=lifespan,
        title="UNav Server",
        version="1.0"
    )

    fastapi_app_instance.include_router(user_router, prefix="/api")
    fastapi_app_instance.include_router(task_router, prefix="/api")

    return fastapi_app_instance


if __name__ == "__main__":
    app.deploy()