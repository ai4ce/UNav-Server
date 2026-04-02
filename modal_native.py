import modal

app = modal.App("unav-server")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("wget", "git", "vim", "cmake", "libeigen3-dev", "libceres-dev")
    .run_commands(
        "wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh",
        "bash /tmp/miniconda.sh -b -p /opt/conda",
        "rm /tmp/miniconda.sh",
    )
    .env({"PATH": "/opt/conda/bin:$PATH"})
    .add_local_file("environment.yml", "/tmp/environment.yml", copy=True)
    .run_commands(
        "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main",
        "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r",
        "conda env create -f /tmp/environment.yml && conda clean -afy",
    )
    # SHELL equivalent - set shell to always use conda env
    .env({"SHELL": "conda run -n unav /bin/bash -c"})
    # Set working directory
    .run_commands("mkdir -p /workspace")
    # Copy project files (matches COPY . /workspace)
    .add_local_dir(
        ".",
        remote_path="/workspace",
        copy=True,
        ignore=[".venv", "__pycache__", ".git", ".modal-cache", "*.egg-info", "node_modules", "MODAL_FIX_ATTEMPTS.md", "MODAL_NATIVE_FIXES.md"],
    )
    # Install external packages - explicitly use Python from unav env
    .run_commands("/opt/conda/envs/unav/bin/pip install --no-deps git+https://github.com/cvg/implicit_dist.git")
    .run_commands("/opt/conda/envs/unav/bin/pip install --no-deps --upgrade git+https://github.com/endeleze/UNav.git")
    # Verify unav is installed
    .run_commands("/opt/conda/envs/unav/bin/pip list | grep unav")
    # Fix torch/torchvision - reinstall with CUDA support
    .run_commands("/opt/conda/bin/conda install -n unav --force-reinstall pytorch torchvision -c pytorch -c nvidia -y")
    # Keep config.py - it's needed by the code (not removing like Dockerfile)
    .run_commands("ls -la /workspace/config.py || echo 'config.py not found'")
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/opt/conda/envs/unav/lib:$LD_LIBRARY_PATH"})
    .env({"PYTHONPATH": "/opt/conda/envs/unav/lib/python3.10/site-packages:/workspace:$PYTHONPATH"})
)


@app.function(
    image=image,
    gpu="A10",
    volumes={
        "/data": modal.Volume.from_name("unav_multifloor"),
    },
)
@modal.web_server(port=5001, startup_timeout=120)
def web():
    import subprocess
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/conda/envs/unav/lib/python3.10/site-packages:/workspace"

    subprocess.Popen(
        [
            "bash",
            "-c",
            "cd /workspace && source /opt/conda/etc/profile.d/conda.sh && conda activate unav && uvicorn main:app --host 0.0.0.0 --port 5001 --log-level info 2>&1 | tee /tmp/uvicorn.log",
        ],
        env=env,
    )
