import modal
from modal.mount import Mount

app = modal.App("unav-server")

image = (
    # Layer 1: Base CUDA image (cached forever)
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    # Layer 2: System packages (cached unless list changes)
    .apt_install("wget", "git", "vim", "cmake", "libeigen3-dev", "libceres-dev")
    # Layer 3: Miniconda installation (cached forever)
    .run_commands(
        "wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh",
        "bash /tmp/miniconda.sh -b -p /opt/conda",
        "rm /tmp/miniconda.sh",
    )
    .env({"PATH": "/opt/conda/bin:$PATH"})
    # Layer 4: Conda environment (cached unless environment.yml changes)
    .add_local_file("environment.yml", "/tmp/environment.yml", copy=True)
    .run_commands(
        "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main",
        "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r",
        "conda env create -f /tmp/environment.yml && conda clean -afy",
    )
    # Layer 5: External pip packages (cached unless URLs change)
    # Must use conda run to ensure packages go into the 'unav' env, not Modal's Python
    .run_commands(
        "conda run -n unav pip install --no-deps git+https://github.com/cvg/implicit_dist.git",
        "conda run -n unav pip install --no-deps --upgrade git+https://github.com/endeleze/UNav.git",
    )
    # Layer 6: Project files (baked into image)
    .add_local_dir(
        ".",
        remote_path="/workspace",
        copy=True,
        ignore=[".venv", "__pycache__", ".git", ".modal-cache", "*.egg-info", "node_modules", "MODAL_FIX_ATTEMPTS.md", "MODAL_NATIVE_FIXES.md"],
    )
)


@app.function(image=image, gpu="A10")
@modal.web_server(port=5001)
def web():
    import subprocess

    subprocess.Popen(
        [
            "bash",
            "-c",
            "source /opt/conda/etc/profile.d/conda.sh && conda activate unav && uvicorn main:app --host 0.0.0.0 --port 5001 --log-level info",
        ]
    )
