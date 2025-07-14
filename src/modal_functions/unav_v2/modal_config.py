from modal import App, Image, Mount, NetworkFileSystem, Volume, Secret
from pathlib import Path

volume = Volume.from_name("unav_multifloor")

MODEL_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
LIGHTGLUE_URL = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth"
DINOSALAD_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
)
# Get the current file's directory
current_dir = Path(__file__).resolve().parent

# Construct the path to the src directory
local_dir = current_dir / ".."


def download_torch_hub_weights():
    import torch

    model_weights = torch.hub.load_state_dict_from_url(MODEL_URL, progress=True)
    torch.save(model_weights, "vgg16_weights.pth")

    lightglue_weights = torch.hub.load_state_dict_from_url(LIGHTGLUE_URL, progress=True)
    torch.save(lightglue_weights, "superpoint_lightglue_v0-1_arxiv-pth")

    dinosalad_weights = torch.hub.load_state_dict_from_url(DINOSALAD_URL, progress=True)
    torch.save(dinosalad_weights, "dinov2_vitb14_weights.pth")


app = App(
    name="unav-server-v2",
    mounts=[
        # Mount.from_local_dir(local_dir.resolve(), remote_path="/root/app"),
        Mount.from_local_file(
            "modal_requirements.txt", remote_path="/root/modal_requirements.txt"
        )
    ],
)


github_secret = Secret.from_name("github-read-private")

unav_image = (
    Image.debian_slim(python_version="3.10")
    .run_commands(
        "apt-get update",
        "apt-get install -y cmake git libgl1-mesa-glx libceres-dev libsuitesparse-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev",
    )
    .run_commands("git clone https://gitlab.com/libeigen/eigen.git eigen")
    .workdir("/eigen")
    .run_commands(
        "git checkout 3.4",
        "mkdir build",
    )
    .workdir("/eigen/build")
    .run_commands(
        "cmake ..",
        "make",
        "make install",
    )
    .workdir("/")
    .run_commands(
        "git clone https://github.com/cvg/implicit_dist.git implicit_dist",
    )
    .run_commands(
        "pip install 'numpy<2.0.0'",
    )
    .workdir("/implicit_dist")
    .run_commands(
        "ls",
        "python3 -m venv .venv",
        ". .venv/bin/activate",
        "pip install . --no-deps",
        "pip freeze",
    )
    .pip_install_private_repos(
        "github.com/ai4ce/unav",
        git_user="surendharpalanisamy",
        secrets=[github_secret],
        extra_options="--no-deps"
    )
    .workdir("/root")
    .run_commands("git clone https://github.com/ai4ce/UNav-Server.git unav_server_v2")
    .workdir("/root/unav_server_v2")
    .run_commands(
        "pwd",  # Debug: show current directory
        "ls -la",  # Debug: show directory contents
        "git branch -a",  # Debug: show available branches
        "git checkout endeleze",
    )

    .run_commands("pip freeze")
    .pip_install("torch==2.1.2")
    .run_function(download_torch_hub_weights)
    .pip_install("psutil")
    .run_commands("pip install -r /modal_requirements.txt",force_build=True)
)
