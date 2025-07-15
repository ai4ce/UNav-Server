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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_weights = torch.hub.load_state_dict_from_url(MODEL_URL, progress=True, map_location=device)
    torch.save(model_weights, "vgg16_weights.pth")

    lightglue_weights = torch.hub.load_state_dict_from_url(LIGHTGLUE_URL, progress=True, map_location=device)
    torch.save(lightglue_weights, "superpoint_lightglue_v0-1_arxiv-pth")

    dinosalad_weights = torch.hub.load_state_dict_from_url(DINOSALAD_URL, progress=True, map_location=device)
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
        extra_options="--no-deps",
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
    .pip_install(
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "dataloaders>=0.0.1",
        "einops>=0.8.1",
        "faiss-gpu>=1.7.2",
        "fast-pytorch-kmeans>=0.2.0.1",
        "h5py>=3.7.0",
        "joblib>=1.1.1",
        "kornia>=0.6.12",
        "lib>=4.0.0",
        "matplotlib>=3.7.1",
        "open3d>=0.19.0",
        "Pillow>=10.0.0",
        "poselib>=2.0.0",
        "POT>=0.9.0",
        "prettytable<=3.11.0",
        "pytorch-lightning>=2.0.6",
        "pytorch-metric-learning>=2.3.0",
        "PyYAML>=6.0",
        "scikit-image>=0.19.2",
        "scikit-learn>=1.2.1",
        "scipy>=1.10.0",
        "Shapely>=2.0.7",
        "timm>=0.4.12",
        "tqdm>=4.65.0",
        "transformers>=4.45.0",
        "tyro>=0.9.22",
        "wandb>=0.19.11",
        "xformers>=0.0.28",
        "psutil",
        "opencv-python==4.10.0.84",
    )
    .run_function(download_torch_hub_weights)
    .run_commands("pip install -r /modal_requirements.txt")
    .pip_install("PyYAML")
)
