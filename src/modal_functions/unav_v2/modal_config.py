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
        Mount.from_local_dir(local_dir.resolve(), remote_path="/root"),
    ],
)


github_secret = Secret.from_name("github-read-private")

unav_image = (
    Image.debian_slim(python_version="3.9")
    .run_commands(
        "apt-get update",
        "apt-get install -y cmake git libgl1-mesa-glx libceres-dev libsuitesparse-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev",
    )
    .run_commands(
        "echo 'DEBUG: Current directory after package installation:' && pwd && ls -la"
    )
    .run_commands("git clone https://github.com/ai4ce/UNav-Server.git unav_server_v2")
    .run_commands(
        "echo 'DEBUG: Current directory after git clone:' && pwd && ls -la && echo 'DEBUG: Contents of unav_server_v2:' && ls -la unav_server_v2/"
    )
    .workdir("/unav_server_v2")
    .run_commands(
        "echo 'DEBUG: Current directory after workdir change:' && pwd && ls -la"
    )
    .run_commands("git checkout endeleze")
    .run_commands(
        "echo 'DEBUG: Current directory after git checkout:' && pwd && ls -la && echo 'DEBUG: Looking for requirements files:' && find . -name '*requirements*.txt' && echo 'DEBUG: Checking root directory:' && ls -la /"
    )
    .run_commands("pip install -r /modal_requirements.txt")
    .run_commands(
        "echo 'DEBUG: Current directory after requirements install:' && pwd && ls -la"
    )
    # .run_commands("pip install unav_pretrained")
    .run_commands("echo 'DEBUG: Final directory state:' && pwd && ls -la")
    .pip_install_private_repos(
        "github.com/ai4ce/UNav_Navigation",
        git_user="surendharpalanisamy",
        secrets=[github_secret],
    )
    # .run_commands("pip install unav_pretrained")
)
