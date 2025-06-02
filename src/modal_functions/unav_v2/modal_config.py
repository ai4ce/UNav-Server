from modal import App, Image, Mount, NetworkFileSystem, Volume
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
        # Mount.from_local_file(
        #     "modal_functions/config.yaml", remote_path="/root/config.yaml"
        # ),
    ],
)

unav_image = (
    Image.debian_slim(python_version="3.9")
    .run_commands(
        "apt-get update",
        "apt-get install -y cmake git libgl1-mesa-glx libceres-dev libsuitesparse-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev",
    )
    .run_commands("git clone https://github.com/ai4ce/UNav-Server.git")
    .run_commands(
        "git checkout endeleze",
        "pip install -r modal_requirements.txt",
        "pip install unav_pretrained",
    )
)
