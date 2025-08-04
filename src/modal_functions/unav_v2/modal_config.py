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
    import os
    import zipfile
    from pathlib import Path

    # Try to import requests, fallback to urllib if not available
    try:
        import requests

        HAS_REQUESTS = True
    except ImportError:
        import urllib.request

        HAS_REQUESTS = False

    # Set up torch hub cache directories
    torch_home = os.environ.get("TORCH_HOME", "/root/.cache/torch")
    hub_dir = os.path.join(torch_home, "hub")
    checkpoints_dir = os.path.join(hub_dir, "checkpoints")

    # Ensure directories exist
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(hub_dir, exist_ok=True)

    # Always download and save with CPU mapping to avoid CUDA device issues
    device = torch.device("cpu")

    print("📦 Predownloading VGG16 weights...")
    model_weights = torch.hub.load_state_dict_from_url(
        MODEL_URL, progress=True, map_location=device
    )
    torch.save(model_weights, os.path.join(checkpoints_dir, "vgg16_weights.pth"))

    print("📦 Predownloading LightGlue SuperPoint weights...")
    # Download LightGlue weights to the exact path torch expects
    lightglue_weights = torch.hub.load_state_dict_from_url(
        LIGHTGLUE_URL, progress=True, map_location=device
    )
    # Save with the exact filename that torch hub expects
    lightglue_path = os.path.join(
        checkpoints_dir, "superpoint_lightglue_v0-1_arxiv-pth"
    )
    torch.save(lightglue_weights, lightglue_path)
    print(f"✅ LightGlue weights saved to: {lightglue_path}")

    print("📦 Predownloading DinoV2 repository and weights...")
    # Download the DinoV2 repository structure that torch hub expects
    dinov2_repo_url = "https://github.com/facebookresearch/dinov2/zipball/main"
    dinov2_zip_path = os.path.join(hub_dir, "main.zip")

    # Download the zip file
    if HAS_REQUESTS:
        response = requests.get(dinov2_repo_url, stream=True)
        with open(dinov2_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        urllib.request.urlretrieve(dinov2_repo_url, dinov2_zip_path)

    # Extract to the expected location
    dinov2_extract_dir = os.path.join(hub_dir, "facebookresearch_dinov2_main")

    # Remove existing directory if it exists
    if os.path.exists(dinov2_extract_dir):
        import shutil

        shutil.rmtree(dinov2_extract_dir)

    with zipfile.ZipFile(dinov2_zip_path, "r") as zip_ref:
        # Extract all files to temporary location
        temp_extract_dir = os.path.join(hub_dir, "temp_dinov2_extract")
        if os.path.exists(temp_extract_dir):
            import shutil

            shutil.rmtree(temp_extract_dir)

        zip_ref.extractall(temp_extract_dir)

        # Find the extracted directory (it will have a name like facebookresearch-dinov2-abc123)
        extracted_dirs = [
            d
            for d in os.listdir(temp_extract_dir)
            if "dinov2" in d and os.path.isdir(os.path.join(temp_extract_dir, d))
        ]
        if extracted_dirs:
            extracted_dir = os.path.join(temp_extract_dir, extracted_dirs[0])

            # Move the contents to the expected location, not the directory itself
            import shutil

            shutil.move(extracted_dir, dinov2_extract_dir)

            # Clean up temp directory
            shutil.rmtree(temp_extract_dir)

            # Verify hubconf.py exists
            hubconf_path = os.path.join(dinov2_extract_dir, "hubconf.py")
            if os.path.exists(hubconf_path):
                print(f"✅ DinoV2 repository extracted successfully with hubconf.py")
            else:
                print(f"⚠️ Warning: hubconf.py not found in {dinov2_extract_dir}")
                print(f"Directory contents: {os.listdir(dinov2_extract_dir)}")
        else:
            print("❌ Error: Could not find dinov2 directory in extracted files")
            print(f"Available directories: {os.listdir(temp_extract_dir)}")

    # Clean up zip file
    os.remove(dinov2_zip_path)
    print(f"✅ DinoV2 repository extracted to: {dinov2_extract_dir}")

    # Also download the DinoV2 pretrained weights
    print("📦 Predownloading DinoV2 pretrained weights...")
    dinosalad_weights = torch.hub.load_state_dict_from_url(
        DINOSALAD_URL, progress=True, map_location=device
    )
    dinov2_weights_path = os.path.join(checkpoints_dir, "dinov2_vitb14_pretrain.pth")
    torch.save(dinosalad_weights, dinov2_weights_path)
    print(f"✅ DinoV2 weights saved to: {dinov2_weights_path}")

    print("🎉 All torch hub models predownloaded successfully!")


app = App(
    name="unav-server-v21",
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
    .pip_install("requests")
    .run_function(download_torch_hub_weights)
    .run_commands("pip install -r /modal_requirements.txt")
    .pip_install("PyYAML")
)
