# Modal Native Image - Fix Tracking

## Approach
Rebuild the Dockerfile as a native `modal.Image` chain. Each step is a cached layer — only changed layers rebuild.

## Layer Caching Strategy

| Layer | Content | Cache Invalidation |
|-------|---------|-------------------|
| 1 | Base CUDA image | Never (pinned tag) |
| 2 | apt packages | When package list changes |
| 3 | Miniconda install | Never |
| 4 | Conda env | When `environment.yml` changes |
| 5 | External pip packages | When git URLs change |
| 6 | torch/torchvision fix | Never (one-time fix) |
| 7 | Project files (copy=True) | When any local file changes |

**Key insight:** Layers 1-6 are heavy but stable. Layer 7 (project files) changes frequently but is fast. Only layer 7 rebuilds on code changes.

---

## Attempts

### Attempt 1: Initial native image conversion
**Status:** Failed
**Error:** `An image tried to run a build step after using image.add_local_*`
**Fix:** Moved `add_local_dir` to the very last step.

---

### Attempt 2: `add_local_file` without `copy=True`
**Status:** Failed
**Error:** Same as Attempt 1 — `add_local_file("environment.yml")` was followed by `.run_commands()`
**Fix:** Added `copy=True` to `add_local_file("environment.yml", copy=True)` so it's baked into the image layer.

---

### Attempt 3: `pip install` without conda prefix
**Status:** Failed
**Error:** `subprocess.CalledProcessError` — pip used Modal's Python 3.13 instead of conda's Python 3.10
**Fix:** Changed to `conda run -n unav pip install ...` to ensure packages go into the correct conda environment.

---

### Attempt 4: `ModuleNotFoundError: No module named 'config'`
**Status:** Failed
**Error:** `config.py` was deleted by `RUN rm -f /workspace/config.py` in Dockerfile but code imports it
**Fix:** Removed `rm -f /workspace/config.py` from the startup command and added `Mount.from_local_file("config.py", ...)` to the function.

---

### Attempt 5: `ModuleNotFoundError: No module named 'main'`
**Status:** Failed
**Error:** uvicorn couldn't find `main.py` because working directory wasn't `/workspace`
**Fix:** Added `cd /workspace &&` prefix to the uvicorn command.

---

### Attempt 6: `RuntimeError: operator torchvision::nms does not exist`
**Status:** Failed
**Error:** `torch` and `torchvision` version mismatch — conda installed incompatible versions
**Fix 1 (failed):** `pip install --force-reinstall torchvision` — pulled wrong CUDA version
**Fix 2 (failed):** `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126` — pulled torch built for CUDA 13, base image has CUDA 12.6.3
**Fix 3 (failed):** `pip install --force-reinstall torch torchvision` — same CUDA mismatch
**Fix 4 (current):** `conda install -n unav pytorch torchvision -c pytorch -y` — uses conda to ensure both packages are from the same channel and compatible

---

### Attempt 7: `OSError: libcudart.so.13: cannot open shared object file`
**Status:** Failed
**Error:** pip pulled torch built for CUDA 13, but base image has CUDA 12.6.3
**Fix:** Added `LD_LIBRARY_PATH` env var: `/usr/local/cuda/lib64:/opt/conda/envs/unav/lib:$LD_LIBRARY_PATH`

---

### Attempt 8: `ImportError: libnvshmem_host.so.3: cannot open shared object file`
**Status:** Failed
**Error:** `--force-reinstall` pulled newer torch expecting libraries not in base image
**Fix:** Removed `--force-reinstall` step, added `LD_LIBRARY_PATH` env var instead

---

### Attempt 9: Volume mounting
**Status:** Success
**Details:** Added `volumes={"/data": modal.Volume.from_name("unav_multifloor")}` to function decorator. Volume contents available at `/data/archive/`, `/data/data/`, `/data/ddata/`, `/data/fl1/`

---

## Current Working Configuration

```python
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
    .run_commands(
        "conda run -n unav pip install --no-deps git+https://github.com/cvg/implicit_dist.git",
        "conda run -n unav pip install --no-deps --upgrade git+https://github.com/endeleze/UNav.git",
    )
    .run_commands(
        "conda install -n unav pytorch torchvision -c pytorch -y",
    )
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/opt/conda/envs/unav/lib:$LD_LIBRARY_PATH"})
    .add_local_dir(
        ".",
        remote_path="/workspace",
        copy=True,
        ignore=[".venv", "__pycache__", ".git", ".modal-cache", "*.egg-info", "node_modules", "MODAL_FIX_ATTEMPTS.md", "MODAL_NATIVE_FIXES.md"],
    )
)
```

## Key Rules (DO NOT BREAK)
1. **`add_local_*` must be the LAST step** in the image chain, or use `copy=True`
2. **Always use `conda run -n unav` prefix** for pip commands — Modal's Python is 3.13, conda env is 3.10
3. **Never use `pip install --force-reinstall torch`** — it pulls wrong CUDA versions
4. **Use conda for torch/torchvision** — `conda install -n unav pytorch torchvision -c pytorch -y`
5. **`add_local_file` needs `copy=True`** if followed by `.run_commands()`
6. **`cd /workspace &&` prefix** needed for uvicorn to find `main.py`
7. **`LD_LIBRARY_PATH`** must include both `/usr/local/cuda/lib64` and `/opt/conda/envs/unav/lib`

---

### Attempt 10: `FileNotFoundError: /data/parameters/DinoV2Salad/ckpts/dino_salad.ckpt`
**Status:** New Issue - Model files not on volume
**Details:** Server started successfully (torch issue resolved!) but model files are missing from the mounted volume `/data`. Need to upload model checkpoints to the volume.

