# MASt3R Integration for Crash Looping Fix

## Approach Tracking

### Modal Server Approaches (tested in modal_config.py)
- [ ] **Option A**: `pip install -e .[mast3r]` before ai4ce/unav install ❌ Failed (no PyPI package)
- [x] **Option B**: Clone mast3r from GitHub and install dependencies (in progress)
- [ ] **Option C**: pip install from mast3r GitHub repo directly

### Current Status
- Modal: Not yet tested
- Local: See local setup below

---

## Problem
Crash looping issue that requires switching the local feature model to MASt3R.

## Solution

### Option 1: Full Setup (with CUDA RoPE acceleration)

```bash
# 1. Clone MASt3R
cd ~/Desktop
git clone --recursive https://github.com/naver/mast3r

# 2. Install dependencies in unav conda environment
conda activate unav
pip install -r ~/Desktop/mast3r/requirements.txt
pip install -r ~/Desktop/mast3r/dust3r/requirements.txt
pip install poselib

# 3. Compile CUDA RoPE (optional, for acceleration)
cd ~/Desktop/mast3r/dust3r/croco/models/curope
python setup.py build_ext --inplace

# 4. Update UNav config
# In server's config.py:
LOCAL_FEATURE_MODEL = "mast3r"
```

### Option 2: Simplified (Recommended)

```bash
# Pull latest code
git pull

# Install with MASt3R support
pip install -e .[mast3r]

# Update config
LOCAL_FEATURE_MODEL = "mast3r"
```

## Key Changes
- Switch `LOCAL_FEATURE_MODEL` from current value to `"mast3r"` in the server config file.
- The simpler Option 2 should be tried first as it requires fewer steps.
