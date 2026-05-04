# Deployment Changes Log

## Project: UNav-Server Modal Deployment
## Date: 2026-05-04
## Objective: Fix Modal 1.4.2 compatibility and BLAS library detection issues with 2025.06 image builder

---

## Changes Made

### 1. Fixed Modal Import Error (unav_modal.py)
**File:** `src/modal_functions/unav_v2/unav_modal.py`
**Line:** 1
**Issue:** `gpu` import no longer exists in Modal 1.4.x
**Change:**
```python
# Before:
from modal import method, gpu, enter

# After:
from modal import method, enter
```
**Reason:** The `gpu` module was removed/restructured in Modal 1.4.x. GPU configuration is handled via `get_gpu_config()` in deploy_config.py instead.

---

### 2. Added BLAS/LAPACK Development Libraries (modal_config.py)
**File:** `src/modal_functions/unav_v2/modal_config.py`
**Line:** 157
**Issue:** CMake couldn't find BLAS library required by SuiteSparse/Ceres
**Change:**
```python
# Before:
"apt-get install -y cmake git libgl1-mesa-glx libceres-dev libsuitesparse-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev"

# After:
"apt-get install -y cmake git libgl1-mesa-glx libceres-dev libsuitesparse-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libblas-dev liblapack-dev"
```
**Reason:** The 2025.06 image builder requires explicit BLAS development headers for CMake to find the libraries during pyimplicitdist build.

---

### 3. Added CMAKE Environment Variables (modal_config.py) - ATTEMPT 1
**File:** `src/modal_functions/unav_v2/modal_config.py`
**Lines:** 179-186
**Issue:** CMake's FindBLAS module couldn't locate BLAS libraries in the new image builder
**Change:**
```python
# Before:
.workdir("/implicit_dist")
.run_commands(
    "ls",
    "python3 -m venv .venv",
    ". .venv/bin/activate",
    "pip install . --no-deps",
    "pip freeze",
)

# After:
.workdir("/implicit_dist")
.env({
    "CMAKE_PREFIX_PATH": "/usr/lib/x86_64-linux-gnu",
    "CMAKE_ARGS": "-DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so -DBLA_VENDOR=ATLAS",
})
.run_commands(
    "ls",
    "python3 -m venv .venv",
    ". .venv/bin/activate",
    "pip install . --no-deps",
    "pip freeze",
)
```
**Reason:** The 2025.06 image builder has different library search paths. CMAKE_ARGS environment variable helps CMake locate BLAS/LAPACK libraries explicitly.
**Status:** ❌ FAILED - Virtual environment doesn't pass env vars to pip install subprocess

---

### 4. Added Inline Environment Variables (modal_config.py) - ATTEMPT 2
**File:** `src/modal_functions/unav_v2/modal_config.py`
**Lines:** 179-189
**Issue:** Virtual environment wasn't passing CMAKE environment variables to pip install subprocess
**Change:**
```python
# Before:
.workdir("/implicit_dist")
.env({
    "CMAKE_PREFIX_PATH": "/usr/lib/x86_64-linux-gnu",
    "CMAKE_ARGS": "-DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so -DBLA_VENDOR=ATLAS",
})
.run_commands(
    "ls",
    "python3 -m venv .venv",
    ". .venv/bin/activate",
    "pip install . --no-deps",
    "pip freeze",
)

# After:
.workdir("/implicit_dist")
.run_commands(
    "ls",
    "python3 -m venv .venv",
    ". .venv/bin/activate && \
     CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu \
     BLA_VENDOR=ATLAS \
     BLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so \
     LAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so \
     pip install . --no-deps",
    "pip freeze",
)
```
**Reason:** Inline environment variables ensure they're passed to the cmake subprocess during pip install. The virtual environment activation and pip install are combined in one command to preserve environment variables.
**Status:** ❌ FAILED - Environment variables still not reaching CMake subprocess

---

### 5. Added Export Statements for Environment Variables (modal_config.py) - ATTEMPT 3
**File:** `src/modal_functions/unav_v2/modal_config.py`
**Lines:** 179-194
**Issue:** Environment variables weren't being exported to subprocesses spawned by setup.py
**Change:**
```python
# Before:
.workdir("/implicit_dist")
.run_commands(
    "ls",
    "python3 -m venv .venv",
    ". .venv/bin/activate && \
     CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu \
     BLA_VENDOR=ATLAS \
     BLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so \
     LAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so \
     pip install . --no-deps",
    "pip freeze",
)

# After:
.workdir("/implicit_dist")
.run_commands(
    "ls",
    "python3 -m venv .venv",
    "export CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu && \
     export BLA_VENDOR=ATLAS && \
     export BLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so && \
     export LAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so && \
     . .venv/bin/activate && \
     pip install . --no-deps",
    "pip freeze",
)
```
**Reason:** Using `export` makes environment variables available to all subprocesses, including the cmake subprocess spawned by the setup.py build process. The exports happen before venv activation to ensure they're inherited.
**Status:** ❌ FAILED - Setup.py spawns cmake with explicit env dict, overriding exports

---

### 6. Added PKG_CONFIG Files (modal_config.py) - ATTEMPT 4
**File:** `src/modal_functions/unav_v2/modal_config.py`
**Lines:** 179-200
**Issue:** All environment variable approaches failed because setup.py passes explicit env to subprocess
**Change:**
```python
# Before:
.workdir("/implicit_dist")
.run_commands(
    "ls",
    "python3 -m venv .venv",
    "export CMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu && \
     export BLA_VENDOR=ATLAS && \
     export BLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so && \
     export LAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so && \
     . .venv/bin/activate && \
     pip install . --no-deps",
    "pip freeze",
)

# After:
.workdir("/implicit_dist")
.run_commands(
    "ls",
    "mkdir -p /usr/lib/x86_64-linux-gnu/pkgconfig",
    "printf '%s\\n' 'Name: blas' 'Description: Basic Linear Algebra Subprograms' 'Version: 3.11.0' 'Libs: -L/usr/lib/x86_64-linux-gnu -lblas' 'Cflags: -I/usr/include' > /usr/lib/x86_64-linux-gnu/pkgconfig/blas.pc",
    "printf '%s\\n' 'Name: lapack' 'Description: Linear Algebra PACKage' 'Version: 3.11.0' 'Libs: -L/usr/lib/x86_64-linux-gnu -llapack' 'Requires: blas' > /usr/lib/x86_64-linux-gnu/pkgconfig/lapack.pc",
    "export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig && \
     . .venv/bin/activate && \
     pip install . --no-deps",
    "pip freeze",
)
```
**Reason:** CMake's FindBLAS module can use pkg-config as a search method. By creating proper .pc files and setting PKG_CONFIG_PATH, CMake should be able to find BLAS/LAPACK through the standard pkg-config mechanism, bypassing the environment variable issues.
**Status:** ❌ FAILED - Shell syntax error with echo command

---

### 7. Fixed PKG_CONFIG file creation using printf (modal_config.py) - ATTEMPT 5
**File:** `src/modal_functions/unav_v2/modal_config.py`
**Lines:** 179-194
**Issue:** Echo command with newlines caused shell syntax errors; venv not created in same step
**Change:**
```python
# Before:
"echo 'Name: blas\nDescription: Basic Linear Algebra Subprograms\nVersion: 3.11.0\nLibs: -L/usr/lib/x86_64-linux-gnu -lblas\nCflags: -I/usr/include' > /usr/lib/x86_64-linux-gnu/pkgconfig/blas.pc",
"echo 'Name: lapack\nDescription: Linear Algebra PACKage\nVersion: 3.11.0\nLibs: -L/usr/lib/x86_64-linux-gnu -llapack\nRequires: blas' > /usr/lib/x86_64-linux-gnu/pkgconfig/lapack.pc",
"export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig && \
 . .venv/bin/activate && \
 pip install . --no-deps",

# After:
"mkdir -p /usr/lib/x86_64-linux-gnu/pkgconfig",
"printf '%s\\n' 'Name: blas' 'Description: Basic Linear Algebra Subprograms' 'Version: 3.11.0' 'Libs: -L/usr/lib/x86_64-linux-gnu -lblas' 'Cflags: -I/usr/include' > /usr/lib/x86_64-linux-gnu/pkgconfig/blas.pc",
"printf '%s\\n' 'Name: lapack' 'Description: Linear Algebra PACKage' 'Version: 3.11.0' 'Libs: -L/usr/lib/x86_64-linux-gnu -llapack' 'Requires: blas' > /usr/lib/x86_64-linux-gnu/pkgconfig/lapack.pc",
"python3 -m venv .venv",
"export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig && \
 . .venv/bin/activate && \
 pip install . --no-deps",
```
**Reason:** printf is more reliable than echo for creating files with multiple lines in shell commands. Also, venv needs to be created in the same run_commands block since each block runs in isolation.
**Status:** ❌ FAILED - Venv not found

---

### 8. Fixed venv creation order (modal_config.py) - ATTEMPT 6
**File:** `src/modal_functions/unav_v2/modal_config.py`
**Lines:** 179-194
**Issue:** Venv creation was in a separate run_commands block, causing it to not exist when activating
**Change:**
```python
# Before:
.workdir("/implicit_dist")
.run_commands(
    "ls",
    "mkdir -p /usr/lib/x86_64-linux-gnu/pkgconfig",
    "printf ... > blas.pc",
    "printf ... > lapack.pc",
    "export PKG_CONFIG_PATH=... && . .venv/bin/activate && pip install . --no-deps",
)

# After:
.workdir("/implicit_dist")
.run_commands(
    "ls",
    "mkdir -p /usr/lib/x86_64-linux-gnu/pkgconfig",
    "printf ... > blas.pc",
    "printf ... > lapack.pc",
    "python3 -m venv .venv",  # Added venv creation
    "export PKG_CONFIG_PATH=... && . .venv/bin/activate && pip install . --no-deps",
)
```
**Reason:** Each run_commands block runs in isolation, so the venv needs to be created in the same block where it's activated.
**Status:** ❌ FAILED - PKG_CONFIG_PATH still not reaching CMake

---

### 9. Removed Virtual Environment, Set Image-Level Environment Variables (modal_config.py) - ATTEMPT 7
**File:** `src/modal_functions/unav_v2/modal_config.py`
**Lines:** 179-194
**Issue:** Setup.py spawns cmake with explicit env dict that overrides all environment variables
**Change:**
```python
# Before:
.workdir("/implicit_dist")
.run_commands(
    "ls",
    "mkdir -p /usr/lib/x86_64-linux-gnu/pkgconfig",
    "printf ... > blas.pc",
    "printf ... > lapack.pc",
    "python3 -m venv .venv",
    "export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig && . .venv/bin/activate && pip install . --no-deps",
)

# After:
.workdir("/implicit_dist")
.env({
    "PKG_CONFIG_PATH": "/usr/lib/x86_64-linux-gnu/pkgconfig",
    "CMAKE_PREFIX_PATH": "/usr/lib/x86_64-linux-gnu",
    "CMAKE_ARGS": "-DBLA_PREFER_PKGCONFIG=ON",
})
.run_commands(
    "ls",
    "mkdir -p /usr/lib/x86_64-linux-gnu/pkgconfig",
    "printf ... > blas.pc",
    "printf ... > lapack.pc",
    "pip install . --no-deps",
    "pip freeze",
)
```
**Reason:** Setup.py explicitly passes an `env` dict to subprocess.check_call(), which overrides the entire environment. By using Image.env() instead of shell exports, the variables are set at the container level and should be inherited by all subprocesses. Also removed virtual environment to simplify the build process.
**Status:** 🔄 TESTING

---

## Deployment Command

```bash
modal deploy --force -m src.modal_functions.unav_v2.unav_modal
```

**Flags:**
- `--force`: Force rebuild of all image layers (bypass cache due to 2025.06 image builder changes)
- `-m src.modal_functions.unav_v2.unav_modal`: Deploy the unav_v2 module

---

## Root Cause Analysis

### Original Error
```
ImportError: cannot import name 'gpu' from 'modal'
```
**Cause:** Modal 1.4.x removed the `gpu` import that existed in 1.1.4

### Secondary Error
```
Failed to find SuiteSparse - Did not find BLAS library (required for SuiteSparse)
```
**Cause:** The 2025.06 image builder changed how system libraries are discovered:
1. Different base image layering
2. PYTHONPATH handling changes
3. CMake search paths not including standard system directories

---

## Testing Strategy

1. Deploy with `--force` flag to ensure fresh image build
2. Monitor the implicit_dist build step closely
3. Verify successful deployment of UnavServer class
4. Test basic functionality via Modal dashboard

---

## Future Improvements

### Base Image Optimization (Recommended)
To reduce rebuild times, consider splitting heavy dependencies into a base image:

```python
# base_image.py - Build once, cache forever
heavy_base = (
    Image.debian_slim(python_version="3.10")
    .apt_install("cmake", "libceres-dev", "libsuitesparse-dev", ...)
    .pip_install("torch>=2.4.0", "faiss-gpu-cu12")
    .run_commands("git clone ...")  # Eigen, MAST3R
)

# unav_modal.py - Use base image
from modal import Image
unav_image = (
    Image.from_registry("modal.com/workspace/unav-base:latest")
    .pip_install("unav-core", "middleware-io")
    .add_local_python_source("src")
)
```

**Benefits:**
- 15-30 min → 30 sec rebuild times
- 4GB+ → 50MB per deploy
- Faster iteration cycles

---

## References

- Modal 2025.06 Image Builder Migration Guide
- CMake FindBLAS Documentation: https://cmake.org/cmake/help/latest/module/FindBLAS.html
- Modal Image Caching: https://modal.com/docs/guide/images

---

## Status

### ✅ Resolved
- [x] Fixed Modal import error (removed `gpu`)
- [x] OpenBLAS replaces ATLAS (solves gfortran static lib dep issue)
- [x] System Eigen instead of building from source (version mismatch fixed)
- [x] Deprecated Ceres API warnings suppressed (removed `-Werror` from CMakeLists.txt)
- [x] LTO disabled (linker error with static libraries)
- [ ] `-fPIC` issue with SuiteSparse static libs unresolved

### Decision: Switch to 2024.10 Image Builder
After 9+ attempts with the 2025.06 image builder, the core issue is:
- **2025.06 builder** causes SuiteSparse static libs to be linked into shared Python modules, but they lack `-fPIC`
- **2025.06 builder** has different CMake search paths and library discovery behavior
- **2024.10 builder** works correctly with these libraries

**Deploy command:**
```bash
MODAL_IMAGE_BUILDER_VERSION=2024.10 modal deploy -m src.modal_functions.unav_v2.unav_modal
```

### All Changes Made (Final State)
- `unav_modal.py`: Removed `gpu` from import
- `modal_config.py`: 
  - Replaced ATLAS + libblas-dev with `libopenblas-dev`
  - Removed custom Eigen build (uses system package `libeigen3-dev`)
  - Removed virtual env creation
  - Setup.py patched: `env=env` → `env=os.environ`
  - CMakeLists.txt patched: removed `-Werror` and `-flto`

**Last Updated:** 2026-05-04
