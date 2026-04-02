# Modal Deployment - Fix Attempts Log

## Problem
Modal cannot detect Python version in the Dockerfile-based image because Python is installed via Conda at `/opt/conda/envs/unav/bin/python3` (non-standard path).

**Error (recurring):**
```
We were unable to determine the version of Python installed in the Image provided for the `web` Function. Please contact Modal support for more assistance.
```

---

## Attempt 1: Basic `from_dockerfile()`
```python
image = modal.Image.from_dockerfile("Dockerfile")
```
**Result:** Failed — Python detection error.

---

## Attempt 2: Added `python_version="3.10"`
```python
image = modal.Image.from_dockerfile("Dockerfile", python_version="3.10")
```
**Result:** Failed — `TypeError: from_dockerfile() got an unexpected keyword argument 'python_version'`

---

## Attempt 3: Changed to `add_python="3.10"`
```python
image = modal.Image.from_dockerfile("Dockerfile", add_python="3.10")
```
**Result:** Failed — same Python detection error. `add_python` adds a Modal-managed Python layer but Modal's scanner still can't detect it after the Dockerfile steps.

---

## Attempt 4: Added `.entrypoint([])`
```python
image = (
    modal.Image.from_dockerfile("Dockerfile", add_python="3.10")
    .entrypoint([])
)
```
**Result:** Failed — `.entrypoint([])` creates a new image layer AFTER `add_python`, which breaks Modal's Python detection.

---

## Attempt 5: Sandbox approach (basic)
```python
sb = modal.Sandbox.create("sleep", "infinity", image=app_image, timeout=3600)
```
**Result:** Failed — `FileNotFoundError: Dockerfile` — Modal's remote build environment doesn't have the project files.

---

## Attempt 6: Added `context_mount` with `modal.Mount`
```python
context_mount = modal.Mount.from_local_dir(".", remote_path="/root", ignore=[...])
```
**Result:** Failed — `AttributeError: module 'modal' has no attribute 'Mount'`

---

## Attempt 7: Changed import to `from modal.mount import Mount`
```python
from modal.mount import Mount
context_mount = Mount.from_local_dir(".", remote_path="/root", ignore=[...])
```
**Result:** Failed — `Mount.from_local_dir` doesn't exist in this Modal version.

---

## Attempt 8: Used `Mount().add_local_dir()`
```python
context_mount = Mount().add_local_dir(".", remote_path="/root", condition=ignore_fn)
```
**Result:** Failed — `TypeError: Mount() takes no arguments`

---

## Attempt 9: Used `Mount._from_local_dir()`
```python
context_mount = Mount._from_local_dir(".", remote_path="/root", condition=ignore_fn)
```
**Result:** Failed — wrong API, still errors.

---

## Attempt 10: Used `context_dir` and `ignore` params on `from_dockerfile()`
```python
image = modal.Image.from_dockerfile(
    "Dockerfile",
    context_dir=".",
    ignore=[".venv", "__pycache__", ".git", ".modal-cache"],
)
```
**Result:** Dockerfile build succeeded (488s). But function still failed with Python detection error.

---

## Attempt 11: Added `image=image` to `@app.function()` decorator
```python
@app.function(image=image, timeout=3600)
```
**Result:** Failed — same Python detection error. The image builds fine but Modal's function wrapper can't find Python.

---

## Attempt 12: Symlink to `/usr/local/bin/python`
```python
.run_commands("ln -s /opt/conda/envs/unav/bin/python3 /usr/local/bin/python")
```
**Result:** Failed — Modal checks `/usr/bin/python`, not `/usr/local/bin/python`.

---

## Attempt 13: Dual symlinks
```python
.run_commands(
    "ln -sf /opt/conda/envs/unav/bin/python3 /usr/bin/python",
    "ln -sf /opt/conda/envs/unav/bin/python3 /usr/local/bin/python",
)
```
**Result:** Failed — same Python detection error. Symlinks are created but Modal's scanner still can't detect the version.

---

## Attempt 14: Separated func_image from app_image
```python
func_image = modal.Image.debian_slim(python_version="3.11")
app_image = modal.Image.from_dockerfile("Dockerfile", context_dir=".", add_python="3.10")

@app.function(image=func_image, timeout=3600)
def run_test():
    sb = modal.Sandbox.create("sleep", "infinity", image=app_image, ...)
```
**Result:** Failed — `FileNotFoundError: Dockerfile` — the function's minimal image doesn't have the project files, so the sandbox can't build the Dockerfile image.

---

## Attempt 15: `add_python="3.10"` + `.entrypoint([])` (no symlink)
```python
image = (
    modal.Image.from_dockerfile("Dockerfile", context_dir=".", add_python="3.10")
    .entrypoint([])
)
```
**Result:** Failed — same Python detection error.

---

## Attempt 16: Removed `.entrypoint([])` (keep only `add_python`)
```python
image = modal.Image.from_dockerfile("Dockerfile", context_dir=".", add_python="3.10")
```
**Result:** Failed — same Python detection error.

---

## Attempt 17: All three combined (`add_python` + symlink + `.entrypoint([])`)
```python
image = (
    modal.Image.from_dockerfile("Dockerfile", context_dir=".", add_python="3.10")
    .run_commands("ln -sf /opt/conda/envs/unav/bin/python3 /usr/bin/python")
    .entrypoint([])
)
```
**Result:** Failed — same Python detection error.

---

## Attempt 18: Removed `add_python`, kept symlink + `.entrypoint([])` (matching Slack thread fix)
```python
image = (
    modal.Image.from_dockerfile("Dockerfile", context_dir=".")
    .run_commands("ln -sf /opt/conda/envs/unav/bin/python3 /usr/bin/python")
    .entrypoint([])
)
```
**Result:** Pending test. Note: The Slack thread fix was for `from_registry()` (pre-built image), not `from_dockerfile()`.

---

## Attempt 19: GitHub Actions build + `from_registry()` (recommended path)
Build image in CI, push to Docker Hub, then:
```python
image = modal.Image.from_registry("yourusername/unav-server:latest")
```
**Status:** Workflow file created (`.github/workflows/docker-build.yml`). Not yet tested.

---

## Root Cause Analysis
Modal's Python version scanner looks for Python in standard system paths (`/usr/bin/python`, `/usr/local/bin/python`, etc.) and expects a standard system Python installation. Conda environments install Python in isolated paths (`/opt/conda/envs/unav/bin/python3`) that Modal's scanner does not traverse.

The Slack thread fix worked for `from_registry()` because pre-built Docker Hub images typically have Python in standard paths. For `from_dockerfile()` with Conda, the issue persists because:
1. `add_python` adds a Modal-managed Python layer, but subsequent Dockerfile steps (Conda install) may override or shadow it.
2. Symlinks don't fool Modal's version detection — it likely runs `python --version` or inspects the binary metadata, not just checks for file existence.

## Recommended Solution
**Attempt 19 (GitHub Actions + `from_registry()`)** is the most reliable path:
- Build the image externally where Conda works normally
- Modal pulls the pre-built image and can detect Python from the standard paths set up by the Dockerfile's `ENV PATH=$CONDA_DIR/bin:$PATH`
