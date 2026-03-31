# Deploying Docker Images on Modal – Complete Guide

Modal supports **three approaches** for using custom Docker images, each with different trade-offs.

---

## Approach 1: `Image.from_dockerfile()` — Build from Dockerfile

Modal builds your Dockerfile in the cloud. No registry needed.

```python
import modal

image = modal.Image.from_dockerfile("Dockerfile", context_dir=".")

app = modal.App("my-app")

@app.function(image=image, gpu="T4")
@modal.web_server(5001)
def serve():
    import subprocess
    subprocess.Popen("uvicorn main:app --host 0.0.0.0 --port 5001", shell=True)
```

**Pros:**
- No registry or push step needed
- Source code and Dockerfile live together
- Modal handles the build

**Cons:**
- Full rebuild on every deploy (Modal's cloud builder doesn't share your local cache)
- Slow for large images with many layers
- Dockerfile must be compatible with Modal's builder (most are; `ONBUILD`, `STOPSIGNAL`, `VOLUME` not supported)

**Best for:** Development, iteration, teams without a container registry.

---

## Approach 2: `Image.from_registry()` — Push to a registry, then reference

Build locally, push to any registry (Docker Hub, GHCR, ECR, GCR), then reference it.

### Step 1: Build and push

```bash
# Build
docker build -t your-username/unav-server:latest .

# Push to Docker Hub
docker push your-username/unav-server:latest

# Or push to GitHub Container Registry
docker tag unav-server ghcr.io/your-org/unav-server:latest
docker push ghcr.io/your-org/unav-server:latest
```

### Step 2: Reference in Modal

```python
import modal

# Public registry
image = modal.Image.from_registry("your-username/unav-server:latest")

# Private Docker Hub (requires a Modal Secret with DOCKER_USERNAME + DOCKER_PASSWORD)
image = modal.Image.from_registry(
    "your-username/unav-server:latest",
    secret=modal.Secret.from_name("dockerhub-creds"),
)

# AWS ECR (requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION in a Secret)
image = modal.Image.from_registry(
    "123456789.dkr.ecr.us-east-1.amazonaws.com/unav-server:latest",
    secret=modal.Secret.from_name("aws-ecr-creds"),
)
```

**Pros:**
- Local Docker cache makes rebuilds fast
- Image is versioned and reusable
- Can use eStargz compression for faster pulls

**Cons:**
- Requires a registry account and push step
- Private registries need Modal Secrets configured

**Best for:** Production, CI/CD pipelines, teams with existing registry infrastructure.

### Speed tip: eStargz compression

For faster cold starts, build with eStargz so Modal only pulls layers it needs:

```bash
docker buildx build \
  --push \
  --cache-to type=registry,ref=your-username/unav-server:cache \
  -t your-username/unav-server:latest \
  --output type=registry,compression=estargz,oci-mediatypes=true,force-compression=true \
  .
```

Supported registries: Docker Hub, AWS ECR, Google Artifact Registry.

---

## Approach 3: Build Modal image programmatically

Define the image entirely in Python using Modal's image builder methods.

```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04")
    .apt_install("wget", "git", "cmake", "libeigen3-dev", "libceres-dev")
    .run_commands("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    .pip_install("fastapi", "uvicorn", "sqlalchemy", "PyJWT")
    .add_local_dir(".", remote_path="/workspace", ignore=[".git", ".env"])
    .workdir("/workspace")
)
```

**Pros:**
- Modal caches each layer independently — only changed layers rebuild
- No Dockerfile or registry needed
- Fine-grained control over each build step

**Cons:**
- Must manually replicate all Dockerfile steps
- Verbose for complex environments (conda, C++ deps, etc.)

**Best for:** Simple Python apps, when you want Modal's layer caching.

---

## Comparison

| | `from_dockerfile()` | `from_registry()` | Programmatic |
|---|---|---|---|
| **Registry needed?** | No | Yes | No |
| **Local cache used?** | No (cloud build) | Yes | No (cloud build) |
| **Rebuild speed** | Slow (full rebuild) | Instant (pull only) | Fast (layer cache) |
| **Setup complexity** | Low | Medium | Medium-High |
| **Best for** | Dev / simple deploys | Production / CI-CD | Simple Python apps |

---

## ENTRYPOINT Note

If your image has an `ENTRYPOINT`, it must `exec "$@"` at some point so Modal's runtime can inject its Python entrypoint. Most standard entrypoints already do this. Example:

```bash
#!/bin/bash
# ... your setup ...
exec "$@"
```

---

## Quick Start for UNav Server

### Development (fastest iteration)
```python
# modal_unav_v2.py
image = modal.Image.from_dockerfile("Dockerfile", context_dir=".")
```

### Production (recommended)
```bash
# 1. Build and push once
docker build -t ghcr.io/your-org/unav-server:latest .
docker push ghcr.io/your-org/unav-server:latest

# 2. Reference in Modal
image = modal.Image.from_registry("ghcr.io/your-org/unav-server:latest")
```
