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
| 5 | Project files | When any local file changes |
| 6 | pip install from git | When git URLs change |

**Key insight:** Layers 1-4 are heavy but stable. Layer 5 (project files) changes frequently but is fast. Layer 6 is fast. Only layer 5+ rebuild on code changes.

## Attempts

### Attempt 1: Initial native image conversion
**File:** `modal_native.py`
**Date:** 2026-04-02
**Status:** Pending test
**Details:** Converted Dockerfile to `modal.Image` chain with proper layering. No conda symlink hacks — Modal handles Python detection natively when using `from_registry()`.
