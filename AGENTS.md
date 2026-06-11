# UNav-Server Agent Guidelines

Serverless indoor navigation via computer vision, deployed on Modal. Core logic is in `src/modal_functions/unav_v2/`.

## Quick Reference

```bash
# Deploy (from repo root)
cd src/modal_functions/unav_v2 && modal deploy unav_modal.py

# Deploy with custom params
UNAV_SCALEDOWN_WINDOW=1800 UNAV_GPU_TYPE=A10 modal deploy unav_modal.py

# Test (requires live Modal deployment)
cd src/modal_functions/unav_v2 && python test_modal_functions.py
```

Modal app name: `Mast3r-UNav-Server`. Class name: `UnavServer`.

## Project Structure

```
UNav-Server/
├── src/modal_functions/
│   ├── unav_v2/                    # Core - all work goes here
│   │   ├── unav_modal.py           # Thin @method/@enter wrappers (Modal entry point)
│   │   ├── localizer.py            # UNavLocalizer: feature extraction, VPR, matching, RANSAC
│   │   ├── modal_config.py         # Modal App, Image, Volume, Secrets
│   │   ├── deploy_config.py        # Env var config: GPU, scaledown, RAM
│   │   ├── destinations_service.py # Destination list for place/building/floor
│   │   ├── logic/                  # Business logic (run_* functions)
│   │   │   ├── init.py             # 3-phase startup + monkey-patching (~500 lines)
│   │   │   ├── navigation.py       # run_planner, run_localize_user (~500 lines)
│   │   │   ├── maps.py             # Lazy map loading per building
│   │   │   ├── places.py           # Filesystem-based place discovery
│   │   │   ├── utils.py            # Serialization, mock localization, trajectory
│   │   │   └── vlm.py             # Gemini VLM text extraction
│   │   ├── server_methods/helpers.py  # Queue bucketing by image shape
│   │   └── test_modal_functions.py    # Integration test against deployed app
│   ├── unav_v1/                    # Legacy (ignore)
│   └── volume_utils/               # One-off volume management scripts
├── unav/                           # Git submodule (unav-core library)
└── docs/TODO.md                    # Technical decisions and history
```

## Architecture

### Logic Extraction Pattern

`unav_modal.py` contains only `@method()` and `@enter()` decorators. All logic lives in `logic/` as `run_*` functions that receive `self` (the UnavServer instance) as first arg.

```python
# unav_modal.py - thin wrapper
@method()
def planner(self, session_id: str, ...):
    return run_planner(self, session_id=session_id, ...)

# logic/navigation.py - actual logic
def run_planner(self, session_id: str, ...) -> Dict[str, Any]:
    ...
```

Internal helpers (get_session, update_session) are called directly from logic modules, not wrapped.

### Three-Phase Initialization

Modal calls these `@enter(snap=False)` methods on container start, in order:
1. `initialize_middleware` → `run_init_middleware` (deferred if no GPU)
2. `initialize_cpu_components` → `run_init_cpu_components` (UNavConfig, FacilityNavigator, places)
3. `initialize_gpu_components` → `run_init_gpu_components` (UNavLocalizer, model weights)

### Lazy Map Loading

Maps are NOT loaded at startup. `run_ensure_maps_loaded()` creates per-building `UNavLocalizer` instances on first request. Tracked in `server.maps_loaded` (set) and `server.selective_localizers` (dict).

### Volume-Based Data

Modal volume `unav_multifloor` mounted at `/root/UNav-IO`. Data root: `/root/UNav-IO/data`. Directory structure: `{place}/{building}/{floor}/` with `boundaries.json` required in each floor dir.

## Environment Variables

**Deploy-time** (Modal class config via `deploy_config.py`):
| Variable | Default | Allowed |
|----------|---------|---------|
| `UNAV_SCALEDOWN_WINDOW` | 300 | any positive int |
| `UNAV_GPU_TYPE` | t4 | t4, a10, a100, h200, any |
| `UNAV_RAM_MB` | 73728 | max 98304 |

**Runtime** (Modal secrets or container env):
| Secret/Env | Used in | Purpose |
|------------|---------|---------|
| `gemini-api-key` | vlm.py | Gemini VLM access |
| `middleware` | init.py | Middleware.io telemetry |
| `MW_API_KEY` | init.py | Middleware.io API key |
| `MW_TARGET` | init.py | Middleware.io endpoint |
| `GEMINI_API_KEY` | vlm.py | Alternative env var for Gemini |
| `PYTHONPATH` | modal_config.py | Must include `/root/mast3r:/root/mast3r/dust3r` |

## Key Gotchas

- **Tests require a live Modal deployment.** `test_modal_functions.py` uses `modal.Cls.from_name()` — cannot run locally without deploying first.
- **LSP errors are expected locally.** Runtime deps (torch, unav, faiss, middleware, google.genai) only exist in the Modal container.
- **Container clones branch `endeleze`** during image build (`modal_config.py`). Changes to other branches don't affect the deployed container unless redeployed.
- **MASt3R symlink workaround** may be needed if the updated `unav` package fails to find DB perspective images. See `docs/ROLLBACK_MASt3R_SYMLINK.md`.
- **`unav/` is a git submodule** for the unav-core library. It's excluded from `.gitignore` but not directly used by the Modal deployment (the container installs it from GitHub).
- **No linter/formatter config** is checked in. `.ruff_cache/` exists but no `ruff.toml` or `pyproject.toml`.

## Testing

```bash
cd src/modal_functions/unav_v2
python test_modal_functions.py
```

Test constants: BUILDING=Langone, PLACE=New_York_University, FLOOR=17_floor. Uses `media/vinay_sample.jpeg` as test image. Tests `get_destinations_list` and `planner` via Modal RPC.

## Deployment

### CLI
```bash
cd src/modal_functions/unav_v2
modal deploy unav_modal.py
```

### GitHub Actions
Actions → "Deploy UNav v2 Modal" → "Run workflow". Requires repo secrets: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.

## Code Conventions

- Logic functions use `run_` prefix. Private helpers use `_` prefix.
- Import order: stdlib → third-party → local. Heavy imports inside functions to avoid import-time overhead.
- Error returns: `{"status": "error", "error": ..., "timing": ...}` dicts.
- Logging: emoji prefixes — 🔧 init, ✅ success, ⚠️ warning, ❌ error.
