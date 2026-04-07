# UNav-Server Agent Guidelines

This document provides guidelines for agents working on the UNav-Server codebase.

## Project Overview

UNav-Server provides a serverless implementation for indoor navigation using computer vision. It leverages Modal for deployment and offers features like visual localization, path planning, and navigation guidance.

## Project Structure

```
UNav-Server/
├── src/
│   └── modal_functions/
│       ├── unav_v1/           # Legacy version
│       ├── unav_v2/           # Current production version
│       │   ├── unav_modal.py          # Main Modal app (~200 lines)
│       │   ├── logic/                  # Extracted business logic
│       │   │   ├── __init__.py         # Exports all run_* functions
│       │   │   ├── navigation.py       # run_planner, run_localize_user
│       │   │   ├── init.py             # Initialization & monkey-patching
│       │   │   ├── places.py           # run_get_places, run_get_fallback_places
│       │   │   ├── maps.py             # run_ensure_maps_loaded
│       │   │   ├── utils.py            # run_safe_serialize, etc.
│       │   │   └── vlm.py               # run_vlm_on_image
│       │   ├── server_methods/
│       │   │   └── helpers.py          # Queue utility functions
│       │   ├── test_modal_functions.py
│       │   ├── modal_config.py
│       │   ├── deploy_config.py
│       │   ├── destinations_service.py
│       │   └── media/                  # Test images
│       └── volume_utils/               # Volume management utilities
├── .github/workflows/                   # CI/CD workflows
├── requirements.txt                     # Python dependencies
└── TODO.md                             # Technical documentation
```

## Build/Lint/Test Commands

### Python Version
- Minimum: Python 3.10+
- Recommended: Python 3.11 (used in CI/CD)

### Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
```bash
# Navigate to the module directory
cd src/modal_functions/unav_v2

# Run a single test file
python test_modal_functions.py

# Run with pytest (if installed)
pytest test_modal_functions.py -v
```

### Deployment Commands
```bash
# Deploy to Modal (from unav_v2 directory)
cd src/modal_functions/unav_v2
modal deploy unav_modal.py

# Deploy with custom parameters
UNAV_SCALEDOWN_WINDOW=600 UNAV_GPU_TYPE=t4 UNAV_RAM_MB=73728 modal deploy unav_modal.py
```

### GitHub Actions Deployment
1. Go to Actions -> "Deploy UNav v2 Modal" -> "Run workflow"
2. Set inputs: scaledown_window, gpu_type, ram_mb
3. Requires secrets: MODAL_TOKEN_ID, MODAL_TOKEN_SECRET

## Code Style Guidelines

### Import Organization
Order: stdlib -> third-party -> local imports, with blank lines between groups.

```python
import os
import json
from typing import Dict, List, Any, Optional

import modal
import cv2
import numpy as np

from .deploy_config import get_scaledown_window
from .logic import run_planner, run_localize_user
```

### Naming Conventions
- **Functions/variables**: snake_case (e.g., `get_destinations_list`, `image_data`)
- **Classes**: PascalCase (e.g., `UnavServer`, `FacilityNavigator`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `BUILDING`, `PLACE`)
- **Logic functions**: prefix with `run_` (e.g., `run_planner`, `run_safe_serialize`)
- **Private methods**: prefix with underscore (e.g., `_configure_middleware_tracing`)

### Type Hints
Use type hints for function parameters and return values.

```python
def get_destinations_list_impl(
    server: Any,
    floor: str = "6_floor",
    place: str = "New_York_City",
    enable_multifloor: bool = False,
) -> Dict[str, Any]:
```

### Refactoring Pattern: Logic Extraction

When extracting code from `unav_modal.py`:

1. **Keep `@method()` decorators in `unav_modal.py`** - Modal requires them
2. **Move logic to `logic/` directory** - Each function gets `run_` prefix
3. **Thin wrapper pattern** - Method in unav_modal.py just calls the logic function

```python
# unav_modal.py - thin wrapper
@method()
def planner(self, session_id: str, ...):
    return run_planner(self, session_id=session_id, ...)

# logic/navigation.py - actual logic
def run_planner(self, session_id: str, ...) -> Dict[str, Any]:
    # Full implementation here
    pass
```

**DO NOT create wrapper methods** for internal functions (e.g., `get_session`, `update_session`) - call `run_*` functions directly from logic modules.

### Error Handling
- Use try/except blocks for operations that may fail
- Catch specific exceptions when possible
- Return error dictionaries for recoverable errors
- Use print statements with emojis for logging

```python
try:
    result = some_function()
except ValueError as e:
    print(f"❌ Invalid value: {e}")
    return {"status": "error", "message": str(e)}
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    raise
```

### Code Formatting
- Maximum line length: 100 characters (soft limit)
- Use 4 spaces for indentation (no tabs)
- Use blank lines to separate logical sections
- Use trailing commas in multi-line collections
- Use f-strings for string interpolation

### Logging Patterns
- `print("🔧 [Phase X] ...")` - Initialization steps
- `print("✅ ...")` - Success messages
- `print("⚠️ ...")` - Warnings
- `print("❌ ...")` - Errors
- `print(f"[DEBUG] ...")` - Debug info

### Testing Guidelines
- Test files: `test_modal_functions.py`
- Use descriptive test parameters (BUILDING, PLACE, FLOOR, etc.)
- Include error handling for Modal class lookup
- Test both success and failure paths when applicable

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| UNAV_SCALEDOWN_WINDOW | 300 | Modal scaledown window (seconds) |
| UNAV_GPU_TYPE | t4 | GPU type (a10, t4, a100, any, h200) |
| UNAV_RAM_MB | 73728 | RAM reservation in MiB |
| MODAL_TOKEN_ID | - | Modal token (GitHub secret) |
| MODAL_TOKEN_SECRET | - | Modal secret (GitHub secret) |

## Notes for Agents

- This is a Modal-based serverless application
- Tests require a deployed Modal app to run against
- The codebase uses the unav-core library internally (runtime dependency - LSP errors are expected locally)
- Code changes may require redeployment to take effect
- Check TODO.md for technical context on implementation decisions
- Runtime imports (torch, unav, middleware, google.genai) only exist in Modal container
- When committing changes, append `Committed by agent (<provider>-<model-name>)` to the commit message (e.g., `Committed by agent (opencode-qwencoder)`, `Committed by agent (claude-code-sonnet)`)
