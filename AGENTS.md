# UNav-Server Agent Guidelines

This document provides guidelines for agents working on the UNav-Server codebase.

## Project Overview

UNav-Server provides a serverless implementation for indoor navigation using computer vision. It leverages Modal for deployment and offers features like visual localization, path planning, and navigation guidance.

## Project Structure

```
UNav-Server/
├── src/
│   └── modal_functions/
│       ├── unav_v1/         # Legacy version
│       ├── unav_v2/         # Current production version
│       │   ├── unav_modal.py       # Main Modal app
│       │   ├── test_modal_functions.py
│       │   ├── modal_config.py
│       │   ├── deploy_config.py
│       │   ├── destinations_service.py
│       │   ├── localizer.py
│       │   └── media/              # Test images
│       └── volume_utils/   # Volume management utilities
├── .github/workflows/      # CI/CD workflows
├── requirements.txt        # Python dependencies
└── TODO.md               # Technical documentation
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
- Standard library imports first
- Third-party imports (modal, cv2, numpy, etc.)
- Local/relative imports last
- Group by type with blank lines between groups

Example:
```python
import os
import json
from typing import Dict, List, Any, Optional

import modal
import cv2
import numpy as np

from .deploy_config import get_scaledown_window
from .modal_config import app
```

### Naming Conventions
- **Functions/variables**: snake_case (e.g., `get_destinations_list`, `image_data`)
- **Classes**: PascalCase (e.g., `UnavServer`, `FacilityNavigator`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `BUILDING`, `PLACE`)
- **Private methods**: prefix with underscore (e.g., `_configure_middleware_tracing`)
- **Internal helper functions**: prefix with underscore (e.g., `_get_queue_key_for_image_shape`)

### Type Hints
- Use type hints for function parameters and return values
- Use Optional for nullable types
- Use Any when type is unknown or complex

Example:
```python
def get_destinations_list_impl(
    server: Any,
    floor: str = "6_floor",
    place: str = "New_York_City",
    enable_multifloor: bool = False,
) -> Dict[str, Any]:
```

### Function Documentation
- Use docstrings for public functions and classes
- Keep docstrings concise but descriptive
- Include Args and Returns sections for complex functions

Example:
```python
def _get_queue_key_for_image_shape(image_shape):
    """Get a queue key based on image shape for bucket-based refinement queue handling."""
    if image_shape is None:
        return "default"
    h, w = image_shape[:2]
    return f"{h}x{w}"
```

### Error Handling
- Use try/except blocks for operations that may fail
- Catch specific exceptions when possible
- Return error dictionaries for recoverable errors
- Use print statements with emojis for logging (e.g., `print("❌ Error: {e}")`)
- Re-raise critical exceptions that should fail the request

Example:
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

### String Formatting
- Use f-strings for string interpolation
- Use os.path.join for path construction

### Class Definitions
- Use Modal decorators (`@app.cls`, `@method`, `@enter`)
- Use type annotations for class attributes
- Initialize optional attributes with default None

Example:
```python
@app.cls(...)
class UnavServer:
    user_sessions: Dict[str, Any] = {}
    tracer: Optional[Any] = None
    
    @enter(snap=False)
    def initialize(self):
        ...
```

### Constants and Configuration
- Use configuration functions from deploy_config.py
- Access via environment variables for deployment
- Define defaults in modal_config.py

### Testing Guidelines
- Test files: `test_modal_functions.py`
- Use descriptive test parameters (BUILDING, PLACE, FLOOR, etc.)
- Include error handling for Modal class lookup
- Test both success and failure paths when applicable

### Logging
- Use print statements with emoji prefixes for status messages
- Common patterns:
  - `print("🔧 [Phase X] ...")`: Initialization steps
  - `print("✅ ..."): Success messages
  - `print("⚠️ ..."): Warnings
  - `print("❌ ..."): Errors
  - `print(f"[DEBUG] ..."): Debug info

### Working with Modal
- Use `modal deploy` from the correct directory
- Configure GPU, memory, and scaledown window via environment variables
- Use volume mounts for persistent data (/root/UNav-IO)
- Handle cold-start scenarios with multi-pass stabilization

### Common Tasks
- **Adding a new method**: Add to UnavServer class in unav_modal.py
- **Configuring deployment**: Modify deploy_config.py
- **Adding test image**: Place in src/modal_functions/unav_v2/media/
- **Volume operations**: Use scripts in volume_utils/

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
- The codebase uses the unav-core library internally
- Code changes may require redeployment to take effect
- Check TODO.md for technical context on implementation decisions