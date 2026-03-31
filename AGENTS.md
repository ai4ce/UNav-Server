# AGENTS.md – UNav Server

## Project Overview

UNav Server is a FastAPI backend for an indoor navigation system. It provides user management (JWT auth, SQLite), session state management, and a task-based API for destination queries, floorplan retrieval, and visual localization/navigation.

## Directory Structure

```
UNav-Server/
├── main.py                  # FastAPI app entry point
├── config.py                # Centralized config (DATA_ROOT, PLACES, models)
├── modal_unav.py            # Modal deployment v1 (rebuilds deps in Modal)
├── modal_unav_v2.py         # Modal deployment v2 (builds from Dockerfile)
├── Dockerfile               # Docker image (conda-based)
├── run_docker_unav_server.sh # Local Docker launch script
├── api/
│   ├── user_api.py          # User auth, profiles, avatars
│   └── task_api.py          # Generic /api/run_task endpoint
├── core/
│   ├── unav_state.py        # Global singletons (localizer, navigator, sessions)
│   ├── task_registry.py     # Task name → function registry
│   ├── i18n_labels.py       # Lazy-loaded i18n labels from JSON
│   └── tasks/
│       ├── unav.py          # Navigation tasks (get_destinations, unav_navigation, etc.)
│       └── general.py       # General tasks (select_unit, select_language)
├── db/
│   └── db.py                # SQLAlchemy models (users, navigation logs)
└── models/
    └── schemas.py           # Pydantic request/response schemas
```

## Build / Run Commands

### Local Docker
```bash
docker build -t unav-server .
chmod +x run_docker_unav_server.sh
./run_docker_unav_server.sh
```

### Modal Deployment
```bash
modal run modal_unav_v2.py      # Test locally
modal deploy modal_unav_v2.py   # Deploy to cloud
```

### Manual (if conda env active)
```bash
uvicorn main:app --host 0.0.0.0 --port 5001
```

There are **no lint, format, or test commands** configured in this repo. No pytest, no pre-commit, no ruff/black. If adding tests, use pytest and place them in a `tests/` directory.

## Code Style Guidelines

### Imports
- Standard library → third-party → local, each group separated by a blank line
- Use explicit imports (`from fastapi import APIRouter, HTTPException`) not wildcard
- Local imports use absolute paths from project root (`from core.unav_state import get_session`)
- In Modal `@asgi_app`/`@web_server` functions, use lazy imports (`import sys; sys.path.insert(0, "/workspace")`)

### Formatting
- 4-space indentation, no tabs
- Line length: no strict limit enforced (follow existing ~100-120 char style)
- No formatter configured; match surrounding file style

### Types
- Use Python type hints on function signatures where practical
- Pydantic models for all API request/response schemas (`models/schemas.py`)
- `dict` is the universal task input/output type (not strict Pydantic for tasks)
- Return `{"error": "message"}` dicts for task-level errors; raise `HTTPException` for API-level errors

### Naming Conventions
- `snake_case` for functions, variables, modules
- `PascalCase` for classes (Pydantic models, SQLAlchemy ORM models)
- `UPPER_SNAKE_CASE` for constants (`DATA_ROOT`, `JWT_SECRET`, `SESSION_TIMEOUT_SECONDS`)
- Task function names match their registry keys (`get_destinations`, `unav_navigation`)

### Error Handling
- **API layer**: raise `HTTPException(status_code=..., detail="...")` for 4xx/5xx
- **Task layer**: return `{"success": False, "error": "...", "stage": "..."}` dicts — never raise
- Use `try/finally` for database session cleanup (`db.close()` in finally block)
- Navigation logging is non-blocking (spawned via `threading.Thread`)
- SMTP/email failures raise `RuntimeError` with descriptive message

### Architecture Patterns
- **Task registry pattern**: all business logic lives in `core/tasks/*.py`, registered in dicts, looked up by name in `/api/run_task`
- **Session state**: in-memory dict keyed by JWT user ID, auto-expired after 30 min
- **Global singletons**: `localizer`, `nav`, `commander`, `LABELS` initialized once at module load in `unav_state.py`
- **Database**: two separate SQLite databases (`users.db`, `log.db`) with separate engines/sessions
- **Config**: all paths and model choices centralized in `config.py`

### Modal Deployment Notes
- `modal_unav_v2.py` is the current deployment file (builds from Dockerfile)
- `config.py` is baked into the image via `.add_local_file()` (the Dockerfile deletes it during build)
- Data volume mounted at `/data`; create via `modal volume create unav-data`
- GPU: T4 by default; change `GPU_MODEL` for A100/H100
- No `allow_concurrent_inputs` — use `@modal.concurrent(max_inputs=N)` instead
