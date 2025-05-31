# UNav Server

The UNav Server backend provides a scalable, secure, and extensible platform for visual localization and indoor navigation services. Built on FastAPI, it exposes RESTful APIs and a generic task execution interface, integrating the UNav localization and navigation core algorithms packaged as a Python library.

---

## Key Features

- **User Authentication and Management:**  
  Robust user registration and login system secured via JWT tokens.

- **Hierarchical Place and Floor Management:**  
  Support for multiple places, buildings, and floors with dynamic retrieval of navigational destinations.

- **Visual Localization and Navigation:**  
  End-to-end pipeline allowing clients to upload query images for localization and route planning to user-selected destinations.

- **Session State Management:**  
  Maintains per-user navigation context and pose refinement states for consistent multi-step navigation workflows.

- **Extensible Task Registry:**  
  Unified backend interface enabling modular task registration and execution with support for image and non-image inputs.

- **Floorplan and Scale Retrieval:**  
  Provides compressed floorplan imagery and scale metrics aligned with current user navigation context for client rendering and metric conversions.

---

## Project Structure

```
UNav_socket/
├── api/                        # REST API route handlers
│   ├── task_api.py             # Generic task execution endpoint
│   └── user_api.py             # User auth and profile management
├── config.py                   # Central configuration (paths, JWT secret, places)
├── core/                       # Core backend logic and task registry
│   ├── task_registry.py        # Centralized task lookup and dispatch
│   ├── tasks/                  # Modular task implementations
│   │   ├── general.py          # General-purpose backend tasks
│   │   └── unav.py             # UNav-specific navigation tasks
│   └── unav_state.py           # Global UNav singleton instances and session store
├── db/                        # Database schema and initialization
│   └── db.py
├── main.py                    # FastAPI application factory and server start
├── models/                    # Pydantic request/response schemas
│   └── schemas.py
└── users.db                   # SQLite database file for user data
```

---

## Installation and Setup

1. **Clone repository and enter directory**

```bash
git clone https://github.com/ai4ce/UNav-Server.git
cd UNav-Server
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install unav_pretrained  # UNav core library
```

4. **Initialize database**

```bash
python -c "from db.db import init_db; init_db()"
```

5. **Configure application**

Edit `config.py` to set environment-specific paths, place/building/floor hierarchy, and secure JWT secret keys.

---

## Running the Server

Use Uvicorn to launch the FastAPI app:

```bash
uvicorn main:app --host 0.0.0.0 --port 5001 --reload
```

For production, omit `--reload` and configure HTTPS and process managers accordingly.

---

## API Endpoints

### User Management

- `POST /api/register` - Register a new user account  
- `POST /api/login` - Authenticate user and retrieve JWT token  
- `POST /api/logout` - Logout user and clear session  
- `GET /api/me` - Retrieve authenticated user information (requires JWT)

### Generic Task Execution

- `POST /api/run_task` - Execute any registered backend task by specifying task name and inputs, supports image file uploads.

### UNav-Specific Services (via `run_task`)

- `get_destinations` - Retrieve all destination points on a given floor  
- `select_destination` - Save user's selected navigation target  
- `select_unit` - Set user preference for distance units (feet/meters)  
- `get_floorplan` - Fetch current floorplan image encoded in base64 for client-side rendering  
- `get_scale` - Retrieve floorplan scale factor (meters or feet per pixel)  
- `unav_navigation` - Perform visual localization from query image and compute navigation path with step-by-step commands

---

## Session and State Management

User-specific session state is held in-memory keyed by user ID extracted from JWT tokens. This includes selected destination, current floor context, preferred units, and localization refinement queues for iterative pose updates.

Sessions expire automatically after a configurable timeout to free resources and maintain security.

---

## Client Integration Example

The following Python snippet demonstrates how a client can interact with the UNav server:

```python
import requests
import json
import base64

SERVER = "http://your_server_ip:5001"
USERNAME = "testuser"
PASSWORD = "testpass"

# Register user (if not registered)
requests.post(f"{SERVER}/api/register", json={"username": USERNAME, "password": PASSWORD})

# Login and retrieve JWT token
resp = requests.post(f"{SERVER}/api/login", json={"username": USERNAME, "password": PASSWORD})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Query available destinations
dest_resp = requests.post(f"{SERVER}/api/run_task", json={
    "task": "get_destinations",
    "inputs": {"place": "New_York_City", "building": "LightHouse", "floor": "6_floor"}
}, headers=headers)
destinations = dest_resp.json()["destinations"]

# Select destination
requests.post(f"{SERVER}/api/run_task", json={
    "task": "select_destination",
    "inputs": {"dest_id": destinations[0]["id"]}
}, headers=headers)

# Select unit
requests.post(f"{SERVER}/api/run_task", json={
    "task": "select_unit",
    "inputs": {"unit": "feet"}
}, headers=headers)

# Get current floorplan image
floorplan_resp = requests.post(f"{SERVER}/api/run_task", json={
    "task": "get_floorplan",
    "inputs": {}
}, headers=headers)
floorplan_b64 = floorplan_resp.json().get("floorplan")
if floorplan_b64:
    with open("floorplan.jpg", "wb") as f:
        f.write(base64.b64decode(floorplan_b64))

# Perform navigation with an image
with open("query_image.jpg", "rb") as f:
    files = {"file": ("query.jpg", f, "image/jpeg")}
    nav_resp = requests.post(f"{SERVER}/api/run_task", data={
        "task": "unav_navigation",
        "inputs": "{}"
    }, files=files, headers=headers)
    print(nav_resp.json())
```

---

## Security Considerations

- Replace default JWT secret with a strong, environment-specific key for production.  
- Secure communications with TLS/HTTPS.  
- Implement rate limiting and input validation to mitigate abuse.

---

## Extensibility

The unified task registry architecture allows easy integration of additional modules or algorithms by registering new tasks without modifying the API interface.

---

## 👤 Maintainer

- **Developer:** Anbang Yang (`ay1620@nyu.edu`)
- **Last updated:** 2025-05-31
