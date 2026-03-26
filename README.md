# 🧭 UNav System – Docker Deployment & API Guide

This guide provides step-by-step instructions for building, configuring, and running the UNav server using Docker. It also documents the main RESTful API endpoints and provides a Python integration example.

---

## 1️⃣ Build the Docker Image

From the project root directory (where `Dockerfile` and `environment.yml` are located), run:

```bash
docker build -t unav-server .
```

The Docker build is pinned to install the upstream `UNav` package from the `unav-server` branch.

---

## 2️⃣ Configure the Startup Script

Edit `run_unav_docker.sh` to set your preferred options:

```bash
DATA_ROOT="/mnt/d/unav/data"   # Absolute path to your local data directory
HOST_PORT=5001                 # Port on your host machine
CONTAINER_PORT=5001            # Port inside the container (usually leave as 5001)
```

---

## 3️⃣ Launch the UNav Server

### On Linux/macOS

```bash
chmod +x run_docker_unav_server.sh
./run_docker_unav_server.sh
```

### On Windows

Open [Git Bash](https://gitforwindows.org/), [WSL](https://docs.microsoft.com/en-us/windows/wsl/), or any Bash-compatible terminal and run:

```bash
wsl bash run_docker_unav_server.sh
```

> **Tips:**  
> - Make sure Docker Desktop is running.  
> - Adjust `DATA_ROOT` as needed.  
> - Remove `--gpus all` in the script if no NVIDIA GPU is available.

---

## 4️⃣ Platform Notes

- On Windows, use WSL or Git Bash for best Bash compatibility.
- Make sure your data directory exists and is accessible to Docker on all platforms.

---

## 5️⃣ API Endpoints

### 👤 User Management

- `POST /api/register` &nbsp;&nbsp;&nbsp;— Register a new user  
- `POST /api/login` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Authenticate and receive a JWT  
- `POST /api/logout` &nbsp;&nbsp;&nbsp;&nbsp;— Log out user, clear session  
- `GET  /api/me` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Get authenticated user info (JWT required)

### 🛠️ Generic Task Execution

- `POST /api/run_task` — Execute a backend task by name and inputs (supports image uploads)

### 📍 UNav-Specific Tasks (via `run_task`)

- `get_destinations` &nbsp;&nbsp;&nbsp;&nbsp;— Get all destination points for a floor  
- `select_destination` &nbsp;&nbsp;&nbsp;— Set the user's selected destination  
- `select_unit` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Set unit preference (feet/meters)  
- `get_floorplan` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Get the current floorplan as base64  
- `get_scale` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Get floorplan scale  
- `unav_navigation` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;— Localize visually and get navigation path

---

## 6️⃣ Session & State Management

- User-specific session state is kept in memory, keyed by the JWT user ID.  
- Includes selected destination, floor context, units, and refinement queues.  
- Sessions auto-expire after a configurable timeout.

---

## 7️⃣ Python Client Example

```python
import requests
import json
import base64

SERVER = "http://your_server_ip:5001"
USERNAME = "testuser"
PASSWORD = "testpass"

# Register user
requests.post(f"{SERVER}/api/register", json={"username": USERNAME, "password": PASSWORD})

# Login and get JWT
resp = requests.post(f"{SERVER}/api/login", json={"username": USERNAME, "password": PASSWORD})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Query destinations
dest_resp = requests.post(f"{SERVER}/api/run_task", json={
    "task": "get_destinations",
    "inputs": {"place": "New_York_City", "building": "LightHouse", "floor": "6_floor"}
}, headers=headers)
destinations = dest_resp.json()["destinations"]

# Select a destination
requests.post(f"{SERVER}/api/run_task", json={
    "task": "select_destination",
    "inputs": {"dest_id": destinations[0]["id"]}
}, headers=headers)

# Select unit
requests.post(f"{SERVER}/api/run_task", json={
    "task": "select_unit",
    "inputs": {"unit": "feet"}
}, headers=headers)

# Get the floorplan image
floorplan_resp = requests.post(f"{SERVER}/api/run_task", json={
    "task": "get_floorplan",
    "inputs": {}
}, headers=headers)
floorplan_b64 = floorplan_resp.json().get("floorplan")
if floorplan_b64:
    with open("floorplan.jpg", "wb") as f:
        f.write(base64.b64decode(floorplan_b64))

# Visual navigation
with open("query_image.jpg", "rb") as f:
    files = {"file": ("query.jpg", f, "image/jpeg")}
    nav_resp = requests.post(f"{SERVER}/api/run_task", data={
        "task": "unav_navigation",
        "inputs": "{}"
    }, files=files, headers=headers)
    print(nav_resp.json())
```

---

**For advanced usage or troubleshooting, please refer to the project repository or open an issue.**
