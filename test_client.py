import requests
import json
import base64

SERVER = "http://128.238.176.102:5001"  # Replace with your server's public IP or localhost for local testing
USERNAME = "testuser"
PASSWORD = "testpass"

def register():
    print("==> Register user")
    r = requests.post(f"{SERVER}/api/register", json={
        "username": USERNAME, "password": PASSWORD
    })
    print("register:", r.status_code, r.text)
    assert r.status_code in [200, 400]  # 200=success, 400=already registered

register()

def login():
    print("==> Login user")
    r = requests.post(f"{SERVER}/api/login", json={
        "username": USERNAME, "password": PASSWORD
    })
    print("login:", r.status_code, r.text)
    token = r.json()["access_token"]
    return token

token = login()
headers = {"Authorization": f"Bearer {token}"}

# 1. Query all available destinations on a floor (get_destinations)
PLACE = "New_York_City"
BUILDING = "LightHouse"
FLOOR = "6_floor"
print("==> Get destinations")
r = requests.post(
    f"{SERVER}/api/run_task",
    json={
        "task": "get_destinations",
        "inputs": {
            "place": PLACE,
            "building": BUILDING,
            "floor": FLOOR
        }
    },
    headers=headers
)
print("destinations:", r.status_code, r.text)
resp = r.json()
dests = resp["destinations"]
assert dests, "No destinations found!"
first_dest_id = dests[0]["id"]

# 2. Select destination (select_destination)
print("==> Select destination")
r = requests.post(
    f"{SERVER}/api/run_task",
    json={
        "task": "select_destination",
        "inputs": {
            "dest_id": first_dest_id
        }
    },
    headers=headers
)
print("select_destination:", r.status_code, r.text)

# 3. Select preferred unit (select_unit)
print("==> Select unit")
r = requests.post(
    f"{SERVER}/api/run_task",
    json={
        "task": "select_unit",
        "inputs": {
            "unit": "feet"
        }
    },
    headers=headers
)
print("select_unit:", r.status_code, r.text)

# 4. Retrieve floorplan for current floor (get_floorplan)
print("==> Get floorplan")
r = requests.post(
    f"{SERVER}/api/run_task",
    json={
        "task": "get_floorplan",
        "inputs": {}
    },
    headers=headers
)
print("get_floorplan:", r.status_code)
resp = r.json()
if r.status_code == 200 and "floorplan" in resp:
    fp = resp["floorplan"]
    with open("floorplan.jpg", "wb") as f:
        f.write(base64.b64decode(fp))
    print("Floorplan image saved as floorplan.jpg")
else:
    print("Failed to get floorplan:", resp.get("error", "Unknown error"))

# 5. Retrieve scale for current floor (get_scale)
print("==> Get scale")
r = requests.post(
    f"{SERVER}/api/run_task",
    json={
        "task": "get_scale",
        "inputs": {}
    },
    headers=headers
)
print("get_scale:", r.status_code, r.text)

# 6. Upload query images for localization and navigation (unav_navigation)
for query_img_path in [
    "/mnt/data/UNav-IO/test/photos/LightHouse/3-2.jpg",
    "/mnt/data/UNav-IO/test/photos/LightHouse/3-5.jpg",
    "/mnt/data/UNav-IO/test/photos/LightHouse/6-2.jpg"
]:
    with open(query_img_path, "rb") as f:
        files = {"file": ("query.jpg", f, "image/jpeg")}
        data = {
            "task": "unav_navigation",
            "inputs": {}
        }
        print(f"==> Navigation (localize and plan) for {query_img_path}")
        r = requests.post(
            f"{SERVER}/api/run_task",
            data=data,      # form-data format for file upload
            files=files,
            headers=headers
        )
    print("navigate:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print("Failed to parse JSON response:", e)
        print("Raw response:", r.text)

def logout(token):
    print("==> Logout user")
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(f"{SERVER}/api/logout", headers=headers)
    print("logout:", r.status_code, r.text)

logout(token)
