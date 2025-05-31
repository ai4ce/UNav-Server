import requests
import json
import base64

SERVER = "http://128.238.176.102:5001"  # 改成服务器公网IP即可
USERNAME = "testuser"
PASSWORD = "testpass"

def register():
    print("==> Register user")
    r = requests.post(f"{SERVER}/api/register", json={
        "username": USERNAME, "password": PASSWORD
    })
    print("register:", r.status_code, r.text)
    assert r.status_code in [200, 400]  # 200=成功，400=已注册

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

# 1. 查询某楼层所有可选目标（get_destinations）
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

# 2. 选择目标 (select_destination)
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

# 3. 选择单位 (select_unit)
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

# 4. 获取 floorplan (get_floorplan)
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
fp = r.json()["floorplan"]
with open("floorplan.jpg", "wb") as f:
    f.write(base64.b64decode(fp))
print("Floorplan image saved as floorplan.jpg")

# 5. 获取 scale (get_scale)
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

# 6. 上传一张图片做导航（unav_navigation）
query_img_path = "/mnt/data/UNav-IO/test/photos/LightHouse/3-2.jpg"  # 换成你实际的测试图片路径
with open(query_img_path, "rb") as f:
    files = {"file": ("query.jpg", f, "image/jpeg")}
    data = {
        "task": "unav_navigation",
        "inputs": {}
    }
    print("==> Navigation (localize and plan)")
    r = requests.post(
        f"{SERVER}/api/run_task",
        data=data,      # form-data
        files=files,
        headers=headers
    )
print("navigate:", r.status_code)
print(json.dumps(r.json(), indent=2))
