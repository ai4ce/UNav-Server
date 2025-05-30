from config import UNavConfig
from navigator.navigator import FacilityNavigator

# Set up your config and navigator
DATA_FINAL_ROOT = "/mnt/data/UNav-IO/data"
PLACES = {
    "New_York_City": {
        "LightHouse": ["3_floor", "4_floor", "6_floor"]
    }
}

config = UNavConfig(
    data_final_root=DATA_FINAL_ROOT,
    places=PLACES
)
nav = FacilityNavigator(config.navigator_config)

# Get all destinations across all maps
dest_dict = nav.list_destinations()
for key, (label, coords) in dest_dict.items():
    print(f"Place: {key[0]}, Building: {key[1]}, Floor: {key[2]}, NodeID: {key[3]} -> Label: {label}, Coordinates: {coords}")