"""
Places-related logic functions.
"""
import os
from typing import Dict, List, Optional


def run_get_places(
    self,
    target_place: Optional[str] = None,
    target_building: Optional[str] = None,
    target_floor: Optional[str] = None,
    enable_multifloor: bool = False,
) -> Dict[str, Dict[str, List[str]]]:
    """Get available places configuration, optionally filtering to a specific floor"""
    try:
        print("📁 Fetching places from data directory...")

        SKIP_FOLDERS = {
            "features",
            "colmap_map",
            ".ipynb_checkpoints",
            "parameters",
        }

        def should_skip_folder(folder_name):
            return (
                folder_name in SKIP_FOLDERS
                or "_old" in folder_name.lower()
                or folder_name.endswith("_old")
            )

        places: Dict[str, Dict[str, List[str]]] = {}

        data_root = getattr(self, "DATA_ROOT", "/root/UNav-IO/data")

        if os.path.exists(data_root):
            for place_name in os.listdir(data_root):
                place_path = os.path.join(data_root, place_name)
                if os.path.isdir(place_path) and not should_skip_folder(place_name):
                    if target_place and place_name != target_place:
                        continue
                    places[place_name] = {}
                    print(f"  ✓ Found place: {place_name}")

                    for building_name in os.listdir(place_path):
                        building_path = os.path.join(place_path, building_name)
                        if os.path.isdir(building_path) and not should_skip_folder(building_name):
                            if target_building and building_name != target_building:
                                continue
                            floors = []

                            for floor_name in os.listdir(building_path):
                                floor_path = os.path.join(building_path, floor_name)
                                if os.path.isdir(floor_path) and not should_skip_folder(floor_name):
                                    if (
                                        place_name == "New_York_City"
                                        and building_name == "LOH"
                                        and floor_name == "9_floor"
                                    ):
                                        print(f"    ⚠️ Skipping {building_name}/{floor_name}: explicitly excluded")
                                        continue

                                    if (
                                        not enable_multifloor
                                        and target_floor
                                        and floor_name != target_floor
                                    ):
                                        continue

                                    boundaries_file = os.path.join(floor_path, "boundaries.json")
                                    if not os.path.exists(boundaries_file):
                                        print(f"    ⚠️ Skipping {building_name}/{floor_name}: missing boundaries.json")
                                        continue

                                    floors.append(floor_name)

                            if floors:
                                places[place_name][building_name] = floors
                                print(f"    ✓ Building: {building_name} with floors: {floors}")

            places = {k: v for k, v in places.items() if v}

            if not enable_multifloor and target_floor:
                for p_name in list(places.keys()):
                    buildings = places[p_name]
                    for b_name in list(buildings.keys()):
                        filtered_floors = [
                            f_name
                            for f_name in buildings[b_name]
                            if f_name == target_floor
                        ]
                        if filtered_floors:
                            buildings[b_name] = filtered_floors
                        else:
                            del buildings[b_name]
                    if not buildings:
                        del places[p_name]

            print(f"✅ Found {len(places)} places with buildings and floors")
            return places
        else:
            print(f"⚠️ Data root {data_root} does not exist, using fallback")
            return run_get_fallback_places()

    except Exception as e:
        print(f"❌ Error fetching places: {e}, using fallback")
        return run_get_fallback_places()


def run_get_fallback_places():
    """Fallback hardcoded places configuration"""
    return {
        "New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"]},
        "New_York_University": {"Langone": ["15_floor", "16_floor", "17_floor"]},
        "Mahidol_University": {"Jubilee": ["fl1", "fl2", "fl3"]},
    }
