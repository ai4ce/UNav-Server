import os
import time
from typing import Dict, List, Optional


def get_places(
    self,
    target_place: Optional[str] = None,
    target_building: Optional[str] = None,
    target_floor: Optional[str] = None,
    enable_multifloor: bool = False,
):
    """Get available places configuration, optionally filtering to a specific floor"""
    try:
        print("📁 Fetching places from data directory...")

        # Define folders to skip at all levels
        SKIP_FOLDERS = {
            "features",
            "colmap_map",
            ".ipynb_checkpoints",
            "parameters",
        }

        def should_skip_folder(folder_name):
            """Check if folder should be skipped based on name patterns"""
            return (
                folder_name in SKIP_FOLDERS
                or "_old" in folder_name.lower()
                or folder_name.endswith("_old")
            )

        places: Dict[str, Dict[str, List[str]]] = {}

        data_root = getattr(self, "DATA_ROOT", "/root/UNav-IO/data")

        # Get all place directories (depth 1 under data/)
        if os.path.exists(data_root):
            for place_name in os.listdir(data_root):
                place_path = os.path.join(data_root, place_name)
                if os.path.isdir(place_path) and not should_skip_folder(place_name):
                    if target_place and place_name != target_place:
                        continue
                    places[place_name] = {}
                    print(f"  ✓ Found place: {place_name}")

                    # Get buildings for this place (depth 2)
                    for building_name in os.listdir(place_path):
                        building_path = os.path.join(place_path, building_name)
                        if os.path.isdir(building_path) and not should_skip_folder(
                            building_name
                        ):
                            if target_building and building_name != target_building:
                                continue
                            floors = []

                            # Get floors for this building (depth 3)
                            for floor_name in os.listdir(building_path):
                                floor_path = os.path.join(building_path, floor_name)
                                if os.path.isdir(
                                    floor_path
                                ) and not should_skip_folder(floor_name):
                                    # Skip specific problematic floor
                                    if (
                                        place_name == "New_York_City"
                                        and building_name == "LOH"
                                        and floor_name == "9_floor"
                                    ):
                                        print(
                                            f"    ⚠️ Skipping {building_name}/{floor_name}: explicitly excluded"
                                        )
                                        continue

                                    if (
                                        not enable_multifloor
                                        and target_floor
                                        and floor_name != target_floor
                                    ):
                                        continue

                                    # Validate that required navigation files exist
                                    boundaries_file = os.path.join(
                                        floor_path, "boundaries.json"
                                    )
                                    if not os.path.exists(boundaries_file):
                                        print(
                                            f"    ⚠️ Skipping {building_name}/{floor_name}: missing boundaries.json"
                                        )
                                        continue

                                    floors.append(floor_name)

                            if floors:  # Only add building if it has floors
                                places[place_name][building_name] = floors
                                print(
                                    f"    ✓ Building: {building_name} with floors: {floors}"
                                )

            # Remove places that have no buildings
            places = {k: v for k, v in places.items() if v}

            if not enable_multifloor and target_floor:
                # Ensure we only return the specific floor requested
                for p_name in list(places.keys()):
                    buildings = places[p_name]
                    for b_name in list(buildings.keys()):
                        filtered_floors = [
                            f_name for f_name in buildings[b_name] if f_name == target_floor
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
            return self._get_fallback_places()

    except Exception as e:
        print(f"❌ Error fetching places: {e}, using fallback")
        return self._get_fallback_places()


def _get_fallback_places(self):
    """Fallback hardcoded places configuration"""
    return {
        "New_York_City": {"LightHouse": ["3_floor", "4_floor", "6_floor"]},
        "New_York_University": {"Langone": ["15_floor", "16_floor", "17_floor"]},
        "Mahidol_University": {"Jubilee": ["fl1", "fl2", "fl3"]},
    }


def ensure_maps_loaded(
    self,
    place: str,
    building: str = None,
    floor: str = None,
    enable_multifloor: bool = False,
):
    """
    Ensure that maps for a specific place/building are loaded.
    When building is specified, loads all floors for that building.
    Creates selective localizer instances for true lazy loading.
    """
    if building:
        if enable_multifloor or not floor:
            map_key = (place, building)
        else:
            map_key = (place, building, floor)
    else:
        map_key = place

    if map_key in self.maps_loaded:
        return  # Already loaded

    print(f"🔄 [Phase 4] Creating selective localizer for: {map_key}")

    # Create selective places config with only the requested location
    if building:
        selective_places = self.get_places(
            target_place=place,
            target_building=building,
            target_floor=floor,
            enable_multifloor=enable_multifloor,
        )
    else:
        selective_places = self.get_places(target_place=place)

    if not selective_places:
        print("⚠️ No matching places found for selective load; skipping localizer creation")
        return

    # Create selective config and localizer
    from unav.config import UNavConfig

    selective_config = UNavConfig(
        data_final_root=self.DATA_ROOT,
        places=selective_places,
        global_descriptor_model=self.FEATURE_MODEL,
        local_feature_model=self.LOCAL_FEATURE_MODEL,
    )

    from unav.localizer.localizer import UNavLocalizer

    selective_localizer = UNavLocalizer(selective_config.localizer_config)

    # Optionally patch the selective localizer too
    if hasattr(self, "tracer") and self.tracer:
        try:
            self._monkey_patch_localizer_methods(selective_localizer)
            # Note: feature extractors are already patched at module level, no need to patch per instance
        except Exception as e:
            print(f"⚠️ Failed to patch selective localizer: {e}")

    # Load maps and features with tracing if available
    if hasattr(self, "tracer") and self.tracer:
        with self.tracer.start_as_current_span(
            "load_maps_and_features_span"
        ) as load_span:
            load_span.add_event("Starting map and feature loading")
            load_span.set_attribute("map_key", str(map_key))
            load_span.set_attribute("selective_places", str(selective_places))

            start_load_time = time.time()
            selective_localizer.load_maps_and_features()
            load_duration = time.time() - start_load_time

            load_span.set_attribute("load_duration_seconds", load_duration)
            load_span.add_event("Map and feature loading completed")
    else:
        print(f"⏱️ Starting load_maps_and_features for: {map_key}")
        start_load_time = time.time()
        selective_localizer.load_maps_and_features()
        load_duration = time.time() - start_load_time
        print(f"⏱️ Completed load_maps_and_features in {load_duration:.2f} seconds")

    # Cache the selective localizer
    self.selective_localizers[map_key] = selective_localizer
    self.maps_loaded.add(map_key)
    print(f"✅ Selective localizer created and maps loaded for: {map_key}")


def ensure_gpu_components_ready(self):
    """
    Ensure GPU components are initialized before processing requests.
    This is called by methods that need the localizer.
    """
    if not hasattr(self, "gpu_components_initialized"):
        print("🔧 GPU components not ready, initializing now...")
        self.initialize_gpu_components()
