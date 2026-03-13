from typing import Any, Dict, Optional, Set


def run_ensure_maps_loaded(
    server: Any,
    place: str,
    building: Optional[str] = None,
    floor: Optional[str] = None,
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

    if map_key in server.maps_loaded:
        return

    print(f"🔄 [Phase 4] Creating selective localizer for: {map_key}")

    if building:
        selective_places = server.get_places(
            target_place=place,
            target_building=building,
            target_floor=floor,
            enable_multifloor=enable_multifloor,
        )
    else:
        selective_places = server.get_places(target_place=place)

    if not selective_places:
        print(
            "⚠️ No matching places found for selective load; skipping localizer creation"
        )
        return

    from unav.config import UNavConfig

    selective_config = UNavConfig(
        data_final_root=server.DATA_ROOT,
        places=selective_places,
        global_descriptor_model=server.FEATURE_MODEL,
        local_feature_model=server.LOCAL_FEATURE_MODEL,
    )

    from unav.localizer.localizer import UNavLocalizer
    import time

    selective_localizer = UNavLocalizer(selective_config.localizer_config)

    if hasattr(server, "tracer") and server.tracer:
        try:
            server._monkey_patch_localizer_methods(selective_localizer)
        except Exception as e:
            print(f"⚠️ Failed to patch selective localizer: {e}")

    if hasattr(server, "tracer") and server.tracer:
        with server.tracer.start_as_current_span(
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

    server.selective_localizers[map_key] = selective_localizer
    server.maps_loaded.add(map_key)
    print(f"✅ Selective localizer created and maps loaded for: {map_key}")
