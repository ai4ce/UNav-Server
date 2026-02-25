from typing import Any


def get_destinations_list_impl(
    server: Any,
    floor: str = "6_floor",
    place: str = "New_York_City",
    building: str = "LightHouse",
    enable_multifloor: bool = False,
):
    """
    Fetch destinations for a place/building/floor.
    When enable_multifloor=True, aggregate destinations from all floors in the building.
    """

    def _collect_floor_destinations(target_floor: str, include_floor: bool = False):
        target_key = (place, building, target_floor)
        pf_target = server.nav.pf_map[target_key]
        destinations = []
        for did in pf_target.dest_ids:
            item = {
                "id": str(did),
                "name": pf_target.labels[did],
                "xy": pf_target.nodes[did],
            }
            if include_floor:
                item["floor"] = target_floor
            destinations.append(item)
        return destinations

    def _run():
        print(f"🎯 [Phase 3] Getting destinations for {place}/{building}/{floor}")

        # Ensure maps are loaded for this location.
        server.ensure_maps_loaded(
            place,
            building,
            floor=floor,
            enable_multifloor=enable_multifloor,
        )

        if enable_multifloor:
            places = server.get_places(
                target_place=place,
                target_building=building,
                enable_multifloor=True,
            )
            building_floors = places.get(place, {}).get(building, [])
            if not building_floors:
                raise ValueError(
                    f"No floors found for place='{place}', building='{building}'"
                )

            destinations = []
            for floor_name in building_floors:
                target_key = (place, building, floor_name)
                if target_key not in server.nav.pf_map:
                    print(
                        f"⚠️ Skipping floor '{floor_name}' because map is not loaded in pf_map"
                    )
                    continue
                destinations.extend(
                    _collect_floor_destinations(
                        target_floor=floor_name, include_floor=True
                    )
                )
        else:
            destinations = _collect_floor_destinations(
                target_floor=floor, include_floor=True
            )

        print(f"✅ Found {len(destinations)} destinations")
        return {"destinations": destinations}

    if hasattr(server, "tracer") and server.tracer:
        with server.tracer.start_as_current_span("get_destinations_list_span"):
            try:
                with server.tracer.start_as_current_span("ensure_maps_loaded"):
                    return _run()
            except Exception as e:
                print(f"❌ Error getting destinations: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                    "type": type(e).__name__,
                }

    try:
        return _run()
    except Exception as e:
        print(f"❌ Error getting destinations: {e}")
        return {"status": "error", "message": str(e), "type": type(e).__name__}
