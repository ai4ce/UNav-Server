from typing import Any

from .logic.places import run_get_places
from .logic.maps import run_ensure_maps_loaded


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
        run_ensure_maps_loaded(
            server=server,
            place=place,
            building=building,
            floor=floor,
            enable_multifloor=enable_multifloor,
        )

        if enable_multifloor:
            places = run_get_places(
                server,
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


def _extract_destinations_from_boundaries(boundaries: dict, floor: str) -> list:
    """Extract destinations from a floor's boundaries.json.

    Mirrors upstream PathFinder._load_data (unav/navigator/pathfinder.py:107-143):
    destination ids are the ordinal of each point shape in the shapes array
    (gaps included), and a point is a destination when group_id == 5.
    """
    destinations = []
    point_idx = 0
    for shape in boundaries.get("shapes", []):
        if shape.get("shape_type") != "point":
            continue
        pts = shape.get("points") or []
        if not pts:
            continue
        if shape.get("group_id") == 5:
            destinations.append(
                {
                    "id": str(point_idx),
                    "name": (shape.get("label") or "").strip(),
                    "xy": (float(pts[0][0]), float(pts[0][1])),
                    "floor": floor,
                }
            )
        point_idx += 1
    return destinations


def get_destinations_list_fs_impl(
    data_root: str,
    floor: str = "6_floor",
    place: str = "New_York_City",
    building: str = "LightHouse",
    enable_multifloor: bool = False,
):
    """Fetch destinations straight from boundaries.json on the volume.

    CPU-only: no torch, no GPU localizer, no FacilityNavigator. Backs the
    lightweight DestinationsServer Modal class so a cold start never waits on
    GPU capacity scheduling. Response shape matches get_destinations_list_impl.
    """
    import json
    import os
    import time
    from types import SimpleNamespace

    _t0 = time.time()
    print(
        f"🎯 [FS] get_destinations_list place={place!r} building={building!r} "
        f"floor={floor!r} enable_multifloor={enable_multifloor}"
    )
    print(f"📁 [FS] data_root={data_root}")

    def _read_floor(floor_name: str) -> list:
        path = os.path.join(data_root, place, building, floor_name, "boundaries.json")
        if not os.path.exists(path):
            print(
                f"⚠️ [FS] Skipping {place}/{building}/{floor_name}: missing boundaries.json"
            )
            return []
        with open(path) as f:
            boundaries = json.load(f)
        dests = _extract_destinations_from_boundaries(boundaries, floor_name)
        print(f"🏷️ [FS] {place}/{building}/{floor_name}: {len(dests)} destinations")
        return dests

    if enable_multifloor:
        floors = (
            run_get_places(
                SimpleNamespace(DATA_ROOT=data_root),
                target_place=place,
                target_building=building,
                enable_multifloor=True,
            )
            .get(place, {})
            .get(building, [])
        )
        if not floors:
            raise ValueError(
                f"No floors found for place='{place}', building='{building}'"
            )

        print(f"🏢 [FS] Aggregating {len(floors)} floor(s): {floors}")
        destinations = []
        for floor_name in floors:
            destinations.extend(_read_floor(floor_name))
    else:
        path = os.path.join(data_root, place, building, floor, "boundaries.json")
        if not os.path.exists(path):
            raise ValueError(
                f"No boundaries.json found for {place}/{building}/{floor}"
            )
        destinations = _read_floor(floor)

    _elapsed_ms = (time.time() - _t0) * 1000
    print(f"✅ [FS] Found {len(destinations)} destinations in {_elapsed_ms:.0f}ms")
    return {"destinations": destinations}
