# Why localization succeeds but path planning fails

## Goal

Investigate and document the cases where `localize_user` returns a valid pose
(`floorplan_pose` is set, `status=success`) but the `planner` still fails downstream.

The symptom reported: **localization works, planning fails.** This doc tracks the
hypotheses, root causes, and fixes for that gap — i.e. what the planner does with a
good pose that can still break.

## Context: how the two are connected

The planner (in `logic/navigation.py`) calls `localize_user` internally before routing.
So a planner error can originate from localization even when the user's own
`localize_user` call looked fine. Two distinct failure shapes:

1. **Localization genuinely failed inside the planner** (e.g. CUDA OOM at
   `batch_local_matching_and_ransac`) — the pose is `None` and the planner can't route.
   This looks like "planning failed" but is really localization dying in the planner
   context.
2. **Localization succeeded but the planner's own post-localization step breaks** —
   pose is present, but destination resolution, map/graph lookup, trajectory building,
   or serialization fails.

This doc focuses on separating those two and fixing #2 (and correctly labeling #1).

## Hypotheses

- [ ] Planner uses a different code path / params for localization than the standalone
      `localize_user` call (e.g. `enable_multifloor`, `top_k`, refinement queue state),
      causing the in-planner localization to behave differently.
- [ ] Destination resolution fails: `destination_id` not found in the loaded map
      (`destinations_service.py`) even though localization succeeded.
- [ ] Pose is returned but in a coordinate frame / map scope the planner's graph
      builder doesn't accept (floor mismatch, `map_scope` handling).
- [ ] Memory pressure: localization succeeded but left the GPU/process in a state where
      the planner's next heavy step OOMs (see historical T4 OOM note below).

## Root cause (confirmed 2026-07-28)

**Localization succeeded, but `destination_id` does not exist on the floor the user
was localized to.** This is the real "localization ok but planning fails" case.

Observed log:
```
🧭 [PATH PLANNING INPUT] start_key=('New_York_University','Langone','17_floor'),
   target_key=('New_York_University','Langone','17_floor', 88), dest_id_in_target=False
❌ [PATH PLANNING] dest_id=88 not found in target floor.
   Available dest_ids (first 20)=[0..19] (total=52)
❌ [PATH PLANNING FAILED] error=No path found, ...
```

- `17_floor` has **52 destinations (ids `0–51`)**. `destination_id=88` is out of range.
- Destination ids are **per-floor, 0-indexed** — see `destinations_service.py:23`
  (`pf_target.dest_ids`) and `unav/navigator/multifloor.py` (`find_path` builds the
  target node from `(place, building, floor, dest_id)`; a missing id → no graph node
  → `NetworkXNoPath` → `{"error": "No path found"}`).
- The client requested a `destination_id` that belongs to a **different floor** than the
  one the user was localized to (the destination list shown to the user was likely
  fetched for a different floor, or the user's true floor differs from the requested
  `floor` param). Annotated `boundaries.json` sources these dest labels per floor
  (e.g. `Kiosk`, `group_id`-based nodes), so each floor's `dest_ids` are independent.

This is an **input/data mismatch**, not a map-loading or routing bug. The diagnostic
logs added to `run_planner` (`logic/navigation.py`) now make this explicit in Modal.

## Fixes attempted

### Attempt 1 — Bump GPU (related, done 2026-07-28)

- Changed `UNAV_GPU_TYPE` from `t4` → **`a10g`**. The T4 OOM during the in-planner
  localization no longer occurs (logs now show `NVIDIA A10`).
- Status: **DONE** (resolved the localization-died-inside-planner shape).

### Attempt 2 — Detailed path-planning diagnostics (done 2026-07-28)

- Added diagnostic logging + structured error to `run_planner` in
  `logic/navigation.py`:
  - `🧭 [PATH PLANNING INPUT]` logs `start_key`, `target_key`, `snapped_xy`,
    `force_walkable`, and whether source/target maps are loaded + `dest_id_in_target`.
  - `❌ [PATH PLANNING]` warnings when the source map / target map isn't loaded, or
    `dest_id` isn't in the target floor (with sample of available `dest_ids`).
  - `❌ [PATH PLANNING FAILED]` logs the exact `error` + map/dest diagnostics, and
    returns a structured `stage="path_planning"` + `debug` block.
  - Wraps `find_path` in try/except to surface `KeyError`/other exceptions with a
    traceback instead of the opaque outer handler.
- Status: **DONE**. This is what surfaced the real root cause (dest_id 88 missing).

### Attempt 3 — Actionable error + cross-floor fallback (proposed)

Make the planner fail loudly and helpfully when `dest_id` is missing on the localized
floor, instead of a generic `No path found`:

1. **Precise error**: if `dest_id_in_target is False`, return
   `status=error, stage="destination_not_found",
    error="destination_id 88 not found on floor 17_floor (valid 0–51)"` with the
   available dest ids — so the client can correct the request.
2. **Cross-floor lookup (optional)**: when `enable_multifloor` and the `dest_id` is not
   on the localized floor, search other floors in the same building/place for that id
   and route there. This needs the destination list the client uses to be floor-scoped
   so ids don't collide across floors.

## Next steps

- Decide whether to implement Attempt 3 (precise error and/or cross-floor fallback).
- Confirm with the client team that the destination list shown to the user is fetched
  for the **same floor the user is localized to**, so ids match `dest_ids` on that floor.
- Re-run `test_modal_functions.py` with a valid `destination_id` (e.g. `0–51`) on
  `17_floor` to confirm an end-to-end successful plan.
