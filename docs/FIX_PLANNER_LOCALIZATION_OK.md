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

## Fixes attempted

### Attempt 1 — Bump GPU (related, in progress 2026-07-28)

- Changed `UNAV_GPU_TYPE` from `t4` → **`a10g`** to rule out GPU OOM during the
  combined localize+plan work.
- Status: **TESTING**. This addresses the "localization died inside planner" shape,
  not necessarily a true localization-ok-but-plan-fails case.

## Next steps

- Reproduce a clean run where `localize_user` returns `status=success` AND `planner`
  returns `status=error`, then capture the exact `stage` / `reason` from
  `[PLANNER RESULT]`.
- Trace the planner path after a successful pose: destination lookup → map/graph
  selection → trajectory → serialization, and log which step fails.
