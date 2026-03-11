# UNav-VIS4ION Alignment Tasks

## Context
Output differences between Modal.com and static GPU are from **wrapper/orchestration layer**, not from unav-core itself.

## Root Cause Analysis

### Why unav-server (static GPU) works better:
1. **Map loading**: Loads ALL floors at startup (`localizer.load_maps_and_features()`)
2. **Session in memory across requests persistence**: Queue persists
3. **Queue works**: Subsequent calls use refinement queue → better accuracy

### Why Modal fails:
1. **Floor-locked maps**: Only loads target floor, not all floors
2. **No session persistence**: `enable_memory_snapshot=False`, each call hits different container
3. **Queue doesn't work**: Session/queue lost on cold-start = **ALWAYS cold start**

This is why X-value drifts - every single localization is cold-start!

## Codex Comparison Results
- Test case: `vianys_640_360_langone_elevator.jpeg`, NYU Langone, destination context set to `15_floor`
- Floor-lock issue: 0/5 successful localizations (floor-locked) → 5/5 successful (multifloor)
- Cold-start stabilization: XY error improved from 160.79 px → 58.90 px → 47.92 px

---

## Changes Implemented (Chronological)

### 1. Enable Multifloor by Default
- **Changed**: `enable_multifloor` default from `False` to `True` in both `planner()` and `localize_user()`
- **Why**: Matches unav-server behavior - loads all floors for the building instead of just target floor
- **Impact**: Fixes 0/5 → 5/5 success rate

### 2. Queue Bucketing by Image Shape
- **Added** helper functions:
  - `_get_queue_key_for_image_shape()` - generates queue key based on `image.shape[:2]` (e.g., "360x640")
  - `_get_refinement_queue_for_map()` - retrieves queue for specific map_key and queue_key
  - `_update_refinement_queue()` - updates queue for specific map_key and queue_key
- **Modified** queue structure to be nested: `{best_map_key: {queue_key: {pairs, initial_poses, pps}}}`
- **Note**: Less critical since queue doesn't persist in serverless anyway

### 3. Cold-Start Multi-Pass Stabilization (v1 - 2 passes)
- **Initial**: Ran 2 localization passes on cold-start, averaged results
- **Why**: Since queue doesn't work in serverless, each request needs self-correction

### 4. Cold-Start Multi-Pass Stabilization (v2 - 3 passes)
- **Changed**: Upgraded from 2 passes to 3 passes
- **Updated**: bootstrap_mode label from "mean_pass2_pass3" to "mean_all_passes"
- **Why**: Better averaging with more samples, diminishing returns after 3 but still improved

### 5. Add Debug Fields
Added `debug_info` to responses:
- `map_scope`: "building_level_multifloor" or "floor_locked"
- `bootstrap_mode`: "mean_all_passes", "single_pass", or "none"
- `bootstrap_passes`: number of passes run
- `queue_key`: image shape bucket key
- `n_frames`: number of frames in queue
- `top_candidates_count`: number of VPR candidates

---

## Technical Details

### Helper Functions (lines 13-42)
```python
def _get_queue_key_for_image_shape(image_shape):
    """Get a queue key based on image shape for bucket-based refinement queue handling."""
    if image_shape is None:
        return "default"
    h, w = image_shape[:2]
    return f"{h}x{w}"

def _get_refinement_queue_for_map(queue_dict, map_key, queue_key):
    """Get the refinement queue for a specific map_key and queue_key."""
    ...

def _update_refinement_queue(queue_dict, map_key, queue_key, new_queue_state):
    """Update the refinement queue for a specific map_key and queue_key."""
    ...
```

### Cold-Start Stabilization Logic
- Triggered when `len(refinement_queue) == 0` (always in serverless)
- Runs 3 bootstrap passes on empty queue
- Averages XY coordinates and angle from all 3 passes
- Updates queue after each pass for next iteration
- Falls back to single pass if fewer than 2 succeed

### Code Locations
- `planner()`: lines ~1470-1520
- `localize_user()`: lines ~1902-1952

---

## Expected Improvements

| Metric | Before | After |
|--------|--------|-------|
| Success rate (floor-locked) | 0/5 | 5/5 |
| XY error (cold-start) | ~160px | ~50px |
| Map scope | floor-locked | building-level |

---

## Future Considerations

1. **Enable memory snapshots**: Could persist queue across cold-starts (but adds ~5-10s restore time)
2. **Client-side queue**: Pass queue with each request
3. **External storage**: Redis for queue persistence
4. **top_k optimization**: Experiment with different top_k values:
   - Default (None) uses config value (~10-20)
   - Lower top_k = faster but fewer candidates
   - Higher top_k = slower but more candidates to match against
   - Consider making it dynamic based on image quality

---

## Notes
- Test image: `vianys_640_360_langone_elevator.jpeg`
- Expected floor: `15_floor`
- Reference mean output should be used for comparison
