# UNav-VIS4ION Alignment Tasks

## Context
Output differences between Modal.com and static GPU are from **wrapper/orchestration layer**, not from unav-core itself.

## Codex Comparison Results
- Test case: `vianys_640_360_langone_elevator.jpeg`, NYU Langone, destination context set to `15_floor`
- Floor-lock issue: 0/5 successful localizations (floor-locked) → 5/5 successful (multifloor)
- Cold-start stabilization: XY error improved from 160.79 px → 58.90 px → 47.92 px

## Changes Implemented

### ✅ 1. Enable Multifloor by Default
- Changed `enable_multifloor` default from `False` to `True` in both `planner()` and `localize_user()` methods
- This ensures building-level multi-floor scope (e.g., 15/16/17 floors) is loaded instead of floor-locked

### ✅ 2. Queue Bucketing by Image Shape
- Added helper functions:
  - `_get_queue_key_for_image_shape()` - generates queue key based on `image.shape[:2]` (e.g., "360x640")
  - `_get_refinement_queue_for_map()` - retrieves queue for specific map_key and queue_key
  - `_update_refinement_queue()` - updates queue for specific map_key and queue_key
- Modified queue structure to be nested: `{best_map_key: {queue_key: {pairs, initial_poses, pps}}}`

### ✅ 3. Cold-Start Multi-Pass Stabilization
- Implemented bootstrap stabilization in both `planner()` and `localize_user()`:
  - Detects cold-start (empty queue)
  - Runs 2 localization passes on empty queue
  - Averages results from both passes (mean of pass2/pass3)
  - Falls back to single pass if only one succeeds
- Added `bootstrap_mode` field: "mean_pass2_pass3", "single_pass", or "none"

### ✅ 4. Add Debug Fields
Added `debug_info` to responses:
- `map_scope`: "building_level_multifloor" or "floor_locked"
- `bootstrap_mode`: stabilization mode used
- `bootstrap_passes`: number of passes run
- `queue_key`: image shape bucket key
- `n_frames`: number of frames in queue
- `top_candidates_count`: number of VPR candidates

## Implementation Details

### Helper Functions (lines 13-42)
```python
def _get_queue_key_for_image_shape(image_shape):
    """Get a queue key based on image shape for bucket-based refinement queue handling."""
    ...

def _get_refinement_queue_for_map(queue_dict, map_key, queue_key):
    """Get the refinement queue for a specific map_key and queue_key (image shape bucket)."""
    ...

def _update_refinement_queue(queue_dict, map_key, queue_key, new_queue_state):
    """Update the refinement queue for a specific map_key and queue_key."""
    ...
```

### Cold-Start Stabilization Logic
- Triggered when `len(refinement_queue) == 0`
- Runs 2 bootstrap passes on empty queue
- Averages XY coordinates and angle from both passes
- Updates queue after each pass for next iteration

## Notes
- Test image: `vianys_640_360_langone_elevator.jpeg`
- Expected floor: `15_floor`
- Reference mean output should be used for comparison
