# Fix: Localization Failure - No Matching Step

## Symptom
Deployment returns `no pose found` with `max_inliers=0` and `results_count=0` (`results_count=0` means `batch_local_matching_and_ransac` returned 0 results). On past failures we still saw the matching step run against DB images; now the pipeline never reaches matching — it skips straight to "no candidates passed".

## Hypothesis
When `local_feature_model == "mast3r"`, the default path in `batch_local_matching_and_ransac` (LightGlue/SuperPoint) is the wrong code path. The MASt3R matcher needs:
- the on-disk query image path
- direct access to DB images via `data_temp_root` / `data_final_root`

Without these, matching produces nothing — `results_count=0`.

## Fixes Tried

### 1. Switch unav-core source to `rizzojr01/unav-backend-core`
- **File**: `src/modal_functions/unav_v2/modal_config.py`
- **Change**: `pip_install_private_repos("github.com/endeleze/UNav.git", ...)` → `pip_install_private_repos("github.com/rizzojr01/unav-backend-core.git", ..., force_build=True)`
- **Why**: The fork exposes the MASt3R matcher (`unav.localizer.tools.matcher.mast3r_matching_and_pnp` and `_resolve_db_image_path`).
- **Status**: Pending verification (not yet deployed).

### 2. Dispatch to MASt3R matcher when `local_feature_model == "mast3r"`
- **File**: `src/modal_functions/unav_v2/localizer.py` (`batch_local_matching_and_ransac` method)
- **Change**: When `local_feature_model == "mast3r"`, dispatch to `mast3r_matching_and_pnp` from `unav.localizer.tools.matcher`, passing:
  - `query_img_path`
  - `candidates_data`
  - `mast3r_matcher=self.local_matcher`
  - `colmap_models=self.all_colmap_models`
  - `max_nn_dist`, `min_inliers`, `max_candidates=10`, `early_stop_inliers=80`
  - `data_roots=[data_temp_root, data_final_root]`
- **Status**: Pending verification.

### 3. Save query image to a temp path and stash it on the localizer
- **File**: `src/modal_functions/unav_v2/logic/navigation.py` (`run_planner`)
- **Change**: When `local_feature_model == "mast3r"`, write the query image to `tempfile.NamedTemporaryFile(suffix=".jpg")` and set `localizer._current_query_img_path`.
- **Status**: Pending verification.

### 4. Debug logging around matching dispatch
- **File**: `src/modal_functions/unav_v2/localizer.py` (inside `localize`, before `batch_local_matching_and_ransac`)
- **Change**: Log `local_feature_model`, `matcher_type`, candidate count, and a sample resolved DB image path (`_resolve_db_image_path`) with `exists` check.
- **Purpose**: On next deploy, confirm whether candidates exist and DB images resolve.
- **Status**: Pending verification.

## Next Deploy Goals
1. Confirm `🧪 [MAST3R DISPATCH]` log appears (means MASt3R code path reached).
2. Confirm `[MAST3R DB LOOKUP] resolved_path=... exists=True` (DB images locatable).
3. Observe `results_count > 0` (matching now actually runs and produces candidates).
4. If still 0 candidates, inspect `top_candidates` for the query — likely image shape / candidate quality issue rather than matcher wiring.

## Environment
- Branch: `add_temp_config`
- GPU: A10
- Test image: `media/vinay_sample.jpeg` (640x360)
- Place/Building/Floor: `New_York_University` / `Langone` / `17_floor`
