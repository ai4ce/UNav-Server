# Localization Performance: ~10s → ~1.5s

How we cut `localize_user` / `planner` localization time from ~10s to
~1.3–1.5s on a warm container (A10 GPU), with no measurable accuracy loss.

## 1. Before (baseline)

Timestamped planner log (Langone / 17_floor, warm container):

| Stage | Time |
|---|---|
| Feature extraction (DinoV2Salad) + VPR + candidate gathering | ~0.2s |
| Image loading (10 pairs @512px, query re-decoded per pair) | ~1.9s |
| **MASt3R batched inference + matching (10 pairs @512px)** | **~7.0s** |
| PnP refine + floorplan transform | ~0.1s |
| **Total localization** | **~9.25s** |

Matching was 100% of the problem; path planning / serialization were <15ms.

## 2. What we did (step by step)

- **Step 1 — Shrink MASt3R input resolution to 384px** (`UNAV_MAST3R_SIZE=384`):
  `_apply_mast3r_tuning()` in `logic/maps.py` patches
  `localizer.local_matcher.config["mast3r_size"]` after construction.
  Compute scales with pixel area: (384/512)² ≈ 56% of the work per forward.
  This alone removes most of the matching cost.

- **Step 2 — Made matching knobs env-tunable at deploy time**:
  `deploy_config.py` getters + `modal_config.py` `Image.env()` bake
  `UNAV_MAST3R_CANDIDATES`, `UNAV_MAST3R_SIZE`,
  `UNAV_MAST3R_EARLY_STOP_INLIERS`, and `UNAV_VPR_TOP_K` into the
  container. `traced_match` reads them at match time — tune without code
  edits, redeploy to apply.

- **Step 3 — Fewer candidates during tuning** (`UNAV_MAST3R_CANDIDATES=5`):
  matched 5 VPR candidates instead of 10, halving the number of pair
  forwards. Accuracy held (338 vs 316 inliers), so the default was later
  restored to 10 candidates — 384px alone delivers most of the speedup.

- **Step 4 — Stop the global fallback localizer from preloading every
  floor** (`logic/init.py`): the global `UNavLocalizer` is only a fallback
  (per-request selective localizers do the real work). It is now built with
  an empty places map, so its background thread no longer eagerly loads
  global features for all places/buildings/floors at container start
  (~50k images of volume IO, ~52s of the first-request cost).

## 3. Results

- **Steady state (warm container, Modal dashboard execution times):**
  - `localize_user`: **~1.25s** (was 9.25s)
  - `planner`: **~1.3s**
- **Accuracy:** same `best_map_key`, `total_inliers` ≥ baseline (338 vs 316),
  consistent pose across repeated calls.
- **First request (cold container):** still dominated by one-time model +
  map loading (GPU init, MASt3R/DinoV2Salad weights, h5 global features,
  lazy COLMAP parse) — separate from matching; mitigated by keeping the
  container warm (`UNAV_SCALEDOWN_WINDOW`).

## 4. How to tune / verify

```bash
cd src/modal_functions/unav_v2
# e.g. more speed: UNAV_MAST3R_CANDIDATES=5; more accuracy: UNAV_MAST3R_SIZE=448
UNAV_MAST3R_SIZE=384 modal deploy -m src.modal_functions.unav_v2.unav_modal
python test_modal_functions.py   # run twice (cold + warm)
```

Compare against the baseline above: `timing["localization"]`, upstream
`timings["match+ransac"]`, `total_inliers`, and `floorplan_pose`.

## Related

- Destinations cold start was reduced separately from 45s → ~3s with a
  CPU-only `DestinationsServer` class (see `unav_modal.py`).
