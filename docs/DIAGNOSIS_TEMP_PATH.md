# Diagnosis: Why `branch_with_db_route_configured` Localization Failed — And the Fix

## Symptom (pre-fix)

```
[✓] Loaded COLMAP model for ('New_York_University', 'Langone', '17_floor'): 12996 frames
[✓] Loaded global features for ('New_York_University', 'Langone', '17_floor'): 12996 images
[✓] Loaded transform matrix for ('New_York_University', 'Langone', '17_floor'): shape=(2, 4)
🧪 [LOCALIZER DEBUG] local_feature_model=mast3r, local_extractor=NoneType(callable=False),
                       local_matcher=MASt3RExtractor(callable=True), global_extractor=GlobalExtractors(callable=True)
⏱️ Localization: 28549.41ms
❌ [PLANNER RESULT] status=error, stage=batch_local_matching_and_ransac,
                  reason=No candidates passed local matching + RANSAC.,
                  best_map_key=None, top_candidates_count=10, results_count=0, max_inliers=0
```

VPR retrieval works. Map loads correctly. But every candidate gets rejected
at the local matching + RANSAC step. 0 inliers.

## Root cause

`UNavLocalizer.batch_local_matching_and_ransac()` on this branch has **no
MASt3R dispatch**. When `local_feature_model="mast3r"`, the code still falls
through to the **SuperPoint RANSAC pipeline** (`batch_local_matching_and_ransac`
imported from `unav.localizer.tools.matcher`). MASt3R's local features and
matcher output have a different structure, so the SuperPoint RANSAC rejects
every candidate → 0 inliers → `best_map_key=None`.

The MASt3R-specific matcher (`mast3r_matching_and_pnp`) exists in
`add_temp_config`'s `unav` fork (`rizzojr01/unav-backend-core`) — but
this branch was pinned to the upstream `endeleze/UNav` repo, which
doesn't expose it. Even if it did, there's no code path that calls it.

## The surgical fix (cherry-picked)

Cherry-picked `5a5d7b9` from `add_temp_config`. That commit is the
entire fix — it bundles three things that have to ship together:

1. **`localizer.py` — MASt3R dispatch** (the actual fix). When
   `local_feature_model == "mast3r"`, call
   `mast3r_matching_and_pnp` instead of the SuperPoint RANSAC.

2. **`modal_config.py` — `unav` repo swap**. Switches the
   `pip_install_private_repos` source from
   `endeleze/UNav` to `rizzojr01/unav-backend-core`. This is
   the fork that exposes `mast3r_matching_and_pnp`. The dispatch
   in (1) would `ImportError` without this.

3. **`logic/navigation.py` — save query image to temp path** and
   stash on the localizer so `mast3r_matching_and_pnp` can load it.

The commit landed cleanly as `98b52e7`.

## Why the other 4 candidates were rejected

| Commit | Why skipped |
|---|---|
| `f29abff` Fix data_temp_root / data_final_root | Reverses our `bff19b9` fix. Per the `bff19b9` message, `mast3r_matching_and_pnp` reads `data_roots = [data_temp_root, data_final_root]` and tries each at `{root}/{p}/{b}/{f}/perspectives/{name}`. The default `data_final_root = /root/UNav-IO/data` has no `perspectives/` folder — so `data_final_root` MUST be set to the temp path, not left as `DATA_ROOT`. Our `bff19b9` is correct; the `f29abff` interpretation was wrong. |
| `aa7c9e9` Add `pp=None` kwarg | Modifies instrumentation wrappers (`_install_upstream_instrumentation`) that don't exist on this branch. The dispatch in `5a5d7b9` calls `mast3r_matching_and_pnp` with `pp=None` *implicit* (i.e. doesn't pass `pp` at all, which is the call-site fix that `0c6edec` documents as correct). |
| `0c6edec` Remove `pp=pp` from wrapper | Same — modifies instrumentation wrappers we don't have. |
| `0a48a98` Patch MASt3RExtractor | 100% debug-log code + a one-line `max_nn_dist` value change inside `_override_mast3r_config`, a function that doesn't exist on this branch. Pure debug noise. |

## Final state of the fix on this branch

After cherry-pick + the `bff19b9` already on the branch:

```python
# logic/init.py:83-86
self.localizor_config = self.config.localizer_config
# Configure MASt3R DB image lookup path (perspectives live in temp folder)
self.localizor_config.data_temp_root = "/root/UNav-IO/mnt/data/UNav-IO/temp"
self.localizor_config.data_final_root = "/root/UNav-IO/mnt/data/UNav-IO/temp"
```

```python
# localizer.py:185-209 — when local_feature_model=mast3r:
return mast3r_matching_and_pnp(
    query_img_path=getattr(self, "_current_query_img_path", None),
    candidates_data=candidates_data,
    mast3r_matcher=self.local_matcher,
    colmap_models=self.all_colmap_models,
    max_nn_dist=...,
    min_inliers=...,
    max_candidates=10,
    early_stop_inliers=80,
    data_roots=[r for r in data_roots if r],   # = [data_temp_root, data_final_root]
)
```

```python
# modal_config.py:175 — pull the matching library
"github.com/rizzojr01/unav-backend-core.git",
```

## Self-correction on earlier diagnosis

In the first session of this conversation, I told you the `data_final_root`
override on `init.py:86` was an "over-correction" and removed it. **That
was wrong.** I was reading the code in isolation, not the matching code
path. The `bff19b9` commit message — and the working `add_temp_config`
branch — both show that `data_final_root` MUST be set to the temp path
for `mast3r_matching_and_pnp` to resolve perspective images. The
`docs/BRANCH_DIFF_TEMP_PATH.md` doc I wrote earlier was based on this
mistaken analysis and has been removed.
