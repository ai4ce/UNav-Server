# Rules for `integrate_backend_snap_to_route`

1. **No more touching or rebuilding `ai4ce/unav`.** The deployed container
   has a cached `unav` install layer from the branch HEAD (`aa60dc9`).
   Don't re-pin the URL, don't add `force_build=True`, don't try to
   upgrade. The deployed `unav` is the source of truth for what
   `mast3r_matching_and_pnp` accepts.

2. **No more syncing `endeleze` branch.** The image was previously cloning
   `origin/endeleze` to `/root/unav_server_v2/`, but that line has been
   removed (see commit `9ad15bf`). Don't re-push `integrate_backend_snap_to_route`
   to `origin/endeleze` and don't `git checkout endeleze` anywhere in
   the image build.

## Why these rules exist

- The `pp` error in `mast3r_matching_and_pnp()` has been chased across
  multiple cached image layers. The deployed `unav` is what it is;
  working around its signature (instead of trying to upgrade it) is
  the only stable path forward.
- The `endeleze` branch was stale and a dead-code clone of the whole
  UNav-Server repo. Pushing to it adds a second source of truth that
  can diverge from the local working copy.
