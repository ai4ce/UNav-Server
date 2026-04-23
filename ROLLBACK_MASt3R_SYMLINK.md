# Rollback: MASt3R Symlink Workaround

> Generated before removing the runtime symlink fix for MASt3R hardcoded paths.
> If the updated `unav` package fails to find DB perspective images after deployment,
> revert `src/modal_functions/unav_v2/logic/init.py` using the code blocks below.

## Context

The `unav` package's old `mast3r_matching_and_pnp()` hardcoded DB image paths like:

```python
db_img_path = f'/mnt/data/UNav-IO/temp/{place}/{building}/{floor}/perspectives/{name}'
```

But our Modal volume mounts data at `/root/UNav-IO`, so the actual temp data lives at
`/root/UNav-IO/mnt/data/UNav-IO/temp/...`.

The workaround below created a runtime symlink inside the Modal container so the
hardcoded `/mnt/data/UNav-IO/...` paths resolved correctly.

## Code to restore (paste back into `init.py`)

### 1. Restore the symlink helper function

Paste this at the top of `init.py`, right after the `from .places import run_get_places` line:

```python
def _setup_mast3r_symlink(data_root: str):
    """
    Create symlink for MASt3R hardcoded paths.

    MASt3R's matcher.py has hardcoded paths pointing to /mnt/data/UNav-IO/temp/
    but our perspectives data is at /root/UNav-IO/mnt/data/UNav-IO/temp/.
    This creates the necessary symlink so MASt3R can find the perspective images.
    """
    import os

    target_path = "/mnt/data/UNav-IO"
    # The perspectives folder is at /root/UNav-IO/mnt/data/UNav-IO/temp/
    source_path = "/root/UNav-IO/mnt/data/UNav-IO"

    # Check if already exists
    if os.path.islink(target_path):
        print(
            f"✅ MASt3R symlink already exists: {target_path} -> {os.readlink(target_path)}"
        )
        return

    # Create parent directories if needed
    parent_dir = os.path.dirname(target_path)
    if not os.path.exists(parent_dir):
        try:
            os.makedirs(parent_dir, exist_ok=True)
            print(f"📁 Created parent directory: {parent_dir}")
        except Exception as e:
            print(f"⚠️ Could not create parent directory {parent_dir}: {e}")

    # Create the symlink
    try:
        os.symlink(source_path, target_path)
        print(f"✅ Created MASt3R symlink: {target_path} -> {source_path}")
    except FileExistsError:
        print(f"⚠️ Path already exists (not a symlink): {target_path}")
    except Exception as e:
        print(f"❌ Failed to create MASt3R symlink: {e}")
```

### 2. Restore the symlink call inside `run_init_cpu_components`

In `run_init_cpu_components`, right after `self.LOCAL_FEATURE_MODEL = "mast3r"`,
re-add the call:

```python
    self.DATA_ROOT = "/root/UNav-IO/data"
    self.FEATURE_MODEL = "DinoV2Salad"
    self.LOCAL_FEATURE_MODEL = "mast3r"

    # Create symlink for MASt3R hardcoded paths
    _setup_mast3r_symlink(self.DATA_ROOT)

    self.PLACES = run_get_places(self)
```

### 3. Remove the new `data_temp_root` config line (if present)

If you previously applied the forward fix, delete this line inside `run_init_cpu_components`:

```python
    self.localizor_config.data_temp_root = "/root/UNav-IO/mnt/data/UNav-IO/temp"
```

## How to apply the rollback

1. Open `src/modal_functions/unav_v2/logic/init.py`.
2. Paste the `_setup_mast3r_symlink` function back in.
3. Re-add `_setup_mast3r_symlink(self.DATA_ROOT)` inside `run_init_cpu_components`.
4. Remove the `self.localizor_config.data_temp_root = ...` line if it exists.
5. Commit and redeploy:
   ```bash
   cd src/modal_functions/unav_v2
   modal deploy unav_modal.py
   ```
