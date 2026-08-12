import os
from typing import List

DEFAULT_UNAV_RAM_MB = 73728
MAX_UNAV_RAM_MB = 98304


def get_scaledown_window() -> int:
    raw_value = os.getenv("UNAV_SCALEDOWN_WINDOW", "300")
    try:
        value = int(raw_value)
        return max(1, value)
    except (TypeError, ValueError):
        print(
            f"⚠️ Invalid UNAV_SCALEDOWN_WINDOW={raw_value!r}; falling back to 300 seconds."
        )
        return 300


def get_gpu_config() -> List[str]:
    raw_value = os.getenv("UNAV_GPU_TYPE", "t4")
    gpu_choice = str(raw_value).strip().lower()

    # Modal docs support GPU shortcodes and "any" for flexible scheduling.
    mapping = {
        "t4": "T4",
        "a10": "A10",
        "a10g": "A10",
        "a100": "A100",
        "h200": "H200",
        "any": "any",
    }

    if gpu_choice not in mapping:
        print(
            f"⚠️ Invalid UNAV_GPU_TYPE={raw_value!r}; expected one of: t4, a10, a100, any. Falling back to t4."
        )
        return ["T4"]

    return [mapping[gpu_choice]]


def get_memory_mb() -> int:
    """
    Resolve UNav memory reservation in MiB for Modal deployment.
    UNAV_RAM_MB is expected to come from a workflow dropdown.
    """
    raw_value = os.getenv("UNAV_RAM_MB", str(DEFAULT_UNAV_RAM_MB))
    try:
        requested_mb = int(raw_value)
        if requested_mb <= 0:
            raise ValueError("must be > 0")
    except (TypeError, ValueError):
        print(
            f"⚠️ Invalid UNAV_RAM_MB={raw_value!r}; falling back to {DEFAULT_UNAV_RAM_MB}."
        )
        return DEFAULT_UNAV_RAM_MB

    max_allowed_mb = MAX_UNAV_RAM_MB
    if requested_mb > max_allowed_mb:
        print(
            f"⚠️ UNAV_RAM_MB={requested_mb} exceeds configured max ({max_allowed_mb}); clamping to {max_allowed_mb}."
        )
        return max_allowed_mb
    return requested_mb


def get_mast3r_candidates() -> int:
    """Max MASt3R candidate images matched per localization. Lower = faster."""
    raw_value = os.getenv("UNAV_MAST3R_CANDIDATES", "5")
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        print(
            f"⚠️ Invalid UNAV_MAST3R_CANDIDATES={raw_value!r}; falling back to 5."
        )
        return 5


def get_mast3r_size() -> int:
    """MASt3R/dust3r input resolution. Lower = faster; clamped to 224..512."""
    raw_value = os.getenv("UNAV_MAST3R_SIZE", "384")
    try:
        return min(max(int(raw_value), 224), 512)
    except (TypeError, ValueError):
        print(f"⚠️ Invalid UNAV_MAST3R_SIZE={raw_value!r}; falling back to 384.")
        return 384


def get_mast3r_early_stop_inliers() -> int:
    """Stop matching more candidates once a map_key reaches this inlier count."""
    raw_value = os.getenv("UNAV_MAST3R_EARLY_STOP_INLIERS", "80")
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        print(
            f"⚠️ Invalid UNAV_MAST3R_EARLY_STOP_INLIERS={raw_value!r}; falling back to 80."
        )
        return 80


def get_vpr_top_k() -> int:
    """VPR candidates retrieved per localization (mast3r caps at 10 upstream)."""
    raw_value = os.getenv("UNAV_VPR_TOP_K", "10")
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        print(f"⚠️ Invalid UNAV_VPR_TOP_K={raw_value!r}; falling back to 10.")
        return 10
