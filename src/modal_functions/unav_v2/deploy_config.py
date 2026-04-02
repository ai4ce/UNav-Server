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
