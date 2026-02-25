import os
from typing import List


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
    raw_value = os.getenv("UNAV_GPU_TYPE", "a10")
    gpu_choice = str(raw_value).strip().lower()

    # Modal docs support GPU shortcodes and "any" for flexible scheduling.
    mapping = {
        "t4": "T4",
        "a10": "A10",
        "a10g": "A10",
        "any": "any",
    }

    if gpu_choice not in mapping:
        print(
            f"⚠️ Invalid UNAV_GPU_TYPE={raw_value!r}; expected one of: t4, a10, any. Falling back to a10."
        )
        return ["A10"]

    return [mapping[gpu_choice]]
