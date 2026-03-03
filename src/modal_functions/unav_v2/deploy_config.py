import os
from typing import List

DEFAULT_UNAV_RAM_MB = 73728


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

    Modes:
    - current: use DEFAULT_UNAV_RAM_MB
    - modal_max: use UNAV_MODAL_MAX_RAM_MB
    - custom: use UNAV_CUSTOM_RAM_MB (must be <= UNAV_MODAL_MAX_RAM_MB)
    """
    option_raw = os.getenv("UNAV_RAM_OPTION", "current")
    option = str(option_raw).strip().lower()
    if option not in {"current", "modal_max", "custom"}:
        print(
            f"⚠️ Invalid UNAV_RAM_OPTION={option_raw!r}; expected current|modal_max|custom. Falling back to current."
        )
        option = "current"

    modal_max_raw = os.getenv("UNAV_MODAL_MAX_RAM_MB", str(DEFAULT_UNAV_RAM_MB))
    custom_raw = os.getenv("UNAV_CUSTOM_RAM_MB", str(DEFAULT_UNAV_RAM_MB))

    def _parse_positive_int(value: str, env_name: str, fallback: int) -> int:
        try:
            parsed = int(value)
            if parsed <= 0:
                raise ValueError("must be > 0")
            return parsed
        except (TypeError, ValueError):
            print(
                f"⚠️ Invalid {env_name}={value!r}; falling back to {fallback}."
            )
            return fallback

    modal_max_mb = _parse_positive_int(
        modal_max_raw,
        "UNAV_MODAL_MAX_RAM_MB",
        DEFAULT_UNAV_RAM_MB,
    )
    custom_mb = _parse_positive_int(
        custom_raw,
        "UNAV_CUSTOM_RAM_MB",
        DEFAULT_UNAV_RAM_MB,
    )

    if option == "current":
        return DEFAULT_UNAV_RAM_MB
    if option == "modal_max":
        return modal_max_mb

    if custom_mb > modal_max_mb:
        print(
            f"⚠️ UNAV_CUSTOM_RAM_MB={custom_mb} exceeds UNAV_MODAL_MAX_RAM_MB={modal_max_mb}; clamping to {modal_max_mb}."
        )
        return modal_max_mb
    return custom_mb
