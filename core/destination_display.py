import re


def _is_id_like(text: str | None) -> bool:
    if text is None:
        return True
    value = str(text).strip()
    return value == "" or re.fullmatch(r"\d+", value) is not None


def _humanize_floor(floor: str | None, lang: str) -> str:
    if not floor:
        return "this floor" if lang != "zh" else "本层"
    value = str(floor)
    m = re.fullmatch(r"(\d+)_floor", value)
    if m:
        num = m.group(1)
        return f"{num}楼" if lang == "zh" else f"floor {num}"
    return value.replace("_", " ")


def humanize_destination_name(name: str | None, *, fallback: str | None = None, category: str | None = None, floor: str | None = None, lang: str = "en") -> str:
    primary = str(name).strip() if name is not None else ""
    backup = str(fallback).strip() if fallback is not None else ""

    if not _is_id_like(primary):
        return primary
    if backup and not _is_id_like(backup):
        return backup

    # Only synthesize a friendly label when we actually know the facility type.
    # Otherwise keep the original unique token/id to avoid collapsing many
    # unrelated destinations into the same generic string like
    # "destination on floor 17".
    if not category:
        return primary or backup

    floor_label = _humanize_floor(floor, lang)
    if lang == "zh":
        mapping = {
            "elevator": f"{floor_label}电梯",
            "restroom": f"{floor_label}洗手间",
            "stairs": f"{floor_label}楼梯",
            "exit": f"{floor_label}出口",
            "service_desk": f"{floor_label}服务台",
        }
        return mapping.get(category, primary or backup or f"{floor_label}目的地")

    mapping = {
        "elevator": f"elevator on {floor_label}",
        "restroom": f"restroom on {floor_label}",
        "stairs": f"stairs on {floor_label}",
        "exit": f"exit on {floor_label}",
        "service_desk": f"service desk on {floor_label}",
    }
    return mapping.get(category, primary or backup or f"destination on {floor_label}")
