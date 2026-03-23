ALLOWED_PREFERENCE_KEYS = {
    "language",
    "unit",
    "preferred_audio_mode",
    "guidance_tempo_multiplier",
    "countdown_enabled",
    "haptic_level",
    "verbosity",
}


def sanitize_preference_patch(patch: dict) -> dict:
    cleaned = {k: v for k, v in patch.items() if k in ALLOWED_PREFERENCE_KEYS and v is not None}
    tempo = cleaned.get("guidance_tempo_multiplier")
    if tempo is not None:
        cleaned["guidance_tempo_multiplier"] = min(1.4, max(0.6, float(tempo)))
    return cleaned
