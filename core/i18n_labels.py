# core/i18n_labels.py
# -*- coding: utf-8 -*-
"""
Thin helper around <DATA_FINAL_ROOT>/_i18n/labels.json.
 - Lazy-load with mtime check
 - Simple getters: get_label(), list_aliases(), resolve_alias()
"""

import json
import os
import threading
from typing import Dict, Any

# Thread-safe lazy cache
_LOCK = threading.RLock()
_LABELS_PATH = None            # set by init_labels(data_root)
_LABELS: Dict[str, Any] = {}
_MTIME: float | None = None

SECTIONS = ("places", "buildings", "floors", "destinations", "aliases")
DEFAULT_LANG = "en"


def init_labels(data_root: str, rel_path: str = "_i18n/labels.json") -> None:
    """Set the absolute path to labels.json."""
    global _LABELS_PATH, _LABELS, _MTIME
    with _LOCK:
        _LABELS_PATH = os.path.join(data_root, rel_path)
        _LABELS = {}
        _MTIME = None


def _ensure_loaded() -> None:
    """Load or reload cache if file changed."""
    global _LABELS, _MTIME
    if _LABELS_PATH is None:
        return
    try:
        st = os.stat(_LABELS_PATH)
        mtime = st.st_mtime
    except FileNotFoundError:
        with _LOCK:
            _LABELS = {s: {} for s in SECTIONS}
            _MTIME = None
        return

    with _LOCK:
        if _MTIME is not None and _MTIME == mtime:
            return
        try:
            raw = json.loads(open(_LABELS_PATH, "r", encoding="utf-8").read())
        except Exception:
            raw = {}
        data: Dict[str, Any] = {}
        for s in SECTIONS:
            v = raw.get(s, {})
            data[s] = v if isinstance(v, dict) else {}
        # ensure presence of all sections
        for s in SECTIONS:
            data.setdefault(s, {})
        _LABELS = data
        _MTIME = mtime


def get_label(section: str, key: str, lang: str, fallback: str) -> str:
    """
    Return localized label for (section, key) in `lang`,
    falling back to 'en' then to provided fallback string.
    """
    _ensure_loaded()
    sec = _LABELS.get(section, {})
    entry = sec.get(key, {})
    if isinstance(entry, dict):
        txt = entry.get(lang) or entry.get(DEFAULT_LANG)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    return fallback

def get_all_labels() -> Dict[str, Dict[str, Any]]:
    """
    Return a copy of all labels data structure:
    {
        "places": {id: {"en": "...", "zh": "...", ...}, ...},
        "buildings": {...},
        "floors": {...},
        "destinations": {...},
        "aliases": {lang: {alias: id, ...}, ...}
    }
    """
    _ensure_loaded()
    with _LOCK:
        return {s: dict(v) for s, v in _LABELS.items()}

def list_aliases(lang: str) -> Dict[str, str]:
    """Return aliases map for a language: alias_text -> canonical_id."""
    _ensure_loaded()
    amap = _LABELS.get("aliases", {}).get(lang, {})
    return amap if isinstance(amap, dict) else {}


def resolve_alias(lang: str, text: str) -> str | None:
    """If `text` is an alias in `lang`, return canonical id, else None."""
    amap = list_aliases(lang)
    return amap.get(text)
