# api/trial_api.py
#
# FastAPI router for uploading TrialRecorder archives produced by the iOS
# app's research-logging module. Each trial is a zipped directory containing:
#
#   meta.json          trial metadata (device, src/dst, counts)
#   arkit.ndjson       ARKit pose stream (~30 Hz rows, JSON per line)
#   frames/*.jpg       continuous camera frames (~2 Hz)
#   queries/q_*.jpg    high-quality VPR query captures
#   queries/q_*.json   server responses + ar_t_at_capture
#
# The zip is extracted into
#     <DATA_ROOT>/trials/<user_id>/<trial_id>/
# and is the authoritative source of truth for offline drift / recovery
# analysis.
#
# Note: this endpoint does NOT validate the zip contents beyond basic path
# safety (no absolute paths, no '..' traversal). Trials are research data,
# not user-facing state, and will be audited offline.

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from config import DATA_ROOT
import socket as _socket

def _get_site_id():
    """Auto-detect site from hostname to avoid ID conflicts between servers."""
    h = _socket.gethostname().lower()
    if "unav" in h or "nyu" in h:
        return "nyu"
    elif "mahidol" in h or "thai" in h:
        return "mahidol"
    else:
        return h.split(".")[0]

SITE_ID = _get_site_id()
from api.user_api import decode_access_token
import io
import logging
import os
import re
import time
import zipfile
import threading
from typing import Optional

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

logger = logging.getLogger("unav.api")

# ---------- HuggingFace auto-sync ----------
# Every uploaded trial is automatically mirrored to HuggingFace for
# centralized research data management. Sync runs in a background thread
# so it never blocks the upload response to the client.

HF_TOKEN = os.environ.get("HF_TOKEN", "" + os.environ.get("HF_TOKEN", "") + "")
HF_REPO = os.environ.get("HF_REPO", "NYU-UNav/foresight-trials")

def _sync_trial_to_hf(trial_dir: str, user_id: str, trial_id: str):
    """Background: upload trial files to HuggingFace dataset repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)

        for root, dirs, files in os.walk(trial_dir):
            for fname in files:
                local_path = os.path.join(root, fname)
                rel_path = os.path.relpath(local_path, trial_dir)
                repo_path = f"trials/{SITE_ID}/{user_id}/{trial_id}/{rel_path}"
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=repo_path,
                        repo_id=HF_REPO,
                        repo_type="dataset",
                    )
                except Exception as e:
                    logger.warning(
                        "HF upload failed for %s: %s", repo_path, e
                    )
        logger.info(
            "[HF] synced trial user=%s trial=%s to %s",
            user_id, trial_id, HF_REPO,
        )
    except ImportError:
        logger.warning("[HF] huggingface_hub not installed, skipping sync")
    except Exception as e:
        logger.warning("[HF] sync failed for trial %s: %s", trial_id, e)

# Trial IDs are produced client-side. We accept a conservative character
# class: letters, digits, underscore, dash. Length bounded to keep filesystem
# paths sane.
_TRIAL_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")

# Safety ceiling. Real trials are typically ~240 MB (10-minute walk at 2 Hz
# camera frames). 2 GiB guards against accidental multi-gigabyte uploads.
_MAX_ZIP_BYTES = 2 * 1024 * 1024 * 1024


def _get_user_id(token: str = Depends(oauth2_scheme)) -> str:
    payload = decode_access_token(token)
    return str(payload["id"])


def _safe_member_path(name: str) -> Optional[str]:
    """Return a safe, normalized relative path for a zip member, or None
    if the member name is suspicious (absolute, traversal, null byte)."""
    if not name:
        return None
    if "\x00" in name:
        return None
    # Normalize path separators to POSIX before normpath so Windows-style
    # zips are handled correctly on the Linux host.
    cleaned = name.replace("\\", "/")
    norm = os.path.normpath(cleaned)
    # After normpath: reject absolute paths and anything that walks out of
    # the trial root (e.g. "..", "../foo", "a/../../etc").
    if norm.startswith("/") or norm.startswith(".."):
        return None
    # Any interior traversal component is also unsafe.
    parts = norm.split(os.sep)
    if any(p == ".." for p in parts):
        return None
    return norm


@router.post("/trials/upload")
async def upload_trial(
    trial_id: str = Form(...),
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
):
    """Upload a single TrialRecorder zip archive.

    Request:
        multipart/form-data with fields:
          trial_id:  client-generated trial identifier (form field)
          file:      the zip archive (file field)

    Response:
        {
          "ok": true,
          "trial_id": "...",
          "path": "<DATA_ROOT>/trials/<user_id>/<trial_id>/",
          "files_written": N,
          "zip_bytes": M
        }
    """
    user_id = _get_user_id(token)

    if not _TRIAL_ID_RE.match(trial_id):
        raise HTTPException(status_code=400, detail="invalid trial_id")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > _MAX_ZIP_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")

    out_dir = os.path.join(DATA_ROOT, "trials", SITE_ID, user_id, trial_id)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    n_written = 0
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                safe = _safe_member_path(info.filename)
                if safe is None:
                    logger.warning(
                        "[trial_upload] rejecting member %r in trial %s",
                        info.filename,
                        trial_id,
                    )
                    continue
                target = os.path.join(out_dir, safe)
                target_dir = os.path.dirname(target)
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(65536)
                        if not chunk:
                            break
                        dst.write(chunk)
                n_written += 1
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="not a valid zip archive")

    elapsed = time.time() - t0
    logger.info(
        "[UNav-API] trial_upload user=%s trial=%s files=%d zip_bytes=%d extract_s=%.2f",
        user_id,
        trial_id,
        n_written,
        len(raw),
        elapsed,
    )

    # Background sync to HuggingFace (non-blocking)
    thread = threading.Thread(
        target=_sync_trial_to_hf,
        args=(out_dir, user_id, trial_id),
        daemon=True,
    )
    thread.start()

    return {
        "ok": True,
        "trial_id": trial_id,
        "path": out_dir,
        "files_written": n_written,
        "zip_bytes": len(raw),
        "hf_sync": "started",
    }
