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
import hashlib
import shutil
import socket as _socket
import tempfile

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


def _count_files(root: str) -> int:
    n = 0
    for _, _, files in os.walk(root):
        n += len(files)
    return n


def _extract_publish_and_sync(
    staged_zip: str, out_dir: str, user_id: str, trial_id: str
) -> None:
    """Background worker: extract a staged zip atomically, publish it as the
    final trial directory, HF-sync, and only then drop the staged zip.

    On failure the staged zip is left on disk so it can be retried (either by
    the next server start or by a manual recover_pending hit)."""
    tmp_extract_dir = out_dir + ".tmp"
    n_written = 0
    try:
        if os.path.exists(tmp_extract_dir):
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
        os.makedirs(tmp_extract_dir, exist_ok=True)

        t0 = time.time()
        with zipfile.ZipFile(staged_zip, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                safe = _safe_member_path(info.filename)
                if safe is None:
                    logger.warning(
                        "[trial_upload] rejecting member %r in trial %s",
                        info.filename, trial_id,
                    )
                    continue
                target = os.path.join(tmp_extract_dir, safe)
                target_dir = os.path.dirname(target)
                if target_dir:
                    os.makedirs(target_dir, exist_ok=True)
                with zf.open(info, "r") as src, open(target, "wb") as dst:
                    while True:
                        piece = src.read(65536)
                        if not piece:
                            break
                        dst.write(piece)
                n_written += 1

        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        try:
            os.rename(tmp_extract_dir, out_dir)
        except OSError:
            # Race: another worker already published this trial_id. Drop
            # ours and treat as success.
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
            logger.info(
                "[UNav-API] async extract race-loser user=%s trial=%s",
                user_id, trial_id,
            )

        # Published successfully — staged zip can go.
        try:
            os.unlink(staged_zip)
        except OSError:
            pass

        logger.info(
            "[UNav-API] async extract OK user=%s trial=%s files=%d extract_s=%.2f",
            user_id, trial_id, n_written, time.time() - t0,
        )

        try:
            _append_attempt_log({
                "ts": _datetime.utcnow().isoformat() + "Z",
                "user_id": user_id,
                "trial_id": trial_id,
                "stage": "server_extracted",
                "zip_bytes": 0,
                "error": "",
            })
        except Exception:
            pass

        try:
            _sync_trial_to_hf(out_dir, user_id, trial_id)
        except Exception as e:
            logger.warning("[HF] sync failed for %s: %s", trial_id, e)
    except Exception as e:
        logger.error(
            "[UNav-API] async extract FAILED user=%s trial=%s err=%s",
            user_id, trial_id, e,
        )
        if os.path.exists(tmp_extract_dir):
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)
        try:
            _append_attempt_log({
                "ts": _datetime.utcnow().isoformat() + "Z",
                "user_id": user_id,
                "trial_id": trial_id,
                "stage": "server_extract_failed",
                "zip_bytes": (
                    os.path.getsize(staged_zip)
                    if os.path.exists(staged_zip)
                    else 0
                ),
                "error": str(e),
            })
        except Exception:
            pass


def _staged_zip_path(staging_root: str, trial_id: str) -> str:
    """Random suffix prevents collision when the same trial_id is uploaded
    twice concurrently (e.g. client retried while the original was still in
    flight)."""
    fd, p = tempfile.mkstemp(
        prefix=f"{trial_id}__", suffix=".zip", dir=staging_root,
    )
    os.close(fd)
    return p


def _trial_id_from_staged(filename: str) -> Optional[str]:
    """Inverse of `_staged_zip_path`: extract the trial_id from a staged zip
    filename (`<trial_id>__<rand>.zip`). Returns None if the filename does
    not match the convention."""
    if not filename.endswith(".zip"):
        return None
    base = filename[:-4]
    if "__" not in base:
        return None
    tid = base.split("__", 1)[0]
    if not _TRIAL_ID_RE.match(tid):
        return None
    return tid


def _recover_pending_uploads() -> None:
    """At server start, scan every user's `.uploading/` directory for:

      1. staged single-shot zips whose extraction never published (server
         crashed mid-extract); re-run extraction;
      2. chunked-upload directories where every chunk has already arrived
         (server crashed before assembly completed); assemble + extract.

    Idempotent: skips anything whose target trial is already extracted."""
    trials_root = os.path.join(DATA_ROOT, "trials")
    if not os.path.isdir(trials_root):
        return
    n_resumed = 0
    n_chunked_resumed = 0
    for site in os.listdir(trials_root):
        site_dir = os.path.join(trials_root, site)
        if not os.path.isdir(site_dir):
            continue
        for user_id in os.listdir(site_dir):
            user_dir = os.path.join(site_dir, user_id)
            staging = os.path.join(user_dir, ".uploading")
            if not os.path.isdir(staging):
                continue
            for fn in os.listdir(staging):
                full = os.path.join(staging, fn)

                # (1) Single-shot staged zips
                if os.path.isfile(full):
                    if fn.endswith(".part"):
                        # Body never finished streaming — unrecoverable.
                        try:
                            os.unlink(full)
                        except OSError:
                            pass
                        continue
                    trial_id = _trial_id_from_staged(fn)
                    if trial_id is None:
                        continue
                    out_dir = os.path.join(user_dir, trial_id)
                    if os.path.exists(os.path.join(out_dir, "meta.json")):
                        try:
                            os.unlink(full)
                        except OSError:
                            pass
                        continue
                    logger.info(
                        "[UNav-API] resuming pending extract user=%s trial=%s zip=%s",
                        user_id, trial_id, full,
                    )
                    threading.Thread(
                        target=_extract_publish_and_sync,
                        args=(full, out_dir, user_id, trial_id),
                        daemon=True,
                    ).start()
                    n_resumed += 1

                # (2) Chunked-upload state directories
                elif os.path.isdir(full) and fn.endswith(".chunks"):
                    trial_id = fn[: -len(".chunks")]
                    if not _TRIAL_ID_RE.match(trial_id):
                        continue
                    out_dir = os.path.join(user_dir, trial_id)
                    if os.path.exists(os.path.join(out_dir, "meta.json")):
                        # Already published; the chunks dir is stale.
                        shutil.rmtree(full, ignore_errors=True)
                        continue
                    manifest = _read_chunks_manifest(full)
                    if manifest is None:
                        continue
                    staged = _assemble_chunks_if_complete(
                        user_id, trial_id, full, manifest,
                    )
                    if staged is None:
                        # Not yet complete — wait for client to send the rest.
                        continue
                    logger.info(
                        "[UNav-API] resuming chunked-extract user=%s trial=%s",
                        user_id, trial_id,
                    )
                    threading.Thread(
                        target=_extract_publish_and_sync,
                        args=(staged, out_dir, user_id, trial_id),
                        daemon=True,
                    ).start()
                    n_chunked_resumed += 1

    if n_resumed or n_chunked_resumed:
        logger.info(
            "[UNav-API] startup recovery: %d single-shot, %d chunked",
            n_resumed, n_chunked_resumed,
        )


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
          "path": "<DATA_ROOT>/trials/<site>/<user_id>/<trial_id>/",
          "files_written": N,        # number of files in the received zip
          "zip_bytes": M,
          "sha1": "<hex>",
          "idempotent": false,
          "extraction": "done" | "deferred" | "skipped",
        }

    Reliability behavior:
      - **Fast-ACK**: returns 200 as soon as the request body is durably on
        disk in `.uploading/`, before extraction. iOS URLSession's idle
        timeout (~60s) only fires when no bytes flow on the connection;
        completing the response right after upload prevents the timeout
        from firing during server-side extract.
      - **Idempotent**: a trial that's already extracted (meta.json present)
        short-circuits to a fast 200 — clients can safely retry after a
        dropped connection without re-uploading hundreds of MB.
      - **Atomic**: extraction goes into <trial_id>.tmp/ and is renamed onto
        <trial_id>/ only after every member lands. A crash mid-extract
        leaves the staged zip in `.uploading/` for the startup recovery
        hook to retry.
      - **Crash-safe**: the staged zip survives until extraction publishes
        successfully. On server start, `_recover_pending_uploads` re-extracts
        any orphans.
    """
    user_id = _get_user_id(token)

    if not _TRIAL_ID_RE.match(trial_id):
        raise HTTPException(status_code=400, detail="invalid trial_id")

    out_dir = os.path.join(DATA_ROOT, "trials", SITE_ID, user_id, trial_id)
    meta_path = os.path.join(out_dir, "meta.json")

    # Idempotency: a previous attempt already landed.
    if os.path.exists(meta_path):
        n_existing = _count_files(out_dir)
        logger.info(
            "[UNav-API] trial_upload IDEMPOTENT user=%s trial=%s files=%d",
            user_id, trial_id, n_existing,
        )
        return {
            "ok": True,
            "trial_id": trial_id,
            "path": out_dir,
            "files_written": n_existing,
            "zip_bytes": 0,
            "sha1": "",
            "idempotent": True,
            "extraction": "skipped",
        }

    user_root = os.path.join(DATA_ROOT, "trials", SITE_ID, user_id)
    staging_root = os.path.join(user_root, ".uploading")
    os.makedirs(staging_root, exist_ok=True)

    staged_zip_path = _staged_zip_path(staging_root, trial_id)
    part_path = staged_zip_path + ".part"

    sha = hashlib.sha1()
    n_bytes = 0

    try:
        # Stream body to a `.part` file in 1 MiB chunks. Bounded memory,
        # incremental size cap, no whole-file slurp.
        with open(part_path, "wb") as out:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                if n_bytes + len(chunk) > _MAX_ZIP_BYTES:
                    raise HTTPException(
                        status_code=413, detail="upload too large",
                    )
                out.write(chunk)
                sha.update(chunk)
                n_bytes += len(chunk)

        if n_bytes == 0:
            raise HTTPException(status_code=400, detail="empty upload")

        # Promote `.part` → `.zip` atomically. Only `.zip` files are picked
        # up by the recovery scanner.
        os.rename(part_path, staged_zip_path)

        # Cheap probe: read the zip's central directory for member count and
        # validate it isn't corrupt before we tell the client we got it.
        try:
            with zipfile.ZipFile(staged_zip_path, "r") as zf:
                n_files = sum(1 for info in zf.infolist() if not info.is_dir())
        except zipfile.BadZipFile:
            try:
                os.unlink(staged_zip_path)
            except OSError:
                pass
            raise HTTPException(
                status_code=400, detail="not a valid zip archive",
            )
    except HTTPException:
        try:
            _append_attempt_log({
                "ts": _datetime.utcnow().isoformat() + "Z",
                "user_id": user_id,
                "trial_id": trial_id,
                "stage": "server_failed",
                "zip_bytes": n_bytes,
                "error": "see server log",
            })
        except Exception:
            pass
        for p in (part_path, staged_zip_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        raise
    except Exception:
        for p in (part_path, staged_zip_path):
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        raise

    # Body durably on disk + zip is well-formed. Hand off extraction to a
    # background worker and return 200 immediately — this is the
    # iOS-timeout fix.
    threading.Thread(
        target=_extract_publish_and_sync,
        args=(staged_zip_path, out_dir, user_id, trial_id),
        daemon=True,
    ).start()

    logger.info(
        "[UNav-API] trial_upload ACCEPTED user=%s trial=%s files=%d zip_bytes=%d sha1=%s",
        user_id, trial_id, n_files, n_bytes, sha.hexdigest()[:10],
    )

    return {
        "ok": True,
        "trial_id": trial_id,
        "path": out_dir,
        "files_written": n_files,
        "zip_bytes": n_bytes,
        "sha1": sha.hexdigest(),
        "idempotent": False,
        "extraction": "deferred",
        "hf_sync": "deferred",
    }


@router.get("/trials/exists")
async def trial_exists(
    trial_id: str,
    token: str = Depends(oauth2_scheme),
):
    """Return whether a fully-extracted trial already lives on the server.

    Used by clients whose previous /trials/upload response was lost (e.g.
    iOS URLSession timed out after the server had already finished writing).
    Cheaper than re-uploading the zip just to find out."""
    user_id = _get_user_id(token)
    if not _TRIAL_ID_RE.match(trial_id):
        raise HTTPException(status_code=400, detail="invalid trial_id")
    out_dir = os.path.join(DATA_ROOT, "trials", SITE_ID, user_id, trial_id)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path):
        return {
            "exists": True,
            "trial_id": trial_id,
            "path": out_dir,
            "files_written": _count_files(out_dir),
        }
    return {"exists": False, "trial_id": trial_id, "path": None}


# ---------- Chunked upload ----------
#
# Single-shot `/trials/upload` becomes unreliable when the zip is large
# enough that any one packet drop kills the whole upload. Chunked upload
# splits the zip into N pieces (typically ~5 MiB each), uploads them
# independently, and assembles them server-side.
#
# Per-chunk request: `POST /trials/upload_chunk` with form fields
#   trial_id, chunk_idx, chunk_total, sha1_full, size_full, file=<bytes>.
#
# Status query: `GET /trials/chunk_status?trial_id=...` returns which
# chunk indices have already landed; clients use this on resume to skip
# re-sending pieces the server already has.
#
# Server-side state (per pending trial):
#   <user_root>/.uploading/<trial_id>.chunks/
#     manifest.json    {chunk_total, sha1_full, size_full, last_seen_at}
#     chunk_000000     raw chunk bytes
#     chunk_000001     ...
#
# When all chunks land, the manifest's sha1_full is verified and the
# assembled zip is handed off to `_extract_publish_and_sync` — exactly the
# same crash-safe extract pipeline used by single-shot uploads.

# Cap any single chunk at 32 MiB. A reasonable client uses ~5 MiB chunks;
# this is just a safety ceiling against runaway uploads.
_MAX_CHUNK_BYTES = 32 * 1024 * 1024


def _chunks_dir(user_id: str, trial_id: str) -> str:
    return os.path.join(
        DATA_ROOT, "trials", SITE_ID, user_id, ".uploading",
        f"{trial_id}.chunks",
    )


def _read_chunks_manifest(chunks_dir: str) -> Optional[dict]:
    p = os.path.join(chunks_dir, "manifest.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


def _write_chunks_manifest(chunks_dir: str, manifest: dict) -> None:
    p = os.path.join(chunks_dir, "manifest.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(manifest, f)
    os.replace(tmp, p)


def _list_received_chunks(chunks_dir: str) -> "list[int]":
    if not os.path.isdir(chunks_dir):
        return []
    out = []
    for fn in os.listdir(chunks_dir):
        if not fn.startswith("chunk_"):
            continue
        try:
            idx = int(fn[len("chunk_"):])
        except ValueError:
            continue
        out.append(idx)
    out.sort()
    return out


def _chunk_path(chunks_dir: str, idx: int) -> str:
    return os.path.join(chunks_dir, f"chunk_{idx:06d}")


def _assemble_chunks_if_complete(
    user_id: str, trial_id: str, chunks_dir: str, manifest: dict,
) -> Optional[str]:
    """Concatenate every chunk into a staged zip if all chunks are present
    and the SHA-1 matches the manifest. Returns the path to the assembled
    zip on success (caller hands it off to extract), or None if not yet
    complete or if assembly failed."""
    received = _list_received_chunks(chunks_dir)
    chunk_total = int(manifest.get("chunk_total", 0))
    if chunk_total <= 0 or len(received) < chunk_total:
        return None
    # All chunks present; verify count and indices are exactly 0..N-1.
    if received != list(range(chunk_total)):
        logger.warning(
            "[trial_upload_chunk] inconsistent chunk set for trial=%s: %r",
            trial_id, received,
        )
        return None

    user_root = os.path.join(DATA_ROOT, "trials", SITE_ID, user_id)
    staging_root = os.path.join(user_root, ".uploading")
    os.makedirs(staging_root, exist_ok=True)
    fd, staged_zip = tempfile.mkstemp(
        prefix=f"{trial_id}__assembled.", suffix=".zip", dir=staging_root,
    )
    os.close(fd)

    sha = hashlib.sha1()
    n_bytes = 0
    try:
        with open(staged_zip, "wb") as out:
            for idx in range(chunk_total):
                with open(_chunk_path(chunks_dir, idx), "rb") as inp:
                    while True:
                        piece = inp.read(1 << 20)
                        if not piece:
                            break
                        out.write(piece)
                        sha.update(piece)
                        n_bytes += len(piece)
    except Exception as e:
        logger.error(
            "[trial_upload_chunk] assembly read failed trial=%s: %s",
            trial_id, e,
        )
        try:
            os.unlink(staged_zip)
        except OSError:
            pass
        return None

    expected_sha = str(manifest.get("sha1_full", "")).lower()
    expected_size = int(manifest.get("size_full", 0))
    actual_sha = sha.hexdigest()
    if expected_sha and actual_sha != expected_sha:
        logger.error(
            "[trial_upload_chunk] sha1 mismatch trial=%s expected=%s got=%s",
            trial_id, expected_sha, actual_sha,
        )
        try:
            os.unlink(staged_zip)
        except OSError:
            pass
        # Wipe the chunks dir so the client can start over cleanly. (If
        # the client retries with the same chunks we'd just loop here.)
        shutil.rmtree(chunks_dir, ignore_errors=True)
        return None
    if expected_size and n_bytes != expected_size:
        logger.error(
            "[trial_upload_chunk] size mismatch trial=%s expected=%d got=%d",
            trial_id, expected_size, n_bytes,
        )
        try:
            os.unlink(staged_zip)
        except OSError:
            pass
        return None

    # Sanity-check the assembled zip is well-formed before declaring victory.
    try:
        with zipfile.ZipFile(staged_zip, "r") as zf:
            n_files = sum(1 for info in zf.infolist() if not info.is_dir())
    except zipfile.BadZipFile as e:
        logger.error(
            "[trial_upload_chunk] assembled zip is corrupt trial=%s: %s",
            trial_id, e,
        )
        try:
            os.unlink(staged_zip)
        except OSError:
            pass
        shutil.rmtree(chunks_dir, ignore_errors=True)
        return None

    logger.info(
        "[trial_upload_chunk] assembled trial=%s chunks=%d bytes=%d files=%d",
        trial_id, chunk_total, n_bytes, n_files,
    )

    # Clear the chunks dir — the assembled zip is the canonical state from
    # here on, and `_extract_publish_and_sync` deletes it after publish.
    shutil.rmtree(chunks_dir, ignore_errors=True)
    return staged_zip


@router.get("/trials/chunk_status")
async def chunk_status(
    trial_id: str,
    token: str = Depends(oauth2_scheme),
):
    """Return which chunks the server already has for a pending chunked
    upload. Lets a client resume after a crash / app-restart by re-sending
    only the missing indices.

    Response:
        {
          "trial_id": "...",
          "chunks_received": [0, 1, 2, 5],   # sorted
          "chunk_total": N | null,
          "completed": false | true,         # true if trial is already extracted
          "files_written": M,                # only when completed
        }
    """
    user_id = _get_user_id(token)
    if not _TRIAL_ID_RE.match(trial_id):
        raise HTTPException(status_code=400, detail="invalid trial_id")

    out_dir = os.path.join(DATA_ROOT, "trials", SITE_ID, user_id, trial_id)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        return {
            "trial_id": trial_id,
            "chunks_received": [],
            "chunk_total": None,
            "completed": True,
            "files_written": _count_files(out_dir),
        }

    chunks_dir = _chunks_dir(user_id, trial_id)
    manifest = _read_chunks_manifest(chunks_dir) or {}
    return {
        "trial_id": trial_id,
        "chunks_received": _list_received_chunks(chunks_dir),
        "chunk_total": manifest.get("chunk_total"),
        "completed": False,
    }


@router.post("/trials/upload_chunk")
async def upload_chunk(
    trial_id: str = Form(...),
    chunk_idx: int = Form(...),
    chunk_total: int = Form(...),
    sha1_full: str = Form(...),
    size_full: int = Form(...),
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
):
    """Receive one chunk of a chunked-upload trial. The first chunk
    establishes the manifest (chunk_total, sha1_full, size_full); later
    chunks must agree. When every index 0..chunk_total-1 has landed, the
    server assembles the zip in-place, verifies SHA-1 against the manifest,
    and runs the same `_extract_publish_and_sync` pipeline as single-shot
    uploads.

    Idempotent in two ways:
      - re-sending the same chunk index just overwrites it (chunks are
        addressed by index, not appended)
      - posting any chunk after the trial has already been published
        returns `completed: true` without doing anything

    Response:
        {
          "ok": true,
          "trial_id": "...",
          "chunk_idx": N,
          "chunks_received": [...],
          "chunk_total": M,
          "completed": true|false,           # true once assembly finished
          "extraction": "deferred"|"skipped",
          "files_written": K,                 # present when completed
        }
    """
    user_id = _get_user_id(token)
    if not _TRIAL_ID_RE.match(trial_id):
        raise HTTPException(status_code=400, detail="invalid trial_id")
    if chunk_total <= 0:
        raise HTTPException(status_code=400, detail="invalid chunk_total")
    if chunk_idx < 0 or chunk_idx >= chunk_total:
        raise HTTPException(
            status_code=400, detail="chunk_idx out of range",
        )
    if size_full <= 0 or size_full > _MAX_ZIP_BYTES:
        raise HTTPException(status_code=413, detail="size_full out of range")

    sha1_full = sha1_full.lower().strip()
    if len(sha1_full) != 40 or any(c not in "0123456789abcdef" for c in sha1_full):
        raise HTTPException(status_code=400, detail="malformed sha1_full")

    out_dir = os.path.join(DATA_ROOT, "trials", SITE_ID, user_id, trial_id)
    if os.path.exists(os.path.join(out_dir, "meta.json")):
        # Trial already published — chunk is moot. Tell the client we're
        # already done so they can stop sending.
        return {
            "ok": True,
            "trial_id": trial_id,
            "chunk_idx": chunk_idx,
            "chunks_received": [],
            "chunk_total": chunk_total,
            "completed": True,
            "extraction": "skipped",
            "files_written": _count_files(out_dir),
        }

    chunks_dir = _chunks_dir(user_id, trial_id)
    os.makedirs(chunks_dir, exist_ok=True)

    # Manifest: created on first chunk, validated on every subsequent chunk.
    manifest = _read_chunks_manifest(chunks_dir)
    if manifest is None:
        manifest = {
            "trial_id": trial_id,
            "chunk_total": chunk_total,
            "sha1_full": sha1_full,
            "size_full": size_full,
            "created_at": _datetime.utcnow().isoformat() + "Z",
        }
        _write_chunks_manifest(chunks_dir, manifest)
    else:
        if (
            int(manifest.get("chunk_total", 0)) != chunk_total
            or str(manifest.get("sha1_full", "")).lower() != sha1_full
            or int(manifest.get("size_full", 0)) != size_full
        ):
            # The client switched plans mid-upload (e.g. re-zipped with a
            # different chunk size or a different file). Refuse with 409
            # so the client knows to call /chunk_status, decide whether to
            # wipe (DELETE/POST), and start over.
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "manifest mismatch",
                    "stored": {
                        "chunk_total": manifest.get("chunk_total"),
                        "sha1_full": manifest.get("sha1_full"),
                        "size_full": manifest.get("size_full"),
                    },
                    "got": {
                        "chunk_total": chunk_total,
                        "sha1_full": sha1_full,
                        "size_full": size_full,
                    },
                },
            )

    # Stream chunk bytes to a `.part` file, then atomic rename. Resending
    # the same idx just overwrites the previous copy.
    chunk_dst = _chunk_path(chunks_dir, chunk_idx)
    chunk_part = chunk_dst + ".part"
    n_bytes = 0
    try:
        with open(chunk_part, "wb") as out:
            while True:
                piece = await file.read(1 << 20)
                if not piece:
                    break
                if n_bytes + len(piece) > _MAX_CHUNK_BYTES:
                    raise HTTPException(
                        status_code=413, detail="chunk too large",
                    )
                out.write(piece)
                n_bytes += len(piece)
        if n_bytes == 0:
            raise HTTPException(status_code=400, detail="empty chunk")
        os.replace(chunk_part, chunk_dst)
    except HTTPException:
        if os.path.exists(chunk_part):
            try:
                os.unlink(chunk_part)
            except OSError:
                pass
        raise
    except Exception:
        if os.path.exists(chunk_part):
            try:
                os.unlink(chunk_part)
            except OSError:
                pass
        raise

    # Touch last_seen so we can later age out stale uploads.
    manifest["last_seen_at"] = _datetime.utcnow().isoformat() + "Z"
    _write_chunks_manifest(chunks_dir, manifest)

    received = _list_received_chunks(chunks_dir)
    logger.info(
        "[trial_upload_chunk] user=%s trial=%s chunk=%d/%d size=%d received=%d/%d",
        user_id, trial_id, chunk_idx, chunk_total, n_bytes,
        len(received), chunk_total,
    )

    # If we just finished, assemble + extract.
    completed = False
    extraction = "in_progress"
    if len(received) == chunk_total and received == list(range(chunk_total)):
        staged = _assemble_chunks_if_complete(user_id, trial_id, chunks_dir, manifest)
        if staged is not None:
            threading.Thread(
                target=_extract_publish_and_sync,
                args=(staged, out_dir, user_id, trial_id),
                daemon=True,
            ).start()
            completed = True
            extraction = "deferred"
            try:
                _append_attempt_log({
                    "ts": _datetime.utcnow().isoformat() + "Z",
                    "user_id": user_id,
                    "trial_id": trial_id,
                    "stage": "chunked_assembled",
                    "zip_bytes": size_full,
                    "error": "",
                })
            except Exception:
                pass

    return {
        "ok": True,
        "trial_id": trial_id,
        "chunk_idx": chunk_idx,
        "chunks_received": received,
        "chunk_total": chunk_total,
        "completed": completed,
        "extraction": extraction,
    }


# ---------- Upload-attempt debug log ----------
# Clients report each stage of the upload pipeline (zip_started,
# upload_started, done, failed) so we have server-side visibility even when
# the zip or HTTP transfer fails silently on the device.
#
# Log file: <DATA_ROOT>/trials/_upload_attempts.jsonl
# View last N: GET /api/trials/upload_log?n=50

import json as _json
from datetime import datetime as _datetime

_ATTEMPT_LOG_LOCK = threading.Lock()

def _append_attempt_log(entry: dict):
    log_path = os.path.join(DATA_ROOT, "trials", "_upload_attempts.jsonl")
    line = _json.dumps(entry, ensure_ascii=False)
    with _ATTEMPT_LOG_LOCK:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


@router.post("/trials/upload_attempt")
async def log_upload_attempt(
    trial_id: str = Form(...),
    stage: str = Form(...),     # zip_started | upload_started | done | failed
    error: str = Form(""),
    zip_bytes: int = Form(0),
    token: str = Depends(oauth2_scheme),
):
    """Client-side upload pipeline heartbeat for server-side debugging."""
    user_id = _get_user_id(token)
    entry = {
        "ts": _datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "trial_id": trial_id,
        "stage": stage,
        "zip_bytes": zip_bytes,
        "error": error,
    }
    try:
        _append_attempt_log(entry)
    except Exception as e:
        logger.warning("[trial_upload_attempt] failed to write log: %s", e)
    logger.info("[trial_upload_attempt] %s", _json.dumps(entry, ensure_ascii=False))
    return {"ok": True}


@router.get("/trials/upload_log")
async def get_upload_log(
    n: int = 50,
    token: str = Depends(oauth2_scheme),
):
    """Return the last N upload attempt log entries (newest first)."""
    _get_user_id(token)  # auth check only
    log_path = os.path.join(DATA_ROOT, "trials", "_upload_attempts.jsonl")
    if not os.path.exists(log_path):
        return {"entries": [], "total": 0}
    with _ATTEMPT_LOG_LOCK:
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()
    entries = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(_json.loads(line))
        except Exception:
            pass
        if len(entries) >= n:
            break
    return {"entries": entries, "total": len(lines)}


@router.post("/trials/recover_pending")
async def recover_pending(token: str = Depends(oauth2_scheme)):
    """Manually re-trigger the staged-zip recovery scan. Useful after a
    server restart if you want to confirm orphans were re-extracted."""
    _get_user_id(token)
    _recover_pending_uploads()
    return {"ok": True}


# Run the recovery scan once at module import time so any staged zips left
# behind by an earlier server crash get re-extracted as soon as the new
# process comes up.
try:
    _recover_pending_uploads()
except Exception as _e:
    logger.warning("[UNav-API] startup recovery scan failed: %s", _e)
