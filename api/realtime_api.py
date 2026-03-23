import json
import os
import urllib.request
import urllib.error

import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordBearer

from api.user_api import decode_access_token

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

OPENAI_REALTIME_URL = os.getenv("OPENAI_REALTIME_URL", "https://api.openai.com/v1/realtime/client_secrets")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "marin")
OPENAI_TRANSCRIBE_URL = os.getenv("OPENAI_TRANSCRIBE_URL", "https://api.openai.com/v1/audio/transcriptions")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")


def get_user_id_from_token(token: str = Depends(oauth2_scheme)) -> str:
    payload = decode_access_token(token)
    return str(payload["id"])


@router.get("/realtime/token")
def create_realtime_token(user_id: str = Depends(get_user_id_from_token)):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server")

    payload = {
        "expires_after": {
            "anchor": "created_at",
            "seconds": 60,
        },
        "session": {
            "type": "realtime",
            "model": OPENAI_REALTIME_MODEL,
            "instructions": (
                "You are UNav Smart Mode, a navigation assistant for indoor wayfinding. "
                "Help users express their destination naturally in any language. "
                "If the user intent is ambiguous, ask concise follow-up questions. "
                "Keep replies short and accessible for blind and low-vision users. "
                f"Current user id: {user_id}."
            ),
            "audio": {
                "output": {
                    "voice": OPENAI_REALTIME_VOICE,
                },
            },
        },
    }

    req = urllib.request.Request(
        OPENAI_REALTIME_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"OpenAI realtime token request failed: {detail}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to create realtime token: {exc}")

    return json.loads(body)


@router.post("/realtime/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    user_id: str = Depends(get_user_id_from_token),
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server")

    payload = {
        "model": OPENAI_TRANSCRIBE_MODEL,
        "response_format": "json",
    }
    if language:
        payload["language"] = language
    if prompt:
        payload["prompt"] = prompt

    try:
        audio_bytes = await audio.read()
        response = requests.post(
            OPENAI_TRANSCRIBE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            data=payload,
            files={
                "file": (
                    audio.filename or f"smart-mode-{user_id}.m4a",
                    audio_bytes,
                    audio.content_type or "audio/m4a",
                )
            },
            timeout=90,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to upload audio for transcription: {exc}")

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI transcription request failed: {response.text}",
        )

    try:
        parsed = response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to parse transcription response: {exc}")

    return {
        "text": parsed.get("text", ""),
        "model": OPENAI_TRANSCRIBE_MODEL,
    }
