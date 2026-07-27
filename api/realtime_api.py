import json
import os
import urllib.request
import urllib.error

import requests
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.security import OAuth2PasswordBearer

from api.user_api import decode_access_token

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

OPENAI_REALTIME_URL = os.getenv("OPENAI_REALTIME_URL", "https://api.openai.com/v1/realtime/client_secrets")
OPENAI_REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
OPENAI_REALTIME_VOICE = os.getenv("OPENAI_REALTIME_VOICE", "marin")
OPENAI_TRANSCRIBE_URL = os.getenv("OPENAI_TRANSCRIBE_URL", "https://api.openai.com/v1/audio/transcriptions")
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
OPENAI_SPEECH_URL = os.getenv("OPENAI_SPEECH_URL", "https://api.openai.com/v1/audio/speech")
OPENAI_SPEECH_MODEL = os.getenv("OPENAI_SPEECH_MODEL", "gpt-4o-mini-tts")
OPENAI_SPEECH_VOICE = os.getenv("OPENAI_SPEECH_VOICE", "coral")


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

@router.post("/realtime/speech")
def synthesize_speech(payload: dict, user_id: str = Depends(get_user_id_from_token)):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured on the server")

    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing text.")
    if len(text) > 4096:
        raise HTTPException(status_code=400, detail="Text is too long for speech synthesis.")

    language = str(payload.get("language", "")).strip() or "auto"
    speech_payload = {
        "model": OPENAI_SPEECH_MODEL,
        "voice": OPENAI_SPEECH_VOICE,
        "input": text,
        "response_format": "mp3",
        "instructions": (
            "You are the spoken voice of UNav, an indoor navigation assistant for blind and low-vision users. "
            "Speak clearly, warmly, and naturally. Keep a calm pace suitable for navigation instructions. "
            "Pronounce place names and floor numbers carefully. "
            f"The user's language or locale is {language}."
        ),
    }

    try:
        response = requests.post(
            OPENAI_SPEECH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(speech_payload),
            timeout=90,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to synthesize speech: {exc}")

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI speech request failed: {response.text}",
        )

    return Response(
        content=response.content,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-UNav-TTS-Model": OPENAI_SPEECH_MODEL,
            "X-UNav-TTS-Voice": OPENAI_SPEECH_VOICE,
        },
    )

