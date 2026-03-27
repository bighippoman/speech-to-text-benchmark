"""
API wrappers for each ASR provider.
Each provider implements transcribe() → TranscriptionResult.
"""

import os
import time
import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class TranscriptionResult:
    text: Optional[str]
    elapsed: float  # Total wall-clock time including network
    error: Optional[str] = None
    raw_status: int = 200


# ── Cohere Transcribe ───────────────────────────────────────────────────────

def transcribe_cohere(filepath: str, language: str, api_key: str) -> TranscriptionResult:
    """Cohere Transcribe API (v2, multipart upload)."""
    url = "https://api.cohere.com/v2/audio/transcriptions"
    start = time.time()
    try:
        with open(filepath, "rb") as f:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(filepath), f)},
                data={
                    "model": "cohere-transcribe-03-2026",
                    "language": language,
                },
                timeout=180,
            )
        elapsed = time.time() - start
        if resp.status_code != 200:
            return TranscriptionResult(
                text=None, elapsed=elapsed,
                error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                raw_status=resp.status_code,
            )
        text = resp.json().get("text", "")
        return TranscriptionResult(text=text, elapsed=elapsed)
    except Exception as e:
        return TranscriptionResult(
            text=None, elapsed=time.time() - start, error=str(e)
        )


# ── OpenAI Whisper ──────────────────────────────────────────────────────────

def transcribe_openai(filepath: str, language: str, api_key: str) -> TranscriptionResult:
    """OpenAI Whisper API (whisper-1, multipart upload)."""
    url = "https://api.openai.com/v1/audio/transcriptions"
    start = time.time()
    try:
        with open(filepath, "rb") as f:
            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(filepath), f)},
                data={"model": "whisper-1", "language": language},
                timeout=180,
            )
        elapsed = time.time() - start
        if resp.status_code != 200:
            return TranscriptionResult(
                text=None, elapsed=elapsed,
                error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                raw_status=resp.status_code,
            )
        text = resp.json().get("text", "")
        return TranscriptionResult(text=text, elapsed=elapsed)
    except Exception as e:
        return TranscriptionResult(
            text=None, elapsed=time.time() - start, error=str(e)
        )


# ── Deepgram Nova-2 ────────────────────────────────────────────────────────

DEEPGRAM_MIME_MAP = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
}


def transcribe_deepgram(filepath: str, language: str, api_key: str) -> TranscriptionResult:
    """Deepgram Nova-2 API (raw binary upload)."""
    ext = os.path.splitext(filepath)[1].lower()
    content_type = DEEPGRAM_MIME_MAP.get(ext, "audio/wav")

    url = (
        "https://api.deepgram.com/v1/listen"
        f"?model=nova-2&language={language}&smart_format=true&punctuate=true"
    )
    start = time.time()
    try:
        with open(filepath, "rb") as f:
            audio_bytes = f.read()
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": content_type,
            },
            data=audio_bytes,
            timeout=180,
        )
        elapsed = time.time() - start
        if resp.status_code != 200:
            return TranscriptionResult(
                text=None, elapsed=elapsed,
                error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                raw_status=resp.status_code,
            )
        data = resp.json()
        # Navigate Deepgram's nested response structure
        try:
            text = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        except (KeyError, IndexError):
            text = ""
        return TranscriptionResult(text=text, elapsed=elapsed)
    except Exception as e:
        return TranscriptionResult(
            text=None, elapsed=time.time() - start, error=str(e)
        )


# ── AssemblyAI ──────────────────────────────────────────────────────────────

def transcribe_assemblyai(filepath: str, language: str, api_key: str) -> TranscriptionResult:
    """
    AssemblyAI API (async: upload → create → poll).
    Elapsed time includes upload + transcription + polling.
    """
    base = "https://api.assemblyai.com/v2"
    headers = {"authorization": api_key}
    start = time.time()

    try:
        # Step 1: Upload audio
        with open(filepath, "rb") as f:
            upload_resp = requests.post(
                f"{base}/upload",
                headers=headers,
                data=f,
                timeout=120,
            )
        if upload_resp.status_code != 200:
            return TranscriptionResult(
                text=None, elapsed=time.time() - start,
                error=f"Upload failed: HTTP {upload_resp.status_code}",
                raw_status=upload_resp.status_code,
            )
        upload_url = upload_resp.json()["upload_url"]

        # Step 2: Create transcription job
        lang_code = language if language != "en" else "en"
        transcript_resp = requests.post(
            f"{base}/transcript",
            headers=headers,
            json={"audio_url": upload_url, "language_code": lang_code},
            timeout=30,
        )
        if transcript_resp.status_code != 200:
            return TranscriptionResult(
                text=None, elapsed=time.time() - start,
                error=f"Create failed: HTTP {transcript_resp.status_code}",
                raw_status=transcript_resp.status_code,
            )
        transcript_id = transcript_resp.json()["id"]

        # Step 3: Poll for completion (max 120s)
        poll_deadline = time.time() + 120
        while time.time() < poll_deadline:
            poll_resp = requests.get(
                f"{base}/transcript/{transcript_id}",
                headers=headers,
                timeout=15,
            )
            result = poll_resp.json()
            status = result.get("status")
            if status == "completed":
                elapsed = time.time() - start
                return TranscriptionResult(
                    text=result.get("text", ""), elapsed=elapsed
                )
            elif status == "error":
                elapsed = time.time() - start
                return TranscriptionResult(
                    text=None, elapsed=elapsed,
                    error=f"Transcription error: {result.get('error', 'unknown')}",
                )
            time.sleep(1.5)

        # Timeout
        return TranscriptionResult(
            text=None, elapsed=time.time() - start,
            error="Polling timeout after 120s",
        )

    except Exception as e:
        return TranscriptionResult(
            text=None, elapsed=time.time() - start, error=str(e)
        )


# ── Dispatcher ──────────────────────────────────────────────────────────────

PROVIDER_FUNCTIONS = {
    "cohere": transcribe_cohere,
    "openai": transcribe_openai,
    "deepgram": transcribe_deepgram,
    "assemblyai": transcribe_assemblyai,
}


def transcribe(provider_id: str, filepath: str, language: str, api_key: str) -> TranscriptionResult:
    """Route to the correct provider function."""
    fn = PROVIDER_FUNCTIONS.get(provider_id)
    if fn is None:
        return TranscriptionResult(
            text=None, elapsed=0.0, error=f"Unknown provider: {provider_id}"
        )
    return fn(filepath, language, api_key)
