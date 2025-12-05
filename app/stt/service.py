"""Speech-to-text service helpers."""

import sys
import uuid
from pathlib import Path
from threading import Lock
from typing import Dict, List

from fastapi import HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import settings

# Ensure the vendored klaam package is importable when the API runs inside Docker.
KLAAM_PATH = Path(__file__).resolve().parent / "klaam-updated"
if str(KLAAM_PATH) not in sys.path:
	sys.path.append(str(KLAAM_PATH))

from klaam import SpeechRecognition  # type: ignore  # noqa: E402


SUPPORTED_LANGUAGES = {"egy", "msa"}
_models: Dict[str, SpeechRecognition] = {}
_model_lock = Lock()


def _get_or_create_model(lang: str) -> SpeechRecognition:
	"""Load a SpeechRecognition model for the requested language (lazy singleton)."""

	normalized_lang = lang.lower()
	if normalized_lang not in SUPPORTED_LANGUAGES:
		raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'. Use one of {sorted(SUPPORTED_LANGUAGES)}")

	if normalized_lang not in _models:
		with _model_lock:
			if normalized_lang not in _models:
				_models[normalized_lang] = SpeechRecognition(lang=normalized_lang)

	return _models[normalized_lang]


async def speech_to_text(file: UploadFile, lang: str = "egy") -> str:
	"""Persist the uploaded file, run inference, and return the transcript."""

	if file.content_type and not file.content_type.startswith("audio/"):
		raise HTTPException(status_code=400, detail="Only audio uploads are supported")

	contents = await file.read()
	if not contents:
		raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

	if len(contents) > settings.MAX_FILE_SIZE:
		raise HTTPException(status_code=413, detail="Audio file exceeds the allowed size limit")

	suffix = Path(file.filename or "audio.wav").suffix or ".wav"
	temp_path = Path(settings.TEMP_DIR) / f"{uuid.uuid4().hex}{suffix}"
	temp_path.write_bytes(contents)

	try:
		recognizer = await run_in_threadpool(_get_or_create_model, lang)
		transcript = await run_in_threadpool(recognizer.transcribe, str(temp_path))
	except HTTPException:
		raise
	except Exception as exc:  # pragma: no cover - defensive log surface
		raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {exc}") from exc
	finally:
		temp_path.unlink(missing_ok=True)

	cleaned_transcript = transcript.strip()
	if not cleaned_transcript:
		raise HTTPException(status_code=500, detail="Model returned an empty transcript")

	return cleaned_transcript


def get_supported_languages() -> List[str]:
	"""Return languages that the API can serve."""

	return sorted(SUPPORTED_LANGUAGES)


def get_loaded_languages() -> List[str]:
	"""Expose which models have already been materialized in memory."""

	with _model_lock:
		return sorted(_models.keys())
