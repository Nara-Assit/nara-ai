from fastapi import APIRouter, File, Form, UploadFile

from app.stt.service import get_loaded_languages, get_supported_languages, speech_to_text

router = APIRouter()


@router.post("/convert")
async def convert_speech_to_text(audio: UploadFile = File(...), lang: str = Form("egy")):
    normalized_lang = lang.lower()
    transcript = await speech_to_text(audio, normalized_lang)
    return {"service": "stt", "language": normalized_lang, "transcript": transcript}


@router.get("/health")
async def health():
    return {
        "service": "stt",
        "status": "ok",
        "supported_languages": get_supported_languages(),
        "loaded_languages": get_loaded_languages(),
    }