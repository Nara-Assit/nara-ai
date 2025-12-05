from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .service import predict_sound, convert_to_wav
import os
import tempfile

router = APIRouter()

ALLOWED_EXTENSIONS = ["wav", "mp3", "m4a", "ogg", "flac"]


@router.post("/predict")
async def predict_sound_endpoint(audio_file: UploadFile = File(...)):
    if not audio_file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = audio_file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    try:
        temp_file.write(await audio_file.read())
        temp_file.close()

        class_name = predict_sound(temp_file.name)
        
        return {"success": True, "predicted_class": class_name}
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@router.get("/classes")
async def get_classes():
    return {"classes": label_classes, "count": len(label_classes)}


@router.get("/info")
async def get_info():
    return {
        "name": "Sound Alert",
        "version": "1.0.0",
        "description": "AI-powered sound classification for deaf & hard-of-hearing assistance",
        "supported_formats": ALLOWED_EXTENSIONS,
        "classes": label_classes,
        "model": "YAMNet + Custom Classifier"
    }


@router.get("/health")
async def health():
    return {"service": "sound_alert", "status": "ok"}