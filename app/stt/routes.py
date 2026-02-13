from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.stt.service import transcribe_audio
from app.config import settings
import os
import tempfile

router = APIRouter()


@router.post("/convert")
async def convert_speech_to_text(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, etc.)")
):
    """Convert Arabic speech to text"""
    
    # Validate file size
    content = await audio.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size: {settings.MAX_FILE_SIZE / (1024*1024)}MB")
    
    # Save to temp file
    temp_file = tempfile.mktemp(suffix=os.path.splitext(audio.filename)[1])
    
    try:
        with open(temp_file, "wb") as f:
            f.write(content)
        
        # Transcribe
        transcription = await transcribe_audio(temp_file)
        
        return JSONResponse({
            "success": True,
            "transcription": transcription,
            "filename": audio.filename
        })
        
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {str(e)}")
        
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)

@router.get("/health")
async def health():
    return {"service": "stt", "status": "healthy", "model": "vosk-ar-mgb2-0.4"}
