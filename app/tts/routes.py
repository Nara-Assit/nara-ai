from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import FileResponse
from app.tts.service import text_to_speech

router = APIRouter()

@router.post("/convert")
async def convert_text_to_speech(
    text: str = Form(...),
    apply_tashkeel: bool = Form(True)
):
    try:
        if not text or len(text.strip()) == 0:
            raise HTTPException(400, "Text cannot be empty")
        
        if len(text) > 5000:
            raise HTTPException(400, "Text too long (max 5000 chars)")
        
        audio_path = await text_to_speech(text.strip(), apply_tashkeel)
        
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename="audio.mp3"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")

@router.get("/health")
async def health():
    return {"service": "tts", "status": "ok"}