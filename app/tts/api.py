from fastapi import APIRouter, Form
from fastapi.responses import FileResponse
import uuid
import os
from .core import auto_tashkeel, generate_tts

router = APIRouter()

@router.post("/tts")
async def tts_endpoint(text: str = Form(...)):
    filename = f"{uuid.uuid4().hex}.mp3"
    output_path = f"/tmp/{filename}"
    final_text = auto_tashkeel(text)
    generate_tts(final_text, output_path)
    return FileResponse(output_path, media_type="audio/mpeg", filename="audio.mp3")