from fastapi import APIRouter

router = APIRouter()

@router.post("/convert")
async def convert_speech_to_text():
    return {"message": "STT - To be implemented"}

@router.get("/health")
async def health():
    return {"service": "stt", "status": "not_implemented"}