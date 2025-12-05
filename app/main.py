from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.tts.routes import router as tts_router
from app.stt.routes import router as stt_router
from app.sign.routes import router as sign_router
from app.sound_alert.routes import router as sound_alert_router
from app.config import settings
import os

os.makedirs(settings.TEMP_DIR, exist_ok=True)

app = FastAPI(title="AI Services API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tts_router, prefix="/api/tts", tags=["TTS"])
app.include_router(stt_router, prefix="/api/stt", tags=["STT"])
app.include_router(sign_router, prefix="/api/sign", tags=["Sign"])
app.include_router(sound_alert_router, prefix="/api/sound-alert", tags=["Sound Alert"])

@app.get("/")
async def root():
    return {"message": "AI Services API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}