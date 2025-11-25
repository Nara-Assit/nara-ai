from fastapi import FastAPI
from app.tts.api import router as tts_router

app = FastAPI()
app.include_router(tts_router)