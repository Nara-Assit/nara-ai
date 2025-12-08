from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.tts.routes import router as tts_router
from app.stt.routes import router as stt_router
from app.sign.routes import router as sign_router
from app.config import settings
import os

# Ngrok setup
from pyngrok import ngrok
import uvicorn

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

@app.get("/")
async def root():
    return {"message": "AI Services API", "version": "1.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Run with optional ngrok
if __name__ == "__main__":
    import uvicorn
    
    port = 8000
    # FIXED: Read the environment variable correctly
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN", None)
    
    # Only start ngrok if token is provided
    if ngrok_token:
        try:
            from pyngrok import ngrok
            ngrok.set_auth_token(ngrok_token)
            public_url = ngrok.connect(port)
            print(f"\n{'='*60}")
            print(f"🚀 Public URL: {public_url}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"⚠️  Ngrok failed: {e}")
            print("Continuing without ngrok...\n")
    else:
        print("ℹ️  No NGROK_AUTH_TOKEN provided, running locally only\n")
    
    # Run uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)