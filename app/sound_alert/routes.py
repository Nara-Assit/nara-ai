"""
Sound Alert API Routes (FastAPI)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .service import SoundAlertService

router = APIRouter(
    prefix="/sound-alert",
    tags=["Sound Alert"]
)

service = SoundAlertService()

ALLOWED_EXTENSIONS = ["wav", "mp3", "m4a", "ogg", "flac"]


@router.post("/predict")
async def predict_sound(
    audio_file: UploadFile = File(...)
):
    """
    Predict sound class from uploaded audio file
    
    Args:
        audio_file: Audio file (wav, mp3, m4a, ogg, flac)
        
    Returns:
        {
            "success": true,
            "predicted_class": "doorbell",
            "confidence": 0.945,
            "all_predictions": {...}
        }
    """
    try:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        ext = audio_file.filename.split(".")[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        audio_bytes = await audio_file.read()
        
        if len(audio_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Run prediction
        result = service.predict_from_bytes(audio_bytes, ext)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Prediction failed'))
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/health")
async def health():
    """Health check endpoint"""
    is_ready = service.is_model_loaded()
    
    return JSONResponse({
        "service": "sound_alert",
        "status": "ok" if is_ready else "error",
        "model_loaded": is_ready
    }, status_code=200 if is_ready else 503)


@router.get("/classes")
async def get_classes():
    """Get list of detectable sound classes"""
    classes = service.get_classes()
    
    return JSONResponse({
        "classes": classes,
        "count": len(classes)
    })


@router.get("/info")
async def get_info():
    """Get service information"""
    return JSONResponse({
        "name": "Sound Alert",
        "version": "1.0.0",
        "description": "AI-powered sound classification for deaf & hard-of-hearing assistance",
        "supported_formats": ALLOWED_EXTENSIONS,
        "classes": service.get_classes(),
        "model": "YAMNet + Custom Classifier"
    })