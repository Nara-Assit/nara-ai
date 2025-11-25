from fastapi import APIRouter

router = APIRouter()

@router.post("/convert")
async def convert_sign_to_text():
    return {"message": "Sign-to-text - To be implemented"}

@router.get("/health")
async def health():
    return {"service": "sign", "status": "not_implemented"}