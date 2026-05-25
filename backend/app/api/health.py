from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}
