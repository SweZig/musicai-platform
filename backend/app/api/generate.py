"""Generate API — MusicGen-integration (Fas 4)"""
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., examples=["dark techno loop with industrial elements, 132 BPM"])
    duration_sec: int = Field(default=8, ge=4, le=30)
    genre: str | None = None


@router.post("/", summary="Generera musikidé med MusicGen")
async def generate_music(request: GenerateRequest):
    """
    Fas 4-funktion — MusicGen-integration implementeras i Fas 4.
    Returnerar placeholder under Fas 1-3.
    """
    return {
        "status": "not_implemented",
        "message": "MusicGen-integration implementeras i Fas 4.",
        "prompt": request.prompt,
    }
