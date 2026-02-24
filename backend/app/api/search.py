"""Search API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db import get_db
from app.schemas.track import TextSearchRequest, SimilarTrackResponse

router = APIRouter()


@router.post(
    "/text",
    response_model=list[SimilarTrackResponse],
    summary="Semantisk text-sökning med CLAP"
)
async def search_by_text(
    request: TextSearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Söker i kartoteket med naturlig text via CLAP text-encoder.
    Exempel: 'energetisk house loop 128 BPM' eller 'mörk ambient med piano'
    """
    # TODO: Fas 3 — Generera text-embedding med CLAP och sök i pgvector
    # Placeholder-svar under Fas 1
    return []
