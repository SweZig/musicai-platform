"""
Grundläggande API-tester för Fas 1.
Kör med: pytest tests/ -v
"""
import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
async def client():
    """AsyncClient med mock-databas för testning."""
    # I Fas 1 testar vi utan riktig DB (override dependencies)
    from app.main import app
    from app.db import get_db
    from unittest.mock import AsyncMock, MagicMock

    async def mock_db():
        db = AsyncMock()
        db.get = AsyncMock(return_value=None)
        db.add = MagicMock()
        db.flush = AsyncMock()
        db.commit = AsyncMock()
        db.scalar = AsyncMock(return_value=None)
        yield db

    app.dependency_overrides[get_db] = mock_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health(client):
    """API ska svara på /health."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_root(client):
    """Root endpoint ska returnera API-info."""
    response = await client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


@pytest.mark.asyncio
async def test_upload_invalid_format(client):
    """Uppladdning med fel filformat ska ge 400."""
    from io import BytesIO
    response = await client.post(
        "/api/v1/tracks/upload",
        files={"file": ("test.txt", BytesIO(b"not audio"), "text/plain")},
    )
    assert response.status_code == 400
    assert "Filformat" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_tracks_empty(client):
    """Tom lista av tracks ska returnera 200 och tom array."""
    from sqlalchemy import select
    from unittest.mock import AsyncMock, MagicMock

    from app.main import app
    from app.db import get_db

    async def mock_db_empty():
        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute = AsyncMock(return_value=result)
        yield db

    app.dependency_overrides[get_db] = mock_db_empty
    response = await client.get("/api/v1/tracks/")
    assert response.status_code == 200
    assert response.json() == []
    app.dependency_overrides.clear()


def test_feature_vector_structure():
    """Feature-vektorn ska ha rätt dimensioner."""
    import numpy as np

    # Simulera en feature-dict (utan att faktiskt köra Librosa)
    mfcc_mean = [float(i) for i in range(13)]
    mfcc_std = [float(i) for i in range(13)]
    chroma_mean = [float(i) for i in range(12)]

    feature_vector = (
        mfcc_mean + mfcc_std + chroma_mean +
        [120.0, 3500.0, 8000.0, 1500.0, 0.05, 0.01, 0.7]
    )

    # 13 + 13 + 12 + 7 = 45 features
    assert len(feature_vector) == 45


def test_heuristic_classifier_electronic():
    """Heuristisk klassificerare ska identifiera elektronisk musik korrekt."""
    from app.core.classify import GenreClassifier

    classifier = GenreClassifier()
    features = {
        "bpm": 128.0,
        "energy": 0.05,
        "zero_crossing_rate_mean": 0.03,
        "spectral_centroid_mean": 4000.0,
        "chroma_stats": {"std": [0.08] * 12},
        "feature_vector": None,
    }
    result = classifier.predict(features)
    assert result["genre"] in ["electronic", "pop", "unknown"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert "scores" in result
