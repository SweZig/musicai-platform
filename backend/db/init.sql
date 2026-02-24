-- Aktivera pgvector-tillägget
CREATE EXTENSION IF NOT EXISTS vector;

-- Skapa IVFFlat-index på embeddings efter att tabellen är skapad
-- (körs via Alembic-migration i Fas 2 när vi har data)
