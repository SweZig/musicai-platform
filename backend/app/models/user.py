"""User-modell"""
import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    plan: Mapped[str] = mapped_column(String(20), default="free")  # free, pro, enterprise
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
