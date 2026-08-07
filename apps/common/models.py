from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import ForeignKey, String, Text, Float, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from apps.common.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> uuid.UUID:
    return uuid.uuid4()


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    status: Mapped[str] = mapped_column(
        String(20), default="PENDING", index=True
    )
    packet_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )

    documents: Mapped[List[Document]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    decision: Mapped[Optional[Decision]] = relationship(
        back_populates="case", uselist=False, cascade="all, delete-orphan"
    )
    events: Mapped[List[AuditEvent]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cases.id"), index=True
    )
    doc_type: Mapped[str] = mapped_column(String(40))
    storage_uri: Mapped[str] = mapped_column(Text)
    original_filename: Mapped[str] = mapped_column(Text, default="")
    extraction: Mapped[dict] = mapped_column(JSONB, default=dict)
    validation: Mapped[dict] = mapped_column(JSONB, default=dict)
    quality_meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    forensics_meta: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    case: Mapped[Case] = relationship(back_populates="documents")
    corrections: Mapped[List[ReviewCorrection]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cases.id"), unique=True
    )
    outcome: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    policy_citations: Mapped[list] = mapped_column(JSONB, default=list)
    decided_by: Mapped[str] = mapped_column(String(100), default="system")
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    case: Mapped[Case] = relationship(back_populates="decision")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("cases.id"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    event_hash: Mapped[str] = mapped_column(String(64))
    prev_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    case: Mapped[Case] = relationship(back_populates="events")


class ReviewCorrection(Base):
    __tablename__ = "review_corrections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_id
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id"), index=True
    )
    reviewer: Mapped[str] = mapped_column(String(100))
    field_name: Mapped[str] = mapped_column(String(60))
    original_value: Mapped[str] = mapped_column(Text, default="")
    corrected_value: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    document: Mapped[Document] = relationship(back_populates="corrections")
