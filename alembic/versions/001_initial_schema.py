"""Initial schema — cases, documents, decisions, audit_events, review_corrections

Revision ID: 001
Revises: None
Create Date: 2026-08-07
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, index=True),
        sa.Column("packet_id", UUID(as_uuid=True), nullable=True, index=True),
    )

    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "case_id",
            UUID(as_uuid=True),
            sa.ForeignKey("cases.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("doc_type", sa.String(40), nullable=False),
        sa.Column("storage_uri", sa.Text, nullable=False),
        sa.Column("original_filename", sa.Text, nullable=False, server_default=""),
        sa.Column("extraction", JSONB, nullable=False, server_default="{}"),
        sa.Column("validation", JSONB, nullable=False, server_default="{}"),
        sa.Column("quality_meta", JSONB, nullable=False, server_default="{}"),
        sa.Column("forensics_meta", JSONB, nullable=True),
        sa.Column("confidence_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "decisions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "case_id",
            UUID(as_uuid=True),
            sa.ForeignKey("cases.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("outcome", sa.String(20), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("policy_citations", JSONB, nullable=False, server_default="[]"),
        sa.Column(
            "decided_by", sa.String(100), nullable=False, server_default="system"
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "audit_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "case_id",
            UUID(as_uuid=True),
            sa.ForeignKey("cases.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("event_type", sa.String(40), nullable=False),
        sa.Column("payload", JSONB, nullable=False, server_default="{}"),
        sa.Column("event_hash", sa.String(64), nullable=False),
        sa.Column("prev_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "review_corrections",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            UUID(as_uuid=True),
            sa.ForeignKey("documents.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("reviewer", sa.String(100), nullable=False),
        sa.Column("field_name", sa.String(60), nullable=False),
        sa.Column("original_value", sa.Text, nullable=False, server_default=""),
        sa.Column("corrected_value", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("review_corrections")
    op.drop_table("audit_events")
    op.drop_table("decisions")
    op.drop_table("documents")
    op.drop_table("cases")
