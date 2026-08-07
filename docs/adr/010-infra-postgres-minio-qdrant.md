# ADR-010: Postgres + MinIO + Qdrant Infrastructure Stack

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 0  

## Context

The original project used only Redis (for Celery) and local filesystem storage. Scaling to a multi-document KYC platform requires:
- Relational data: cases, documents, decisions, audit events with foreign keys and JSONB
- Object storage: document images with presigned URLs for review UI
- Vector storage: embedding-based retrieval for policy RAG

### Alternatives Considered

**Database:**
- SQLite: No concurrent write support, no async driver
- MongoDB: JSONB in Postgres is equivalent for our use case, SQL is more portable
- MySQL: Lacks JSONB natively, weaker async ecosystem

**Object Storage:**
- S3 directly: Requires AWS account, not self-hostable for dev
- Local filesystem: No presigned URLs, no HTTP access, doesn't scale

**Vector Store:**
- Pinecone: Cloud-only, cost, vendor lock-in
- Weaviate: Heavier footprint, more complex API
- ChromaDB: No persistent storage mode at the time, less mature

## Decision

| Service | Image | Purpose |
|---------|-------|---------|
| PostgreSQL 16 | `postgres:16-alpine` | Cases, documents, decisions, audit events |
| MinIO | `minio/minio:latest` | S3-compatible document storage |
| Qdrant | `qdrant/qdrant:v1.7.4` | Vector store for policy RAG |
| Prometheus | `prom/prometheus:v2.48.1` | Metrics collection |

All services run in docker-compose with named volumes for persistence. API and worker services connect via internal Docker networking.

**SQLAlchemy async** with `asyncpg` driver for non-blocking DB operations. Pool size 10, max overflow 20.

**MinIO** implements the same `Storage` protocol as `LocalStorage` — swappable via config.

## Consequences

**Positive:**
- Complete self-hosted stack — `docker compose up` starts everything
- Production-like environment locally
- MinIO is S3-compatible — easy migration to AWS S3 later
- Qdrant is lightweight (~200MB) with REST+gRPC APIs

**Negative:**
- docker-compose now has 7 services — higher local resource usage (~2GB RAM)
- Postgres requires Alembic migrations for schema changes
- MinIO adds another set of credentials to manage

**Risks:**
- Qdrant v1.7.4 is pinned — may need updates for bug fixes
- asyncpg requires PostgreSQL-specific connection strings
