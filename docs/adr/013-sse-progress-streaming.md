# ADR-013: Server-Sent Events for Case Progress

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 9  

## Context

Multi-document KYC cases take 10-30 seconds to process. Clients need real-time progress updates (which documents are done, which are still processing) without polling.

### Alternatives Considered

- **WebSocket**: Full-duplex is unnecessary — progress is server→client only. More complex connection management
- **Long polling**: Higher latency, more HTTP overhead, harder to implement correctly
- **Polling**: Simple but wastes bandwidth, slower to reflect state changes (poll interval delay)
- **gRPC streaming**: Requires protobuf setup, not browser-friendly without gRPC-web

## Decision

Use **Server-Sent Events (SSE)** at `GET /v1/cases/{case_id}/progress`:

Event types:
- `started`: case_id, document_count
- `document_complete`: job_id, status (SUCCESS/FAILED), progress (e.g., "2/5")
- `case_complete`: case_id, total, succeeded count
- `timeout`: case_id (after 120s)
- `error`: message

Implementation: async generator yielding `text/event-stream` formatted events. Polls Celery `AsyncResult` every 1 second internally. `X-Accel-Buffering: no` header prevents nginx buffering.

## Consequences

**Positive:**
- Native browser support via `EventSource` API — no client library needed
- Unidirectional (server→client) matches our use case exactly
- Auto-reconnect built into the SSE spec
- Simpler than WebSocket — no upgrade handshake, works through proxies

**Negative:**
- 1-second internal polling of Celery results — not true event-driven (would need Redis pub/sub for that)
- 120-second timeout per connection — long-lived connections may be killed by load balancers
- No backpressure — if client can't consume, events queue in memory

**Risks:**
- Connection limits — each SSE connection holds a server thread/task. Mitigated by uvicorn's async model (thousands of concurrent connections possible)
