# ADR-012: In-Process Prometheus Metrics Without External Library

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 9  

## Context

Production monitoring requires metrics: documents processed, latency percentiles, queue depth, auto-clear rate. Prometheus is the industry standard for cloud-native metrics.

### Alternatives Considered

- **`prometheus_client` library**: Standard but adds dependency, complex histogram implementation
- **StatsD + Graphite**: Older stack, less common in modern deployments
- **Custom JSON metrics endpoint**: Not Prometheus-compatible, requires custom dashboards
- **No metrics**: Not acceptable for production workload

## Decision

Implement a **lightweight in-process `MetricsCollector`** that formats output in Prometheus text exposition format without requiring the `prometheus_client` library:

- **Counters**: documents_processed, succeeded, failed, auto_cleared, review, rejected, api_requests
- **Summaries**: processing_duration_seconds, api_latency_seconds (with p50/p90/p95/p99 quantiles)
- **Gauges**: queue_depth, active_workers

Exposed at `GET /metrics` with `text/plain; version=0.0.4` content type. FastAPI middleware tracks request latency automatically.

Prometheus scrapes every 10 seconds via `config/prometheus.yml`.

## Consequences

**Positive:**
- Zero additional dependencies
- Standard Prometheus format — works with Grafana dashboards out of the box
- Latency middleware captures every API request automatically
- Lightweight — no background threads or push mechanisms

**Negative:**
- No histogram bucketing (using summary quantiles instead — less accurate for aggregation across instances)
- In-memory storage — metrics reset on process restart
- Quantile computation is O(n log n) on each scrape (mitigated by capping at 10K samples)
- Single-process only — no cross-worker aggregation

**Risks:**
- Memory growth from histogram samples — mitigated by 10K cap with 5K tail retention
