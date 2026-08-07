from __future__ import annotations

import time
from typing import Any, Dict

from fastapi import APIRouter, Request, Response
from fastapi.routing import APIRoute


class MetricsCollector:
    """In-process Prometheus-style metrics for KYC pipeline."""

    def __init__(self) -> None:
        self._counters: Dict[str, float] = {
            "kyc_documents_processed_total": 0,
            "kyc_documents_succeeded_total": 0,
            "kyc_documents_failed_total": 0,
            "kyc_documents_auto_cleared_total": 0,
            "kyc_documents_review_total": 0,
            "kyc_documents_rejected_total": 0,
            "kyc_api_requests_total": 0,
        }
        self._histograms: Dict[str, list] = {
            "kyc_processing_duration_seconds": [],
            "kyc_api_latency_seconds": [],
        }
        self._gauges: Dict[str, float] = {
            "kyc_queue_depth": 0,
            "kyc_active_workers": 0,
        }

    def inc(self, name: str, value: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, value: float) -> None:
        self._histograms.setdefault(name, []).append(value)
        if len(self._histograms[name]) > 10000:
            self._histograms[name] = self._histograms[name][-5000:]

    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    def format_prometheus(self) -> str:
        lines = []

        for name, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        for name, value in sorted(self._gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        for name, values in sorted(self._histograms.items()):
            if not values:
                continue
            lines.append(f"# TYPE {name} summary")
            sorted_v = sorted(values)
            count = len(sorted_v)
            total = sum(sorted_v)
            lines.append(f"{name}_count {count}")
            lines.append(f"{name}_sum {total:.6f}")
            for q in (0.5, 0.9, 0.95, 0.99):
                idx = min(int(q * count), count - 1)
                lines.append(f'{name}{{quantile="{q}"}} {sorted_v[idx]:.6f}')

        return "\n".join(lines) + "\n"


metrics = MetricsCollector()


def create_metrics_router() -> APIRouter:
    router = APIRouter()

    @router.get("/metrics", response_class=Response)
    async def prometheus_metrics():
        return Response(
            content=metrics.format_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return router
