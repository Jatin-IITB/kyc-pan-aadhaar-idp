from __future__ import annotations

import logging
from typing import Any, Dict, List

from services.cross_doc.contradiction import ContradictionDetector
from services.cross_doc.address_normalizer import IndianAddressNormalizer
from services.graph.state import CaseState

logger = logging.getLogger(__name__)


def cross_doc_node(state: CaseState) -> CaseState:
    packet_docs: List[Dict[str, Any]] = state.get("packet_documents", [])
    if len(packet_docs) < 2:
        return {**state, "cross_doc_result": {"skipped": True, "reason": "single_document"}}

    detector = ContradictionDetector()
    addr_norm = IndianAddressNormalizer()

    docs_for_check = []
    for doc in packet_docs:
        ext = doc.get("extraction", {})
        fields = {}
        for k, v in ext.items():
            if isinstance(v, dict) and "value" in v:
                fields[k] = v["value"]
            elif isinstance(v, str):
                fields[k] = v
        docs_for_check.append({"doc_type": doc.get("doc_type", "unknown"), "fields": fields})

    contradiction_result = detector.detect(docs_for_check)

    addresses = []
    for doc in docs_for_check:
        for addr_key in ("address", "permanent_address", "correspondence_address", "billing_address"):
            val = doc["fields"].get(addr_key)
            if val:
                addresses.append({"doc_type": doc["doc_type"], "raw": val, "normalized": addr_norm.normalize(val)})

    address_consistency = 1.0
    if len(addresses) >= 2:
        scores = []
        for i in range(len(addresses)):
            for j in range(i + 1, len(addresses)):
                scores.append(addr_norm.compare(addresses[i]["normalized"], addresses[j]["normalized"]))
        address_consistency = sum(scores) / len(scores) if scores else 1.0

    result = {
        "skipped": False,
        "contradictions": contradiction_result["contradictions"],
        "consistency_score": contradiction_result["consistency_score"],
        "address_consistency": address_consistency,
        "recommendation": contradiction_result["recommendation"],
        "documents_checked": len(packet_docs),
    }

    logger.info("cross_doc: %d docs, consistency=%.2f, addr=%.2f, rec=%s",
                len(packet_docs), result["consistency_score"], address_consistency, result["recommendation"])

    return {**state, "cross_doc_result": result}
