from __future__ import annotations

from services.graph.state import CaseState
from services.preprocessing.quality import resize_if_huge


def ingest_node(state: CaseState) -> CaseState:
    img = resize_if_huge(state["image_bgr"])
    return {"image_bgr": img}
