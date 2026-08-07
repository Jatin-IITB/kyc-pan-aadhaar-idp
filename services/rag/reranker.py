# services/rag/reranker.py
from __future__ import annotations

from typing import Any, Dict, List


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder model that scores
    each ``(query, passage)`` pair for relevance.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Score each chunk against *query* with the cross-encoder and
        return the top-*k* results sorted by descending score.

        Each returned dict is a copy of the input chunk with an added
        ``rerank_score`` key.
        """
        if not chunks:
            return []

        self._ensure_model()

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self._model.predict(pairs)

        scored_chunks: List[Dict[str, Any]] = []
        for chunk, score in zip(chunks, scores):
            entry = dict(chunk)
            entry["rerank_score"] = float(score)
            scored_chunks.append(entry)

        scored_chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored_chunks[:top_k]
