# services/rag/retriever.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import logging

from services.rag.indexer import canonical_chunk_id

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid search over a Qdrant collection: combines dense vector
    search with sparse BM25 scoring, merged via Reciprocal Rank Fusion.

    The BM25 corpus is loaded from Qdrant at initialization and kept
    in-memory for fast sparse matching.
    """

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str = "kyc_policies",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        *,
        encoder: Any = None,
        qdrant_client: Any = None,
    ) -> None:
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Lazy-loaded
        self._encoder: Any = encoder
        self._qdrant: Any = qdrant_client
        self._bm25: Any = None
        self._corpus: List[Dict[str, Any]] = []
        self._corpus_loaded = False

    # ------------------------------------------------------------------
    # Internal bootstrap
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load encoder, Qdrant client, and BM25 corpus on first use."""
        self._ensure_dense_loaded()
        self._ensure_corpus_loaded()

    def _ensure_qdrant_loaded(self) -> None:
        from qdrant_client import QdrantClient

        if self._qdrant is None:
            self._qdrant = QdrantClient(url=self.qdrant_url)

    def _ensure_dense_loaded(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._ensure_qdrant_loaded()
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model)

    def _ensure_corpus_loaded(self) -> None:
        self._ensure_qdrant_loaded()
        if not self._corpus_loaded:
            self._load_bm25_corpus()

    def _load_bm25_corpus(self) -> None:
        """Scroll the entire Qdrant collection into memory and build a
        BM25 index over the ``text`` payloads."""
        from rank_bm25 import BM25Okapi

        records, _next = self._qdrant.scroll(
            collection_name=self.collection_name,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        self._corpus = []
        tokenized: List[List[str]] = []
        for rec in records:
            payload = rec.payload or {}
            text = payload.get("text", "")
            source_file = payload.get("source_file", "")
            section_header = payload.get("section_header", "")
            section_chunk_index = payload.get("section_chunk_index", 0)
            self._corpus.append(
                {
                    "id": str(rec.id),
                    "chunk_id": payload.get("chunk_id")
                    or canonical_chunk_id(
                        source_file,
                        section_header,
                        section_chunk_index,
                    ),
                    "text": text,
                    "source_file": source_file,
                    "section_header": section_header,
                    "chunk_index": payload.get("chunk_index", 0),
                    "section_chunk_index": section_chunk_index,
                }
            )
            tokenized.append(text.lower().split())

        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

        self._corpus_loaded = True
        logger.info("BM25 corpus loaded: {} documents", len(self._corpus))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Run hybrid search (dense + sparse) and return up to *top_k*
        chunks ranked by Reciprocal Rank Fusion."""
        return self.retrieve_rrf(query, top_k=top_k)

    def retrieve_dense(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return dense-only results using the production embedding index."""
        return self._dense_search(query, top_k=top_k)

    def retrieve_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return BM25-only results without loading the embedding model."""
        return self._sparse_search(query, top_k=top_k)

    def retrieve_rrf(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return dense + BM25 results fused with reciprocal rank fusion."""
        dense_results = self._dense_search(query, top_k=top_k)
        sparse_results = self._sparse_search(query, top_k=top_k)
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:top_k]

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Encode *query* with the embedding model and search Qdrant."""
        self._ensure_dense_loaded()
        vector = self._encoder.encode(query).tolist()
        # qdrant-client removed .search() in 1.x in favour of .query_points();
        # keep the old call for older pinned clients so both resolve.
        if hasattr(self._qdrant, "query_points"):
            hits = self._qdrant.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=True,
            ).points
        else:  # pragma: no cover - legacy client
            hits = self._qdrant.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
        results: List[Dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            source_file = payload.get("source_file", "")
            section_header = payload.get("section_header", "")
            section_chunk_index = payload.get("section_chunk_index", 0)
            results.append(
                {
                    "id": str(hit.id),
                    "chunk_id": payload.get("chunk_id")
                    or canonical_chunk_id(
                        source_file,
                        section_header,
                        section_chunk_index,
                    ),
                    "text": payload.get("text", ""),
                    "source_file": source_file,
                    "section_header": section_header,
                    "chunk_index": payload.get("chunk_index", 0),
                    "section_chunk_index": section_chunk_index,
                    "score": float(hit.score),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Sparse (BM25) search
    # ------------------------------------------------------------------

    def _sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Score the in-memory corpus with BM25 and return top results."""
        self._ensure_corpus_loaded()
        if self._bm25 is None or not self._corpus:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        scored_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results: List[Dict[str, Any]] = []
        for idx in scored_indices:
            if scores[idx] <= 0:
                continue
            entry = self._corpus[idx]
            results.append(
                {
                    "id": entry["id"],
                    "chunk_id": entry["chunk_id"],
                    "text": entry["text"],
                    "source_file": entry["source_file"],
                    "section_header": entry["section_header"],
                    "chunk_index": entry["chunk_index"],
                    "section_chunk_index": entry["section_chunk_index"],
                    "score": float(scores[idx]),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        dense: List[Dict[str, Any]],
        sparse: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Merge two ranked lists using RRF.  Each document receives
        score = sum(1 / (k + rank)) across both lists.

        Args:
            dense:  Results from dense search (ordered by score desc).
            sparse: Results from sparse search (ordered by score desc).
            k:      RRF constant (default 60 per the original paper).
        """
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(sparse, start=1):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=lambda did: rrf_scores[did], reverse=True)
        merged: List[Dict[str, Any]] = []
        for doc_id in sorted_ids:
            entry = dict(doc_map[doc_id])
            entry["rrf_score"] = rrf_scores[doc_id]
            merged.append(entry)
        return merged
