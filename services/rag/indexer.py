# services/rag/indexer.py
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

import logging

logger = logging.getLogger(__name__)

_POINT_NAMESPACE = uuid.UUID("eef0b2db-ec75-4c6f-a828-9fe930f0bc87")


def canonical_chunk_id(
    source_file: str,
    section_header: str,
    section_chunk_index: int,
) -> str:
    """Return the stable, human-readable identity of a policy chunk.

    The ID is derived from policy structure rather than a retrieval run, so
    authored relevance labels remain valid across re-indexes.
    """
    return f"{source_file}::{section_header}::{section_chunk_index}"


class PolicyIndexer:
    """Reads Markdown policy documents, chunks them, embeds with
    sentence-transformers, and upserts into a Qdrant collection.

    Chunking strategy:
      1. Split on ``## `` section boundaries first.
      2. Within each section, split by approximate token count
         (512 tokens with 50-token overlap; tokens ≈ words / 0.75).
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

        # Lazy-loaded at first use
        self._encoder: Any = encoder
        self._qdrant: Any = qdrant_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_policy_chunks(self, policies_dir: Path) -> List[Dict[str, Any]]:
        """Build structurally identified chunks without loading ML services."""
        all_chunks: List[Dict[str, Any]] = []
        for md_path in sorted(Path(policies_dir).glob("*.md")):
            text = md_path.read_text(encoding="utf-8")
            all_chunks.extend(self._chunk_markdown(text, source=md_path.name))
        return all_chunks

    def index_policies(self, policies_dir: Path) -> int:
        """Read all ``.md`` files from *policies_dir*, chunk, embed, and
        upsert to Qdrant.  Returns the total number of indexed chunks.
        """
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from sentence_transformers import SentenceTransformer

        if self._encoder is None:
            self._encoder = SentenceTransformer(self.embedding_model)
        if self._qdrant is None:
            self._qdrant = QdrantClient(url=self.qdrant_url)

        md_files = sorted(Path(policies_dir).glob("*.md"))
        if not md_files:
            logger.warning("No .md files found in {}", policies_dir)
            return 0

        all_chunks = self.load_policy_chunks(policies_dir)
        if not all_chunks:
            return 0

        texts = [c["text"] for c in all_chunks]
        embeddings = self._encoder.encode(texts, show_progress_bar=False)
        dim = embeddings.shape[1]

        # Recreate collection (idempotent seed workflow)
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if self.collection_name in existing:
            self._qdrant.delete_collection(self.collection_name)

        self._qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=str(uuid.uuid5(_POINT_NAMESPACE, all_chunks[i]["chunk_id"])),
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": all_chunks[i]["chunk_id"],
                    "text": all_chunks[i]["text"],
                    "source_file": all_chunks[i]["source_file"],
                    "section_header": all_chunks[i]["section_header"],
                    "chunk_index": all_chunks[i]["chunk_index"],
                    "section_chunk_index": all_chunks[i]["section_chunk_index"],
                },
            )
            for i in range(len(all_chunks))
        ]

        batch_size = 64
        for start in range(0, len(points), batch_size):
            self._qdrant.upsert(
                collection_name=self.collection_name,
                points=points[start : start + batch_size],
            )

        logger.info("Indexed {} chunks from {} files", len(points), len(md_files))
        return len(points)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_markdown(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Split *text* on ``## `` section headers, then subdivide each
        section into chunks of ~512 tokens with 50-token overlap.

        Returns a list of dicts with keys:
          ``chunk_id``, ``text``, ``source_file``, ``section_header``,
          ``chunk_index``, ``section_chunk_index``.
        """
        max_tokens = 512
        overlap_tokens = 50

        sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
        chunks: List[Dict[str, Any]] = []
        global_idx = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            header_match = re.match(r"^## (.+)", section)
            header = header_match.group(1).strip() if header_match else "(preamble)"

            words = section.split()
            if not words:
                continue

            # Approximate token count: tokens ≈ words / 0.75
            max_words = int(max_tokens * 0.75)
            overlap_words = int(overlap_tokens * 0.75)

            pos = 0
            section_chunk_idx = 0
            while pos < len(words):
                end = min(pos + max_words, len(words))
                chunk_text = " ".join(words[pos:end])
                chunks.append(
                    {
                        "chunk_id": canonical_chunk_id(
                            source,
                            header,
                            section_chunk_idx,
                        ),
                        "text": chunk_text,
                        "source_file": source,
                        "section_header": header,
                        "chunk_index": global_idx,
                        "section_chunk_index": section_chunk_idx,
                    }
                )
                global_idx += 1
                section_chunk_idx += 1
                pos = end - overlap_words if end < len(words) else end

        return chunks
