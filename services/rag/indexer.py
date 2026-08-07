# services/rag/indexer.py
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

import logging

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Lazy-loaded at first use
        self._encoder: Any = None
        self._qdrant: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        all_chunks: List[Dict[str, Any]] = []
        for md_path in md_files:
            text = md_path.read_text(encoding="utf-8")
            chunks = self._chunk_markdown(text, source=md_path.name)
            all_chunks.extend(chunks)

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
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "text": all_chunks[i]["text"],
                    "source_file": all_chunks[i]["source_file"],
                    "section_header": all_chunks[i]["section_header"],
                    "chunk_index": all_chunks[i]["chunk_index"],
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
          ``text``, ``source_file``, ``section_header``, ``chunk_index``.
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
            while pos < len(words):
                end = min(pos + max_words, len(words))
                chunk_text = " ".join(words[pos:end])
                chunks.append(
                    {
                        "text": chunk_text,
                        "source_file": source,
                        "section_header": header,
                        "chunk_index": global_idx,
                    }
                )
                global_idx += 1
                pos = end - overlap_words if end < len(words) else end

        return chunks
