#!/usr/bin/env python3
"""Seed the Qdrant vector store with KYC policy documents.

Usage:
    python -m scripts.seed_qdrant --qdrant-url http://localhost:6333
    python -m scripts.seed_qdrant --policies-dir config/policies
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index KYC policy Markdown files into Qdrant."
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL (default: http://localhost:6333)",
    )
    parser.add_argument(
        "--policies-dir",
        type=str,
        default="config/policies",
        help="Directory containing .md policy files (default: config/policies)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="kyc_policies",
        help="Qdrant collection name (default: kyc_policies)",
    )
    args = parser.parse_args()

    policies_path = Path(args.policies_dir)
    if not policies_path.is_dir():
        logger.error("Policies directory does not exist: {}", policies_path)
        sys.exit(1)

    from services.rag.indexer import PolicyIndexer

    indexer = PolicyIndexer(
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
    )
    count = indexer.index_policies(policies_path)
    logger.info("Done. Indexed {} chunks into collection '{}'.", count, args.collection)


if __name__ == "__main__":
    main()
