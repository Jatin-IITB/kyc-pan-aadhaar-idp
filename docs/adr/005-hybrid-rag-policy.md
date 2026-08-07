# ADR-005: Hybrid RAG for Policy Compliance Verification

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 5  

## Context

KYC must comply with Indian regulations: RBI KYC Master Direction 2016, PMLA Rules 2005, Aadhaar e-KYC guidelines, SEBI KYC requirements. Hardcoding rules is brittle — regulations change, and different institutions may have different interpretations.

### Alternatives Considered

- **Hardcoded rule engine**: Brittle, requires code changes for regulation updates
- **Pure dense retrieval**: Misses keyword-specific regulatory language (section numbers, specific terms)
- **Pure BM25**: Misses semantic similarity for paraphrased requirements
- **Fine-tuned compliance model**: Requires labeled compliance training data, expensive to maintain

## Decision

Implement a **hybrid RAG pipeline** with citation-grounded verification:

1. **Indexing**: Chunk policy markdown files (512 tokens, 50 overlap) → embed with `BAAI/bge-small-en-v1.5` → upsert to Qdrant
2. **Retrieval**: Dense (Qdrant vector) + Sparse (BM25 in-memory) → Reciprocal Rank Fusion (k=60)
3. **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` scores top candidates
4. **Verification**: LLM judges each requirement against retrieved policy chunks → PASS/FAIL/NOT_APPLICABLE per check, with section citations

Policy documents stored as markdown in `config/policies/` — updatable without code changes.

## Consequences

**Positive:**
- Regulations updatable by editing markdown files + re-indexing
- Citations provide auditability — each decision traceable to a specific regulation section
- Hybrid retrieval handles both exact regulatory language and semantic paraphrases
- Graceful degradation — if Qdrant/embedding model unavailable, node is skipped

**Negative:**
- Requires Qdrant running (added infrastructure)
- Embedding model (`bge-small-en-v1.5`) + cross-encoder add ~500MB to deployment
- Initial indexing requires `scripts/seed_qdrant.py` execution
- LLM-as-judge may misinterpret complex regulatory nuance

**Risks:**
- Policy markdown files are summaries, not full legal text — may miss edge cases
- Mitigation: designed for operational filtering, not legal opinions
