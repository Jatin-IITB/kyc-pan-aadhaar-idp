# ADR-050 — RAG Evidence Harness

**Status:** Proposed — baseline run pending  
**Date:** 2026-09-04  
**Phase:** 12 / WS-A

## Context

ADR-005 selected dense retrieval, BM25, reciprocal-rank fusion, and
cross-encoder reranking without measuring whether each component improved
retrieval. The policy corpus also had no authored relevance set, retrieval
metrics, citation-support measurement, or regression workflow.

The evaluation must not derive labels from retriever output. Doing so would
measure self-consistency rather than retrieval quality.

## Decision

### D1: Structural labels and stable chunk identities

`data/rag_eval/golden.jsonl` contains 72 independently authored questions:

- 42 single-section lookups
- 10 multi-section synthesis questions
- 10 unsupported questions that should abstain
- 10 near-miss questions with lexical distractors

Labels reference deterministic IDs of the form
`source_file::section_header::section_chunk_index`. The indexer now persists
those IDs and uses a UUIDv5 Qdrant point ID, so re-indexing cannot invalidate
the golden labels. Fit/report assignment depends only on the SHA-256 of the
question ID. The initial split is 30 fit / 42 report, and the authored dataset
hash is:

`e08b42a24650703d8be1941310f8d521b44c449eb8d375d0e12ef4a8191e54ae`

### D2: Five-way ablation on the real retrieval components

`tools/eval/rag_eval.py` builds the production chunks in an ephemeral Qdrant
collection and reports recall@1/5/10, MRR, nDCG@10, negative abstention, and
latency for:

1. dense only
2. BM25 only
3. dense + BM25 with RRF
4. RRF + cross-encoder (the shipped architecture)
5. BM25 + cross-encoder

The report split is the certification split. Both split metrics and individual
rankings are retained for diagnosis.

### D3: Citation support is separate from retrieval relevance

`tools/eval/rag_faithfulness.py` runs actual `PolicyVerifier` checks, resolves
each emitted citation back to the indexed policy corpus, and asks `qwen3:8b`
whether the cited chunk entails the verdict. The rubric disallows outside
knowledge and distinguishes `SUPPORTED`, `UNSUPPORTED`, and `UNVERIFIABLE`.

A deterministic 20-verdict sample is reserved for human review. Human labels
carry a hash of the exact verdict, so agreement cannot silently be reported
against changed model output. qwen thinking is disabled for this structured
classification; temperature and seed are fixed at zero.

### D4: Preserve downstream decision and audit compatibility

The existing graph consumers read `policy_result.compliant` and
`policy_result.citations`, while the verifier originally returned only
`overall_status` and nested checks. The verifier now emits both schemas:
`compliant` is true only for `COMPLIANT`, and citations are flattened for the
audit ledger. Retrieval failures return `REQUIRES_REVIEW` with
`compliant: false`, preventing an unavailable policy store from silently
auto-clearing a case.

### D5: Baseline and gates are a follow-up measurement commit

No retrieval or citation number is claimed in this commit. The designated
model host is the Windows evaluation machine. After it produces the baseline,
the next commit will:

1. record the five-way fit/report table and citation-support result here;
2. commit the 20 human labels and judge agreement;
3. add measured floors under the new `rag:` threshold block; and
4. change the RAG workflow from evidence publication to `--check`.

This separation avoids inventing an aspirational gate or committing a
tautological zero floor before the baseline exists.

## Reproduction on the Windows model host

This workstream evaluates pretrained models; it does not train a RAG model.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install "transformers==4.37.2"
.\.venv\Scripts\python -m tools.eval.rag_eval

ollama pull qwen3:8b
.\.venv\Scripts\python -m tools.eval.rag_eval --faithfulness
```

Outputs are written to `eval/rag_metrics.json`. The citation command also
requires the committed human-review labels before judge agreement can be
certified.

## Limitations

- The corpus is six substantive sections from two summary documents. Results
  characterize this toy corpus, not a full RBI Master Direction.
- The model repositories are named but their upstream Hugging Face revisions
  are not yet pinned.
- Negative abstention is measured as an empty retrieval result; the shipped
  dense/RRF path currently has no explicit relevance threshold.
- Citation judging remains model-dependent and is not CI-safe without a local
  qwen runner.

## README deltas

Do not update the README yet. After the baseline lands, replace the unmeasured
hybrid-RAG claim with the report-split ablation result, citation support rate,
20-case judge agreement, dataset hash, and the toy-corpus qualification.
