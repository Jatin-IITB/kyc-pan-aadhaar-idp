# ADR-007: SHA-256 Hash-Chained Immutable Audit Ledger

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 7  

## Context

KYC processing decisions must be auditable for regulatory compliance. The audit trail must be:
1. **Immutable** — events cannot be retroactively modified without detection
2. **Complete** — every processing stage produces an event
3. **Verifiable** — integrity can be checked independently
4. **Replayable** — state at any point can be reconstructed

### Alternatives Considered

- **Database-only audit log**: Mutable — a DB admin could modify records undetected
- **Blockchain**: Overkill for single-organization audit, introduces consensus overhead
- **Append-only file log**: No integrity verification, no chain guarantee
- **Event sourcing framework (Eventuous, etc.)**: Additional dependency, learning curve

## Decision

Implement a **hash-chained audit ledger** inspired by blockchain principles but without the consensus overhead:

- **Hash function**: SHA-256
- **Chain formula**: `event_hash = SHA-256(prev_hash || canonical_json(payload))`
- **Canonical JSON**: sorted keys, no whitespace, `default=str` for non-serializable types
- **Genesis**: first event's `prev_hash` = `"0" * 64`
- **10 event types**: quality_check, classification, extraction, validation, forensics, policy_check, cross_doc_check, decision, correction, review

Verification: walk the chain, recompute each hash, compare. Any modification to any event's payload breaks the chain from that point forward.

`AuditReplayer` reconstructs case state at any audit point by applying events in sequence — enables "what was the state at step N?" queries.

## Consequences

**Positive:**
- Tamper detection without external infrastructure — just SHA-256
- Full state reconstruction at any point via replay
- Diff between two points shows exactly what changed
- Regulatory auditors can independently verify chain integrity

**Negative:**
- Hash chain is per-case, not global — a case's entire chain could be deleted (mitigated by DB persistence)
- Canonical JSON serialization adds slight overhead per event
- Chain length grows linearly with processing stages (8-10 events per case)

**Risks:**
- If `_canonical_json` implementation changes, old chains won't verify — must freeze the serialization format
