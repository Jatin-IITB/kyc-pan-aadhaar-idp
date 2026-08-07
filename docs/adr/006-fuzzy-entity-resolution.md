# ADR-006: Multi-Algorithm Fuzzy Entity Resolution

**Status:** Accepted  
**Date:** 2026-08-07  
**Phase:** 6  

## Context

When a KYC packet contains multiple documents (PAN + Aadhaar + Passport), the system must verify that they belong to the same person. Indian names have common variations:
- Middle name abbreviation: "RAHUL KUMAR SHARMA" vs "RAHUL K SHARMA"
- Transliteration differences: "SHARMA" vs "SARMA"
- Spelling variants: "MOHAMMAD" vs "MOHAMMED"

A simple string equality check would flag most legitimate packets as mismatches.

### Alternatives Considered

- **Exact match**: Too strict, >50% of legitimate packets would be flagged
- **Single algorithm (Levenshtein)**: Doesn't handle token reordering or abbreviations well
- **ML-based entity resolution**: Requires labeled training data of Indian name pairs
- **Third-party identity resolution API**: Cost, privacy, external dependency

## Decision

Implement a **multi-algorithm scoring ensemble** for name matching:

| Algorithm | Weight | Handles |
|-----------|--------|---------|
| Jaro-Winkler | 0.30 | Character-level typos, prefix similarity |
| Soundex | 0.10 | Phonetic variants (SHARMA/SARMA) |
| Token-Set Ratio | 0.30 | Word reordering, subset matching |
| SequenceMatcher | 0.30 | Overall structural similarity |

Combined score thresholds:
- `>= 0.85`: No mismatch flagged
- `>= 0.65`: Same person (but mismatch noted as phonetic variant)
- `< 0.65`: Different person

DOB comparison: exact date match required (CRITICAL severity if mismatch).  
Gender comparison: exact match required (HIGH severity if mismatch).  
Address comparison: `IndianAddressNormalizer` with pincode/state extraction + abbreviation expansion.

### Threshold: 0.65

Calibrated against common Indian name variant patterns. "RAHUL KUMAR SHARMA" vs "RAHUL K SHARMA" scores ~0.69, correctly identified as same person. "RAHUL SHARMA" vs "SURESH PATEL" scores <0.5, correctly identified as different person.

## Consequences

**Positive:**
- Handles the most common Indian name variations without false rejections
- No ML training data required — pure algorithmic approach
- Each algorithm contributes a different perspective on similarity
- Contradiction detection prevents identity fraud across documents

**Negative:**
- Thresholds are empirically chosen — may need tuning for specific name distributions
- Does not handle transliteration across scripts (Hindi → English variations)
- No learning from corrections — thresholds are static

**Risks:**
- Edge cases: very short names (2-3 chars) may have unreliable scores
- Mitigation: contradictions are flagged for human review, not auto-rejected
