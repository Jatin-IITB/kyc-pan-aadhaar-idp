# ADR-018: Celery Solo Pool for VLM Inference

**Status:** Accepted  
**Date:** 2026-08-08

## Context

The VLM extractor (minicpm-v via Ollama) works correctly when called directly but crashes the Celery worker with SIGSEGV (signal 11) when run through the pipeline. The crash occurs because Celery's default `prefork` pool uses `fork()` to create worker processes. After the VLM inference completes and returns results, the forked process attempts to clean up GPU/Metal memory mappings that were inherited from the parent process, triggering a segmentation fault.

This is a known issue with fork-based multiprocessing and GPU frameworks (Metal, CUDA). The parent process initializes GPU memory, `fork()` copies the memory mappings, and when the child exits, the GPU driver attempts to free memory it doesn't own.

## Decision

Use `--pool=solo` for the Celery worker when VLM inference is enabled. The solo pool executes tasks in the main process without forking, avoiding the GPU memory conflict entirely.

Additional fixes made alongside:
1. **LLM cleaner default URL**: Changed from `host.docker.internal:11434` (Docker-only) to `localhost:11434` for local development compatibility.
2. **LLM config passthrough**: Pipeline loader now reads `llm.model` and `llm.timeout_s` from `config/models.yaml` and passes them to `LLMKycCleaner`, instead of relying on hardcoded defaults.
3. **LLM rescue skip**: When `schema_valid` is already True, the LLM rescue node returns immediately instead of making a 20s+ Ollama call. This reduced pipeline time from ~132s to ~49s.
4. **Logger format strings**: Fixed `{}` to `%s` in `llm_rescue.py` and `extract_vlm.py` logger calls.

## Consequences

- **Pros**: VLM inference works reliably in Celery; no SIGSEGV crashes; 49s pipeline time.
- **Cons**: Solo pool processes one task at a time. For production, use a separate solo-pool worker for VLM tasks and a prefork-pool worker for non-VLM tasks via Celery routing.
- **Production pattern**: Route VLM tasks to a dedicated `vlm` queue with `--pool=solo --concurrency=1 -Q vlm`, and keep a prefork worker for lightweight tasks.
