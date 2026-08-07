# ADR-015: Use Ollama /api/chat Instead of Hardcoded Prompt Tokens

## Status
Accepted

## Context
The PolicyVerifier was constructing prompts with hardcoded Llama 3 chat tokens (`<|begin_of_text|>`, `<|start_header_id|>`, etc.) and sending them via the `/api/generate` endpoint. This broke model portability — switching to a different model (Mistral, Gemma, Qwen) would produce garbled output because each model family uses different special tokens.

## Decision
Switch PolicyVerifier from `/api/generate` with manual prompt templating to `/api/chat` with structured `messages` array (system + user roles). Ollama's chat endpoint handles model-specific tokenization internally, making the code model-agnostic.

## Consequences
- PolicyVerifier works with any Ollama-supported model without code changes
- System/user role separation is cleaner and matches the LLM's training format
- Response parsing changes from `resp["response"]` to `resp["message"]["content"]`
- Also fixed loguru→stdlib logging in the same file for consistency
