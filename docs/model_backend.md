# Model Backend Guide

This runtime keeps model-family details behind engine-level target backends. The
backends live under `dflash_mlx/engine/`; this is not a top-level plugin
framework.

## What is a TargetOps backend?

A `TargetOps` backend is the owner of target-model runtime mechanics for one
structural family. It provides the operations that `spec_epoch` needs without
letting `spec_epoch` know whether the target is hybrid GDN, attention-only, or a
future family.

A backend owns:

- target text-model unwrapping;
- token embedding access;
- logits projection and any family-specific logits post-processing;
- target cache construction;
- hidden-state capture;
- verify-block forward;
- target hook installation;
- rollback, trim, or no-op cache transaction behavior;
- model-specific capability reporting through `capabilities_for(...)`.

The current registered backends are:

- `QwenGdnTargetOps` in `dflash_mlx/engine/target_qwen_gdn.py`;
- `Gemma4TargetOps` in `dflash_mlx/engine/target_gemma4.py`.

## When do you need a new backend?

Add a backend when the target model changes runtime mechanics, not when the model
size changes.

Use the existing backend for Qwen variants only while they share the same
structural contract. Add a new backend when a model family requires different
cache semantics, mask scheduling, hidden capture, logits post-processing, shared
KV behavior, or rollback policy.

Examples that need a real backend before product support:

- Llama with sliding/full attention cache rules;
- future Gemma variants with different cache or logits rules.

## How to add a model family

1. Add one backend file under `dflash_mlx/engine/`, for example
   `target_llama.py` or `target_gemma4.py`.
2. register in `TARGET_BACKENDS` by adding the backend class in
   `dflash_mlx/engine/target_ops.py`.
3. Implement `supports_model(...)` strictly. Unknown models must fail loudly.
4. Implement `model_type(...)` so resolver errors are useful.
5. Implement `text_model(...)`, `embed_tokens(...)`, and
   `logits_from_hidden(...)`.
6. Implement `make_cache(...)` for that family’s real active target cache.
7. Implement `forward_with_hidden_capture(...)` with the family’s real masks,
   RoPE, shared-KV, and per-layer input behavior.
8. Implement rollback/trim as the family requires. If recurrent rollback is not
   supported, make that explicit in `capabilities_for(...)` and keep rollback a
   safe no-op or trim-only transaction.
9. Add resolver tests.
10. Add cache construction tests.
11. Add logits/hidden parity tests against the model’s own MLX forward path
    before claiming product support.

## Checklist

- backend file added under `dflash_mlx/engine/`;
- backend registered in `TARGET_BACKENDS`;
- strict `supports_model(...)` implemented;
- `capabilities_for(...)` reports loaded-model capabilities, not just backend
  defaults;
- text-model unwrap, embeddings, logits, cache, hidden capture, verify, and
  rollback are implemented;
- resolver rejects unsupported Llama/Mistral/unknown models until their
  backend exists;
- cache tests cover expected cache entry types;
- parity tests pass before README or product claims change.

## Anti-patterns

- no `if gemma` / `if llama` / `if qwen` branches in `runtime.py` or
  `spec_epoch.py`;
- no backend per model size, such as `target_qwen_27b.py` or
  `target_qwen_35b.py`;
- no hidden fallback to the Qwen backend for unknown models;
- no “loads therefore supported” claim;
- no compatibility wrapper that only routes old module names to the new backend;
- no paged-KV or chunk-level L2 work inside a model-backend patch.
