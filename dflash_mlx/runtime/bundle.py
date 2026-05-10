# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dflash_mlx.draft_backend import DraftBackend, make_draft_backend
from dflash_mlx.engine.target_ops import TargetOps, bind_draft_to_target
from dflash_mlx.runtime import VerifyConfig
from dflash_mlx.runtime.loading import (
    load_draft_bundle,
    load_target_bundle,
)
from dflash_mlx.runtime.registry import (
    ModelSupportSpec,
    resolve_model_support_spec,
    resolve_optional_draft_ref,
    supported_base_models,
    validate_support_spec_family,
)


@dataclass(frozen=True)
class RuntimeBundle:
    target_model: Any
    tokenizer: Any
    target_meta: dict[str, Any]
    draft_model: Any
    draft_meta: dict[str, Any]
    draft_backend: DraftBackend
    target_ops: TargetOps
    resolved_model_ref: str
    resolved_draft_ref: str
    support_spec: ModelSupportSpec | None


def load_runtime_bundle(
    *,
    model_ref: str | Path,
    draft_ref: str | None,
    draft_quant: str | None = None,
    verify_config: VerifyConfig | None = None,
    split_full_attention_sdpa: bool | None = None,
    quantize_kv_cache: bool = False,
    lazy: bool = True,
) -> RuntimeBundle:
    support_spec = resolve_model_support_spec(str(model_ref))
    resolved_draft_ref = resolve_optional_draft_ref(str(model_ref), draft_ref)
    if not resolved_draft_ref:
        raise ValueError(
            f"No DFlash draft model found for '{model_ref}'.\n"
            f"Use --draft to specify one, or check https://huggingface.co/z-lab for available drafts.\n"
            f"Supported base models: {supported_base_models()}"
        )
    target_bundle = load_target_bundle(
        model_ref,
        lazy=lazy,
        split_full_attention_sdpa=split_full_attention_sdpa,
        quantize_kv_cache=quantize_kv_cache,
        verify_config=verify_config,
    )
    target_model = target_bundle.model
    tokenizer = target_bundle.tokenizer
    target_meta = target_bundle.meta
    target_ops = target_bundle.target_ops
    resolved_model_ref = str(target_meta.get("resolved_model_ref") or model_ref)
    if support_spec is None:
        support_spec = resolve_model_support_spec(resolved_model_ref)

    if support_spec is not None:
        validate_support_spec_family(
            model_ref=resolved_model_ref,
            support_spec=support_spec,
            actual_family=target_ops.family(target_model),
        )

    draft_model, draft_meta = load_draft_bundle(
        resolved_draft_ref,
        lazy=lazy,
        draft_quant=draft_quant,
    )
    draft_backend = make_draft_backend()
    bind_draft_to_target(draft_model, target_model, target_ops=target_ops)
    return RuntimeBundle(
        target_model=target_model,
        tokenizer=tokenizer,
        target_meta=target_meta,
        draft_model=draft_model,
        draft_meta=draft_meta,
        draft_backend=draft_backend,
        target_ops=target_ops,
        resolved_model_ref=resolved_model_ref,
        resolved_draft_ref=resolved_draft_ref,
        support_spec=support_spec,
    )
