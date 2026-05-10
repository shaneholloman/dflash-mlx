# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.cache.codecs import PrefixSnapshotBuilder
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.draft_backend import DraftBackend
from dflash_mlx.engine.target_ops import TargetOps
from dflash_mlx.model import DFlashDraftModel


def get_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_token_ids = list(getattr(tokenizer, "eos_token_ids", None) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in eos_token_ids:
        eos_token_ids.append(int(eos_token_id))
    return eos_token_ids


def _prepare_prompt_tokens(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))


def build_suppress_token_mask(
    vocab_size: int,
    suppress_token_ids: Optional[list[int]],
) -> Optional[mx.array]:
    token_ids = sorted(
        {
            int(token_id)
            for token_id in (suppress_token_ids or [])
            if 0 <= int(token_id) < vocab_size
        }
    )
    if not token_ids:
        return None
    vocab_indices = mx.arange(vocab_size, dtype=mx.int32)
    token_array = mx.array(token_ids, dtype=mx.int32)
    return mx.any(mx.equal(vocab_indices[:, None], token_array[None, :]), axis=1)


def greedy_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    floor = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(suppress_token_mask, floor, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)


def _eval_logits_and_captured(
    logits: mx.array,
    captured: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)


def _ns_to_us(ns: int | float) -> float:
    return float(ns) / 1_000.0


@dataclass(frozen=True)
class VerifyConfig:
    mode: str = "auto"
    enable_qmm: bool = True

    @classmethod
    def from_mode(cls, mode: str | None) -> "VerifyConfig":
        resolved = (mode or "auto").strip().lower()
        if resolved not in ("auto", "off"):
            raise ValueError("verify mode must be auto or off")
        return cls(mode=resolved)


def stream_dflash_generate(
    *,
    target_model: Any = None,
    target_ops: TargetOps | None = None,
    tokenizer: Any = None,
    draft_model: DFlashDraftModel | None = None,
    draft_backend: DraftBackend | None = None,
    prompt: str = "",
    max_new_tokens: int = 0,
    use_chat_template: bool = False,
    block_tokens: int | None = None,
    stop_token_ids: list[int] | None = None,
    suppress_token_ids: list[int] | None = None,
    prompt_tokens_override: list[int] | None = None,
    quantize_kv_cache: bool = False,
    prefix_snapshot: DFlashPrefixSnapshot | None = None,
    prefix_snapshot_builder: PrefixSnapshotBuilder | None = None,
    stable_prefix_len: int | None = None,
    prefix_cache_active: bool = False,
    runtime_context: Any = None,
) -> Iterator[dict[str, Any]]:
    if runtime_context is None:
        raise ValueError("runtime_context is required")
    if target_ops is None:
        raise ValueError("target_ops is required")
    if draft_backend is None:
        raise ValueError("draft_backend is required")
    gen_stream = mx.default_stream(mx.default_device())
    with mx.stream(gen_stream):
        yield from _stream_dflash_generate_impl(
            target_model=target_model,
            target_ops=target_ops,
            tokenizer=tokenizer,
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            block_tokens=block_tokens,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens_override,
            quantize_kv_cache=quantize_kv_cache,
            prefix_snapshot=prefix_snapshot,
            prefix_snapshot_builder=prefix_snapshot_builder,
            stable_prefix_len=stable_prefix_len,
            prefix_cache_active=prefix_cache_active,
            runtime_context=runtime_context,
        )


from dflash_mlx.engine.fallback import stream_baseline_generate
from dflash_mlx.engine.spec_epoch import (
    stream_dflash_generate_impl as _stream_dflash_generate_impl,
)
