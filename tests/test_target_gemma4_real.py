# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx
import pytest
from mlx_lm.generate import generate_step

from dflash_mlx.engine.events import SummaryEvent, TokenEvent
from dflash_mlx.engine.target_gemma4 import Gemma4TargetOps
from dflash_mlx.runtime import VerifyConfig, stream_dflash_generate
from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.loading import load_draft_bundle, load_target_bundle
from dflash_mlx.runtime.context import build_offline_runtime_context

pytestmark = pytest.mark.skipif(
    os.environ.get("DFLASH_RUN_GEMMA4_REAL_MODEL_TESTS") != "1",
    reason="set DFLASH_RUN_GEMMA4_REAL_MODEL_TESTS=1 to run local Gemma4 parity tests",
)


def _local_snapshot(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        pytest.skip(f"huggingface_hub unavailable: {exc}")
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"local Gemma4 model not present for {repo_id}: {exc}")


def _target_ref() -> str:
    return os.environ.get(
        "DFLASH_REAL_GEMMA4_TARGET",
        "mlx-community/gemma-4-31b-it-4bit",
    )


def _draft_ref() -> str:
    return os.environ.get(
        "DFLASH_REAL_GEMMA4_DRAFT",
        "z-lab/gemma-4-31B-it-DFlash",
    )


@lru_cache(maxsize=1)
def _load_target():
    return load_target_bundle(
        _local_snapshot(_target_ref()),
        lazy=True,
        verify_config=VerifyConfig.from_mode("off"),
    )


def _token_ids(tokenizer) -> mx.array:
    ids = _prompt_token_ids(tokenizer)
    return mx.array(ids[:8], dtype=mx.uint32)[None]


def _prompt_token_ids(tokenizer) -> list[int]:
    return list(tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hello in one short sentence."}],
        tokenize=True,
        add_generation_prompt=True,
    ))


def _mlx_lm_greedy_tokens(model, prompt_ids: list[int], max_tokens: int) -> list[int]:
    prompt = mx.array(prompt_ids, dtype=mx.uint32)
    tokens: list[int] = []
    for token, _ in generate_step(prompt, model, max_tokens=max_tokens):
        tokens.append(int(token))
        if len(tokens) >= max_tokens:
            break
    return tokens


def test_real_gemma4_target_ops_load_cache_and_logits_parity():
    target_bundle = _load_target()
    model = target_bundle.model
    tokenizer = target_bundle.tokenizer
    meta = target_bundle.meta
    ops = target_bundle.target_ops
    assert isinstance(ops, Gemma4TargetOps)
    assert ops.family(model) == "gemma4_swa"
    assert meta["verify_linear_enabled"] is False

    caches = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    expected_caches = model.make_cache()
    assert len(caches) == len(expected_caches)
    assert [type(cache) for cache in caches] == [type(cache) for cache in expected_caches]

    tokens = _token_ids(tokenizer)
    capture_keys = {
        int(layer_id) + 1
        for layer_id in getattr(_load_draft_for_capture(), "target_layer_ids", ())
    }
    native = model(tokens, cache=None)
    custom, captured = ops.forward_with_hidden_capture(
        model,
        input_ids=tokens,
        cache=None,
        capture_layer_ids=capture_keys,
    )
    mx.eval(native, custom, *captured.values())

    assert sorted(captured) == sorted(capture_keys)
    assert float(mx.max(mx.abs(native - custom)).item()) <= 1e-6
    assert int(mx.argmax(native[:, -1, :], axis=-1).item()) == int(
        mx.argmax(custom[:, -1, :], axis=-1).item()
    )

    native_cache = model.make_cache()
    ops_cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    native_logits = model(tokens, cache=native_cache)
    ops_logits, _ = ops.forward_with_hidden_capture(model, input_ids=tokens, cache=ops_cache)
    for _ in range(8):
        native_next = mx.argmax(native_logits[:, -1, :], axis=-1).astype(mx.uint32)
        ops_next = mx.argmax(ops_logits[:, -1, :], axis=-1).astype(mx.uint32)
        mx.eval(native_next, ops_next)
        assert int(native_next.item()) == int(ops_next.item())
        native_logits = model(native_next[:, None], cache=native_cache)
        ops_logits, _ = ops.forward_with_hidden_capture(
            model,
            input_ids=ops_next[:, None],
            cache=ops_cache,
        )


@lru_cache(maxsize=1)
def _load_draft_for_capture():
    draft_model, _ = load_draft_bundle(
        _local_snapshot(_draft_ref()),
        lazy=True,
    )
    return draft_model


def test_real_gemma4_dflash_matches_mlx_lm_greedy_tokens():
    target_path = _local_snapshot(_target_ref())
    draft_path = _local_snapshot(_draft_ref())
    runtime_context = build_offline_runtime_context(
        target_fa_window=0,
        verify_mode="auto",
    )
    bundle = load_runtime_bundle(
        model_ref=target_path,
        draft_ref=draft_path,
        verify_config=VerifyConfig.from_mode("auto"),
    )
    target_model = bundle.target_model
    tokenizer = bundle.tokenizer
    target_meta = bundle.target_meta
    draft_model = bundle.draft_model
    draft_backend = bundle.draft_backend
    target_ops = bundle.target_ops
    assert target_meta["verify_linear_enabled"] is True
    assert int(target_meta.get("verify_linear_swapped") or 0) > 0

    max_tokens = 16
    prompt_ids = _prompt_token_ids(tokenizer)
    expected_tokens = _mlx_lm_greedy_tokens(target_model, prompt_ids, max_tokens)
    mx.clear_cache()

    generated_tokens: list[int] = []
    summary = None
    for event in stream_dflash_generate(
        target_model=target_model,
        target_ops=target_ops,
        tokenizer=tokenizer,
        draft_model=draft_model,
        draft_backend=draft_backend,
        prompt="",
        max_new_tokens=max_tokens,
        use_chat_template=False,
        stop_token_ids=[],
        prompt_tokens_override=prompt_ids,
        runtime_context=runtime_context,
    ):
        if isinstance(event, TokenEvent):
            generated_tokens.append(int(event.token_id))
        elif isinstance(event, SummaryEvent):
            summary = event
            break

    assert summary is not None
    assert generated_tokens == expected_tokens
    assert list(summary.generated_token_ids) == expected_tokens
    assert int(summary.generation_tokens) == max_tokens
    assert int(summary.accepted_from_draft) > 0
