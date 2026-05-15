# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx
import pytest

from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps

pytestmark = pytest.mark.skipif(
    os.environ.get("DFLASH_RUN_REAL_MODEL_TESTS") != "1",
    reason="set DFLASH_RUN_REAL_MODEL_TESTS=1 to run local real-model parity tests",
)

def _local_model_path() -> str:
    repo_id = os.environ.get(
        "DFLASH_REAL_QWEN_GDN_MODEL",
        "mlx-community/Qwen3.6-27B-4bit",
    )
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        pytest.skip(f"huggingface_hub unavailable: {exc}")
    try:
        return snapshot_download(repo_id, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"local Qwen hybrid model not present for {repo_id}: {exc}")

@lru_cache(maxsize=1)
def _load_model():
    from mlx_lm.utils import load

    model, tokenizer = load(_local_model_path(), lazy=True)
    return model, tokenizer

def _token_ids(tokenizer) -> mx.array:
    ids = list(tokenizer.encode("Hello world. Write one short sentence."))
    if len(ids) < 6:
        ids = [1, 2, 3, 4, 5, 6]
    return mx.array(ids[:6], dtype=mx.uint32)[None]

def _assert_close(lhs: mx.array, rhs: mx.array, *, atol: float = 1e-3) -> None:
    mx.eval(lhs, rhs)
    max_abs = float(mx.abs(lhs - rhs).max())
    assert max_abs <= atol

def _argmax_token(logits: mx.array) -> int:
    token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(token)
    return int(token.item())

def _greedy_ids(model, prompt_ids: list[int], max_tokens: int) -> list[int]:
    cache = model.make_cache()
    logits = model(mx.array(prompt_ids, dtype=mx.uint32)[None], cache=cache)[:, -1, :]
    mx.eval(logits)
    out = []
    for _ in range(max_tokens):
        token = int(mx.argmax(logits, axis=-1).item())
        out.append(token)
        logits = model(mx.array([[token]], dtype=mx.uint32), cache=cache)[:, -1, :]
        mx.eval(logits)
    return out

def test_real_qwen_gdn_hooks_preserve_stock_greedy_trajectory():
    model, tokenizer = _load_model()
    ops = resolve_target_ops(model)
    assert isinstance(ops, QwenGdnTargetOps)

    prompt_ids = list(
        tokenizer.encode("Solve carefully and give the final answer: What is 1234 + 5678?")
    )
    before = _greedy_ids(model, prompt_ids, max_tokens=36)
    ops.install_speculative_hooks(model)
    after = _greedy_ids(model, prompt_ids, max_tokens=36)

    assert after == before

def test_real_qwen_gdn_target_ops_forward_cache_and_rollback_parity():
    model, tokenizer = _load_model()
    ops = resolve_target_ops(model)
    assert isinstance(ops, QwenGdnTargetOps)
    assert ops.family(model) == "hybrid_gdn"

    tokens = _token_ids(tokenizer)
    prompt = tokens[:, :3]
    verify = tokens[:, 3:5]
    next_token = tokens[:, 5:6]

    native_logits = model(prompt, cache=None)
    ops_logits, captured = ops.forward_with_hidden_capture(model, input_ids=prompt, cache=None)
    mx.eval(native_logits, ops_logits)
    assert captured
    _assert_close(ops_logits, native_logits)

    native_last_cache = model.make_cache()
    ops_last_cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    native_last_cached = model(prompt, cache=native_last_cache)
    ops_last_cached, ops_last_captured = ops.forward_with_hidden_capture(
        model,
        input_ids=prompt,
        cache=ops_last_cache,
        capture_layer_ids={1},
        logits_last_only=True,
    )
    mx.eval(native_last_cached, ops_last_cached, *ops_last_captured.values())
    assert tuple(ops_last_cached.shape[:2]) == (1, 1)
    assert -1 in ops_last_captured
    _assert_close(ops_last_cached, native_last_cached[:, -1:, :])
    native_after_last = model(verify[:, :1], cache=native_last_cache)
    ops_after_last, _ = ops.forward_with_hidden_capture(
        model,
        input_ids=verify[:, :1],
        cache=ops_last_cache,
    )
    _assert_close(ops_after_last, native_after_last)

    native_cache = model.make_cache()
    ops_cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    model(prompt, cache=native_cache)
    ops.forward_with_hidden_capture(model, input_ids=prompt, cache=ops_cache)
    native_next = model(verify[:, :1], cache=native_cache)
    ops_next, _ = ops.forward_with_hidden_capture(model, input_ids=verify[:, :1], cache=ops_cache)
    _assert_close(ops_next, native_next)

    tx_cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    clean_cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    ops.forward_with_hidden_capture(model, input_ids=prompt, cache=tx_cache)
    ops.forward_with_hidden_capture(model, input_ids=prompt, cache=clean_cache)
    ops.arm_rollback(tx_cache, prefix_len=int(prompt.shape[1]))
    ops.verify_block(
        target_model=model,
        verify_ids=verify,
        target_cache=tx_cache,
        capture_layer_ids=None,
    )
    ops.restore_after_acceptance(
        tx_cache,
        target_len=int(prompt.shape[1]) + 1,
        acceptance_length=0,
        drafted_tokens=1,
    )
    ops.forward_with_hidden_capture(model, input_ids=verify[:, :1], cache=clean_cache)
    tx_after, _ = ops.forward_with_hidden_capture(model, input_ids=next_token, cache=tx_cache)
    clean_after, _ = ops.forward_with_hidden_capture(model, input_ids=next_token, cache=clean_cache)
    _assert_close(tx_after, clean_after, atol=2e-3)

def test_real_qwen36_27b_quantized_target_kv_one_token_decode_smoke():
    model, tokenizer = _load_model()
    ops = resolve_target_ops(model)
    assert isinstance(ops, QwenGdnTargetOps)

    tokens = _token_ids(tokenizer)
    prompt = tokens[:, :5]
    next_token = tokens[:, 5:6]

    fp_cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    quant_cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=True,
        target_fa_window=0,
    )

    ops.forward_with_hidden_capture(model, input_ids=prompt, cache=fp_cache)
    ops.forward_with_hidden_capture(model, input_ids=prompt, cache=quant_cache)
    fp_next, _ = ops.forward_with_hidden_capture(model, input_ids=next_token, cache=fp_cache)
    quant_next, _ = ops.forward_with_hidden_capture(model, input_ids=next_token, cache=quant_cache)
    mx.eval(fp_next, quant_next)

    assert _argmax_token(quant_next) == _argmax_token(fp_next)
    max_abs = float(mx.abs(quant_next[:, -1, :] - fp_next[:, -1, :]).max())
    assert max_abs <= 2.0
