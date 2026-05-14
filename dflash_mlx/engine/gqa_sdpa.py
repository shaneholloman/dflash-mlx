# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention

_GQA_SDPA_STREAMS: dict[int, list[Any]] = {}
_GQA_MASK_CACHE_MAX = 8
_GQA_MASK_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def tail_causal_mask(q_len: int, kv_len: int) -> mx.array:
    q_pos = mx.arange(kv_len - q_len, kv_len)[:, None]
    k_pos = mx.arange(kv_len)[None, :]
    return k_pos <= q_pos


def _cached_gqa_mask(key: tuple[Any, ...], source: Any, mask: Any) -> Any:
    if len(_GQA_MASK_CACHE) >= _GQA_MASK_CACHE_MAX:
        _GQA_MASK_CACHE.clear()
    _GQA_MASK_CACHE[key] = (source, mask)
    return mask


def repeat_gqa_mask(mask: Any, *, q_len: int, kv_len: int, gqa: int) -> Any:
    if mask is None:
        return None
    if isinstance(mask, str) and mask == "causal":
        key = ("causal", int(q_len), int(kv_len), int(gqa))
        cached = _GQA_MASK_CACHE.get(key)
        if cached is not None:
            return cached[1]
        mask = tail_causal_mask(q_len, kv_len)
        reps = [1] * mask.ndim
        reps[-2] = int(gqa)
        return _cached_gqa_mask(key, None, mx.tile(mask, tuple(reps)))
    if not isinstance(mask, mx.array):
        return mask
    if int(mask.shape[-2]) != q_len:
        return mask
    key = (
        "array",
        id(mask),
        tuple(int(dim) for dim in mask.shape),
        str(mask.dtype),
        int(gqa),
    )
    cached = _GQA_MASK_CACHE.get(key)
    if cached is not None and cached[0] is mask:
        return cached[1]
    reps = [1] * mask.ndim
    reps[-2] = int(gqa)
    return _cached_gqa_mask(key, mask, mx.tile(mask, tuple(reps)))


def grouped_gqa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any] = None,
) -> mx.array:
    batch_size, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, kv_len, _ = keys.shape
    if kv_heads <= 0 or query_heads == kv_heads or query_heads % kv_heads != 0:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )
    gqa = query_heads // kv_heads
    grouped_queries = queries.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, kv_heads, gqa * q_len, head_dim)
    grouped_mask = repeat_gqa_mask(mask, q_len=q_len, kv_len=kv_len, gqa=gqa)
    output = scaled_dot_product_attention(
        grouped_queries,
        keys,
        values,
        cache=cache,
        scale=scale,
        mask=grouped_mask,
    )
    return output.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, query_heads, q_len, head_dim)


def per_head_gqa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: Optional[Any],
    gqa: int,
) -> mx.array:
    batch_size, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, _, _ = keys.shape
    grouped_queries = queries.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, kv_heads, gqa * q_len, head_dim)
    outputs = [
        scaled_dot_product_attention(
            grouped_queries[:, head : head + 1, :, :],
            keys[:, head : head + 1, :, :],
            values[:, head : head + 1, :, :],
            cache=None,
            scale=scale,
            mask=mask,
        )
        for head in range(kv_heads)
    ]
    output = mx.concatenate(outputs, axis=1)
    return output.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, query_heads, q_len, head_dim)


def _gqa_streams_for(kv_heads: int) -> list[Any]:
    if kv_heads not in _GQA_SDPA_STREAMS:
        _GQA_SDPA_STREAMS[kv_heads] = [mx.new_stream(mx.gpu) for _ in range(kv_heads)]
    return _GQA_SDPA_STREAMS[kv_heads]


def async_per_head_gqa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask: Optional[Any],
    gqa: int,
) -> mx.array:
    batch_size, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, _, _ = keys.shape
    streams = _gqa_streams_for(kv_heads)
    grouped_queries = queries.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, kv_heads, gqa * q_len, head_dim)
    outputs = []
    for head in range(kv_heads):
        with mx.stream(streams[head]):
            output = scaled_dot_product_attention(
                grouped_queries[:, head : head + 1, :, :],
                keys[:, head : head + 1, :, :],
                values[:, head : head + 1, :, :],
                cache=None,
                scale=scale,
                mask=mask,
            )
            mx.async_eval(output)
            outputs.append(output)
    output = mx.concatenate(outputs, axis=1)
    return output.reshape(
        batch_size,
        kv_heads,
        gqa,
        q_len,
        head_dim,
    ).reshape(batch_size, query_heads, q_len, head_dim)
