# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import mlx.core as mx

from dflash_mlx.kernels import (
    _gated_delta_tree_ops,
    _gated_delta_tree_tape_ops,
    _gated_delta_ops_with_tape,
    gated_delta_kernel_with_tape,
    _tree_depthwise_conv1d_ops,
    gated_delta_tree_kernel,
    gated_delta_tree_tape_kernel,
    tape_replay_kernel,
    tree_depthwise_conv1d,
)


def _inputs(*, vectorized_g: bool):
    batch = 1
    steps = 4
    h_k = 1
    h_v = 2
    d_k = 32
    d_v = 4
    q = mx.arange(batch * steps * h_k * d_k, dtype=mx.float32).reshape(
        batch,
        steps,
        h_k,
        d_k,
    ) / 100.0
    k = (mx.arange(batch * steps * h_k * d_k, dtype=mx.float32) + 3).reshape(
        batch,
        steps,
        h_k,
        d_k,
    ) / 110.0
    v = (mx.arange(batch * steps * h_v * d_v, dtype=mx.float32) + 5).reshape(
        batch,
        steps,
        h_v,
        d_v,
    ) / 20.0
    beta = mx.full((batch, steps, h_v), 0.25, dtype=mx.float32)
    state = mx.zeros((batch, h_v, d_v, d_k), dtype=mx.float32)
    parent_ids = mx.array([-1, 0, 1, 1], dtype=mx.int32)
    if vectorized_g:
        g = mx.full((batch, steps, h_v, d_k), 0.91, dtype=mx.float32)
    else:
        g = mx.full((batch, steps, h_v), 0.91, dtype=mx.float32)
    return q, k, v, g, beta, state, parent_ids


def _assert_close(lhs: mx.array, rhs: mx.array, *, atol: float = 1e-4) -> None:
    mx.eval(lhs, rhs)
    assert float(mx.max(mx.abs(lhs - rhs)).item()) <= atol


def test_gated_delta_tree_kernel_matches_reference_scalar_g():
    q, k, v, g, beta, state, parent_ids = _inputs(vectorized_g=False)

    actual_y, actual_states = gated_delta_tree_kernel(q, k, v, g, beta, state, parent_ids)
    expected_y, expected_states = _gated_delta_tree_ops(q, k, v, g, beta, state, parent_ids)

    _assert_close(actual_y, expected_y.astype(q.dtype))
    _assert_close(actual_states, expected_states)


def test_gated_delta_tree_kernel_matches_reference_vector_g():
    q, k, v, g, beta, state, parent_ids = _inputs(vectorized_g=True)

    actual_y, actual_states = gated_delta_tree_kernel(q, k, v, g, beta, state, parent_ids)
    expected_y, expected_states = _gated_delta_tree_ops(q, k, v, g, beta, state, parent_ids)

    _assert_close(actual_y, expected_y.astype(q.dtype))
    _assert_close(actual_states, expected_states)


def test_gated_delta_tree_kernel_keeps_state_dtype_with_bf16_inputs():
    q, k, v, g, beta, state, parent_ids = _inputs(vectorized_g=False)
    q = q.astype(mx.bfloat16)
    k = k.astype(mx.bfloat16)
    v = v.astype(mx.bfloat16)
    g = g.astype(mx.bfloat16)
    beta = beta.astype(mx.bfloat16)

    _actual_y, actual_states = gated_delta_tree_kernel(q, k, v, g, beta, state, parent_ids)
    _expected_y, expected_states = _gated_delta_tree_ops(q, k, v, g, beta, state, parent_ids)
    mx.eval(actual_states, expected_states)

    assert actual_states.dtype == state.dtype
    _assert_close(actual_states, expected_states, atol=1e-5)


def test_gated_delta_tape_kernel_keeps_state_dtype_with_bf16_inputs():
    q, k, v, g, beta, state, _parent_ids = _inputs(vectorized_g=False)
    q = q.astype(mx.bfloat16)
    k = k.astype(mx.bfloat16)
    v = v.astype(mx.bfloat16)
    g = g.astype(mx.bfloat16)
    beta = beta.astype(mx.bfloat16)

    _actual_y, actual_state, actual_tape = gated_delta_kernel_with_tape(
        q,
        k,
        v,
        g,
        beta,
        state,
    )
    _expected_y, expected_state, _expected_tape = _gated_delta_ops_with_tape(
        q,
        k,
        v,
        g,
        beta,
        state,
    )
    mx.eval(actual_state, expected_state, actual_tape)

    assert actual_state.dtype == state.dtype
    _assert_close(actual_state, expected_state)
    assert actual_tape.dtype == mx.float32


def test_gated_delta_tree_tape_kernel_keeps_tape_f32_with_bf16_inputs():
    q, k, v, g, beta, state, parent_ids = _inputs(vectorized_g=False)
    q = q.astype(mx.bfloat16)
    k = k.astype(mx.bfloat16)
    v = v.astype(mx.bfloat16)
    g = g.astype(mx.bfloat16)
    beta = beta.astype(mx.bfloat16)

    actual_y, actual_tape = gated_delta_tree_tape_kernel(
        q,
        k,
        v,
        g,
        beta,
        state,
        parent_ids,
    )
    expected_y, expected_tape = _gated_delta_tree_tape_ops(
        q,
        k,
        v,
        g,
        beta,
        state,
        parent_ids,
    )
    mx.eval(actual_y, expected_y, actual_tape, expected_tape)

    _assert_close(actual_y, expected_y.astype(q.dtype))
    _assert_close(actual_tape, expected_tape, atol=1e-5)
    assert actual_tape.dtype == mx.float32


def test_gated_delta_tree_tape_kernel_replays_accepted_path():
    q, k, v, g, beta, state, parent_ids = _inputs(vectorized_g=False)
    accepted_slots = mx.array([0, 1, 3], dtype=mx.int32)

    actual_y, actual_tape = gated_delta_tree_tape_kernel(q, k, v, g, beta, state, parent_ids)
    expected_y, expected_states = _gated_delta_tree_ops(q, k, v, g, beta, state, parent_ids)
    replayed_state = tape_replay_kernel(
        mx.take(actual_tape, accepted_slots, axis=1),
        mx.take(k, accepted_slots, axis=1),
        mx.take(g, accepted_slots, axis=1),
        state,
        None,
    )

    _assert_close(actual_y, expected_y)
    _assert_close(replayed_state, expected_states[:, 3, ...])
    assert actual_tape.dtype == mx.float32


def test_tree_depthwise_conv1d_matches_reference_ops():
    batch = 1
    steps = 4
    conv_dim = 6
    kernel = 4
    qkv = mx.arange(batch * steps * conv_dim, dtype=mx.float32).reshape(
        batch,
        steps,
        conv_dim,
    ) / 100.0
    prefix = (mx.arange(batch * (kernel - 1) * conv_dim, dtype=mx.float32) + 2).reshape(
        batch,
        kernel - 1,
        conv_dim,
    ) / 90.0
    weight = (mx.arange(conv_dim * kernel, dtype=mx.float32) + 1).reshape(
        conv_dim,
        kernel,
        1,
    ) / 80.0
    path_indices = mx.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [2, 3, 4, 6],
        ],
        dtype=mx.int32,
    )

    actual = tree_depthwise_conv1d(qkv, prefix, weight, None, path_indices)
    expected = _tree_depthwise_conv1d_ops(qkv, prefix, weight, None, path_indices)

    _assert_close(actual, expected)
