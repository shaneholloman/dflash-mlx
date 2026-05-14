# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Optional

import mlx.core as mx

def _make_gated_delta_kernel_with_tape(*, has_mask: bool = False, vectorized: bool = False):
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y, tape: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto tape_ = innovation_tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          float delta = 0.0f;
          if ({mask_source}) {{
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              kv_mem += state[i] * k_[s_idx];
            }}
            kv_mem = simd_sum(kv_mem);

            delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }}
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {{
              y[dv_idx] = static_cast<InT>(out);
            }}
          }}
          if (thread_index_in_simdgroup == 0) {{
            tape_[dv_idx] = delta;
          }}
          for (int i = 0; i < n_per_t; ++i) {{
            state[i] = static_cast<float>(static_cast<InT>(state[i]));
          }}
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          tape_ += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}

        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """

    inputs = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"gated_delta_tape{suffix}",
        input_names=inputs,
        output_names=["y", "state_out", "innovation_tape"],
        source=source,
    )

_gated_delta_tape_kernel = _make_gated_delta_kernel_with_tape(has_mask=False, vectorized=False)
_gated_delta_tape_kernel_masked = _make_gated_delta_kernel_with_tape(has_mask=True, vectorized=False)
_gated_delta_tape_kernel_vec = _make_gated_delta_kernel_with_tape(has_mask=False, vectorized=True)
_gated_delta_tape_kernel_vec_masked = _make_gated_delta_kernel_with_tape(has_mask=True, vectorized=True)

def _gated_delta_ops_with_tape(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Hv % Hk != 0:
        raise ValueError(f"Cannot align K heads {Hk} to V heads {Hv}")
    repeat_factor = Hv // Hk
    if repeat_factor > 1:
        q = mx.repeat(q, repeat_factor, axis=2)
        k = mx.repeat(k, repeat_factor, axis=2)

    outputs = []
    tape = []
    for t in range(T):
        old_state = state
        if g.ndim == 4:
            decay = g[:, t, :, None, :]
        elif g.ndim == 3:
            decay = g[:, t, :, None, None]
        else:
            raise ValueError(f"Unsupported gating shape {g.shape}")
        decayed_state = state * decay
        kv_mem = (decayed_state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        new_state = decayed_state + k[:, t, :, None, :] * delta[..., None]
        y = (new_state * q[:, t, :, None, :]).sum(axis=-1)
        if mask is not None:
            step_mask = mask[:, t][:, None, None, None]
            y_mask = mask[:, t][:, None, None]
            new_state = mx.where(step_mask, new_state, old_state)
            delta = mx.where(y_mask, delta, mx.zeros_like(delta))
            y = mx.where(y_mask, y, mx.zeros_like(y))
        state = new_state
        outputs.append(y)
        tape.append(delta.astype(mx.float32))
    return mx.stack(outputs, axis=1), state, mx.stack(tape, axis=1)

def gated_delta_kernel_with_tape(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
):
    if not mx.metal.is_available():
        return _gated_delta_ops_with_tape(q, k, v, g, beta, state, mask)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk < 32 or Dk % 32 != 0:
        return _gated_delta_ops_with_tape(q, k, v, g, beta, state, mask)

    input_type = q.dtype

    if g.ndim == 4:
        kernel = _gated_delta_tape_kernel_vec
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_tape_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _gated_delta_tape_kernel
        inputs = [q, k, v, g, beta, state, T]
        if mask is not None:
            kernel = _gated_delta_tape_kernel_masked
            inputs.append(mask)

    if kernel is None:
        return _gated_delta_ops_with_tape(q, k, v, g, beta, state, mask)

    return kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape, (B, T, Hv, Dv)],
        output_dtypes=[input_type, input_type, mx.float32],
    )


def _make_gated_delta_tree_kernel(*, vectorized: bool = False):
    if not mx.metal.is_available():
        return None

    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {{
          int parent = parent_ids[t];
          auto tree_state_slot = tree_states + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          auto parent_state_slot = parent < 0
            ? i_state
            : tree_states + (((b_idx * T + parent) * Hv + hv_idx) * Dv + dv_idx) * Dk;

          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = static_cast<float>(parent_state_slot[s_idx]);
          }}

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * {g_access};
            kv_mem += state[i] * k_[s_idx];
          }}
          kv_mem = simd_sum(kv_mem);

          float delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }}
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {{
            y[dv_idx] = static_cast<InT>(out);
          }}
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            tree_state_slot[s_idx] = static_cast<InT>(state[i]);
          }}

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
    """

    suffix = "_vec" if vectorized else ""
    return mx.fast.metal_kernel(
        name=f"gated_delta_tree{suffix}",
        input_names=["q", "k", "v", "g", "beta", "state_in", "parent_ids", "T"],
        output_names=["y", "tree_states"],
        source=source,
    )


_gated_delta_tree_kernel = _make_gated_delta_tree_kernel(vectorized=False)
_gated_delta_tree_kernel_vec = _make_gated_delta_tree_kernel(vectorized=True)


def _make_gated_delta_tree_tape_kernel(*, vectorized: bool = False):
    if not mx.metal.is_available():
        return None

    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y, tape: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;

        {g_comment}
        {g_setup}
        auto beta_ = beta + b_idx * T * Hv;

        float history[MaxT][n_per_t];

        for (int t = 0; t < T; ++t) {{
          int parent = parent_ids[t];
          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = parent < 0 ? static_cast<float>(i_state[s_idx]) : history[parent][i];
          }}

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * {g_access};
            kv_mem += state[i] * k_[s_idx];
          }}
          kv_mem = simd_sum(kv_mem);

          float delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }}
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {{
            y[dv_idx] = static_cast<InT>(out);
            tape_[dv_idx] = delta;
          }}
          for (int i = 0; i < n_per_t; ++i) {{
            state[i] = static_cast<float>(static_cast<InT>(state[i]));
            history[t][i] = state[i];
          }}

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          tape_ += Hv * Dv;
          {g_advance}
          beta_ += Hv;
        }}
    """

    suffix = "_vec" if vectorized else ""
    return mx.fast.metal_kernel(
        name=f"gated_delta_tree_tape{suffix}",
        input_names=["q", "k", "v", "g", "beta", "state_in", "parent_ids", "T"],
        output_names=["y", "tape"],
        source=source,
    )


_gated_delta_tree_tape_kernel = _make_gated_delta_tree_tape_kernel(vectorized=False)
_gated_delta_tree_tape_kernel_vec = _make_gated_delta_tree_tape_kernel(vectorized=True)


def _gated_delta_tree_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parent_ids: mx.array,
):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Hv % Hk != 0:
        raise ValueError(f"Cannot align K heads {Hk} to V heads {Hv}")
    repeat_factor = Hv // Hk
    if repeat_factor > 1:
        q = mx.repeat(q, repeat_factor, axis=2)
        k = mx.repeat(k, repeat_factor, axis=2)

    parent_list = [int(parent) for parent in parent_ids.tolist()]
    outputs = []
    states = []
    for t, parent in enumerate(parent_list):
        state_in = state if parent < 0 else states[parent]
        if g.ndim == 4:
            decay = g[:, t, :, None, :]
        elif g.ndim == 3:
            decay = g[:, t, :, None, None]
        else:
            raise ValueError(f"Unsupported gating shape {g.shape}")
        decayed_state = state_in * decay
        kv_mem = (decayed_state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state_t = decayed_state + k[:, t, :, None, :] * delta[..., None]
        y = (state_t * q[:, t, :, None, :]).sum(axis=-1)
        outputs.append(y)
        states.append(mx.contiguous(state_t))
    return mx.stack(outputs, axis=1), mx.stack(states, axis=1)


def _gated_delta_tree_tape_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parent_ids: mx.array,
):
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Hv % Hk != 0:
        raise ValueError(f"Cannot align K heads {Hk} to V heads {Hv}")
    repeat_factor = Hv // Hk
    if repeat_factor > 1:
        q = mx.repeat(q, repeat_factor, axis=2)
        k = mx.repeat(k, repeat_factor, axis=2)

    parent_list = [int(parent) for parent in parent_ids.tolist()]
    outputs = []
    tape = []
    states = []
    for t, parent in enumerate(parent_list):
        state_in = state if parent < 0 else states[parent]
        if g.ndim == 4:
            decay = g[:, t, :, None, :]
        elif g.ndim == 3:
            decay = g[:, t, :, None, None]
        else:
            raise ValueError(f"Unsupported gating shape {g.shape}")
        decayed_state = state_in * decay
        kv_mem = (decayed_state * k[:, t, :, None, :]).sum(axis=-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state_t = decayed_state + k[:, t, :, None, :] * delta[..., None]
        y = (state_t * q[:, t, :, None, :]).sum(axis=-1)
        outputs.append(y)
        tape.append(delta.astype(mx.float32))
        states.append(mx.contiguous(state_t.astype(q.dtype).astype(mx.float32)))
    return mx.stack(outputs, axis=1), mx.stack(tape, axis=1)


def gated_delta_tree_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parent_ids: mx.array,
):
    if not mx.metal.is_available():
        return _gated_delta_tree_ops(q, k, v, g, beta, state, parent_ids)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk < 32 or Dk % 32 != 0:
        return _gated_delta_tree_ops(q, k, v, g, beta, state, parent_ids)

    input_type = q.dtype
    state_type = state.dtype
    parent_ids = parent_ids.astype(mx.int32)
    if g.ndim == 4:
        kernel = _gated_delta_tree_kernel_vec
    else:
        kernel = _gated_delta_tree_kernel
    if kernel is None:
        return _gated_delta_tree_ops(q, k, v, g, beta, state, parent_ids)

    return kernel(
        inputs=[q, k, v, g, beta, state, parent_ids, T],
        template=[
            ("InT", input_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv, Dk)],
        output_dtypes=[input_type, state_type],
    )


def gated_delta_tree_tape_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parent_ids: mx.array,
):
    if not mx.metal.is_available():
        return _gated_delta_tree_tape_ops(q, k, v, g, beta, state, parent_ids)

    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if Dk < 32 or Dk % 32 != 0:
        return _gated_delta_tree_tape_ops(q, k, v, g, beta, state, parent_ids)

    input_type = q.dtype
    parent_ids = parent_ids.astype(mx.int32)
    if g.ndim == 4:
        kernel = _gated_delta_tree_tape_kernel_vec
    else:
        kernel = _gated_delta_tree_tape_kernel
    if kernel is None:
        return _gated_delta_tree_tape_ops(q, k, v, g, beta, state, parent_ids)

    return kernel(
        inputs=[q, k, v, g, beta, state, parent_ids, T],
        template=[
            ("InT", input_type),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
            ("MaxT", T),
        ],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), (B, T, Hv, Dv)],
        output_dtypes=[input_type, mx.float32],
    )


def _make_tree_depthwise_conv1d_kernel(*, has_bias: bool = False):
    if not mx.metal.is_available():
        return None

    bias_source = " + static_cast<float>(bias[c])" if has_bias else ""
    input_names = ["qkv", "prefix", "weight", "path_indices"]
    if has_bias:
        input_names.append("bias")

    source = f"""
        auto c = thread_position_in_grid.x;
        auto t = thread_position_in_grid.y;
        auto b = thread_position_in_grid.z;
        constexpr int prefix_len = Kernel - 1;
        auto qkv_base = qkv + b * T * C;
        float acc = 0.0f;
        for (int w = 0; w < Kernel; ++w) {{
          int path_slot = path_indices[t * Kernel + w];
          float x = 0.0f;
          if (path_slot < prefix_len) {{
            x = static_cast<float>(prefix[(b * prefix_len + path_slot) * C + c]);
          }} else {{
            int qkv_slot = path_slot - prefix_len;
            x = static_cast<float>(qkv_base[qkv_slot * C + c]);
          }}
          acc += x * static_cast<float>(weight[(c * Kernel + w)]);
        }}
        acc = acc{bias_source};
        acc = acc / (1.0f + exp(-acc));
        out[(b * T + t) * C + c] = static_cast<InT>(acc);
    """
    return mx.fast.metal_kernel(
        name=f"tree_depthwise_conv1d{'_bias' if has_bias else ''}",
        input_names=input_names,
        output_names=["out"],
        source=source,
    )


_tree_depthwise_conv1d_kernel = _make_tree_depthwise_conv1d_kernel(has_bias=False)
_tree_depthwise_conv1d_kernel_bias = _make_tree_depthwise_conv1d_kernel(has_bias=True)


def _tree_depthwise_conv1d_ops(
    qkv: mx.array,
    prefix: mx.array,
    weight: mx.array,
    bias: Optional[mx.array],
    path_indices: mx.array,
) -> mx.array:
    qkv_pool = mx.concatenate([prefix, qkv], axis=1)
    windows = mx.take(qkv_pool, path_indices.astype(mx.int32), axis=1)
    kernel_weight = weight[..., 0].T
    conv = (windows * kernel_weight[None, None, :, :]).sum(axis=2)
    if bias is not None:
        conv = conv + bias[None, None, :]
    return conv * mx.sigmoid(conv)


def tree_depthwise_conv1d(
    qkv: mx.array,
    prefix: mx.array,
    weight: mx.array,
    bias: Optional[mx.array],
    path_indices: mx.array,
) -> mx.array:
    if not mx.metal.is_available():
        return _tree_depthwise_conv1d_ops(qkv, prefix, weight, bias, path_indices)

    if weight.ndim != 3 or int(weight.shape[-1]) != 1:
        return _tree_depthwise_conv1d_ops(qkv, prefix, weight, bias, path_indices)
    B, T, C = qkv.shape
    if int(weight.shape[0]) != C:
        return _tree_depthwise_conv1d_ops(qkv, prefix, weight, bias, path_indices)
    kernel_size = int(weight.shape[1])
    if int(prefix.shape[1]) != kernel_size - 1:
        return _tree_depthwise_conv1d_ops(qkv, prefix, weight, bias, path_indices)

    flat_weight = mx.contiguous(weight[..., 0])
    inputs = [qkv, prefix, flat_weight, path_indices.astype(mx.int32)]
    kernel = _tree_depthwise_conv1d_kernel
    if bias is not None:
        kernel = _tree_depthwise_conv1d_kernel_bias
        inputs.append(bias)
    if kernel is None:
        return _tree_depthwise_conv1d_ops(qkv, prefix, weight, bias, path_indices)

    (out,) = kernel(
        inputs=inputs,
        template=[
            ("InT", qkv.dtype),
            ("T", T),
            ("C", C),
            ("Kernel", kernel_size),
        ],
        grid=(C, T, B),
        threadgroup=(128, 1, 1),
        output_shapes=[qkv.shape],
        output_dtypes=[qkv.dtype],
    )
    return out


def _make_tape_replay_kernel(*, has_mask: bool = False, vectorized: bool = False):
    if not mx.metal.is_available():
        return None

    mask_source = "mask[b_idx * T + t]" if has_mask else "true"

    if vectorized:
        g_comment = "// g: [B, T, Hv, Dk]"
        g_setup = "auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;"
        g_access = "g_[s_idx]"
        g_advance = "g_ += Hv * Dk;"
    else:
        g_comment = "// g: [B, T, Hv]"
        g_setup = "auto g_ = g + b_idx * T * Hv;"
        g_access = "g_[hv_idx]"
        g_advance = "g_ += Hv;"

    source = f"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // tape: [B, T, Hv, Dv]
        auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        // k: [B, T, Hk, Dk]
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        {g_comment}
        {g_setup}

        for (int t = 0; t < T; ++t) {{
          if ({mask_source}) {{
            auto delta = static_cast<float>(tape_[dv_idx]);
            for (int i = 0; i < n_per_t; ++i) {{
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * {g_access};
              state[i] = state[i] + k_[s_idx] * delta;
            }}
            for (int i = 0; i < n_per_t; ++i) {{
              state[i] = static_cast<float>(static_cast<InT>(state[i]));
            }}
          }}
          tape_ += Hv * Dv;
          k_ += Hk * Dk;
          {g_advance}
        }}

        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """

    inputs = ["tape", "k", "g", "state_in", "T"]
    if has_mask:
        inputs.append("mask")

    suffix = ""
    if vectorized:
        suffix += "_vec"
    if has_mask:
        suffix += "_mask"

    return mx.fast.metal_kernel(
        name=f"tape_replay{suffix}",
        input_names=inputs,
        output_names=["state_out"],
        source=source,
    )

_tape_replay_kernel = _make_tape_replay_kernel(
    has_mask=False, vectorized=False
)
_tape_replay_kernel_masked = _make_tape_replay_kernel(
    has_mask=True, vectorized=False
)
_tape_replay_kernel_vec = _make_tape_replay_kernel(
    has_mask=False, vectorized=True
)
_tape_replay_kernel_vec_masked = _make_tape_replay_kernel(
    has_mask=True, vectorized=True
)

def _tape_replay_ops(
    tape: mx.array,
    k: mx.array,
    g: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> mx.array:
    _, _, hk, _ = k.shape
    hv = tape.shape[2]
    if hv % hk != 0:
        raise ValueError(f"Cannot align K heads {hk} to tape heads {hv}")
    repeat_factor = hv // hk
    if repeat_factor > 1:
        k = mx.repeat(k, repeat_factor, axis=2)

    for t in range(int(tape.shape[1])):
        prev_state = state
        if g.ndim == 4:
            decay = g[:, t, :, None, :]
        elif g.ndim == 3:
            decay = g[:, t, :, None, None]
        else:
            raise ValueError(f"Unsupported gating shape {g.shape}")
        delta = tape[:, t, :, :, None]
        k_t = k[:, t, :, None, :]
        state = state * decay
        state = state + delta * k_t
        if mask is not None:
            step_mask = mask[:, t][:, None, None, None]
            state = mx.where(step_mask, state, prev_state)
    return state

def tape_replay_kernel(
    tape: mx.array,
    k: mx.array,
    g: mx.array,
    state: mx.array,
    mask: Optional[mx.array] = None,
) -> mx.array:
    if not mx.metal.is_available():
        return _tape_replay_ops(tape, k, g, state, mask)

    bsz, steps, hk, dk = k.shape
    hv, dv = tape.shape[2:]
    input_type = state.dtype
    if dk < 32 or dk % 32 != 0:
        return _tape_replay_ops(tape, k, g, state, mask)

    if g.ndim == 4:
        kernel = _tape_replay_kernel_vec
        inputs = [tape, k, g, state, steps]
        if mask is not None:
            kernel = _tape_replay_kernel_vec_masked
            inputs.append(mask)
    else:
        kernel = _tape_replay_kernel
        inputs = [tape, k, g, state, steps]
        if mask is not None:
            kernel = _tape_replay_kernel_masked
            inputs.append(mask)

    if kernel is None:
        return _tape_replay_ops(tape, k, g, state, mask)

    (state_out,) = kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("Dk", dk),
            ("Dv", dv),
            ("Hk", hk),
            ("Hv", hv),
        ],
        grid=(32, dv, bsz * hv),
        threadgroup=(32, 4, 1),
        output_shapes=[state.shape],
        output_dtypes=[input_type],
    )
    return state_out
