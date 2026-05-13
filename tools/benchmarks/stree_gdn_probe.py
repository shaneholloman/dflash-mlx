from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx

from dflash_mlx.kernels import gated_delta_tree_kernel


def _make_tree_tape_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto tape_ = tape + b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        float history[MaxT][n_per_t];

        for (int t = 0; t < T; ++t) {
          int parent = parent_ids[t];
          float state[n_per_t];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = parent < 0 ? static_cast<float>(i_state[s_idx]) : history[parent][i];
          }

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          float delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            history[t][i] = state[i];
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
            tape_[dv_idx] = delta;
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          tape_ += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_tree_local_tape_probe",
        input_names=["q", "k", "v", "g", "beta", "state_in", "parent_ids", "T"],
        output_names=["y", "tape"],
        source=source,
    )


_TREE_TAPE_KERNEL = _make_tree_tape_kernel()


def _make_tree_conv_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        auto c = thread_position_in_grid.x;
        auto t = thread_position_in_grid.y;
        auto b = thread_position_in_grid.z;
        auto prefix_len = Kernel - 1;
        auto qkv_pool = prefix;
        auto qkv_base = qkv + b * T * C;
        float acc = 0.0f;
        for (int w = 0; w < Kernel; ++w) {
          int path_slot = path_indices[t * Kernel + w];
          float x = 0.0f;
          if (path_slot < prefix_len) {
            x = static_cast<float>(prefix[(b * prefix_len + path_slot) * C + c]);
          } else {
            int qkv_slot = path_slot - prefix_len;
            x = static_cast<float>(qkv_base[qkv_slot * C + c]);
          }
          acc += x * static_cast<float>(weight[w * C + c]);
        }
        acc += static_cast<float>(bias[c]);
        // silu
        acc = acc / (1.0f + exp(-acc));
        out[(b * T + t) * C + c] = static_cast<InT>(acc);
    """
    return mx.fast.metal_kernel(
        name="tree_depthwise_conv_probe",
        input_names=["qkv", "prefix", "weight", "bias", "path_indices"],
        output_names=["out"],
        source=source,
    )


_TREE_CONV_KERNEL = _make_tree_conv_kernel()


def _tree_path_indices(parent_ids: list[int], slot_index: int) -> list[int]:
    path: list[int] = []
    cursor = int(slot_index)
    while cursor >= 0:
        path.append(cursor)
        cursor = int(parent_ids[cursor])
    return list(reversed(path))


def _tree_conv_window_indices(
    parent_ids: list[int],
    *,
    conv_kernel_size: int,
) -> mx.array:
    kernel = int(conv_kernel_size)
    keep = kernel - 1
    rows: list[list[int]] = []
    for slot_index in range(len(parent_ids)):
        path = _tree_path_indices(parent_ids, slot_index)
        path_tail = path[-kernel:]
        prefix_need = kernel - len(path_tail)
        prefix_start = keep - prefix_need
        rows.append(
            list(range(prefix_start, keep))
            + [keep + int(path_slot) for path_slot in path_tail]
        )
    return mx.array(rows, dtype=mx.int32)


def _tree_conv_current_ops(
    qkv: mx.array,
    prefix: mx.array,
    weight: mx.array,
    bias: mx.array,
    path_indices: mx.array,
) -> mx.array:
    qkv_pool = mx.concatenate([prefix, qkv], axis=1)
    windows = mx.take(qkv_pool, path_indices, axis=1)
    y = (windows * weight[None, None, :, :]).sum(axis=2) + bias[None, None, :]
    return y * mx.sigmoid(y)


def _tree_conv_direct_kernel(
    qkv: mx.array,
    prefix: mx.array,
    weight: mx.array,
    bias: mx.array,
    path_indices: mx.array,
) -> mx.array:
    if _TREE_CONV_KERNEL is None:
        return _tree_conv_current_ops(qkv, prefix, weight, bias, path_indices)
    batch, tree_size, conv_dim = qkv.shape
    kernel = int(weight.shape[0])
    out = _TREE_CONV_KERNEL(
        inputs=[qkv, prefix, weight, bias, path_indices.astype(mx.int32)],
        template=[
            ("InT", qkv.dtype),
            ("T", tree_size),
            ("C", conv_dim),
            ("Kernel", kernel),
        ],
        grid=(conv_dim, tree_size, batch),
        threadgroup=(128, 1, 1),
        output_shapes=[qkv.shape],
        output_dtypes=[qkv.dtype],
    )
    return out[0] if isinstance(out, (list, tuple)) else out


def _tree_tape_kernel(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parent_ids: mx.array,
) -> tuple[mx.array, mx.array]:
    if _TREE_TAPE_KERNEL is None:
        y, states = _exact_ops(q, k, v, g, beta, state, parent_ids)
        return y, mx.zeros((*y.shape[:3], y.shape[3]), dtype=mx.float32)
    if g.ndim != 3:
        raise ValueError("probe tape kernel only supports scalar GDN gates")
    batch, tree_size, hk, dk = k.shape
    hv, dv = v.shape[2:]
    if hv % hk != 0:
        raise ValueError(f"cannot align Hk={hk} to Hv={hv}")
    if dk < 32 or dk % 32 != 0:
        raise ValueError("--dk must be a multiple of 32 for the probe kernel")
    parent_ids = parent_ids.astype(mx.int32)
    return _TREE_TAPE_KERNEL(
        inputs=[q, k, v, g, beta, state, parent_ids, tree_size],
        template=[
            ("InT", q.dtype),
            ("Dk", dk),
            ("Dv", dv),
            ("Hk", hk),
            ("Hv", hv),
            ("MaxT", tree_size),
        ],
        grid=(32, dv, batch * hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(batch, tree_size, hv, dv), (batch, tree_size, hv, dv)],
        output_dtypes=[q.dtype, mx.float32],
    )


def _parent_ids(tree: str, size: int) -> mx.array:
    if size <= 0:
        raise ValueError("--tree-size must be positive")
    parents = [-1]
    if tree == "chain":
        parents.extend(range(size - 1))
    elif tree == "root":
        parents.extend(0 for _ in range(size - 1))
    elif tree == "binary":
        parents.extend((index - 1) // 2 for index in range(1, size))
    else:
        raise ValueError(f"unknown tree shape: {tree}")
    return mx.array(parents, dtype=mx.int32)


def _repeat_kv_heads(x: mx.array, hv: int) -> mx.array:
    hk = int(x.shape[2])
    if hv % hk != 0:
        raise ValueError(f"cannot align Hk={hk} to Hv={hv}")
    repeat = hv // hk
    return mx.repeat(x, repeat, axis=2) if repeat > 1 else x


def _exact_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parent_ids: mx.array,
) -> tuple[mx.array, mx.array]:
    hv = int(v.shape[2])
    q = _repeat_kv_heads(q, hv)
    k = _repeat_kv_heads(k, hv)
    parents = [int(value) for value in parent_ids.tolist()]
    outputs = []
    states = []
    for slot, parent in enumerate(parents):
        parent_state = state if parent < 0 else states[parent]
        decay = g[:, slot, :, None, None] if g.ndim == 3 else g[:, slot, :, None, :]
        decayed = parent_state * decay
        kv_mem = (decayed * k[:, slot, :, None, :]).sum(axis=-1)
        delta = (v[:, slot] - kv_mem) * beta[:, slot, :, None]
        state_t = decayed + k[:, slot, :, None, :] * delta[..., None]
        y = (state_t * q[:, slot, :, None, :]).sum(axis=-1)
        outputs.append(y)
        states.append(mx.contiguous(state_t))
    return mx.stack(outputs, axis=1), mx.stack(states, axis=1)


def _diag_only_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parent_ids: mx.array,
) -> tuple[mx.array, mx.array]:
    hv = int(v.shape[2])
    q = _repeat_kv_heads(q, hv)
    k = _repeat_kv_heads(k, hv)
    parents = [int(value) for value in parent_ids.tolist()]
    outputs = []
    states = []
    for slot, parent in enumerate(parents):
        parent_state = state if parent < 0 else states[parent]
        decay = g[:, slot, :, None, None] if g.ndim == 3 else g[:, slot, :, None, :]
        add = k[:, slot, :, None, :] * (v[:, slot, :, :, None] * beta[:, slot, :, None, None])
        state_t = parent_state * decay + add
        y = (state_t * q[:, slot, :, None, :]).sum(axis=-1)
        outputs.append(y)
        states.append(mx.contiguous(state_t))
    return mx.stack(outputs, axis=1), mx.stack(states, axis=1)


def _full_affine_ops(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    parent_ids: mx.array,
) -> tuple[mx.array, mx.array]:
    hv = int(v.shape[2])
    q = _repeat_kv_heads(q, hv)
    k = _repeat_kv_heads(k, hv)
    parents = [int(value) for value in parent_ids.tolist()]
    dk = int(k.shape[-1])
    eye = mx.eye(dk, dtype=mx.float32)
    outputs = []
    states = []
    for slot, parent in enumerate(parents):
        parent_state = state if parent < 0 else states[parent]
        k_t = k[:, slot].astype(mx.float32)
        if g.ndim == 3:
            g_t = mx.broadcast_to(g[:, slot, :, None].astype(mx.float32), k_t.shape)
        else:
            g_t = g[:, slot].astype(mx.float32)
        beta_t = beta[:, slot].astype(mx.float32)
        diagonal = eye[None, None, :, :] * g_t[:, :, None, :]
        correction = (
            beta_t[:, :, None, None]
            * (g_t * k_t)[:, :, :, None]
            * k_t[:, :, None, :]
        )
        transition = diagonal - correction
        add = k_t[:, :, None, :] * (
            v[:, slot].astype(mx.float32)[:, :, :, None] * beta_t[:, :, None, None]
        )
        state_t = mx.matmul(parent_state.astype(mx.float32), transition) + add
        state_t = state_t.astype(state.dtype)
        y = (state_t * q[:, slot, :, None, :]).sum(axis=-1)
        outputs.append(y)
        states.append(mx.contiguous(state_t))
    return mx.stack(outputs, axis=1), mx.stack(states, axis=1)


def _time_fn(
    fn: Callable[[], tuple[mx.array, mx.array]],
    *,
    warmup: int,
    repeat: int,
) -> dict[str, float]:
    for _ in range(warmup):
        y, states = fn()
        mx.eval(y, states)
    mx.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        y, states = fn()
        mx.eval(y, states)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000.0)
    samples.sort()
    count = len(samples)
    return {
        "min_us": samples[0],
        "p25_us": samples[count // 4],
        "median_us": samples[len(samples) // 2],
        "p75_us": samples[(3 * count) // 4],
        "max_us": samples[-1],
        "mean_us": statistics.fmean(samples),
    }


def _max_abs(a: mx.array, b: mx.array) -> float:
    value = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
    mx.eval(value)
    return float(value.item())


def _mean_abs(a: mx.array, b: mx.array) -> float:
    value = mx.mean(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
    mx.eval(value)
    return float(value.item())


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.conv_probe:
        return run_conv(args)

    mx.random.seed(args.seed)
    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float16
    parent_ids = _parent_ids(args.tree, args.tree_size)
    shape_qk = (args.batch, args.tree_size, args.k_heads, args.dk)
    shape_v = (args.batch, args.tree_size, args.v_heads, args.dv)
    q = (args.activation_scale * mx.random.normal(shape_qk)).astype(dtype)
    k = (args.activation_scale * mx.random.normal(shape_qk)).astype(dtype)
    v = (args.activation_scale * mx.random.normal(shape_v)).astype(dtype)
    g = mx.sigmoid(
        args.gate_scale * mx.random.normal((args.batch, args.tree_size, args.v_heads))
        + args.gate_shift
    )
    beta = mx.sigmoid(args.beta_scale * mx.random.normal((args.batch, args.tree_size, args.v_heads)))
    state = (
        args.state_scale * mx.random.normal((args.batch, args.v_heads, args.dv, args.dk))
    ).astype(mx.float32)

    exact_y, exact_states = gated_delta_tree_kernel(q, k, v, g, beta, state, parent_ids)
    mx.eval(exact_y, exact_states)
    ops_y, ops_states = _exact_ops(q, k, v, g, beta, state, parent_ids)
    mx.eval(ops_y, ops_states)
    diag_y, diag_states = _diag_only_ops(q, k, v, g, beta, state, parent_ids)
    mx.eval(diag_y, diag_states)

    results: dict[str, object] = {
        "config": {
            "batch": args.batch,
            "tree": args.tree,
            "tree_size": args.tree_size,
            "k_heads": args.k_heads,
            "v_heads": args.v_heads,
            "dk": args.dk,
            "dv": args.dv,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "activation_scale": args.activation_scale,
            "state_scale": args.state_scale,
            "gate_scale": args.gate_scale,
            "gate_shift": args.gate_shift,
            "beta_scale": args.beta_scale,
        },
        "parity": {
            "kernel_vs_exact_ops_y_max_abs": _max_abs(exact_y, ops_y),
            "kernel_vs_exact_ops_y_mean_abs": _mean_abs(exact_y, ops_y),
            "kernel_vs_exact_ops_state_max_abs": _max_abs(exact_states, ops_states),
            "kernel_vs_exact_ops_state_mean_abs": _mean_abs(exact_states, ops_states),
            "diag_only_y_max_abs": _max_abs(exact_y, diag_y),
            "diag_only_y_mean_abs": _mean_abs(exact_y, diag_y),
            "diag_only_state_max_abs": _max_abs(exact_states, diag_states),
            "diag_only_state_mean_abs": _mean_abs(exact_states, diag_states),
        },
        "timings_us": {
            "current_metal_tree_kernel": _time_fn(
                lambda: gated_delta_tree_kernel(q, k, v, g, beta, state, parent_ids),
                warmup=args.warmup,
                repeat=args.repeat,
            ),
            "current_mlx_exact_ops": _time_fn(
                lambda: _exact_ops(q, k, v, g, beta, state, parent_ids),
                warmup=args.warmup,
                repeat=args.repeat,
            ),
            "diag_only_lossy_ops": _time_fn(
                lambda: _diag_only_ops(q, k, v, g, beta, state, parent_ids),
                warmup=args.warmup,
                repeat=args.repeat,
            ),
        },
    }
    if args.full_affine:
        affine_y, affine_states = _full_affine_ops(q, k, v, g, beta, state, parent_ids)
        mx.eval(affine_y, affine_states)
        results["parity"]["full_affine_y_max_abs"] = _max_abs(exact_y, affine_y)
        results["parity"]["full_affine_state_max_abs"] = _max_abs(exact_states, affine_states)
        results["timings_us"]["full_affine_exact_ops"] = _time_fn(
            lambda: _full_affine_ops(q, k, v, g, beta, state, parent_ids),
            warmup=args.warmup,
            repeat=args.repeat,
        )
    if args.local_tape_kernel:
        tape_y, tape = _tree_tape_kernel(q, k, v, g, beta, state, parent_ids)
        mx.eval(tape_y, tape)
        results["parity"]["local_tape_y_max_abs"] = _max_abs(exact_y, tape_y)
        results["timings_us"]["local_state_tape_kernel"] = _time_fn(
            lambda: _tree_tape_kernel(q, k, v, g, beta, state, parent_ids),
            warmup=args.warmup,
            repeat=args.repeat,
        )
        results["output_bytes"] = {
            "current_tree_states": int(exact_states.nbytes),
            "local_tape": int(tape.nbytes),
        }
    return results


def run_conv(args: argparse.Namespace) -> dict[str, object]:
    mx.random.seed(args.seed)
    dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float16
    parent_ids = [int(value) for value in _parent_ids(args.tree, args.tree_size).tolist()]
    path_indices = _tree_conv_window_indices(parent_ids, conv_kernel_size=args.conv_kernel)
    qkv = (args.activation_scale * mx.random.normal((args.batch, args.tree_size, args.conv_dim))).astype(
        dtype
    )
    prefix = (
        args.activation_scale
        * mx.random.normal((args.batch, max(0, args.conv_kernel - 1), args.conv_dim))
    ).astype(dtype)
    weight = (args.activation_scale * mx.random.normal((args.conv_kernel, args.conv_dim))).astype(dtype)
    bias = (args.activation_scale * mx.random.normal((args.conv_dim,))).astype(dtype)
    current = _tree_conv_current_ops(qkv, prefix, weight, bias, path_indices)
    direct = _tree_conv_direct_kernel(qkv, prefix, weight, bias, path_indices)
    mx.eval(current, direct)
    itemsize = 2 if args.dtype in ("bf16", "fp16") else 4
    window_bytes = int(args.batch * args.tree_size * args.conv_kernel * args.conv_dim * itemsize)
    return {
        "config": {
            "batch": args.batch,
            "tree": args.tree,
            "tree_size": args.tree_size,
            "conv_kernel": args.conv_kernel,
            "conv_dim": args.conv_dim,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "activation_scale": args.activation_scale,
        },
        "parity": {
            "direct_y_max_abs": _max_abs(current, direct),
            "direct_y_mean_abs": _mean_abs(current, direct),
        },
        "output_bytes": {
            "current_windows": window_bytes,
            "direct_output": int(current.nbytes),
        },
        "timings_us": {
            "current_concat_take_depthwise": _time_fn(
                lambda: (_tree_conv_current_ops(qkv, prefix, weight, bias, path_indices), current),
                warmup=args.warmup,
                repeat=args.repeat,
            ),
            "direct_tree_conv_kernel": _time_fn(
                lambda: (_tree_conv_direct_kernel(qkv, prefix, weight, bias, path_indices), current),
                warmup=args.warmup,
                repeat=args.repeat,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(".artifacts/dflash/microbench/stree_gdn_probe.json"))
    parser.add_argument("--tree", choices=("binary", "chain", "root"), default="binary")
    parser.add_argument("--tree-size", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--k-heads", type=int, default=8)
    parser.add_argument("--v-heads", type=int, default=8)
    parser.add_argument("--dk", type=int, default=128)
    parser.add_argument("--dv", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activation-scale", type=float, default=0.05)
    parser.add_argument("--state-scale", type=float, default=0.05)
    parser.add_argument("--gate-scale", type=float, default=0.5)
    parser.add_argument("--gate-shift", type=float, default=1.0)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--full-affine", action="store_true")
    parser.add_argument("--local-tape-kernel", action="store_true")
    parser.add_argument("--conv-probe", action="store_true")
    parser.add_argument("--conv-dim", type=int, default=10240)
    parser.add_argument("--conv-kernel", type=int, default=4)
    args = parser.parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
