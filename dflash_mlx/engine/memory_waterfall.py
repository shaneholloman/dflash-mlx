# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import inspect
from typing import Any, Optional

from dflash_mlx.diagnostics import DiagnosticsConfig
from dflash_mlx.observability.memory import process_memory_bytes

_GB = 1_000_000_000.0
_MISSING = object()

def memory_waterfall_enabled(diagnostics: Optional[DiagnosticsConfig] = None) -> bool:
    return bool(diagnostics is not None and diagnostics.memory_waterfall)

def should_sample_cycle(cycle: int) -> bool:
    return int(cycle) in (1, 2, 4, 8) or (int(cycle) > 0 and int(cycle) % 16 == 0)

def collect_memory_waterfall(
    *,
    phase: str,
    target_cache: Any = None,
    draft_cache: Any = None,
    target_hidden: Any = None,
    gen_hidden_chunks: Any = None,
    prefix_cache_memory: Optional[dict[str, Any]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    process = process_memory_bytes()
    target = target_cache_bytes(target_cache)
    draft = draft_cache_bytes(draft_cache)
    prefix = prefix_cache_memory_fields(prefix_cache_memory)
    hidden_seen: set[int] = set()
    target_hidden_bytes = tree_nbytes(target_hidden, hidden_seen)
    gen_hidden_chunks_bytes = tree_nbytes(gen_hidden_chunks, hidden_seen)
    out: dict[str, Any] = {
        "memory_phase": str(phase),
        **process,
        **target,
        **draft,
        "target_hidden_active_bytes": int(target_hidden_bytes),
        "gen_hidden_chunks_bytes": int(gen_hidden_chunks_bytes),
        **prefix,
    }
    for key, value in list(out.items()):
        if key.endswith("_bytes"):
            out[key[:-6] + "_gb"] = _to_gb(value)
    for key in (
        "target_fa_kv_bytes",
        "target_gdn_state_bytes",
        "rollback_tape_bytes",
        "draft_kv_bytes",
        "target_hidden_active_bytes",
        "gen_hidden_chunks_bytes",
        "l1_snapshot_bytes",
        "l2_disk_bytes",
    ):
        out.setdefault(key[:-6] + "_gb", 0.0)
    if extra:
        out.update(extra)
    return out

def target_cache_bytes(target_cache: Any) -> dict[str, int]:
    out = {
        "target_fa_kv_bytes": 0,
        "target_gdn_state_bytes": 0,
        "rollback_tape_bytes": 0,
    }
    for cache in _iter_cache_entries(target_cache):
        if _looks_recurrent(cache):
            out["target_gdn_state_bytes"] += tree_nbytes(getattr(cache, "cache", None))
            out["rollback_tape_bytes"] += tree_nbytes(
                [
                    getattr(cache, "_tape", None),
                    getattr(cache, "_tape_k", None),
                    getattr(cache, "_tape_g", None),
                    getattr(cache, "_tape_qkv", None),
                    getattr(cache, "_snapshot", None),
                ]
            )
        else:
            out["target_fa_kv_bytes"] += _kv_cache_nbytes(cache)
    return out

def draft_cache_bytes(draft_cache: Any) -> dict[str, int]:
    total = 0
    for cache in _iter_cache_entries(draft_cache):
        total += _kv_cache_nbytes(cache)
    if total == 0:
        total = _kv_cache_nbytes(draft_cache)
    return {"draft_kv_bytes": int(total)}

def prefix_cache_memory_bytes(prefix_cache_memory: Optional[dict[str, Any]]) -> dict[str, int]:
    out = {
        "l1_snapshot_bytes": 0,
        "l1_snapshot_fa_kv_bytes": 0,
        "l1_snapshot_gdn_state_bytes": 0,
        "l1_snapshot_target_hidden_bytes": 0,
        "l1_snapshot_last_logits_bytes": 0,
        "l2_disk_bytes": 0,
        "prefix_prunes": 0,
        "cross_kind_prunes": 0,
        "byte_budget_evictions": 0,
        "l2_hits": 0,
        "l2_writes": 0,
        "l2_misses": 0,
    }
    if prefix_cache_memory is None:
        return out
    for key in out:
        out[key] = int(prefix_cache_memory.get(key, out[key]) or 0)
    return out

def prefix_cache_memory_fields(prefix_cache_memory: Optional[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = prefix_cache_memory_bytes(prefix_cache_memory)
    for key, value in list(out.items()):
        if key.endswith("_bytes"):
            out[key[:-6] + "_gb"] = _to_gb(value)
    return out

def merge_memory_waterfall_peak(
    current: Optional[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(current or {})
    for key, value in payload.items():
        if key.endswith("_bytes") or key.endswith("_gb"):
            try:
                if float(value) > float(merged.get(key, -1)):
                    merged[key] = value
            except (TypeError, ValueError):
                pass
        elif key in (
            "prefix_prunes",
            "cross_kind_prunes",
            "byte_budget_evictions",
            "l2_hits",
            "l2_writes",
            "l2_misses",
        ):
            try:
                if int(value) > int(merged.get(key, -1)):
                    merged[key] = int(value)
            except (TypeError, ValueError):
                pass
    return merged

def format_memory_waterfall_summary(payload: dict[str, Any]) -> str:
    keys = {
        "mlx_active": "mlx_active_gb",
        "mlx_cache": "mlx_cache_gb",
        "untracked": "untracked_gb",
        "target_fa_kv": "target_fa_kv_gb",
        "target_gdn_state": "target_gdn_state_gb",
        "rollback_tape": "rollback_tape_gb",
        "draft_kv": "draft_kv_gb",
        "target_hidden": "target_hidden_active_gb",
        "gen_hidden_chunks": "gen_hidden_chunks_gb",
        "l1_snapshots": "l1_snapshot_gb",
        "l2_disk": "l2_disk_gb",
    }
    buckets = {
        name: float(value)
        for name, key in keys.items()
        if (value := payload.get(key)) is not None
    }
    unknown = [name for name, key in keys.items() if payload.get(key) is None]
    top = sorted(buckets.items(), key=lambda item: item[1], reverse=True)[:5]
    rendered = ",".join(f"{name}:{value:.2f}GB" for name, value in top if value > 0.0)
    if not rendered:
        rendered = "none"
    suffix = f" unknown={','.join(unknown)}" if unknown else ""
    return f"[dflash] memory-waterfall summary top={rendered}{suffix}"

def tree_nbytes(value: Any, seen: Optional[set[int]] = None) -> int:
    if value is None:
        return 0
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if _is_array_like(value):
        return int(getattr(value, "nbytes", 0) or 0)
    if isinstance(value, dict):
        return sum(tree_nbytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return sum(tree_nbytes(v, seen) for v in value)
    return 0

def _to_gb(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / _GB
    except (TypeError, ValueError):
        return 0.0

def _is_array_like(value: Any) -> bool:
    return hasattr(value, "nbytes") and hasattr(value, "shape")

def _iter_cache_entries(cache: Any) -> list[Any]:
    if cache is None:
        return []
    if isinstance(cache, (list, tuple)):
        return list(cache)
    return [cache]

def _looks_recurrent(cache: Any) -> bool:
    name = type(cache).__name__.lower()
    return "recurrent" in name or hasattr(cache, "_tape") or hasattr(cache, "_tape_k")

def _kv_cache_nbytes(cache: Any) -> int:
    if cache is None:
        return 0
    total = 0
    if hasattr(cache, "keys") or hasattr(cache, "values"):
        total += tree_nbytes([getattr(cache, "keys", None), getattr(cache, "values", None)])
    state = None
    if inspect.getattr_static(cache, "state", _MISSING) is not _MISSING:
        try:
            state = getattr(cache, "state")
        except AttributeError:
            keys_attr = inspect.getattr_static(cache, "keys", _MISSING)
            values_attr = inspect.getattr_static(cache, "values", _MISSING)
            if (
                keys_attr is _MISSING
                or values_attr is _MISSING
                or getattr(cache, "keys") is not None
                or getattr(cache, "values") is not None
            ):
                raise
    if state is not None:
        total += tree_nbytes(state)
    return int(total)
