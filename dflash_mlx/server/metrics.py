# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import ctypes
import os
import resource
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from dflash_mlx.bench_logger import log_post as _bench_log_post
from dflash_mlx.cache.manager import sync_runtime_cache_manager
from dflash_mlx.diagnostics import DiagnosticsConfig
from dflash_mlx.server.prefix_cache_manager import build_prefix_key

_GB = 1_000_000_000.0
_LIVE_LOCK = threading.Lock()
_LIVE_STARTED_AT = time.time()
_LIVE_SERVER: dict[str, Any] = {
    "version": None,
    "model": None,
    "draft": None,
    "mode": "dflash",
    "profile": None,
}
_LIVE_RUNTIME: dict[str, Any] = {
    "prefill_step_size": None,
    "prefix_cache_enabled": None,
    "target_fa_window": None,
    "diagnostics": None,
}
_LIVE_PREFIX_CONFIG: dict[str, Any] = {
    "enabled": False,
    "max_entries": None,
    "max_bytes": None,
}
_LIVE_METAL_LIMITS: dict[str, Any] = {
    "wired_limit_bytes": None,
}
_LIVE_LAST_REQUEST: Optional[dict[str, Any]] = None
_LIVE_CURRENT_REQUEST: Optional[dict[str, Any]] = None
_LIVE_RECENT_REQUESTS = deque(maxlen=32)
_LIVE_REQUEST_FINISH_TIMES = deque(maxlen=512)
_LIVE_LAST_PREFIX: dict[str, Optional[int]] = {
    "restored": None,
    "computed": None,
}
_LIVE_TOTALS: dict[str, int] = {
    "requests": 0,
    "generated_tokens": 0,
    "prefill_tokens_physical": 0,
    "prefill_tokens_restored": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}

def configure_live_metrics(
    *,
    version: str,
    model_provider: Any,
) -> None:
    cli_args = model_provider.cli_args
    runtime_config = getattr(cli_args, "runtime_config", None)
    model_key = getattr(model_provider, "model_key", None) or ()
    target_ref = model_key[0] if len(model_key) > 0 else getattr(cli_args, "model", None)
    draft_ref = model_key[2] if len(model_key) > 2 else getattr(cli_args, "draft_model", None)
    target_fa_window = _int_or_none(getattr(runtime_config, "target_fa_window", None))
    prefix_cache_enabled = bool(
        runtime_config is not None
        and getattr(runtime_config, "prefix_cache", False)
        and int(target_fa_window or 0) <= 0
    )
    runtime_context = getattr(cli_args, "runtime_context", None)
    cache_identity: Any = tuple(model_key)
    draft_model = getattr(model_provider, "draft_model", None)
    if runtime_context is not None and draft_model is not None:
        cache_identity = build_prefix_key(model_provider, draft_model, runtime_context)
    sync_runtime_cache_manager(runtime_context, cache_identity=cache_identity)
    metal_limits = getattr(cli_args, "metal_limits", None)
    with _LIVE_LOCK:
        global _LIVE_STARTED_AT
        _LIVE_STARTED_AT = time.time()
        _LIVE_SERVER.update(
            {
                "version": version,
                "model": str(target_ref) if target_ref is not None else None,
                "draft": str(draft_ref) if draft_ref is not None else None,
                "mode": "dflash",
                "profile": getattr(runtime_config, "profile", None),
            }
        )
        _LIVE_RUNTIME.update(
            {
                "prefill_step_size": _int_or_none(
                    getattr(runtime_config, "prefill_step_size", None)
                ),
                "prefix_cache_enabled": prefix_cache_enabled,
                "target_fa_window": (
                    None if int(target_fa_window or 0) <= 0 else int(target_fa_window)
                ),
                "diagnostics": getattr(cli_args, "diagnostics", None),
            }
        )
        _LIVE_PREFIX_CONFIG.update(
            {
                "enabled": prefix_cache_enabled,
                "max_entries": _int_or_none(
                    getattr(runtime_config, "prefix_cache_max_entries", None)
                ),
                "max_bytes": _int_or_none(
                    getattr(runtime_config, "prefix_cache_max_bytes", None)
                ),
            }
        )
        _LIVE_METAL_LIMITS.update(
            {
                "wired_limit_bytes": _int_or_none(
                    getattr(metal_limits, "wired_bytes", None)
                ),
            }
        )

def get_live_metrics_payload(*, prefix_cache_manager: Optional[Any] = None) -> dict[str, Any]:
    prefix_stats = _prefix_cache_payload(prefix_cache_manager)
    now = time.time()
    with _LIVE_LOCK:
        server = dict(_LIVE_SERVER)
        server["uptime_s"] = max(0.0, now - _LIVE_STARTED_AT)
        started_at = _LIVE_STARTED_AT
        runtime = dict(_LIVE_RUNTIME)
        last_request = None if _LIVE_LAST_REQUEST is None else dict(_LIVE_LAST_REQUEST)
        current_request = _current_request_payload(_LIVE_CURRENT_REQUEST, now)
        recent_requests = [dict(request) for request in _LIVE_RECENT_REQUESTS]
        totals = dict(_LIVE_TOTALS)
        metal_limits = dict(_LIVE_METAL_LIMITS)
        finish_times = list(_LIVE_REQUEST_FINISH_TIMES)
    payload = {
        "server": server,
        "runtime": runtime,
        "memory": _memory_payload(metal_limits),
        "current_request": current_request,
        "last_request": last_request,
        "recent_requests": recent_requests,
        "rates": _rates_payload(
            totals=totals,
            current_request=current_request,
            last_request=last_request,
            finish_times=finish_times,
            started_at=started_at,
            now=now,
        ),
        "prefix_cache": prefix_stats,
        "totals": totals,
    }
    if prefix_cache_manager is not None:
        stats = prefix_cache_manager.stats()
        payload["totals"]["cache_hits"] = int(
            stats.get("exact_hits", 0) + stats.get("prefix_hits", 0)
        )
        payload["totals"]["cache_misses"] = int(stats.get("misses", 0))
    return payload

def start_live_request(
    *,
    request_id: int,
    mode_used: str,
    prompt_tokens: Optional[int],
    max_tokens: int,
    cache_hit_tokens: int = 0,
    cache_lookup_ms: float = 0.0,
) -> None:
    current = {
        "request_id": int(request_id),
        "mode_used": mode_used,
        "state": "queued",
        "prompt_tokens": _int_or_none(prompt_tokens),
        "max_tokens": int(max_tokens),
        "generated_tokens": 0,
        "cache_hit_tokens": int(cache_hit_tokens),
        "cache_lookup_ms": float(cache_lookup_ms),
        "prefill_tokens_processed": None,
        "prefill_tokens_total": _int_or_none(prompt_tokens),
        "prefill_s": None,
        "decode_s": None,
        "decode_tok_s": None,
        "acceptance_rate": None,
        "cycles": None,
        "finish_reason": None,
        "_started_at": time.time(),
        "_prefill_done_at": None,
    }
    with _LIVE_LOCK:
        global _LIVE_CURRENT_REQUEST
        _LIVE_CURRENT_REQUEST = current

def update_live_request(
    *,
    request_id: int,
    state: Optional[str] = None,
    generated_tokens: Optional[int] = None,
    prefill_tokens_processed: Optional[int] = None,
    prefill_tokens_total: Optional[int] = None,
    prefill_s: Optional[float] = None,
    prefill_done: bool = False,
    decode_tok_s: Optional[float] = None,
    acceptance_rate: Optional[float] = None,
    cycles: Optional[int] = None,
    finish_reason: Optional[str] = None,
) -> None:
    with _LIVE_LOCK:
        current = _LIVE_CURRENT_REQUEST
        if current is None or int(current.get("request_id", -1)) != int(request_id):
            return
        if state is not None:
            current["state"] = state
        if generated_tokens is not None:
            current["generated_tokens"] = int(generated_tokens)
        if prefill_tokens_processed is not None:
            current["prefill_tokens_processed"] = int(prefill_tokens_processed)
        if prefill_tokens_total is not None:
            current["prefill_tokens_total"] = int(prefill_tokens_total)
        if prefill_s is not None:
            current["prefill_s"] = float(prefill_s)
        if prefill_done:
            current["_prefill_done_at"] = time.time()
        if decode_tok_s is not None:
            current["decode_tok_s"] = float(decode_tok_s)
        if acceptance_rate is not None:
            current["acceptance_rate"] = float(acceptance_rate)
        if cycles is not None:
            current["cycles"] = int(cycles)
        if finish_reason is not None:
            current["finish_reason"] = finish_reason

def clear_live_request(*, request_id: int) -> None:
    with _LIVE_LOCK:
        global _LIVE_CURRENT_REQUEST
        if (
            _LIVE_CURRENT_REQUEST is not None
            and int(_LIVE_CURRENT_REQUEST.get("request_id", -1)) == int(request_id)
        ):
            _LIVE_CURRENT_REQUEST = None

def record_target_only_request(
    *,
    request_id: int,
    mode_used: str,
    wall_ms: float,
    max_tokens: int,
) -> None:
    last_request = _last_request_payload(
        request_id=request_id,
        prompt_tokens=None,
        generated_tokens=None,
        wall_ms=wall_ms,
        prefill_ms=None,
        decode_ms=None,
        prefill_tok_s_physical=None,
        prefill_tok_s_apparent=None,
        decode_tok_s=None,
        acceptance_rate=None,
        cycles=None,
        finish_reason=None,
    )
    last_request["mode_used"] = mode_used
    last_request["max_tokens"] = int(max_tokens)
    with _LIVE_LOCK:
        _LIVE_TOTALS["requests"] += 1
        _finish_live_request_unlocked(last_request)

def reset_live_metrics_for_tests() -> None:
    with _LIVE_LOCK:
        global _LIVE_STARTED_AT, _LIVE_LAST_REQUEST, _LIVE_CURRENT_REQUEST
        _LIVE_STARTED_AT = time.time()
        _LIVE_SERVER.update(
            {
                "version": None,
                "model": None,
                "draft": None,
                "mode": "dflash",
                "profile": None,
            }
        )
        _LIVE_RUNTIME.update(
            {
                "prefill_step_size": None,
                "prefix_cache_enabled": None,
                "target_fa_window": None,
                "diagnostics": None,
            }
        )
        _LIVE_PREFIX_CONFIG.update(
            {"enabled": False, "max_entries": None, "max_bytes": None}
        )
        _LIVE_METAL_LIMITS.update({"wired_limit_bytes": None})
        _LIVE_LAST_REQUEST = None
        _LIVE_CURRENT_REQUEST = None
        _LIVE_RECENT_REQUESTS.clear()
        _LIVE_REQUEST_FINISH_TIMES.clear()
        _LIVE_LAST_PREFIX.update({"restored": None, "computed": None})
        for key in _LIVE_TOTALS:
            _LIVE_TOTALS[key] = 0

def write_summary_line(
    *,
    summary_event: dict[str, Any],
    prompt_token_count: int,
) -> None:
    generation_tokens = int(summary_event.get("generation_tokens", 0) or 0)
    elapsed_us = float(summary_event.get("elapsed_us", 0.0) or 0.0)
    phase_timings_us = dict(summary_event.get("phase_timings_us") or {})
    prefill_us = float(phase_timings_us.get("prefill", 0.0) or 0.0)
    prefill_tok_s = (
        prompt_token_count / (prefill_us / 1_000_000.0)
        if prefill_us > 0.0
        else 0.0
    )
    decode_s = max(0.0, (elapsed_us - prefill_us) / 1_000_000.0)
    tok_s = (generation_tokens / decode_s) if decode_s > 0.0 else 0.0
    acceptance_pct = float(summary_event.get("acceptance_ratio", 0.0) or 0.0) * 100.0
    total_s = elapsed_us / 1_000_000.0
    _write_observability_line(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] {tok_s:.1f} tok/s | "
        f"prefill {prefill_tok_s:.1f} tok/s | "
        f"{acceptance_pct:.1f}% accepted | {generation_tokens} tokens | "
        f"{total_s:.1f}s | prompt: {prompt_token_count} tokens\n"
    )


def write_post_request_memory_line(*, request_id: int) -> None:
    try:
        memory = get_memory_snapshot()
    except Exception as exc:
        _write_observability_line(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] req#{request_id} memory snapshot failed: {exc}\n"
        )
        return
    unavailable = [
        key
        for key in (
            "mlx_active_gb",
            "mlx_cache_gb",
            "mlx_peak_gb",
            "rss_gb",
            "rss_peak_gb",
        )
        if memory.get(key) is None
    ]
    if unavailable:
        _write_observability_line(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] req#{request_id} memory snapshot partial "
            f"unavailable={','.join(unavailable)}\n"
        )
    mlx_active_gb = _float_or_none(memory.get("mlx_active_gb")) or 0.0
    mlx_cache_gb = _float_or_none(memory.get("mlx_cache_gb")) or 0.0
    mlx_peak_gb = _float_or_none(memory.get("mlx_peak_gb")) or 0.0
    rss_now_gb = _float_or_none(memory.get("rss_gb")) or 0.0
    rss_peak_gb = _float_or_none(memory.get("rss_peak_gb")) or 0.0
    untracked_gb = max(0.0, rss_now_gb - mlx_active_gb - mlx_cache_gb)
    _write_observability_line(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] req#{request_id} "
        f"mlx_active={mlx_active_gb:.2f} mlx_cache={mlx_cache_gb:.2f} "
        f"mlx_peak={mlx_peak_gb:.2f} rss_now={rss_now_gb:.2f} "
        f"rss_peak={rss_peak_gb:.2f} untracked={untracked_gb:.2f} GB\n"
    )

def _write_observability_line(line: str) -> None:
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception as exc:
        fallback = (
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] observability stderr write failed: {exc}\n"
        )
        try:
            os.write(2, fallback.encode())
        except OSError:
            return


def record_request_metrics(
    *,
    request_id: int,
    summary_event: Optional[dict[str, Any]],
    request_start_ns: int,
    request_done_ns: int,
    first_token_ns: Optional[int],
    prefill_done_ns: Optional[int],
    prompt_token_count: int,
    live_token_count: int,
    cache_lookup_ms: float,
    cache_hit_tokens: int,
    cache_insert_ms: float,
    finish_reason: Optional[str],
    max_tokens: int,
    prompt_regime: Optional[dict[str, Any]] = None,
    memory_waterfall_peak: Optional[dict[str, Any]] = None,
    diagnostics: Optional[DiagnosticsConfig] = None,
    prefill_event: Optional[dict[str, Any]] = None,
    runtime_config: Optional[Any] = None,
) -> None:
    wall_ms = (request_done_ns - request_start_ns) / 1e6
    ttft_ms = (
        (first_token_ns - request_start_ns) / 1e6
        if first_token_ns is not None
        else None
    )
    prefill_ms = (
        (prefill_done_ns - request_start_ns) / 1e6
        if prefill_done_ns is not None
        else None
    )
    prefill_tok_s = (
        prompt_token_count / (float(prefill_ms) / 1_000.0)
        if prefill_ms is not None and float(prefill_ms) > 0.0
        else None
    )
    decode_ms = (
        (request_done_ns - prefill_done_ns) / 1e6
        if prefill_done_ns is not None
        else None
    )
    fallback_used = bool(summary_event.get("fallback_ar") if summary_event else False)
    generation_tokens = int(
        (summary_event or {}).get("generation_tokens", live_token_count) or 0
    )
    acceptance_ratio = float((summary_event or {}).get("acceptance_ratio", 0.0) or 0.0)
    cycles_completed = int((summary_event or {}).get("cycles_completed", 0) or 0)
    tokens_per_cycle = float((summary_event or {}).get("tokens_per_cycle", 0.0) or 0.0)
    prefill_phase_timings_us = _prefill_phase_timings(prefill_event)
    prefill_accounting = _prefill_accounting(prefill_event)
    runtime_config_payload = _runtime_config_payload(runtime_config)
    logical_prefill_tokens = int(
        prefill_accounting.get("logical_ctx_tokens", prompt_token_count)
    )
    physical_prefill_tokens = prefill_accounting.get("physical_prefill_tokens")
    restored_prefill_tokens = prefill_accounting.get("prefill_tokens_restored")
    computed_prefill_tokens = prefill_accounting.get("prefill_tokens_computed")
    prefill_tok_s_apparent = (
        logical_prefill_tokens / (float(prefill_ms) / 1_000.0)
        if prefill_ms is not None and float(prefill_ms) > 0.0
        else None
    )
    prefill_tok_s_physical = (
        int(physical_prefill_tokens) / (float(prefill_ms) / 1_000.0)
        if physical_prefill_tokens is not None
        and prefill_ms is not None
        and float(prefill_ms) > 0.0
        else None
    )
    decode_tok_s = (
        generation_tokens / (float(decode_ms) / 1_000.0)
        if decode_ms is not None and float(decode_ms) > 0.0
        else None
    )
    _record_live_request(
        request_id=request_id,
        prompt_token_count=prompt_token_count,
        generation_tokens=generation_tokens,
        wall_ms=wall_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        prefill_tok_s_physical=prefill_tok_s_physical,
        prefill_tok_s_apparent=prefill_tok_s_apparent,
        decode_tok_s=decode_tok_s,
        acceptance_ratio=acceptance_ratio,
        cycles_completed=cycles_completed,
        finish_reason=finish_reason,
        cache_hit_tokens=cache_hit_tokens,
        runtime_config=runtime_config,
        physical_prefill_tokens=physical_prefill_tokens,
        restored_prefill_tokens=restored_prefill_tokens,
        computed_prefill_tokens=computed_prefill_tokens,
    )
    _bench_log_post(
        diagnostics.trace if diagnostics is not None else None,
        request_id=request_id,
        mode_used="dflash_fallback" if fallback_used else "dflash",
        prompt_tokens=int(prompt_token_count),
        generated_tokens=generation_tokens,
        wall_ms=wall_ms,
        ttft_ms=ttft_ms,
        prefill_ms=prefill_ms,
        prefill_tok_s=prefill_tok_s,
        prefill_tok_s_physical=prefill_tok_s_physical,
        prefill_tok_s_apparent=prefill_tok_s_apparent,
        decode_tok_s=decode_tok_s,
        decode_ms=decode_ms,
        cache_lookup_ms=cache_lookup_ms,
        cache_hit_tokens=cache_hit_tokens,
        cache_insert_ms=cache_insert_ms,
        acceptance_ratio=acceptance_ratio,
        tokens_per_cycle=tokens_per_cycle,
        cycles_completed=cycles_completed,
        finish_reason=finish_reason,
        max_tokens=int(max_tokens),
        prompt_regime=prompt_regime or {},
        prefill_event=prefill_event or {},
        **prefill_accounting,
        prefill_phase_timings_us=prefill_phase_timings_us,
        phase_timings_us=dict((summary_event or {}).get("phase_timings_us") or {}),
        runtime_config=runtime_config_payload,
        memory_waterfall_peak=memory_waterfall_peak or {},
    )
    _append_diagnostics_summary(
        request_id=request_id,
        wall_ms=wall_ms,
        ttft_ms=ttft_ms,
        prefill_ms=prefill_ms,
        prefill_tok_s=prefill_tok_s,
        decode_ms=decode_ms,
        prompt_token_count=prompt_token_count,
        generation_tokens=generation_tokens,
        acceptance_ratio=acceptance_ratio,
        tokens_per_cycle=tokens_per_cycle,
        cycles_completed=cycles_completed,
        cache_hit_tokens=cache_hit_tokens,
        runtime_config=runtime_config_payload,
        diagnostics=diagnostics,
    )

def _append_diagnostics_summary(
    *,
    request_id: int,
    wall_ms: float,
    ttft_ms: Optional[float],
    prefill_ms: Optional[float],
    prefill_tok_s: Optional[float],
    decode_ms: Optional[float],
    prompt_token_count: int,
    generation_tokens: int,
    acceptance_ratio: float,
    tokens_per_cycle: float,
    cycles_completed: int,
    cache_hit_tokens: int,
    runtime_config: dict[str, Any],
    diagnostics: Optional[DiagnosticsConfig],
) -> None:
    if diagnostics is None or diagnostics.run_dir is None:
        return
    path = Path(diagnostics.run_dir) / "summary.md"
    prefill_step_size = runtime_config.get("prefill_step_size", "")
    line = (
        f"| {request_id} | {prompt_token_count} | {generation_tokens} | "
        f"{_fmt_ms(wall_ms)} | {_fmt_ms(ttft_ms)} | {_fmt_ms(prefill_ms)} | "
        f"{_fmt_float(prefill_tok_s)} | {prefill_step_size} | "
        f"{_fmt_ms(decode_ms)} | {acceptance_ratio:.3f} | "
        f"{tokens_per_cycle:.2f} | {cycles_completed} | {cache_hit_tokens} |\n"
    )
    try:
        current = path.read_text(errors="ignore") if path.exists() else ""
        with path.open("a") as fp:
            if "| request |" not in current:
                fp.write(
                    "\n## Requests\n\n"
                    "| request | prompt_tokens | generated | wall_ms | ttft_ms | "
                    "prefill_ms | prefill_tok_s | prefill_step | decode_ms | "
                    "acceptance | tokens_cycle | cycles | cache_hit_tokens |\n"
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
                )
            fp.write(line)
    except OSError:
        pass

def _prefill_phase_timings(prefill_event: Optional[dict[str, Any]]) -> dict[str, float]:
    if not prefill_event:
        return {}
    out: dict[str, float] = {}
    for key, value in prefill_event.items():
        if key.startswith("phase_") and key.endswith("_us"):
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                pass
    return out

def _prefill_accounting(prefill_event: Optional[dict[str, Any]]) -> dict[str, int]:
    if not prefill_event:
        return {}
    keys = (
        "logical_ctx_tokens",
        "physical_prefill_tokens",
        "prefill_tokens_restored",
        "prefill_tokens_computed",
    )
    out: dict[str, int] = {}
    for key in keys:
        value = prefill_event.get(key)
        if value is None:
            continue
        try:
            out[key] = int(value)
        except (TypeError, ValueError):
            pass
    return out

def _runtime_config_payload(runtime_config: Optional[Any]) -> dict[str, Any]:
    if runtime_config is None:
        return {}
    keys = (
        "profile",
        "prefill_step_size",
        "draft_sink_size",
        "draft_window_size",
        "verify_len_cap",
        "prefix_cache",
        "prefix_cache_l2",
        "target_fa_window",
        "dflash_max_ctx",
        "memory_waterfall",
        "max_snapshot_tokens",
        "clear_cache_boundaries",
        "verify_mode",
    )
    payload: dict[str, Any] = {}
    for key in keys:
        if hasattr(runtime_config, key):
            payload[key] = getattr(runtime_config, key)
    return payload

def _record_live_request(
    *,
    request_id: int,
    prompt_token_count: int,
    generation_tokens: int,
    wall_ms: float,
    prefill_ms: Optional[float],
    decode_ms: Optional[float],
    prefill_tok_s_physical: Optional[float],
    prefill_tok_s_apparent: Optional[float],
    decode_tok_s: Optional[float],
    acceptance_ratio: float,
    cycles_completed: int,
    finish_reason: Optional[str],
    cache_hit_tokens: int,
    runtime_config: Optional[Any],
    physical_prefill_tokens: Optional[int],
    restored_prefill_tokens: Optional[int],
    computed_prefill_tokens: Optional[int],
) -> None:
    last_request = _last_request_payload(
        request_id=request_id,
        prompt_tokens=int(prompt_token_count),
        generated_tokens=int(generation_tokens),
        wall_ms=wall_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        prefill_tok_s_physical=prefill_tok_s_physical,
        prefill_tok_s_apparent=prefill_tok_s_apparent,
        decode_tok_s=decode_tok_s,
        acceptance_rate=float(acceptance_ratio),
        cycles=int(cycles_completed),
        finish_reason=finish_reason,
    )
    cache_enabled = bool(
        runtime_config is not None
        and getattr(runtime_config, "prefix_cache", False)
        and int(getattr(runtime_config, "target_fa_window", 0) or 0) <= 0
    )
    with _LIVE_LOCK:
        _LIVE_TOTALS["requests"] += 1
        _LIVE_TOTALS["generated_tokens"] += int(generation_tokens)
        if physical_prefill_tokens is not None:
            _LIVE_TOTALS["prefill_tokens_physical"] += int(physical_prefill_tokens)
        if restored_prefill_tokens is not None:
            _LIVE_TOTALS["prefill_tokens_restored"] += int(restored_prefill_tokens)
        _LIVE_LAST_PREFIX.update(
            {
                "restored": _int_or_none(restored_prefill_tokens),
                "computed": _int_or_none(computed_prefill_tokens),
            }
        )
        if cache_enabled:
            if int(cache_hit_tokens) > 0:
                _LIVE_TOTALS["cache_hits"] += 1
            else:
                _LIVE_TOTALS["cache_misses"] += 1
        _finish_live_request_unlocked(last_request)

def _finish_live_request_unlocked(last_request: dict[str, Any]) -> None:
    global _LIVE_LAST_REQUEST, _LIVE_CURRENT_REQUEST
    _LIVE_LAST_REQUEST = dict(last_request)
    _LIVE_RECENT_REQUESTS.append(dict(last_request))
    _LIVE_REQUEST_FINISH_TIMES.append(time.time())
    if (
        _LIVE_CURRENT_REQUEST is not None
        and _LIVE_CURRENT_REQUEST.get("request_id") == last_request.get("request_id")
    ):
        _LIVE_CURRENT_REQUEST = None

def _rates_payload(
    *,
    totals: dict[str, int],
    current_request: Optional[dict[str, Any]],
    last_request: Optional[dict[str, Any]],
    finish_times: list[float],
    started_at: float,
    now: float,
) -> dict[str, Optional[float]]:
    uptime_s = max(0.0, now - started_at)
    requests = int(totals.get("requests", 0) or 0)
    recent_window_s = min(60.0, uptime_s)
    recent_cutoff = now - 60.0
    recent_requests = sum(1 for ts in finish_times if ts >= recent_cutoff)
    source = current_request if current_request is not None else last_request
    return {
        "requests_per_s": _rate(requests, uptime_s),
        "requests_per_s_60s": _rate(recent_requests, recent_window_s),
        "generated_tokens_per_s": _rate(
            int(totals.get("generated_tokens", 0) or 0),
            uptime_s,
        ),
        "prefill_tokens_physical_per_s": _rate(
            int(totals.get("prefill_tokens_physical", 0) or 0),
            uptime_s,
        ),
        "prefill_tokens_restored_per_s": _rate(
            int(totals.get("prefill_tokens_restored", 0) or 0),
            uptime_s,
        ),
        "active_decode_tok_s": _float_or_none(
            None if source is None else source.get("decode_tok_s")
        ),
    }

def _current_request_payload(
    current: Optional[dict[str, Any]],
    now: float,
) -> Optional[dict[str, Any]]:
    if current is None:
        return None
    started_at = float(current.get("_started_at", now) or now)
    prefill_done_at = current.get("_prefill_done_at")
    payload = {
        key: value
        for key, value in current.items()
        if not str(key).startswith("_")
    }
    payload["elapsed_s"] = max(0.0, now - started_at)
    if prefill_done_at is not None:
        payload["decode_s"] = max(0.0, now - float(prefill_done_at))
    return payload

def _last_request_payload(
    *,
    request_id: int,
    prompt_tokens: Optional[int],
    generated_tokens: Optional[int],
    wall_ms: Optional[float],
    prefill_ms: Optional[float],
    decode_ms: Optional[float],
    prefill_tok_s_physical: Optional[float],
    prefill_tok_s_apparent: Optional[float],
    decode_tok_s: Optional[float],
    acceptance_rate: Optional[float],
    cycles: Optional[int],
    finish_reason: Optional[str],
) -> dict[str, Any]:
    return {
        "request_id": int(request_id),
        "prompt_tokens": _int_or_none(prompt_tokens),
        "generated_tokens": _int_or_none(generated_tokens),
        "wall_s": _seconds_or_none(wall_ms),
        "prefill_s": _seconds_or_none(prefill_ms),
        "decode_s": _seconds_or_none(decode_ms),
        "prefill_tok_s_physical": _float_or_none(prefill_tok_s_physical),
        "prefill_tok_s_apparent": _float_or_none(prefill_tok_s_apparent),
        "decode_tok_s": _float_or_none(decode_tok_s),
        "acceptance_rate": _float_or_none(acceptance_rate),
        "cycles": _int_or_none(cycles),
        "finish_reason": finish_reason,
    }

def _prefix_cache_payload(prefix_cache_manager: Optional[Any]) -> dict[str, Any]:
    with _LIVE_LOCK:
        config = dict(_LIVE_PREFIX_CONFIG)
        last_request = None if _LIVE_LAST_REQUEST is None else dict(_LIVE_LAST_REQUEST)
        last_prefix = dict(_LIVE_LAST_PREFIX)
    if prefix_cache_manager is None:
        if not config.get("enabled"):
            return {
                "entries": None,
                "max_entries": config.get("max_entries"),
                "bytes": None,
                "max_bytes": config.get("max_bytes"),
                "hits": None,
                "misses": None,
                "insertions": None,
                "evictions": None,
                "prefill_tokens_saved": None,
                "last_restored_tokens": None,
                "last_computed_tokens": None,
            }
        return {
            "entries": 0,
            "max_entries": config.get("max_entries"),
            "bytes": 0,
            "max_bytes": config.get("max_bytes"),
            "hits": 0,
            "misses": 0,
            "insertions": 0,
            "evictions": 0,
            "prefill_tokens_saved": 0,
            "last_restored_tokens": None,
            "last_computed_tokens": None,
        }
    stats = prefix_cache_manager.stats()
    return {
        "entries": _int_or_none(stats.get("current_entries")),
        "max_entries": _int_or_none(stats.get("max_entries")),
        "bytes": _int_or_none(stats.get("current_bytes")),
        "max_bytes": _int_or_none(stats.get("max_bytes")),
        "hits": int(stats.get("exact_hits", 0) + stats.get("prefix_hits", 0)),
        "misses": _int_or_none(stats.get("misses")),
        "insertions": _int_or_none(stats.get("insertions")),
        "evictions": _int_or_none(stats.get("evictions")),
        "prefill_tokens_saved": _int_or_none(stats.get("prefill_tokens_saved")),
        "last_restored_tokens": (
            None if last_request is None else _int_or_none(last_prefix.get("restored"))
        ),
        "last_computed_tokens": (
            None if last_request is None else _int_or_none(last_prefix.get("computed"))
        ),
    }

def _memory_payload(metal_limits: dict[str, Any]) -> dict[str, Any]:
    wired_limit_bytes = metal_limits.get("wired_limit_bytes")
    rss_bytes = _current_rss_bytes()
    rss_gb = None if rss_bytes is None else float(rss_bytes) / _GB
    return {
        "rss_gb": rss_gb,
        "rss_peak_gb": _rss_peak_gb(),
        "mlx_active_gb": _mlx_memory_gb("get_active_memory"),
        "mlx_cache_gb": _mlx_memory_gb("get_cache_memory"),
        "mlx_peak_gb": _mlx_memory_gb("get_peak_memory"),
        "wired_gb": None,
        "wired_limit_gb": (
            None if wired_limit_bytes is None else float(wired_limit_bytes) / _GB
        ),
    }

def get_memory_snapshot() -> dict[str, Any]:
    return _memory_payload({"wired_limit_bytes": None})

def _current_rss_bytes() -> Optional[int]:
    if sys.platform == "darwin":
        return _darwin_proc_resident_size_bytes() or _darwin_task_resident_size_bytes()
    try:
        with open("/proc/self/statm") as fp:
            fields = fp.read().split()
        if len(fields) < 2:
            return None
        return int(fields[1]) * int(resource.getpagesize())
    except Exception:
        return None

def _darwin_proc_resident_size_bytes() -> Optional[int]:
    class ProcTaskInfo(ctypes.Structure):
        _fields_ = [
            ("pti_virtual_size", ctypes.c_uint64),
            ("pti_resident_size", ctypes.c_uint64),
            ("pti_total_user", ctypes.c_uint64),
            ("pti_total_system", ctypes.c_uint64),
            ("pti_threads_user", ctypes.c_uint64),
            ("pti_threads_system", ctypes.c_uint64),
            ("pti_policy", ctypes.c_int32),
            ("pti_faults", ctypes.c_int32),
            ("pti_pageins", ctypes.c_int32),
            ("pti_cow_faults", ctypes.c_int32),
            ("pti_messages_sent", ctypes.c_int32),
            ("pti_messages_received", ctypes.c_int32),
            ("pti_syscalls_mach", ctypes.c_int32),
            ("pti_syscalls_unix", ctypes.c_int32),
            ("pti_csw", ctypes.c_int32),
            ("pti_threadnum", ctypes.c_int32),
            ("pti_numrunning", ctypes.c_int32),
            ("pti_priority", ctypes.c_int32),
        ]

    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        info = ProcTaskInfo()
        result = proc_pidinfo(
            os.getpid(),
            4,  # PROC_PIDTASKINFO
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if result < ctypes.sizeof(info) or info.pti_resident_size <= 0:
            return None
        return int(info.pti_resident_size)
    except Exception:
        return None

def _darwin_task_resident_size_bytes() -> Optional[int]:
    class TimeValue(ctypes.Structure):
        _fields_ = [
            ("seconds", ctypes.c_int32),
            ("microseconds", ctypes.c_int32),
        ]

    class TaskBasicInfo64(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time", TimeValue),
            ("system_time", TimeValue),
            ("policy", ctypes.c_int32),
            ("suspend_count", ctypes.c_int32),
        ]

    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        task_info = libc.task_info
        task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        task_info.restype = ctypes.c_int32
        libc.mach_task_self.restype = ctypes.c_uint32
        info = TaskBasicInfo64()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int32))
        result = task_info(
            libc.mach_task_self(),
            5,  # TASK_BASIC_INFO_64
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if result != 0 or info.resident_size <= 0 or info.resident_size == 0xFFFFFFFF:
            return None
        return int(info.resident_size)
    except Exception:
        return None

def _rss_peak_gb() -> Optional[float]:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            value *= 1024.0
        return value / _GB
    except Exception:
        return None

def _mlx_memory_gb(name: str) -> Optional[float]:
    try:
        import mlx.core as mx

        getter = getattr(mx, name, None)
        if getter is None:
            return None
        return float(getter()) / _GB
    except Exception:
        return None

def _seconds_or_none(value_ms: Optional[float]) -> Optional[float]:
    if value_ms is None:
        return None
    return float(value_ms) / 1_000.0

def _rate(count: int, elapsed_s: float) -> Optional[float]:
    if elapsed_s <= 0.0:
        return None
    return float(count) / float(elapsed_s)

def _float_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _int_or_none(value: Optional[Any]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _fmt_ms(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.1f}"

def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"
