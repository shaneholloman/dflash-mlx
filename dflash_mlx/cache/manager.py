# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.prefix_l2 import DFlashPrefixL2Cache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot

_DFLASH_RUNTIME_CACHE_MANAGER: Optional["RuntimeCacheManager"] = None
_DFLASH_RUNTIME_CACHE_CONFIG_KEY: Optional[tuple[Any, ...]] = None
_DFLASH_RUNTIME_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PrefixCacheLookupResult:
    matched_tokens: int
    snapshot: Optional[DFlashPrefixSnapshot]
    elapsed_ms: float


class RuntimeCacheManager:
    def __init__(self, cache: DFlashPrefixCache) -> None:
        self._cache = cache

    def set_trace_config(self, trace_config: Any) -> None:
        self._cache.set_trace_config(trace_config)

    def shutdown(self) -> None:
        self._cache.shutdown()

    def stats(self) -> dict[str, Any]:
        return self._cache.stats()

    def memory_waterfall_bytes(self) -> dict[str, int]:
        return self._cache.memory_waterfall_bytes()

    def lookup(
        self,
        tokens: list[int] | tuple[int, ...],
        key: DFlashPrefixKey,
    ) -> PrefixCacheLookupResult:
        lookup_t0 = time.perf_counter_ns()
        matched_len, snapshot = self._cache.lookup(tokens, key)
        return PrefixCacheLookupResult(
            matched_tokens=int(matched_len),
            snapshot=snapshot,
            elapsed_ms=(time.perf_counter_ns() - lookup_t0) / 1e6,
        )

    def maybe_insert_snapshot(
        self,
        snapshot: Any,
        *,
        key: DFlashPrefixKey,
        kind: str,
        require_logits: bool,
    ) -> float:
        if not isinstance(snapshot, DFlashPrefixSnapshot):
            raise TypeError(f"expected DFlashPrefixSnapshot, got {type(snapshot).__name__}")
        if snapshot.key != key:
            raise ValueError("prefix snapshot key does not match request key")
        if snapshot.kind != kind:
            raise ValueError(f"expected {kind!r} prefix snapshot, got {snapshot.kind!r}")
        if require_logits and snapshot.last_logits is None:
            return 0.0
        insert_t0 = time.perf_counter_ns()
        try:
            self._cache.insert(snapshot)
        except Exception as exc:
            _log_insert_failure(kind, exc)
            raise
        elapsed_ms = (time.perf_counter_ns() - insert_t0) / 1e6
        if kind == "generation":
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"[dflash] end-of-request snapshot saved "
                f"({snapshot.prefix_len} tokens)\n"
            )
            sys.stderr.flush()
        return elapsed_ms

    def log_stats(self, label: str = "") -> None:
        _format_stats_line(self._cache, label)


def get_runtime_cache_manager(
    runtime_context: Optional[Any] = None,
    *,
    cache_identity: Any = None,
) -> Optional[RuntimeCacheManager]:
    global _DFLASH_RUNTIME_CACHE_CONFIG_KEY, _DFLASH_RUNTIME_CACHE_MANAGER
    if runtime_context is None:
        return None
    if _runtime_cache_disabled(runtime_context):
        _clear_runtime_cache_manager()
        return None
    config_key = _prefix_cache_config_key(runtime_context, cache_identity=cache_identity)
    trace_config = runtime_context.diagnostics.trace
    manager = _DFLASH_RUNTIME_CACHE_MANAGER
    if manager is not None and _DFLASH_RUNTIME_CACHE_CONFIG_KEY == config_key:
        manager.set_trace_config(trace_config)
        return manager
    with _DFLASH_RUNTIME_CACHE_LOCK:
        manager = _DFLASH_RUNTIME_CACHE_MANAGER
        if manager is not None and _DFLASH_RUNTIME_CACHE_CONFIG_KEY == config_key:
            manager.set_trace_config(trace_config)
            return manager
        _shutdown_manager(manager)
        _DFLASH_RUNTIME_CACHE_MANAGER = RuntimeCacheManager(
            _make_prefix_cache(runtime_context)
        )
        _DFLASH_RUNTIME_CACHE_CONFIG_KEY = config_key
    return _DFLASH_RUNTIME_CACHE_MANAGER


def sync_runtime_cache_manager(
    runtime_context: Optional[Any] = None,
    *,
    cache_identity: Any = None,
) -> Optional[RuntimeCacheManager]:
    global _DFLASH_RUNTIME_CACHE_CONFIG_KEY, _DFLASH_RUNTIME_CACHE_MANAGER
    if runtime_context is None:
        return None
    if _runtime_cache_disabled(runtime_context):
        _clear_runtime_cache_manager()
        return None
    manager = _DFLASH_RUNTIME_CACHE_MANAGER
    if manager is None:
        return None
    config_key = _prefix_cache_config_key(runtime_context, cache_identity=cache_identity)
    if _DFLASH_RUNTIME_CACHE_CONFIG_KEY != config_key:
        _clear_runtime_cache_manager()
        return None
    manager.set_trace_config(runtime_context.diagnostics.trace)
    return manager


def current_runtime_cache_manager() -> Optional[RuntimeCacheManager]:
    return _DFLASH_RUNTIME_CACHE_MANAGER


def shutdown_runtime_cache_manager() -> None:
    _clear_runtime_cache_manager(raise_on_error=False)


def _clear_runtime_cache_manager(*, raise_on_error: bool = True) -> None:
    global _DFLASH_RUNTIME_CACHE_CONFIG_KEY, _DFLASH_RUNTIME_CACHE_MANAGER
    with _DFLASH_RUNTIME_CACHE_LOCK:
        manager = _DFLASH_RUNTIME_CACHE_MANAGER
        config_key = _DFLASH_RUNTIME_CACHE_CONFIG_KEY
        _DFLASH_RUNTIME_CACHE_MANAGER = None
        _DFLASH_RUNTIME_CACHE_CONFIG_KEY = None
    try:
        _shutdown_manager(manager, raise_on_error=raise_on_error)
    except Exception:
        with _DFLASH_RUNTIME_CACHE_LOCK:
            _DFLASH_RUNTIME_CACHE_MANAGER = manager
            _DFLASH_RUNTIME_CACHE_CONFIG_KEY = config_key
        raise


def _prefix_cache_config_key(
    runtime_context: Any,
    *,
    cache_identity: Any = None,
) -> tuple[Any, ...]:
    runtime_config = runtime_context.runtime
    return (
        cache_identity,
        int(runtime_config.prefix_cache_max_entries),
        int(runtime_config.prefix_cache_max_bytes),
        int(runtime_config.max_snapshot_tokens),
        bool(runtime_config.prefix_cache_l2),
        str(runtime_config.prefix_cache_l2_dir),
        int(runtime_config.prefix_cache_l2_max_bytes),
    )


def _runtime_cache_disabled(runtime_context: Any) -> bool:
    runtime_config = runtime_context.runtime
    return bool(runtime_config.target_fa_window > 0 or not runtime_config.prefix_cache)


def _shutdown_manager(
    manager: Optional[RuntimeCacheManager],
    *,
    raise_on_error: bool = True,
) -> None:
    if manager is None:
        return
    try:
        manager.shutdown()
    except Exception as exc:
        sys.stderr.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] runtime cache manager shutdown failed: {exc}\n"
        )
        sys.stderr.flush()
        if raise_on_error:
            raise


def _make_prefix_cache(runtime_context: Any) -> DFlashPrefixCache:
    runtime_config = runtime_context.runtime
    max_entries = int(runtime_config.prefix_cache_max_entries)
    max_bytes = int(runtime_config.prefix_cache_max_bytes)
    l2: Optional[DFlashPrefixL2Cache] = None
    if runtime_config.prefix_cache_l2:
        try:
            l2 = DFlashPrefixL2Cache(
                cache_dir=runtime_config.prefix_cache_l2_dir,
                max_bytes=runtime_config.prefix_cache_l2_max_bytes,
            )
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix L2 cache enabled "
                f"(dir={l2.cache_dir}, max_bytes={runtime_config.prefix_cache_l2_max_bytes}, "
                f"writable={l2.writable})\n"
            )
        except Exception as exc:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix L2 cache disabled: {exc}\n"
            )
            l2 = None
    trace_config = runtime_context.diagnostics.trace if runtime_context is not None else None
    cache = DFlashPrefixCache(
        max_entries=max_entries,
        max_bytes=max_bytes,
        l2=l2,
        trace_config=trace_config,
        max_snapshot_tokens=runtime_config.max_snapshot_tokens,
    )
    sys.stderr.write(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix cache enabled "
        f"(max_entries={max_entries}, max_bytes={max_bytes})\n"
    )
    sys.stderr.flush()
    return cache


def _format_stats_line(cache: DFlashPrefixCache, label: str = "") -> None:
    stats = cache.stats()
    prefix = f" [{label}]" if label else ""
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix-cache-stats{prefix} "
        f"entries={stats['current_entries']}/{stats['max_entries']} "
        f"bytes={stats['current_bytes']}/{stats['max_bytes']} "
        f"hits={stats['exact_hits']}+{stats['prefix_hits']} "
        f"misses={stats['misses']} "
        f"insertions={stats['insertions']} "
        f"evictions={stats['evictions']} "
        f"prefill_tokens_saved={stats['prefill_tokens_saved']}"
    )
    l2 = stats.get("l2")
    if l2:
        line += (
            f" l2_hits={stats.get('l2_hits', 0)} l2_misses={stats.get('l2_misses', 0)} "
            f"l2_writes={l2.get('writes', 0)} l2_bytes={l2.get('current_bytes', 0)}/{l2.get('max_bytes', 0)}"
        )
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _log_insert_failure(kind: str, exc: Exception) -> None:
    if kind == "prefill":
        msg = f"[dflash] prefix cache insert failed: {exc}"
    else:
        msg = f"[dflash] end-of-request snapshot failed: {exc}"
    sys.stderr.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    sys.stderr.flush()
