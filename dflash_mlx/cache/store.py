# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import threading
import time
from typing import Any

from dflash_mlx.diagnostics import TraceConfig
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.prefix_l2 import DFlashPrefixL2Cache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.observability.cache import record_cache_event


class PrefixSnapshotStore:
    def __init__(
        self,
        *,
        l1: DFlashPrefixCache,
        l2: DFlashPrefixL2Cache | None = None,
    ) -> None:
        self._l1 = l1
        self._l2 = l2
        self._lock = threading.Lock()
        self._stats: dict[str, int] = {
            "l2_hits": 0,
            "l2_misses": 0,
            "l2_exact_hits": 0,
            "l2_prefix_hits": 0,
            "l2_prefill_tokens_saved": 0,
        }
        self._trace_config: TraceConfig | None = None

    def set_trace_config(self, trace_config: TraceConfig | None) -> None:
        self._trace_config = trace_config
        self._l1.set_trace_config(trace_config)

    def lookup(
        self,
        req_tokens: list[int] | tuple[int, ...],
        key: DFlashPrefixKey,
    ) -> tuple[int, DFlashPrefixSnapshot | None]:
        req_tuple = tuple(int(t) for t in req_tokens)
        t_start = time.perf_counter_ns()
        matched_len, snapshot = self._l1.lookup(req_tuple, key)
        if snapshot is not None or matched_len > 0:
            return matched_len, snapshot
        if self._l2 is None:
            return 0, None

        l2_snapshot = self._l2.lookup(req_tuple, key)
        if l2_snapshot is None:
            with self._lock:
                self._stats["l2_misses"] += 1
            return 0, None

        l2_len = len(l2_snapshot.token_ids)
        promote = self._l1.insert_with_evictions(l2_snapshot, skip_too_long=False)
        self._write_snapshots_to_l2(promote.removed_snapshots)
        exact = l2_len == len(req_tuple)
        with self._lock:
            self._stats["l2_hits"] += 1
            if exact:
                self._stats["l2_exact_hits"] += 1
            else:
                self._stats["l2_prefix_hits"] += 1
            self._stats["l2_prefill_tokens_saved"] += int(l2_len)
        record_cache_event(
            self._trace_config,
            op="lookup",
            result="l2_hit",
            req_tokens=len(req_tuple),
            matched_len=int(l2_len),
            entries=int(self._l1.stats().get("current_entries", 0)),
            elapsed_us=(time.perf_counter_ns() - t_start) / 1_000.0,
        )
        return l2_len, l2_snapshot

    def insert(self, snapshot: DFlashPrefixSnapshot) -> bool:
        inserted_l2_admitted = False
        result = self._l1.insert_with_evictions(
            snapshot,
            skip_too_long=self._l2 is None,
        )
        if self._l2 is not None:
            inserted_l2_admitted = self._write_snapshots_to_l2(
                result.removed_snapshots,
                inserted_snapshot=result.inserted_evicted_snapshot,
            )
        return bool(result.admitted or inserted_l2_admitted)

    def stats(self) -> dict[str, Any]:
        out = self._l1.stats()
        with self._lock:
            stats = dict(self._stats)
        out["l2_hits"] = int(stats["l2_hits"])
        out["l2_misses"] = int(stats["l2_misses"])
        out["exact_hits"] = int(out.get("exact_hits", 0)) + int(stats["l2_exact_hits"])
        out["prefix_hits"] = int(out.get("prefix_hits", 0)) + int(stats["l2_prefix_hits"])
        out["prefill_tokens_saved"] = int(out.get("prefill_tokens_saved", 0)) + int(
            stats["l2_prefill_tokens_saved"]
        )
        if self._l2 is not None:
            out["l2"] = self._l2.stats()
        return out

    def memory_waterfall_bytes(self) -> dict[str, int]:
        out = self._l1.memory_waterfall_bytes()
        out.setdefault("l2_disk_bytes", 0)
        out.setdefault("l2_hits", 0)
        out.setdefault("l2_writes", 0)
        out.setdefault("l2_misses", 0)
        with self._lock:
            out["l2_hits"] = int(self._stats["l2_hits"])
            out["l2_misses"] = int(self._stats["l2_misses"])
        if self._l2 is not None:
            l2 = self._l2.stats()
            out["l2_disk_bytes"] = int(l2.get("current_bytes", 0) or 0)
            out["l2_writes"] = int(l2.get("writes", 0) or 0)
        return out

    def clear(self) -> None:
        self._l1.clear()
        if self._l2 is not None:
            self._l2.clear()

    def shutdown(self) -> None:
        self._l1.shutdown()
        if self._l2 is not None:
            self._l2.shutdown(wait=True)

    def _write_snapshots_to_l2(
        self,
        snapshots: tuple[DFlashPrefixSnapshot, ...],
        *,
        inserted_snapshot: DFlashPrefixSnapshot | None = None,
    ) -> bool:
        if self._l2 is None:
            return False
        inserted_admitted = False
        for snapshot in snapshots:
            admitted = self._l2.insert_async(snapshot)
            if snapshot is inserted_snapshot:
                inserted_admitted = bool(admitted)
        return inserted_admitted
