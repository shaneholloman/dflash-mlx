# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

from dflash_mlx.diagnostics import TraceConfig


class _DiagnosticsJsonlWriter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._dir: str | None = None
        self._post_fp = None
        self._cycle_fp = None
        self._cache_fp = None
        self._initialized = False

    def enabled(self, trace: TraceConfig | None) -> bool:
        return self._init_if_needed(trace)

    def log_post(self, trace: TraceConfig | None, **fields: Any) -> None:
        if self._init_if_needed(trace):
            self._write("post", dict(fields))

    def log_cycle(self, trace: TraceConfig | None, **fields: Any) -> None:
        if self._init_if_needed(trace):
            self._write("cycle", dict(fields))

    def log_cache(self, trace: TraceConfig | None, **fields: Any) -> None:
        if self._init_if_needed(trace):
            self._write("cache", dict(fields))

    def _init_if_needed(self, trace: TraceConfig | None) -> bool:
        desired_dir = trace.log_dir if trace is not None else None
        desired = os.fspath(desired_dir) if desired_dir is not None else None
        if self._initialized and self._dir == desired:
            return self._dir is not None
        with self._lock:
            desired_dir = trace.log_dir if trace is not None else None
            desired = os.fspath(desired_dir) if desired_dir is not None else None
            if self._initialized and self._dir == desired:
                return self._dir is not None
            self._close_locked()
            if not desired:
                self._dir = None
                self._initialized = True
                return False
            try:
                os.makedirs(desired, exist_ok=True)
            except OSError as exc:
                _report_observability_failure(
                    f"diagnostics directory unavailable ({desired})",
                    exc,
                )
                self._dir = None
                self._initialized = True
                return False
            self._dir = desired
            self._initialized = True
            return True

    def _close_locked(self) -> None:
        for fp in (self._post_fp, self._cycle_fp, self._cache_fp):
            if fp is None:
                continue
            try:
                fp.close()
            except OSError as exc:
                _report_observability_failure("diagnostics file close failed", exc)
        self._post_fp = None
        self._cycle_fp = None
        self._cache_fp = None

    def _fp_for_kind(self, kind: str):
        if self._dir is None:
            return None
        if kind == "post":
            if self._post_fp is None:
                self._post_fp = open(
                    os.path.join(self._dir, "post_events.jsonl"),
                    "a",
                    buffering=1,
                )
            return self._post_fp
        if kind == "cycle":
            if self._cycle_fp is None:
                self._cycle_fp = open(
                    os.path.join(self._dir, "cycle_events.jsonl"),
                    "a",
                    buffering=1,
                )
            return self._cycle_fp
        if self._cache_fp is None:
            self._cache_fp = open(
                os.path.join(self._dir, "cache_events.jsonl"),
                "a",
                buffering=1,
            )
        return self._cache_fp

    def _write(self, kind: str, payload: dict[str, Any]) -> None:
        payload.setdefault("ts", time.time())
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        with self._lock:
            try:
                fp = self._fp_for_kind(kind)
                if fp is None:
                    return
                fp.write(line)
            except OSError as exc:
                _report_observability_failure(f"diagnostics {kind} write failed", exc)
                self._close_locked()
                self._dir = None
                self._initialized = True


_WRITER = _DiagnosticsJsonlWriter()


def log_post(trace: TraceConfig | None, **fields: Any) -> None:
    _WRITER.log_post(trace, **fields)


def log_cycle(trace: TraceConfig | None, **fields: Any) -> None:
    _WRITER.log_cycle(trace, **fields)


def log_cache(trace: TraceConfig | None, **fields: Any) -> None:
    _WRITER.log_cache(trace, **fields)


def enabled(trace: TraceConfig | None) -> bool:
    return _WRITER.enabled(trace)


def report_observability_failure(message: str, exc: BaseException) -> None:
    _report_observability_failure(message, exc)


def _report_observability_failure(message: str, exc: BaseException) -> None:
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"[dflash] {message}: {exc}\n"
    )
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except OSError:
        return
