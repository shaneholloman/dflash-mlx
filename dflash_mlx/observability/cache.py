# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any

from dflash_mlx.diagnostics import TraceConfig
from dflash_mlx.observability.writer import log_cache


def record_cache_event(trace: TraceConfig | None, **fields: Any) -> None:
    log_cache(trace, **fields)


def live_prefix_cache_payload(
    *,
    stats: dict[str, Any] | None,
    config: dict[str, Any],
    last_request: dict[str, Any] | None,
    last_prefix: dict[str, Any],
) -> dict[str, Any]:
    if stats is None:
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


def live_prefix_cache_totals(stats: dict[str, Any] | None) -> dict[str, int]:
    if stats is None:
        return {}
    return {
        "cache_hits": int(stats.get("exact_hits", 0) + stats.get("prefix_hits", 0)),
        "cache_misses": int(stats.get("misses", 0)),
    }


def _int_or_none(value: Any | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
