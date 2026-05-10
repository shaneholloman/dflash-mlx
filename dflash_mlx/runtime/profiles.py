# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dflash_mlx.runtime.config import GiB as _GiB
from dflash_mlx.runtime.config import PROFILES as _PROFILES
from dflash_mlx.runtime.config import RuntimeProfile as _RuntimeProfile

__all__ = ["format_profiles"]


def format_profiles() -> str:
    rows = [
        "profile      prefill  draft_window  verify_cap  prefix  L1           clear     max_snapshot  L2          verify  note"
    ]
    notes = {
        "balanced": "recommended default for normal agentic coding",
        "fast": "more throughput, higher memory",
        "low-memory": "lower memory pressure, slower prefill",
        "long-session": "prefix/L2 oriented for multi-turn revisits",
    }
    for profile in _PROFILES.values():
        rows.append(
            " ".join(
                [
                    f"{profile.name:<12}",
                    f"{profile.prefill_step_size:<8}",
                    f"{profile.draft_sink_size}+{profile.draft_window_size:<8}",
                    f"{_verify_cap(profile.verify_len_cap):<10}",
                    f"{_on_off(profile.prefix_cache):<7}",
                    (
                        f"{profile.prefix_cache_max_entries}x"
                        f"{_format_gib(profile.prefix_cache_max_bytes):<10}"
                    ),
                    f"{_clear_policy(profile.clear_cache_boundaries):<9}",
                    f"{profile.max_snapshot_tokens:<13}",
                    f"{_l2_summary(profile):<11}",
                    f"{profile.verify_mode:<7}",
                    notes[profile.name],
                ]
            )
        )
    return "\n".join(rows)


def _on_off(value: bool) -> str:
    return "on" if value else "off"


def _clear_policy(enabled: bool) -> str:
    return "boundary" if enabled else "off"


def _format_gib(value: int) -> str:
    if value % _GiB == 0:
        return f"{value // _GiB}GiB"
    return f"{value / _GiB:.1f}GiB"


def _l2_summary(profile: _RuntimeProfile) -> str:
    if not profile.prefix_cache_l2:
        return "off"
    return f"on/{_format_gib(profile.prefix_cache_l2_max_bytes)}"


def _verify_cap(value: int) -> str:
    return "block" if int(value) == 0 else str(int(value))
