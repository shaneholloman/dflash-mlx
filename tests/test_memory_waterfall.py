# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import mlx.core as mx
import pytest

from dflash_mlx.engine import memory_waterfall
from dflash_mlx.observability import memory as memory_obs
from tools.benchmarks.analyze_trace import (
    classify_verdict,
    compute_phase_deltas,
    load_memory_events,
    render_summary,
    summarize_events,
)
from dflash_mlx.engine.memory_waterfall import (
    collect_memory_waterfall,
    draft_cache_bytes,
    prefix_cache_memory_bytes,
    target_cache_bytes,
)

class FakeKVCache:
    def __init__(self):
        self.keys = mx.zeros((1, 2, 3, 4), dtype=mx.float16)
        self.values = mx.zeros((1, 2, 3, 4), dtype=mx.float16)

class FakeStateKVCache:
    def __init__(self):
        self._state = (
            mx.zeros((1, 2, 3, 4), dtype=mx.float16),
            mx.zeros((1, 2, 3, 4), dtype=mx.float16),
        )

    @property
    def state(self):
        return self._state

class BrokenStateKVCache:
    @property
    def state(self):
        raise RuntimeError("broken state contract")

class EmptyPropertyStateKVCache:
    keys = None
    values = None

    @property
    def state(self):
        raise AttributeError("empty cache has no materialized state")

class StateOnlyAttributeErrorCache:
    @property
    def state(self):
        raise AttributeError("broken state accessor")

class FakeRecurrentRollbackCache:
    def __init__(self):
        self.cache = [mx.zeros((1, 2, 3), dtype=mx.float16)]
        self._tape = [mx.zeros((1, 2, 3), dtype=mx.float16)]
        self._tape_k = None
        self._tape_g = None
        self._tape_qkv = mx.zeros((1, 2), dtype=mx.float16)
        self._snapshot = None

def test_target_cache_bytes_splits_fa_gdn_and_tape():
    out = target_cache_bytes([FakeKVCache(), FakeRecurrentRollbackCache()])
    assert out["target_fa_kv_bytes"] == 96
    assert out["target_gdn_state_bytes"] == 12
    assert out["rollback_tape_bytes"] == 16

def test_draft_cache_bytes_counts_keys_values_and_state():
    out = draft_cache_bytes([FakeKVCache(), FakeStateKVCache()])
    assert out["draft_kv_bytes"] == 192

def test_draft_cache_bytes_fails_fast_on_broken_state_property():
    with pytest.raises(RuntimeError, match="broken state contract"):
        draft_cache_bytes([BrokenStateKVCache()])

def test_draft_cache_bytes_treats_empty_property_state_as_zero():
    out = draft_cache_bytes([EmptyPropertyStateKVCache()])

    assert out["draft_kv_bytes"] == 0

def test_draft_cache_bytes_fails_fast_on_state_only_attribute_error():
    with pytest.raises(AttributeError, match="broken state accessor"):
        draft_cache_bytes([StateOnlyAttributeErrorCache()])

def test_prefix_cache_memory_bytes_breakdown_and_stats():
    out = prefix_cache_memory_bytes(
        {
            "l1_snapshot_bytes": 100,
            "l1_snapshot_fa_kv_bytes": 10,
            "l1_snapshot_gdn_state_bytes": 20,
            "l1_snapshot_target_hidden_bytes": 30,
            "l1_snapshot_last_logits_bytes": 40,
            "l2_disk_bytes": 700,
            "prefix_prunes": 2,
            "cross_kind_prunes": 3,
            "byte_budget_evictions": 4,
            "l2_hits": 5,
            "l2_misses": 6,
            "l2_writes": 8,
        }
    )
    assert out["l1_snapshot_bytes"] == 100
    assert out["l1_snapshot_fa_kv_bytes"] == 10
    assert out["l1_snapshot_gdn_state_bytes"] == 20
    assert out["l1_snapshot_target_hidden_bytes"] == 30
    assert out["l1_snapshot_last_logits_bytes"] == 40
    assert out["l2_disk_bytes"] == 700
    assert out["prefix_prunes"] == 2
    assert out["cross_kind_prunes"] == 3
    assert out["byte_budget_evictions"] == 4
    assert out["l2_hits"] == 5
    assert out["l2_writes"] == 8

def test_collect_memory_waterfall_handles_missing_fields():
    out = collect_memory_waterfall(
        phase="test",
        target_cache=None,
        draft_cache=None,
        target_hidden=None,
        gen_hidden_chunks=None,
        prefix_cache_memory=None,
    )
    assert out["memory_phase"] == "test"
    assert out["target_fa_kv_gb"] == 0.0
    assert out["target_gdn_state_gb"] == 0.0
    assert out["rollback_tape_gb"] == 0.0
    assert out["draft_kv_gb"] == 0.0
    assert out["l1_snapshot_gb"] == 0.0
    assert out["l2_disk_gb"] == 0.0

def test_collect_memory_waterfall_preserves_unknown_process_memory(monkeypatch):
    monkeypatch.setattr(
        memory_waterfall,
        "process_memory_bytes",
        lambda: {
            "rss_bytes": None,
            "system_wired_bytes": None,
            "mlx_active_bytes": None,
            "mlx_cache_bytes": None,
            "mlx_peak_bytes": None,
            "untracked_bytes": None,
        },
    )

    out = collect_memory_waterfall(phase="test")

    assert out["rss_bytes"] is None
    assert out["rss_gb"] is None
    assert out["mlx_active_gb"] is None
    assert out["untracked_gb"] is None
    assert out["target_fa_kv_gb"] == 0.0

def test_collect_memory_waterfall_deduplicates_hidden_buckets():
    hidden = mx.zeros((1, 2, 3), dtype=mx.float16)
    out = collect_memory_waterfall(
        phase="test",
        target_hidden=hidden,
        gen_hidden_chunks=[hidden],
    )
    assert out["target_hidden_active_bytes"] == hidden.nbytes
    assert out["gen_hidden_chunks_bytes"] == 0

def test_current_rss_bytes_returns_none_when_current_probe_is_unavailable(monkeypatch):
    def missing_proc(*_args, **_kwargs):
        raise OSError("no proc")

    monkeypatch.setattr(memory_obs.sys, "platform", "linux")
    monkeypatch.setattr(memory_obs.builtins, "open", missing_proc)
    monkeypatch.setattr(
        memory_obs.resource,
        "getrusage",
        lambda _kind: SimpleNamespace(ru_maxrss=1234),
    )
    monkeypatch.setattr(memory_obs.resource, "getpagesize", lambda: 4096)

    assert memory_obs.current_rss_bytes() is None
    assert memory_obs.rss_peak_bytes() == 1234 * 1024


def test_current_rss_bytes_propagates_programmer_errors(monkeypatch):
    def broken_open(*_args, **_kwargs):
        raise TypeError("broken open contract")

    monkeypatch.setattr(memory_obs.sys, "platform", "linux")
    monkeypatch.setattr(memory_obs.builtins, "open", broken_open)

    with pytest.raises(TypeError, match="broken open contract"):
        memory_obs.current_rss_bytes()


def test_system_wired_bytes_falls_back_on_vm_stat_failure(monkeypatch):
    def fail_vm_stat(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["vm_stat"])

    monkeypatch.setattr(memory_obs.sys, "platform", "darwin")
    monkeypatch.setattr(memory_obs.subprocess, "check_output", fail_vm_stat)

    assert memory_obs.system_wired_bytes() is None


def test_system_wired_bytes_parses_vm_stat(monkeypatch):
    out = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages wired down: 2.\n"
    )
    monkeypatch.setattr(memory_obs.sys, "platform", "darwin")
    monkeypatch.setattr(memory_obs.subprocess, "check_output", lambda *_args, **_kwargs: out)

    assert memory_obs.system_wired_bytes() == 32768


def test_system_wired_bytes_propagates_programmer_errors(monkeypatch):
    def broken_check_output(*_args, **_kwargs):
        raise TypeError("broken helper contract")

    monkeypatch.setattr(memory_obs.sys, "platform", "darwin")
    monkeypatch.setattr(memory_obs.subprocess, "check_output", broken_check_output)

    with pytest.raises(TypeError, match="broken helper contract"):
        memory_obs.system_wired_bytes()


def test_darwin_task_vm_info_bytes_parses_footprint_ledgers(monkeypatch):
    class FakeFunc:
        def __init__(self, impl):
            self.impl = impl

        def __call__(self, *args):
            return self.impl(*args)

    def fake_task_info(_task, flavor, info_ptr, _count_ptr):
        assert flavor == 22
        info = info_ptr._obj
        info.phys_footprint = 5_000_000_000
        info.device = 3_000_000_000
        info.internal = 1_000_000_000
        info.compressed = 500_000_000
        info.resident_size = 4_000_000_000
        return 0

    fake_libc = SimpleNamespace(
        task_info=FakeFunc(fake_task_info),
        mach_task_self=FakeFunc(lambda: 1),
    )
    monkeypatch.setattr(memory_obs.sys, "platform", "darwin")
    monkeypatch.setattr(memory_obs.ctypes, "CDLL", lambda *_args, **_kwargs: fake_libc)

    payload = memory_obs.darwin_task_vm_info_bytes()

    assert payload["phys_footprint"] == 5_000_000_000
    assert payload["device"] == 3_000_000_000
    assert payload["internal"] == 1_000_000_000
    assert payload["compressed"] == 500_000_000


def test_mlx_memory_bytes_fallbacks_and_contract_errors(monkeypatch):
    monkeypatch.setattr(memory_obs.mx, "missing_memory_counter", None, raising=False)
    assert memory_obs.mlx_memory_bytes("missing_memory_counter") is None

    monkeypatch.setattr(
        memory_obs.mx,
        "runtime_failing_memory_counter",
        lambda: (_ for _ in ()).throw(RuntimeError("mlx unavailable")),
        raising=False,
    )
    assert memory_obs.mlx_memory_bytes("runtime_failing_memory_counter") is None

    monkeypatch.setattr(memory_obs.mx, "bad_memory_counter", lambda: object(), raising=False)
    with pytest.raises(TypeError):
        memory_obs.mlx_memory_bytes("bad_memory_counter")

def test_analyzer_loads_jsonl_and_classifies_target_hidden(tmp_path):
    path = tmp_path / "cycle_events.jsonl"
    rows = [
        {"memory_phase": "before_verify_cycle", "target_hidden_active_gb": 2.0, "target_fa_kv_gb": 1.0},
        {"memory_phase": "after_cleanup", "mlx_cache_gb": 0.5},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    events = load_memory_events(path)
    summary = summarize_events(events)
    assert len(events) == 2
    assert summary["peaks_gb"]["target_hidden_active"] == 2.0
    assert summary["verdict"] == "optimize target_hidden first"

def test_analyzer_classifies_l1_fa_and_allocator():
    assert classify_verdict({"l1_snapshot": 3.0, "target_fa_kv": 1.0}) == "optimize L1/L2 pressure"
    assert classify_verdict({"target_fa_kv": 3.0, "target_hidden_active": 1.0}) == "paged/quantized KV worth investigating"
    assert classify_verdict({"mlx_cache": 3.0, "target_fa_kv": 1.0}) == "allocator/scratch/cache policy first"

def test_analyzer_delta_reports_positive_phase_growth():
    events = [
        {"memory_phase": "after_target_cache_create", "mlx_active_gb": 2.0},
        {"memory_phase": "after_prefill_chunk", "mlx_active_gb": 14.0, "target_fa_kv_gb": 1.0},
        {"memory_phase": "after_cleanup", "mlx_active_gb": 10.0, "mlx_cache_gb": 3.0},
    ]
    deltas = compute_phase_deltas(events)
    assert deltas["by_phase"]["after_prefill_chunk"]["mlx_active"] == 12.0
    assert deltas["by_phase"]["after_cleanup"]["mlx_cache"] == 3.0
    rendered = render_summary(summarize_events(events), include_delta=True)
    assert "after_target_cache_create -> after_prefill_chunk: mlx_active +12.000GB" in rendered


def test_memory_waterfall_summary_reports_unknown_buckets():
    rendered = memory_waterfall.format_memory_waterfall_summary(
        {
            "mlx_active_gb": None,
            "mlx_cache_gb": None,
            "untracked_gb": None,
            "target_fa_kv_gb": 1.0,
            "draft_kv_gb": 0.25,
            "target_hidden_active_gb": 0.0,
        }
    )

    assert "target_fa_kv:1.00GB" in rendered
    assert "draft_kv:0.25GB" in rendered
    assert "mlx_active:0.00GB" not in rendered
    assert "unknown=mlx_active,mlx_cache,untracked" in rendered
