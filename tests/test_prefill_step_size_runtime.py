# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from dflash_mlx.cache.codecs import PrefixSnapshotBuilder
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.engine import spec_epoch
from dflash_mlx.runtime_context import (
    build_runtime_context,
    runtime_config_from_profile,
)


class _FakeTargetOps:
    def __init__(self) -> None:
        self.forward_lengths: list[int] = []

    def capabilities_for(self, _target_model):
        return SimpleNamespace(supports_prefix_snapshot=True)

    def make_cache(self, *_args, **_kwargs):
        return []

    def forward_with_hidden_capture(
        self,
        _target_model,
        *,
        input_ids,
        cache,
        capture_layer_ids,
    ):
        del cache, capture_layer_ids
        batch, seq_len = input_ids.shape
        self.forward_lengths.append(int(seq_len))
        logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
        hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
        return logits, hidden

    def extract_context_feature(self, hidden_states, target_layer_id_list):
        layer_id = int(target_layer_id_list[0]) + 1
        return hidden_states[layer_id]

    def cleanup_generation_caches(self, *_args) -> None:
        return None


class _FakeDraftBackend:
    def make_cache(self, **_kwargs):
        return []


def test_yield_pause_tracker_records_enabled_pause(monkeypatch):
    ticks = iter([10, 17, 100, 140])
    monkeypatch.setattr(spec_epoch.time, "perf_counter_ns", lambda: next(ticks))

    tracker = spec_epoch._YieldPauseTracker(enabled=True)

    mark = tracker.mark()
    tracker.done(mark)
    assert tracker.pause_ns == 7

    mark = tracker.mark()
    tracker.done(mark)
    assert tracker.pause_ns == 47


def test_yield_pause_tracker_disabled_noops(monkeypatch):
    calls = []
    monkeypatch.setattr(
        spec_epoch.time,
        "perf_counter_ns",
        lambda: calls.append("called") or 1,
    )

    tracker = spec_epoch._YieldPauseTracker(enabled=False)

    assert tracker.mark() == 0
    tracker.done(10)
    assert tracker.pause_ns == 0
    assert calls == []


def test_runtime_prefill_chunks_use_configured_step_size():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(10)),
            runtime_context=context,
        )
    )

    assert target_ops.forward_lengths == [4, 4, 1, 1]
    assert [
        event["tokens_processed"]
        for event in events
        if event.get("event") == "prefill_progress"
    ] == [4, 8, 9, 10]
    prefill_event = next(event for event in events if event.get("event") == "prefill")
    assert prefill_event["logical_ctx_tokens"] == 10
    assert prefill_event["physical_prefill_tokens"] == 10
    assert prefill_event["prefill_tokens_restored"] == 0
    assert prefill_event["prefill_tokens_computed"] == 10
    assert events[-1]["event"] == "summary"


def test_runtime_prefill_accounting_reports_warm_restore():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
    )
    prefix_snapshot = DFlashPrefixSnapshot(
        token_ids=tuple(range(6)),
        fa_states=(),
        gdn_states=(),
        target_hidden_chunks=(mx.zeros((1, 6, 2), dtype=mx.float32),),
        target_hidden_chunk_spans=((0, 6),),
        target_hidden_total_len=6,
        last_logits=mx.zeros((1, 8), dtype=mx.float32),
        key=DFlashPrefixKey(
            target_model_id="target",
            draft_model_id="draft",
            capture_layer_ids=(0,),
            draft_sink_size=64,
            draft_window_size=1024,
            target_fa_window=0,
        ),
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(10)),
            prefix_snapshot=prefix_snapshot,
            stable_prefix_len=6,
            runtime_context=context,
        )
    )

    assert target_ops.forward_lengths == [4]
    prefill_event = next(event for event in events if event.get("event") == "prefill")
    assert prefill_event["logical_ctx_tokens"] == 10
    assert prefill_event["physical_prefill_tokens"] == 4
    assert prefill_event["prefill_tokens_restored"] == 6
    assert prefill_event["prefill_tokens_computed"] == 4


def test_prefill_snapshot_event_carries_snapshot_not_live_arrays():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
    )
    key = DFlashPrefixKey(
        target_model_id="target",
        draft_model_id="draft",
        capture_layer_ids=(0,),
        draft_sink_size=64,
        draft_window_size=1024,
        target_fa_window=0,
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=[1, 2],
            prefix_snapshot_builder=PrefixSnapshotBuilder(
                key=key,
                draft_model=draft_model,
                draft_sink_size=64,
                draft_window_size=1024,
            ),
            runtime_context=context,
        )
    )

    snapshot_event = next(
        event for event in events if event.get("event") == "prefill_snapshot_ready"
    )
    assert snapshot_event["snapshot"].key == key
    assert snapshot_event["snapshot"].token_ids == (1, 2)
    assert "target_cache" not in snapshot_event
    assert "target_hidden" not in snapshot_event
    assert "last_logits" not in snapshot_event


def test_prefill_snapshot_event_requires_builder():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=[1, 2],
            runtime_context=context,
        )
    )

    assert "prefill_snapshot_ready" not in {
        event.get("event") for event in events
    }


def test_active_prefix_cache_requires_snapshot_builder():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    context = build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )
    draft_model = SimpleNamespace(
        target_layer_ids=[0],
        block_size=4,
        mask_token_id=0,
    )

    try:
        list(
            spec_epoch.stream_dflash_generate_impl(
                target_model=object(),
                target_ops=target_ops,
                tokenizer=object(),
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="unused",
                max_new_tokens=0,
                prompt_tokens_override=[1, 2],
                prefix_cache_active=True,
                runtime_context=context,
            )
        )
    except ValueError as exc:
        assert "prefix_snapshot_builder is required" in str(exc)
    else:
        raise AssertionError("active prefix cache without snapshot builder must fail")
