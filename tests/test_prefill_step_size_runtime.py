# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from dflash_mlx.cache.codecs import PrefixSnapshotBuilder
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.manager import RuntimeCacheManager
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.cache.snapshot_service import SnapshotService
from dflash_mlx.cache.store import PrefixSnapshotStore
from dflash_mlx.engine.events import (
    CycleCompleteEvent,
    MemoryWaterfallEvent,
    PrefillCompleteEvent,
    PrefillProgressEvent,
    SnapshotPublishedEvent,
    SummaryEvent,
    TokenEvent,
)
from dflash_mlx.engine import spec_epoch
from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
from dflash_mlx.runtime.context import (
    build_runtime_context,
    runtime_config_from_profile,
)


class _FakeTargetOps:
    def __init__(self) -> None:
        self.forward_lengths: list[int] = []
        self.logits_last_only_flags: list[bool] = []
        self.cleanup_calls = 0

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
        logits_last_only=False,
    ):
        del cache, capture_layer_ids
        batch, seq_len = input_ids.shape
        self.forward_lengths.append(int(seq_len))
        self.logits_last_only_flags.append(bool(logits_last_only))
        logits_len = 1 if logits_last_only else seq_len
        logits = mx.zeros((batch, logits_len, 8), dtype=mx.float32)
        hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
        return logits, hidden

    def extract_context_feature(self, hidden_states, target_layer_id_list):
        layer_id = int(target_layer_id_list[0]) + 1
        return hidden_states[layer_id]

    def cleanup_generation_caches(self, *_args) -> None:
        self.cleanup_calls += 1

    def arm_rollback(self, *_args, **_kwargs) -> None:
        return None

    def verify_block(
        self,
        *,
        target_model,
        verify_ids,
        target_cache,
        capture_layer_ids,
    ):
        del target_model, target_cache, capture_layer_ids
        batch, seq_len = verify_ids.shape
        logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
        hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
        return logits, hidden

    def restore_after_acceptance(self, *_args, **_kwargs) -> int:
        return 0


class _FakeDraftBackend:
    def make_cache(self, **_kwargs):
        return []

    def draft_greedy(self, **_kwargs):
        block_len = int(_kwargs["block_len"])
        return mx.zeros((max(0, block_len - 1),), dtype=mx.uint32)


def _runtime_context(
    diagnostics_config: DiagnosticsConfig | None = None,
    verify_mode: str | None = None,
):
    return build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
            verify_mode=verify_mode,
        ),
        diagnostics_config=diagnostics_config,
    )


def _draft_model(*, block_size: int = 4):
    return SimpleNamespace(
        target_layer_ids=[0],
        block_size=block_size,
        mask_token_id=0,
        project_target_hidden=lambda value: value,
    )


class _TokenizingTokenizer:
    def __init__(self):
        self.encode_calls: list[str] = []
        self.template_calls: list[tuple[list[dict[str, str]], bool, bool]] = []

    def encode(self, prompt):
        self.encode_calls.append(prompt)
        return [11, 12]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.template_calls.append((messages, tokenize, add_generation_prompt))
        return [21, 22, 23]


def _prefix_key() -> DFlashPrefixKey:
    return DFlashPrefixKey(
        target_model_id="target",
        draft_model_id="draft",
        capture_layer_ids=(0,),
        draft_sink_size=64,
        draft_window_size=1024,
        target_fa_window=0,
    )


def _snapshot_builder(draft_model):
    return PrefixSnapshotBuilder(
        key=_prefix_key(),
        draft_model=draft_model,
        draft_sink_size=64,
        draft_window_size=1024,
    )


def _snapshot_service(draft_model, *, cache=None, builder=None):
    cache = cache if cache is not None else DFlashPrefixCache(max_entries=4)
    builder = builder if builder is not None else _snapshot_builder(draft_model)
    return SnapshotService(
        cache_manager=RuntimeCacheManager(PrefixSnapshotStore(l1=cache)),
        builder=builder,
    )


def test_session_request_derives_token_views():
    token_ids = [1, 2]

    request = spec_epoch._SessionRequest.from_tokens(
        prompt_tokens=token_ids,
        max_new_tokens=3,
        block_tokens=None,
        stop_token_ids=[7],
        suppress_token_ids=None,
        prefix_snapshot=None,
        snapshot_service=None,
        stable_prefix_len=None,
        prefix_cache_active=False,
    )
    token_ids.append(99)

    assert request.prompt_tokens == (1, 2)
    assert request.prompt_len == 2
    assert tuple(int(x) for x in request.prompt_array.reshape(-1).tolist()) == (1, 2)
    assert request.stop_token_array is not None
    assert tuple(int(x) for x in request.stop_token_array.tolist()) == (7,)


def test_dflash_stream_tokenizes_plain_prompt_without_override():
    target_ops = _FakeTargetOps()
    tokenizer = _TokenizingTokenizer()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=tokenizer,
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="plain",
            max_new_tokens=0,
            use_chat_template=False,
            runtime_context=_runtime_context(),
        )
    )

    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    assert tokenizer.encode_calls == ["plain"]
    assert tokenizer.template_calls == []
    assert prefill_event.prompt_token_count == 2


def test_dflash_stream_tokenizes_chat_template_without_override():
    target_ops = _FakeTargetOps()
    tokenizer = _TokenizingTokenizer()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=tokenizer,
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="chat",
            max_new_tokens=0,
            use_chat_template=True,
            runtime_context=_runtime_context(),
        )
    )

    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    assert tokenizer.encode_calls == []
    assert tokenizer.template_calls == [
        ([{"role": "user", "content": "chat"}], True, True)
    ]
    assert prefill_event.prompt_token_count == 3


def test_dflash_max_ctx_fallback_skips_session_request_materialization(monkeypatch):
    target_ops = _FakeTargetOps()

    class FakeTarget:
        def __call__(self, input_ids, cache):
            del cache
            batch, seq_len = input_ids.shape
            return mx.zeros((batch, seq_len, 8), dtype=mx.float32)

    context = build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            prefix_cache=False,
            prefix_cache_l2=False,
            dflash_max_ctx=2,
        )
    )
    calls = []
    original_from_tokens = spec_epoch._SessionRequest.from_tokens

    def tracked_from_tokens(*args, **kwargs):
        calls.append((args, kwargs))
        return original_from_tokens(*args, **kwargs)

    monkeypatch.setattr(spec_epoch._SessionRequest, "from_tokens", tracked_from_tokens)

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=FakeTarget(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=1,
            prompt_tokens_override=[1, 2],
            runtime_context=context,
        )
    )

    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    summary_event = next(event for event in events if isinstance(event, SummaryEvent))
    assert calls == []
    assert prefill_event.fallback_ar is True
    assert summary_event.fallback_ar is True


def _prefix_snapshot(*, token_ids=(1, 2), fa_states=(), gdn_states=(), last_logits=None):
    return DFlashPrefixSnapshot(
        token_ids=tuple(token_ids),
        fa_states=fa_states,
        gdn_states=gdn_states,
        target_hidden_chunks=(mx.zeros((1, len(token_ids), 2), dtype=mx.float32),),
        target_hidden_chunk_spans=((0, len(token_ids)),),
        target_hidden_total_len=len(token_ids),
        last_logits=last_logits,
        key=_prefix_key(),
    )


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
        project_target_hidden=lambda value: value,
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
    assert target_ops.logits_last_only_flags == [True, True, True, True]
    assert [
        event.tokens_processed
        for event in events
        if isinstance(event, PrefillProgressEvent)
    ] == [4, 8, 9, 10]
    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    assert prefill_event.logical_ctx_tokens == 10
    assert prefill_event.physical_prefill_tokens == 10
    assert prefill_event.prefill_tokens_restored == 0
    assert prefill_event.prefill_tokens_computed == 10
    assert isinstance(events[-1], SummaryEvent)


def test_clear_cache_boundaries_false_skips_mlx_clear_cache(monkeypatch):
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    calls = []
    monkeypatch.setattr(
        spec_epoch.mx,
        "clear_cache",
        lambda: calls.append("clear"),
        raising=False,
    )

    context = build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            clear_cache_boundaries=False,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(6)),
            runtime_context=context,
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))
    assert summary.clear_cache_boundaries is False
    assert calls == []


def test_clear_cache_boundaries_true_clears_safe_boundaries(monkeypatch):
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    calls = []
    monkeypatch.setattr(
        spec_epoch.mx,
        "clear_cache",
        lambda: calls.append("clear"),
        raising=False,
    )

    context = build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            clear_cache_boundaries=True,
            prefix_cache=False,
            prefix_cache_l2=False,
        )
    )

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=0,
            prompt_tokens_override=list(range(6)),
            runtime_context=context,
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))
    assert summary.clear_cache_boundaries is True
    assert calls == ["clear", "clear", "clear", "clear"]


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
        project_target_hidden=lambda value: value,
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
    assert target_ops.logits_last_only_flags == [True]
    prefill_event = next(event for event in events if isinstance(event, PrefillCompleteEvent))
    assert prefill_event.logical_ctx_tokens == 10
    assert prefill_event.physical_prefill_tokens == 4
    assert prefill_event.prefill_tokens_restored == 6
    assert prefill_event.prefill_tokens_computed == 4


def test_warm_exact_hit_skips_prefill_republish():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    prefix_snapshot = _prefix_snapshot(
        token_ids=(1, 2),
        last_logits=mx.zeros((1, 8), dtype=mx.float32),
    )
    cache = DFlashPrefixCache(max_entries=4)
    snapshot_service = _snapshot_service(draft_model, cache=cache)
    context = build_runtime_context(
        runtime_config_from_profile(
            profile="balanced",
            prefill_step_size=4,
            prefix_cache=True,
            prefix_cache_l2=False,
        )
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
            prefix_snapshot=prefix_snapshot,
            snapshot_service=snapshot_service,
            stable_prefix_len=2,
            prefix_cache_active=True,
            runtime_context=context,
        )
    )

    assert not any(
        isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
        for event in events
    )


def test_prefix_snapshot_hydration_failure_raises():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    prefix_snapshot = _prefix_snapshot(
        fa_states=(
            (
                mx.zeros((1, 1, 2, 2), dtype=mx.float32),
                mx.zeros((1, 1, 2, 2), dtype=mx.float32),
                2,
            ),
        ),
        gdn_states=(None,),
        last_logits=mx.zeros((1, 8), dtype=mx.float32),
    )

    with pytest.raises(RuntimeError, match="prefix snapshot hydrate failed for 2 tokens"):
        list(
            spec_epoch.stream_dflash_generate_impl(
                target_model=object(),
                target_ops=target_ops,
                tokenizer=object(),
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="unused",
                max_new_tokens=0,
                prompt_tokens_override=[1, 2, 3],
                prefix_snapshot=prefix_snapshot,
                stable_prefix_len=2,
                runtime_context=_runtime_context(),
            )
        )


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
        project_target_hidden=lambda value: value,
    )
    key = DFlashPrefixKey(
        target_model_id="target",
        draft_model_id="draft",
        capture_layer_ids=(0,),
        draft_sink_size=64,
        draft_window_size=1024,
        target_fa_window=0,
    )
    cache = DFlashPrefixCache(max_entries=4)
    snapshot_service = _snapshot_service(
        draft_model,
        cache=cache,
        builder=PrefixSnapshotBuilder(
            key=key,
            draft_model=draft_model,
            draft_sink_size=64,
            draft_window_size=1024,
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
            prompt_tokens_override=[1, 2],
            snapshot_service=snapshot_service,
            runtime_context=context,
        )
    )

    snapshot_event = next(
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
    )
    matched, snapshot = cache.lookup([1, 2], key)
    assert snapshot_event.admitted is True
    assert snapshot_event.prefix_len == 2
    assert matched == 2
    assert snapshot is not None
    assert snapshot.key == key
    assert snapshot.token_ids == (1, 2)


def test_prefill_snapshot_service_build_failure_propagates():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()

    class BrokenBuilder:
        def build(self, **_kwargs):
            raise RuntimeError("snapshot builder broken")

    with pytest.raises(RuntimeError, match="snapshot builder broken"):
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
                snapshot_service=_snapshot_service(
                    draft_model,
                    builder=BrokenBuilder(),
                ),
                runtime_context=_runtime_context(),
            )
        )


def test_prefill_snapshot_service_requires_logits_for_warm_exact_hit():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    prefix_snapshot = _prefix_snapshot()

    with pytest.raises(ValueError, match="prefill snapshot requires last_logits"):
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
                prefix_snapshot=prefix_snapshot,
                stable_prefix_len=2,
                snapshot_service=_snapshot_service(draft_model),
                runtime_context=_runtime_context(),
            )
        )


def test_warm_exact_hit_without_snapshot_service_requires_logits_before_decode():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    prefix_snapshot = _prefix_snapshot()

    with pytest.raises(
        RuntimeError,
        match="prefill logits unavailable after prefix snapshot restore",
    ):
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
                prefix_snapshot=prefix_snapshot,
                stable_prefix_len=2,
                runtime_context=_runtime_context(),
            )
        )


def test_generation_snapshot_service_build_failure_propagates():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    delegate = _snapshot_builder(draft_model)

    class GenerationFailingBuilder:
        def build(self, **kwargs):
            if kwargs["kind"] == "generation":
                raise RuntimeError("generation snapshot broken")
            return delegate.build(**kwargs)

    failing_builder = GenerationFailingBuilder()
    failing_builder.key = delegate.key

    with pytest.raises(RuntimeError, match="generation snapshot broken"):
        list(
            spec_epoch.stream_dflash_generate_impl(
                target_model=object(),
                target_ops=target_ops,
                tokenizer=object(),
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="unused",
                max_new_tokens=1,
                prompt_tokens_override=[1, 2],
                snapshot_service=_snapshot_service(
                    draft_model,
                    builder=failing_builder,
                ),
                runtime_context=_runtime_context(),
            )
        )


def test_request_state_preserves_decode_summary_and_generation_snapshot():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    cache = DFlashPrefixCache(max_entries=4)
    key = _prefix_key()
    snapshot_service = _snapshot_service(
        draft_model,
        cache=cache,
        builder=PrefixSnapshotBuilder(
            key=key,
            draft_model=draft_model,
            draft_sink_size=64,
            draft_window_size=1024,
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
            max_new_tokens=3,
            prompt_tokens_override=[1, 2],
            snapshot_service=snapshot_service,
            runtime_context=_runtime_context(),
        )
    )

    token_events = [event for event in events if isinstance(event, TokenEvent)]
    summary = next(event for event in events if isinstance(event, SummaryEvent))
    generation_snapshot = next(
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "generation"
    )

    assert [event.token_id for event in token_events] == [0, 0, 0]
    assert summary.generated_token_ids == (0, 0, 0)
    assert summary.generation_tokens == 3
    assert summary.cycles_completed == 1
    assert summary.accepted_from_draft == 2
    assert summary.acceptance_history == (2,)
    assert generation_snapshot.kind == "generation"
    assert generation_snapshot.prefix_len == 5
    matched, snapshot = cache.lookup([1, 2, 0, 0, 0, 99], key)
    assert matched == 5
    assert snapshot is not None
    assert snapshot.kind == "generation"
    assert snapshot.token_ids == (1, 2, 0, 0, 0)


def test_generation_snapshot_skipped_for_truncated_stable_prefix():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    cache = DFlashPrefixCache(max_entries=4)
    key = _prefix_key()
    snapshot_service = _snapshot_service(
        draft_model,
        cache=cache,
        builder=PrefixSnapshotBuilder(
            key=key,
            draft_model=draft_model,
            draft_sink_size=64,
            draft_window_size=1024,
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
            max_new_tokens=3,
            prompt_tokens_override=[1, 2, 3],
            snapshot_service=snapshot_service,
            stable_prefix_len=2,
            runtime_context=_runtime_context(),
        )
    )

    prefill_snapshot = next(
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "prefill"
    )
    generation_snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "generation"
    ]
    matched, snapshot = cache.lookup([1, 2, 3, 0, 0, 0, 99], key)

    assert prefill_snapshot.prefix_len == 2
    assert generation_snapshots == []
    assert matched == 2
    assert snapshot is not None
    assert snapshot.kind == "prefill"


def test_generation_snapshot_skipped_when_request_policy_disallows_it():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()
    draft_model = _draft_model()
    cache = DFlashPrefixCache(max_entries=4)
    key = _prefix_key()
    snapshot_service = _snapshot_service(
        draft_model,
        cache=cache,
        builder=PrefixSnapshotBuilder(
            key=key,
            draft_model=draft_model,
            draft_sink_size=64,
            draft_window_size=1024,
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
            max_new_tokens=3,
            prompt_tokens_override=[1, 2],
            snapshot_service=snapshot_service,
            publish_generation_snapshot=False,
            runtime_context=_runtime_context(),
        )
    )

    generation_snapshots = [
        event
        for event in events
        if isinstance(event, SnapshotPublishedEvent) and event.kind == "generation"
    ]
    matched, snapshot = cache.lookup([1, 2, 0, 0, 0, 99], key)

    assert generation_snapshots == []
    assert matched == 2
    assert snapshot is not None
    assert snapshot.kind == "prefill"


def test_request_state_preserves_async_prefetched_draft_reuse():
    target_ops = _FakeTargetOps()
    draft_model = _draft_model()

    class _RecordingDraftBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[int, bool]] = []

        def make_cache(self, **_kwargs):
            return []

        def draft_greedy(self, **kwargs):
            block_len = int(kwargs["block_len"])
            self.calls.append((block_len, bool(kwargs["async_launch"])))
            return mx.zeros((max(0, block_len - 1),), dtype=mx.uint32)

    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls == [(4, True), (2, True)]
    assert summary.generated_token_ids == (0, 0, 0, 0, 0, 0)
    assert summary.generation_tokens == 6
    assert summary.cycles_completed == 2
    assert summary.accepted_from_draft == 4
    assert summary.acceptance_history == (3, 1)


def test_profile_cycle_events_disable_async_prefetch_and_match_summary():
    target_ops = _FakeTargetOps()
    draft_model = _draft_model()

    class _RecordingDraftBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[int, bool]] = []

        def make_cache(self, **_kwargs):
            return []

        def draft_greedy(self, **kwargs):
            block_len = int(kwargs["block_len"])
            self.calls.append((block_len, bool(kwargs["async_launch"])))
            return mx.zeros((max(0, block_len - 1),), dtype=mx.uint32)

    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(
                diagnostics_config=DiagnosticsConfig(
                    trace=TraceConfig(cycle_events=True),
                ),
            ),
        )
    )

    cycle_events = [event for event in events if isinstance(event, CycleCompleteEvent)]
    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls == [(4, False), (2, False)]
    assert [(event.cycle, event.block_len, event.commit_count, event.acceptance_len) for event in cycle_events] == [
        (1, 4, 4, 3),
        (2, 2, 2, 1),
    ]
    assert summary.cycle_profile_us == tuple(cycle_events)
    assert summary.cycle_profile_totals_us is not None
    assert set(summary.cycle_profile_totals_us) == {
        "draft",
        "verify",
        "acceptance",
        "hidden_extraction",
        "rollback",
        "other",
        "cycle_total",
    }


def test_adaptive_verify_mode_drops_and_recovers_block_len():
    draft_model = _draft_model(block_size=8)

    class _PatternTargetOps(_FakeTargetOps):
        def __init__(self) -> None:
            super().__init__()
            self.verify_lengths: list[int] = []
            self._cycle = 0

        def verify_block(
            self,
            *,
            target_model,
            verify_ids,
            target_cache,
            capture_layer_ids,
        ):
            del target_model, target_cache, capture_layer_ids
            batch, seq_len = verify_ids.shape
            self.verify_lengths.append(int(seq_len))
            self._cycle += 1
            if self._cycle <= 12:
                pattern = [1] + [0] * (seq_len - 1)
            else:
                pattern = [0] * seq_len
            logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
            for pos, token_id in enumerate(pattern[:seq_len]):
                logits[:, pos, int(token_id)] = 1.0
            hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
            return logits, hidden

    class _RecordingDraftBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[int, bool]] = []

        def make_cache(self, **_kwargs):
            return []

        def draft_greedy(self, **kwargs):
            block_len = int(kwargs["block_len"])
            self.calls.append((block_len, bool(kwargs["async_launch"])))
            return mx.zeros((max(0, block_len - 1),), dtype=mx.uint32)

    target_ops = _PatternTargetOps()
    draft_backend = _RecordingDraftBackend()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=draft_backend,
            prompt="unused",
            max_new_tokens=48,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(verify_mode="adaptive"),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert draft_backend.calls[:17] == [(8, True)] * 12 + [(4, True)] * 4 + [(8, True)]
    assert target_ops.verify_lengths[:17] == [8] * 12 + [4] * 4 + [8]
    assert summary.block_tokens == 8
    assert summary.verify_len_cap == 8
    assert summary.acceptance_history[:17] == (0,) * 12 + (3,) * 4 + (7,)
    assert summary.adaptive_block_reductions == 1
    assert summary.adaptive_block_cycles == 4
    assert summary.adaptive_block_min == 4


def test_adaptive_verify_mode_ignores_mixed_commit_windows():
    draft_model = _draft_model(block_size=8)

    class _MixedTargetOps(_FakeTargetOps):
        def __init__(self) -> None:
            super().__init__()
            self.verify_lengths: list[int] = []
            self._cycle = 0

        def verify_block(
            self,
            *,
            target_model,
            verify_ids,
            target_cache,
            capture_layer_ids,
        ):
            del target_model, target_cache, capture_layer_ids
            batch, seq_len = verify_ids.shape
            self.verify_lengths.append(int(seq_len))
            self._cycle += 1
            accepted = 5 if self._cycle % 4 == 0 else 1
            pattern = [0] + [0] * min(accepted, seq_len - 1)
            pattern += [1] * max(0, seq_len - len(pattern))
            logits = mx.zeros((batch, seq_len, 8), dtype=mx.float32)
            for pos, token_id in enumerate(pattern[:seq_len]):
                logits[:, pos, int(token_id)] = 1.0
            hidden = {1: mx.zeros((batch, seq_len, 2), dtype=mx.float32)}
            return logits, hidden

    class _RecordingDraftBackend:
        def make_cache(self, **_kwargs):
            return []

        def draft_greedy(self, **kwargs):
            block_len = int(kwargs["block_len"])
            return mx.zeros((max(0, block_len - 1),), dtype=mx.uint32)

    target_ops = _MixedTargetOps()

    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=target_ops,
            tokenizer=object(),
            draft_model=draft_model,
            draft_backend=_RecordingDraftBackend(),
            prompt="unused",
            max_new_tokens=64,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(verify_mode="adaptive"),
        )
    )

    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert 4 not in target_ops.verify_lengths
    assert summary.adaptive_block_reductions == 0
    assert summary.adaptive_block_cycles == 0
    assert summary.adaptive_block_min is None


def test_memory_waterfall_decode_cycle_order_is_stable():
    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_FakeTargetOps(),
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            runtime_context=_runtime_context(
                diagnostics_config=DiagnosticsConfig(memory_waterfall=True),
            ),
        )
    )

    phases = [
        event.fields["memory_phase"]
        for event in events
        if isinstance(event, MemoryWaterfallEvent)
    ]
    decode_phases = [
        phase
        for phase in phases
        if phase in {"before_verify_cycle", "after_verify_cycle", "after_rollback"}
    ]

    assert decode_phases == [
        "before_verify_cycle",
        "after_verify_cycle",
        "after_rollback",
        "before_verify_cycle",
        "after_verify_cycle",
        "after_rollback",
    ]


def test_stop_token_breaks_after_current_committed_segment():
    events = list(
        spec_epoch.stream_dflash_generate_impl(
            target_model=object(),
            target_ops=_FakeTargetOps(),
            tokenizer=object(),
            draft_model=_draft_model(),
            draft_backend=_FakeDraftBackend(),
            prompt="unused",
            max_new_tokens=6,
            prompt_tokens_override=[1, 2],
            stop_token_ids=[0],
            runtime_context=_runtime_context(),
        )
    )

    token_events = [event for event in events if isinstance(event, TokenEvent)]
    summary = next(event for event in events if isinstance(event, SummaryEvent))

    assert [event.generated_tokens for event in token_events] == [1, 2, 3, 4]
    assert summary.generated_token_ids == (0, 0, 0, 0)
    assert summary.generation_tokens == 4
    assert summary.cycles_completed == 1
    assert summary.acceptance_history == (3,)


def test_stream_close_cleans_session_caches():
    target_ops = _FakeTargetOps()
    draft_backend = _FakeDraftBackend()

    stream = spec_epoch.stream_dflash_generate_impl(
        target_model=object(),
        target_ops=target_ops,
        tokenizer=object(),
        draft_model=_draft_model(),
        draft_backend=draft_backend,
        prompt="unused",
        max_new_tokens=6,
        prompt_tokens_override=[1, 2],
        runtime_context=_runtime_context(),
    )

    assert target_ops.cleanup_calls == 0
    first_event = next(stream)
    assert isinstance(first_event, PrefillProgressEvent)
    assert target_ops.cleanup_calls == 0

    stream.close()
    assert target_ops.cleanup_calls == 1


def test_prefill_snapshot_publication_skips_without_snapshot_service():
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
        project_target_hidden=lambda value: value,
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

    assert not any(isinstance(event, SnapshotPublishedEvent) for event in events)


def test_active_prefix_cache_requires_snapshot_service():
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
        project_target_hidden=lambda value: value,
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
        assert "snapshot_service is required" in str(exc)
        assert target_ops.cleanup_calls == 1
    else:
        raise AssertionError("active prefix cache without snapshot service must fail")
