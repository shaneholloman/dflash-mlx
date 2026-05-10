# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

from dflash_mlx.engine.events import (
    PrefillCompleteEvent,
    SnapshotPublishedEvent,
    SummaryEvent,
)
from tools.benchmarks import _prefix_cache_multiturn as multiturn
from tools.benchmarks import _prefix_l2_long_session as l2_long_session


class ClosableEvents:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


def _runtime_context():
    return SimpleNamespace(runtime=SimpleNamespace(draft_sink_size=64, draft_window_size=1024))


def _prefill_complete() -> PrefillCompleteEvent:
    return PrefillCompleteEvent(
        prefill_us=100.0,
        prompt_token_count=3,
        logical_ctx_tokens=3,
        physical_prefill_tokens=3,
        prefill_tokens_restored=0,
        prefill_tokens_computed=3,
    )


def test_multiturn_helper_reads_snapshot_publication_metadata(monkeypatch):
    events = ClosableEvents(
        [
            _prefill_complete(),
            SnapshotPublishedEvent(
                kind="prefill",
                snapshot_boundary=3,
                prefix_len=3,
                insert_ms=0.2,
                admitted=True,
                from_snapshot=True,
                snap_prefix_len=2,
            ),
            SummaryEvent(
                elapsed_us=100.0,
                prompt_token_count=3,
                generated_token_ids=(),
                generation_tokens=0,
                accepted_from_draft=0,
                acceptance_ratio=0.0,
                cycles_completed=0,
                phase_timings_us={},
            ),
        ]
    )
    monkeypatch.setattr(multiturn, "stream_dflash_generate", lambda **_kwargs: events)

    result = multiturn._run_one_turn(
        target_model=object(),
        target_ops=object(),
        tokenizer=object(),
        draft_model=SimpleNamespace(target_layer_ids=()),
        draft_backend=object(),
        prompt_tokens=[1, 2, 3],
        max_tokens=4,
        runtime_context=_runtime_context(),
    )

    assert events.closed is True
    assert result["inserted"] is True
    assert result["from_snapshot"] is True
    assert result["snap_prefix_len"] == 2


def test_l2_long_session_helper_reads_snapshot_publication_metadata(monkeypatch):
    class FakeCache:
        def lookup(self, _tokens, _key):
            return 2, None

    events = ClosableEvents(
        [
            _prefill_complete(),
            SnapshotPublishedEvent(
                kind="prefill",
                snapshot_boundary=3,
                prefix_len=3,
                insert_ms=0.2,
                admitted=True,
                from_snapshot=True,
                snap_prefix_len=2,
            ),
            SummaryEvent(
                elapsed_us=100.0,
                prompt_token_count=3,
                generated_token_ids=(),
                generation_tokens=0,
                accepted_from_draft=0,
                acceptance_ratio=0.0,
                cycles_completed=0,
                phase_timings_us={},
            ),
        ]
    )
    monkeypatch.setattr(l2_long_session, "stream_dflash_generate", lambda **_kwargs: events)

    result = l2_long_session._run_turn(
        target_model=object(),
        target_ops=object(),
        tokenizer=object(),
        draft_model=SimpleNamespace(target_layer_ids=()),
        draft_backend=object(),
        prompt_tokens=[1, 2, 3],
        max_tokens=4,
        cache=FakeCache(),
        key=object(),
        runtime_context=_runtime_context(),
    )

    assert events.closed is True
    assert result["matched_lookup"] == 2
    assert result["from_snapshot"] is True
