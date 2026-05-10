# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from queue import SimpleQueue
from types import SimpleNamespace

import pytest

from dflash_mlx.engine.events import (
    MemoryWaterfallEvent,
    PrefillCompleteEvent,
    SnapshotPublishedEvent,
    SummaryEvent,
    TokenEvent,
)
from dflash_mlx.server.request_loop import consume_dflash_events

class FakeDetokenizer:
    def __init__(self):
        self.last_segment = ""
        self.reset_calls = 0
        self.tokens = []

    def reset(self):
        self.reset_calls += 1
        self.last_segment = ""
        self.tokens = []

    def add_token(self, token):
        self.tokens.append(int(token))
        self.last_segment = f"T{int(token)}"

    def finalize(self):
        self.last_segment = ""

class FakeTokenizer:
    def __init__(self):
        self.detokenizer = FakeDetokenizer()

class FakeContext:
    _should_stop = False

class ClosableEvents:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True

def test_consume_dflash_events_streams_pending_token_and_summary():
    rqueue = SimpleQueue()
    events = ClosableEvents(
        [
            PrefillCompleteEvent(
                prefill_us=1.0,
                prompt_token_count=3,
                logical_ctx_tokens=3,
                physical_prefill_tokens=3,
                prefill_tokens_restored=0,
                prefill_tokens_computed=3,
            ),
            TokenEvent(token_id=10, generated_tokens=1, acceptance_ratio=0.5, cycles_completed=1),
            TokenEvent(token_id=11, generated_tokens=2, acceptance_ratio=0.5, cycles_completed=1),
            SummaryEvent(
                elapsed_us=10.0,
                prompt_token_count=3,
                generated_token_ids=(10, 11),
                generation_tokens=2,
                accepted_from_draft=1,
                acceptance_ratio=0.5,
                cycles_completed=1,
                phase_timings_us={},
            ),
        ]
    )

    result = consume_dflash_events(
        event_iter=events,
        rqueue=rqueue,
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
    )

    responses = []
    while not rqueue.empty():
        responses.append(rqueue.get())

    assert events.closed is True
    assert result.summary_event is not None
    assert result.live_token_count == 2
    assert result.finish_reason == "stop"
    assert len(responses) == 2
    assert responses[0].token == 10
    assert responses[0].text == "T10"
    assert responses[1].token == 11
    assert responses[1].text == "T11"


def test_consume_dflash_events_ignores_snapshot_publication_metadata():
    prefix_flow = SimpleNamespace(lookup_ms=0.1, hit_tokens=2, insert_ms=0.5)
    rqueue = SimpleQueue()
    events = ClosableEvents(
        [
            SnapshotPublishedEvent(
                kind="prefill",
                snapshot_boundary=3,
                prefix_len=3,
                insert_ms=0.2,
                admitted=True,
            ),
            SnapshotPublishedEvent(
                kind="generation",
                snapshot_boundary=5,
                prefix_len=5,
                insert_ms=0.3,
                admitted=True,
            ),
            SummaryEvent(
                elapsed_us=10.0,
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

    result = consume_dflash_events(
        event_iter=events,
        rqueue=rqueue,
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
        prefix_flow=prefix_flow,
    )

    assert events.closed is True
    assert rqueue.empty()
    assert result.summary_event is not None
    assert result.cache_lookup_ms == 0.1
    assert result.cache_hit_tokens == 2
    assert result.cache_insert_ms == 0.5


def test_consume_dflash_events_rejects_stale_dict_events():
    rqueue = SimpleQueue()
    events = ClosableEvents([{"event": "token", "token_id": 10}])

    with pytest.raises(TypeError, match="Unsupported DFlash engine event: dict"):
        consume_dflash_events(
            event_iter=events,
            rqueue=rqueue,
            ctx=FakeContext(),
            tokenizer=FakeTokenizer(),
            prompt=[1, 2, 3],
            max_tokens=16,
            eos_token_ids=set(),
            request_start_ns=0,
        )

    assert events.closed is True
    assert rqueue.empty()


def test_memory_waterfall_events_are_enriched_with_prefix_cache_memory():
    class FakePrefixFlow:
        lookup_ms = 0.0
        hit_tokens = 0
        insert_ms = 0.0

        def prefix_cache_memory_bytes(self):
            return {
                "l1_snapshot_bytes": 123,
                "l1_snapshot_target_hidden_bytes": 45,
            }

    runtime_context = SimpleNamespace(
        diagnostics=SimpleNamespace(memory_waterfall=True, trace=None)
    )
    rqueue = SimpleQueue()
    events = ClosableEvents(
        [
            MemoryWaterfallEvent(fields={"memory_phase": "during_decode"}),
            SummaryEvent(
                elapsed_us=10.0,
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

    result = consume_dflash_events(
        event_iter=events,
        rqueue=rqueue,
        ctx=FakeContext(),
        tokenizer=FakeTokenizer(),
        prompt=[1, 2, 3],
        max_tokens=16,
        eos_token_ids=set(),
        request_start_ns=0,
        prefix_flow=FakePrefixFlow(),
        runtime_context=runtime_context,
    )

    assert result.memory_waterfall_peak is not None
    assert result.memory_waterfall_peak["l1_snapshot_bytes"] == 123
    assert result.memory_waterfall_peak["l1_snapshot_target_hidden_bytes"] == 45
    assert result.memory_waterfall_peak["l1_snapshot_gb"] > 0.0
