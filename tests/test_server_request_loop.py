# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from queue import SimpleQueue
from types import SimpleNamespace

import pytest

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
            {"event": "prefill", "prompt_token_count": 3},
            {"event": "token", "token_id": 10, "acceptance_ratio": 0.5},
            {"event": "token", "token_id": 11, "acceptance_ratio": 0.5},
            {
                "event": "summary",
                "generated_token_ids": [10, 11],
                "generation_tokens": 2,
            },
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


def test_consume_dflash_events_routes_snapshot_events_without_streaming():
    class FakePrefixFlow:
        lookup_ms = 0.0
        hit_tokens = 0
        insert_ms = 0.0
        cache = None

        def __init__(self):
            self.prefill_snapshots = []
            self.generation_snapshots = []

        def handle_prefill_snapshot(self, snapshot):
            self.prefill_snapshots.append(snapshot)

        def handle_generation_snapshot(self, snapshot):
            self.generation_snapshots.append(snapshot)

    prefix_flow = FakePrefixFlow()
    rqueue = SimpleQueue()
    prefill_snapshot = object()
    generation_snapshot = object()
    events = ClosableEvents(
        [
            {"event": "prefill_snapshot_ready", "snapshot": prefill_snapshot},
            {"event": "generation_snapshot_ready", "snapshot": generation_snapshot},
            {"event": "summary", "generated_token_ids": [], "generation_tokens": 0},
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
    assert prefix_flow.prefill_snapshots == [prefill_snapshot]
    assert prefix_flow.generation_snapshots == [generation_snapshot]
    assert result.summary_event is not None


def test_consume_dflash_events_requires_snapshot_payload_for_snapshot_events():
    class FakePrefixFlow:
        lookup_ms = 0.0
        hit_tokens = 0
        insert_ms = 0.0
        cache = None

        def handle_prefill_snapshot(self, snapshot):
            raise AssertionError("missing snapshot should fail before flow call")

        def handle_generation_snapshot(self, snapshot):
            raise AssertionError("missing snapshot should fail before flow call")

    events = ClosableEvents([{"event": "prefill_snapshot_ready"}])

    with pytest.raises(KeyError, match="snapshot") as exc_info:
        consume_dflash_events(
            event_iter=events,
            rqueue=SimpleQueue(),
            ctx=FakeContext(),
            tokenizer=FakeTokenizer(),
            prompt=[1, 2, 3],
            max_tokens=16,
            eos_token_ids=set(),
            request_start_ns=0,
            prefix_flow=FakePrefixFlow(),
        )

    assert exc_info.value.args == ("snapshot",)
    assert events.closed is True


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
            {"event": "memory_waterfall", "memory_phase": "during_decode"},
            {"event": "summary", "generated_token_ids": [], "generation_tokens": 0},
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
