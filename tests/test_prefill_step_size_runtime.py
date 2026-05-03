# Copyright 2026 bstnxbt
# MIT License - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from dflash_mlx.engine import spec_epoch
from dflash_mlx.runtime_context import (
    build_runtime_context,
    runtime_config_from_profile,
)


class _FakeTargetOps:
    def __init__(self) -> None:
        self.forward_lengths: list[int] = []

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


def test_runtime_prefill_chunks_use_configured_step_size(monkeypatch):
    target_ops = _FakeTargetOps()
    monkeypatch.setattr(spec_epoch, "resolve_target_ops", lambda _model: target_ops)
    monkeypatch.setattr(spec_epoch, "make_draft_backend", lambda: _FakeDraftBackend())

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
            tokenizer=object(),
            draft_model=draft_model,
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
    assert events[-1]["event"] == "summary"
