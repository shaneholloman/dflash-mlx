# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from dflash_mlx.engine.target_features import TargetFeatureStore


def _snapshot(hidden):
    total_len = int(hidden.shape[1])
    return SimpleNamespace(
        target_hidden_chunks=(hidden,),
        target_hidden_chunk_spans=((0, total_len),),
        target_hidden_total_len=total_len,
    )


def test_target_feature_store_hydrates_snapshot_and_preserves_prompt_width():
    cached = mx.ones((1, 3, 2), dtype=mx.float32)
    store = TargetFeatureStore(prompt_len=5)

    hidden = store.hydrate_from_snapshot(_snapshot(cached), snap_prefix_len=3)

    assert hidden.shape == (1, 5, 2)
    assert mx.all(hidden[:, :3, :] == 1).item()
    assert mx.all(hidden[:, 3:, :] == 0).item()
    assert store.current_hidden is hidden


def test_target_feature_store_writes_prompt_slices_and_prefix_view():
    store = TargetFeatureStore(prompt_len=4)
    first = mx.ones((1, 2, 3), dtype=mx.float32)
    second = mx.ones((1, 2, 3), dtype=mx.float32) * 2

    store.write_prompt_slice(start=0, end=2, features=first)
    hidden = store.write_prompt_slice(start=2, end=4, features=second)
    prefix = store.prefix_view(3)

    assert hidden.shape == (1, 4, 3)
    assert prefix.shape == (1, 3, 3)
    assert mx.all(prefix[:, :2, :] == 1).item()
    assert mx.all(prefix[:, 2:3, :] == 2).item()


def test_target_feature_store_requires_current_hidden():
    store = TargetFeatureStore(prompt_len=1)
    with pytest.raises(RuntimeError, match="target hidden features are unavailable"):
        store.require_current_hidden()

    features = mx.ones((1, 1, 2))
    store.write_prompt_slice(start=0, end=1, features=features)

    assert store.require_current_hidden() is store.current_hidden


def test_target_feature_store_generation_snapshot_uses_frozen_prefill_and_chunks():
    store = TargetFeatureStore(prompt_len=2)
    prefill = mx.ones((1, 2, 2), dtype=mx.float32)
    first = mx.ones((1, 1, 2), dtype=mx.float32) * 3
    second = mx.ones((1, 2, 2), dtype=mx.float32) * 4

    store.write_prompt_slice(start=0, end=2, features=prefill)
    store.freeze_prefill_for_snapshot(enabled=True)
    store.commit_generation(first, collect_snapshot=True)
    store.commit_generation(second, collect_snapshot=True)

    out = store.generation_snapshot_hidden()

    assert store.current_hidden is second
    assert out.shape == (1, 5, 2)
    assert mx.all(out[:, :2, :] == 1).item()
    assert mx.all(out[:, 2:3, :] == 3).item()
    assert mx.all(out[:, 3:, :] == 4).item()


def test_target_feature_store_generation_snapshot_is_none_when_not_collected():
    store = TargetFeatureStore(prompt_len=1)
    store.write_prompt_slice(start=0, end=1, features=mx.ones((1, 1, 2)))
    store.freeze_prefill_for_snapshot(enabled=False)
    store.commit_generation(mx.ones((1, 1, 2)), collect_snapshot=False)

    assert store.generation_snapshot_hidden() is None
    assert store.generation_chunks == ()
