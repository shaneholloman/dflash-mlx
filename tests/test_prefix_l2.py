# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models.cache import RotatingKVCache

from dflash_mlx.cache.codecs import hydrate_target_cache, serialize_target_cache
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.cache.prefix_l2 import (
    L2_FILE_SUFFIX,
    L2_LAYOUT_ROOT,
    L2_SCHEMA_VERSION,
    DFlashPrefixL2Cache,
    _deserialize,
    _format_filename,
    _key_hash,
    _runtime_layout_hash,
    _serialize,
    _token_hash,
)
from dflash_mlx.cache.store import PrefixSnapshotStore
from dflash_mlx.diagnostics import TraceConfig
from tests.test_prefix_cache import (
    _make_full_hidden_snapshot,
    _make_key,
    _make_rotating_cache_populated,
    _make_synthetic_snapshot,
)

def _all_snapshot_files(cache_dir: Path) -> list[Path]:
    root = cache_dir / L2_LAYOUT_ROOT
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.name.endswith(L2_FILE_SUFFIX):
            out.append(p)
    return out

def _bucket_dir(cache_dir: Path, key) -> Path:
    kh = _key_hash(key)
    return cache_dir / L2_LAYOUT_ROOT / _runtime_layout_hash() / kh[:2] / kh


def _store(
    *,
    l2: DFlashPrefixL2Cache,
    max_entries: int = 4,
    max_bytes: int = 10**9,
) -> PrefixSnapshotStore:
    return PrefixSnapshotStore(
        l1=DFlashPrefixCache(max_entries=max_entries, max_bytes=max_bytes),
        l2=l2,
    )


def _write_tampered_snapshot_file(
    cache_dir: Path,
    key,
    token_ids: list[int],
    tamper,
) -> Path:
    snap = _make_synthetic_snapshot(token_ids, key)
    arrays, meta_dict = _serialize(snap)
    meta = json.loads(meta_dict["dflash_meta"])
    replacement = tamper(meta)
    if replacement is not None:
        meta = replacement
    meta_dict["dflash_meta"] = json.dumps(meta)
    bucket = _bucket_dir(cache_dir, key)
    bucket.mkdir(parents=True, exist_ok=True)
    path = bucket / _format_filename(
        token_len=len(token_ids),
        token_hash=_token_hash(token_ids),
        kind="prefill",
        fp_short="0" * 16,
    )
    mx.save_safetensors(str(path), arrays, metadata=meta_dict)
    return path

def _set_metadata_field(name: str, value):
    def _tamper(meta):
        meta[name] = value

    return _tamper

def _set_key_metadata_field(name: str, value):
    def _tamper(meta):
        meta["key"][name] = value

    return _tamper

def _replace_metadata(value):
    def _tamper(_meta):
        return value

    return _tamper

def _wait_writes(l2: DFlashPrefixL2Cache, expected: int, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if l2.stats()["writes"] >= expected:
            return
        time.sleep(0.05)
    raise AssertionError(
        f"L2 writer did not catch up: writes={l2.stats()['writes']} expected>={expected}"
    )

class TestBfloat16RoundTrip:
    def test_bf16_arrays_preserved_through_save_load(self, tmp_path):
        from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
        n_layers = 4
        head, head_dim, seq = 8, 32, 16

        ref = mx.random.normal((1, head, seq, head_dim))
        bf = ref.astype(mx.bfloat16)
        fa_states = tuple(
            (bf, bf, 0) for _ in range(n_layers)
        )
        last_logits_bf = mx.random.normal((1, 1024)).astype(mx.bfloat16)
        snap = DFlashPrefixSnapshot(
            token_ids=tuple(range(seq)),
            fa_states=fa_states,
            gdn_states=tuple([None] * n_layers),
            target_hidden_chunks=tuple(),
            target_hidden_chunk_spans=tuple(),
            target_hidden_total_len=0,
            last_logits=last_logits_bf,
            key=_make_key(),
            kind="prefill",
        )
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            l2._write_one(snap)
            loaded = l2.lookup(tuple(range(seq)), snap.key)
            assert loaded is not None
            assert loaded.fa_states[0][0].dtype == mx.bfloat16
            assert loaded.last_logits.dtype == mx.bfloat16

            assert mx.all(loaded.fa_states[0][0] == bf).item()
            assert mx.all(loaded.last_logits == last_logits_bf).item()
        finally:
            l2.shutdown()

class TestSerializationRoundTrip:
    def test_full_snapshot_roundtrip(self, tmp_path):
        snap = _make_synthetic_snapshot([1, 2, 3, 4, 5], _make_key())
        arrays, meta_dict = _serialize(snap)
        path = tmp_path / "rt.safetensors"
        mx.save_safetensors(str(path), arrays, metadata=meta_dict)
        loaded_arrays, loaded_meta_dict = mx.load(
            str(path), format="safetensors", return_metadata=True
        )
        loaded_meta = json.loads(loaded_meta_dict["dflash_meta"])
        rehydrated = _deserialize(loaded_arrays, loaded_meta)
        assert rehydrated.token_ids == snap.token_ids
        assert rehydrated.kind == snap.kind
        assert rehydrated.key == snap.key
        assert rehydrated.target_hidden_total_len == snap.target_hidden_total_len
        assert len(rehydrated.fa_states) == len(snap.fa_states)
        assert len(rehydrated.gdn_states) == len(snap.gdn_states)

    def test_post_restore_mutation_isolation(self, tmp_path):
        snap = _make_synthetic_snapshot([1, 2, 3], _make_key())
        arrays, meta_dict = _serialize(snap)
        path = tmp_path / "iso.safetensors"
        mx.save_safetensors(str(path), arrays, metadata=meta_dict)
        loaded_arrays, loaded_meta_dict = mx.load(
            str(path), format="safetensors", return_metadata=True
        )
        loaded_meta = json.loads(loaded_meta_dict["dflash_meta"])
        snap1 = _deserialize(loaded_arrays, loaded_meta)
        snap2 = _deserialize(loaded_arrays, loaded_meta)
        k1, _v1, _ = snap1.fa_states[0]
        before = float(k1[0, 0, 0, 0].item())
        del snap2
        del loaded_arrays
        assert float(k1[0, 0, 0, 0].item()) == before

    def test_rotating_snapshot_l2_roundtrip_hydrates(self, tmp_path):
        src = [_make_rotating_cache_populated(n_tokens=7, max_size=4, keep=1)]
        fa, gdn = serialize_target_cache(src)
        snap = _make_full_hidden_snapshot(
            token_ids=tuple(range(7)),
            fa_states=fa,
            gdn_states=gdn,
            target_hidden=mx.zeros((1, 7, 4)),
            last_logits=mx.zeros((1, 10)),
            key=_make_key(),
        )
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            l2._write_one(snap)
            loaded = l2.lookup(tuple(range(7)), snap.key)
            assert loaded is not None
            hydrated = hydrate_target_cache(
                loaded,
                [RotatingKVCache(max_size=4, keep=1)],
            )
            assert isinstance(hydrated[0], RotatingKVCache)
            assert hydrated[0].offset == 7
            assert mx.all(
                src[0]._temporal_order(src[0].keys)
                == hydrated[0]._temporal_order(hydrated[0].keys)
            ).item()
        finally:
            l2.shutdown()

class TestL2Lifecycle:
    def test_disabled_when_no_l2_attached(self):
        cache = DFlashPrefixCache(max_entries=2, max_bytes=10**9)
        snap = _make_synthetic_snapshot([1, 2, 3], _make_key())
        cache.insert(snap)
        stats = cache.stats()
        assert "l2" not in stats

    def test_evict_promotes_to_l2_disk(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            cache = _store(max_entries=1, max_bytes=10**9, l2=l2)
            key = _make_key()
            cache.insert(_make_synthetic_snapshot([1, 2, 3], key))
            cache.insert(_make_synthetic_snapshot([4, 5, 6], key))
            _wait_writes(l2, expected=1)
            assert len(_all_snapshot_files(tmp_path)) == 1
        finally:
            l2.shutdown()

    def test_pruned_prefix_promotes_to_l2_disk(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            cache = _store(max_entries=4, max_bytes=10**9, l2=l2)
            key = _make_key()
            cache.insert(_make_synthetic_snapshot([1, 2, 3], key))
            cache.insert(_make_synthetic_snapshot([1, 2, 3, 4], key))
            _wait_writes(l2, expected=1)

            matched, hydrated = cache.lookup([1, 2, 3], key)

            assert matched == 3
            assert hydrated is not None
            assert hydrated.token_ids == (1, 2, 3)
        finally:
            l2.shutdown()

    def test_l2_lookup_promotes_to_l1(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([7, 8, 9, 10], key))
            cache = _store(max_entries=4, max_bytes=10**9, l2=l2)
            matched, hydrated = cache.lookup([7, 8, 9, 10], key)
            assert matched == 4
            assert hydrated is not None
            assert hydrated.token_ids == (7, 8, 9, 10)
            stats = cache.stats()
            assert stats["l2_hits"] == 1
            assert stats["current_entries"] == 1
        finally:
            l2.shutdown()

    def test_l2_lookup_hit_logs_cache_event_from_store_trace(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path / "l2", max_bytes=10**9)
        try:
            trace_dir = tmp_path / "trace"
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([7, 8, 9, 10], key))
            cache = _store(max_entries=4, max_bytes=10**9, l2=l2)
            cache.set_trace_config(TraceConfig(log_dir=trace_dir))

            matched, hydrated = cache.lookup([7, 8, 9, 10], key)

            assert matched == 4
            assert hydrated is not None
            rows = (trace_dir / "cache_events.jsonl").read_text().splitlines()
            assert rows
            event = json.loads(rows[-1])
            assert event["op"] == "lookup"
            assert event["result"] == "l2_hit"
            assert event["matched_len"] == 4
        finally:
            l2.shutdown()

    def test_l2_lookup_miss_when_key_mismatch(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key_a = _make_key(target_model_id="model-a")
            key_b = _make_key(target_model_id="model-b")
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key_a))
            cache = _store(max_entries=4, max_bytes=10**9, l2=l2)
            matched, hydrated = cache.lookup([1, 2, 3], key_b)
            assert matched == 0
            assert hydrated is None
            stats = cache.stats()
            assert stats["l2_hits"] == 0
            assert stats["l2_misses"] == 1
        finally:
            l2.shutdown()

    def test_l2_lookup_miss_when_token_prefix_diverges(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3, 4, 5], key))
            cache = _store(max_entries=4, max_bytes=10**9, l2=l2)
            matched, hydrated = cache.lookup([1, 2, 99, 4, 5], key)
            assert matched == 0
            assert hydrated is None
        finally:
            l2.shutdown()

    def test_exact_hit_requires_prefill_kind(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            snap = _make_synthetic_snapshot([1, 2, 3], key)
            snap.kind = "generation"
            snap.last_logits = None
            l2._write_one(snap)
            res = l2.lookup((1, 2, 3), key)
            assert res is None
        finally:
            l2.shutdown()

    def test_prefix_hit_accepts_generation_kind(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            snap = _make_synthetic_snapshot([1, 2, 3], key)
            snap.kind = "generation"
            snap.last_logits = None
            l2._write_one(snap)

            res = l2.lookup((1, 2, 3, 4, 5), key)
            assert res is not None
            assert res.token_ids == (1, 2, 3)
        finally:
            l2.shutdown()

class TestFailureModes:
    def test_corrupt_file_rejected_and_deleted(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            bucket = _bucket_dir(tmp_path, key)
            bucket.mkdir(parents=True, exist_ok=True)

            name = _format_filename(
                token_len=3,
                token_hash=_token_hash([1, 2, 3]),
                kind="prefill",
                fp_short="0" * 16,
            )
            bad = bucket / name
            bad.write_bytes(b"not a safetensors file")
            cache = _store(max_entries=4, max_bytes=10**9, l2=l2)
            matched, hydrated = cache.lookup([1, 2, 3], key)
            assert matched == 0
            assert hydrated is None
            assert not bad.exists()
            stats = l2.stats()
            assert stats["load_errors"] >= 1
        finally:
            l2.shutdown()

    def test_schema_version_mismatch_rejected(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            snap = _make_synthetic_snapshot([1, 2], key)
            arrays, meta_dict = _serialize(snap)
            tampered = json.loads(meta_dict["dflash_meta"])
            tampered["schema_version"] = L2_SCHEMA_VERSION + 99
            meta_dict["dflash_meta"] = json.dumps(tampered)
            bucket = _bucket_dir(tmp_path, key)
            bucket.mkdir(parents=True, exist_ok=True)
            stale = bucket / _format_filename(
                token_len=2,
                token_hash=_token_hash([1, 2]),
                kind="prefill",
                fp_short="0" * 16,
            )
            mx.save_safetensors(str(stale), arrays, metadata=meta_dict)
            cache = _store(max_entries=4, max_bytes=10**9, l2=l2)
            matched, _ = cache.lookup([1, 2], key)
            assert matched == 0
            assert not stale.exists()
            stats = l2.stats()
            assert stats["schema_rejects"] >= 1
        finally:
            l2.shutdown()

    def test_runtime_version_mismatch_rejected(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            snap = _make_synthetic_snapshot([1, 2], key)
            arrays, meta_dict = _serialize(snap)
            tampered = json.loads(meta_dict["dflash_meta"])
            tampered["runtime_version"] = "0.0.0-not-a-real-version"
            meta_dict["dflash_meta"] = json.dumps(tampered)
            bucket = _bucket_dir(tmp_path, key)
            bucket.mkdir(parents=True, exist_ok=True)
            stale = bucket / _format_filename(
                token_len=2,
                token_hash=_token_hash([1, 2]),
                kind="prefill",
                fp_short="0" * 16,
            )
            mx.save_safetensors(str(stale), arrays, metadata=meta_dict)
            cache = _store(max_entries=4, max_bytes=10**9, l2=l2)
            matched, _ = cache.lookup([1, 2], key)
            assert matched == 0
            assert not stale.exists()
            stats = l2.stats()
            assert stats["schema_rejects"] >= 1
        finally:
            l2.shutdown()

    @pytest.mark.parametrize(
        "tamper",
        [
            _set_metadata_field("schema_version", "bad-int"),
            _set_metadata_field("schema_version", "1"),
            _set_metadata_field("schema_version", 1.0),
            _set_metadata_field("schema_version", True),
            _replace_metadata(["not", "a", "dict"]),
            _set_metadata_field("token_ids", None),
            _set_metadata_field("token_ids", ["1", 2]),
            _set_metadata_field("token_ids", [1.0, 2]),
            _set_metadata_field("token_ids", [True, 2]),
            _set_key_metadata_field("capture_layer_ids", ["10", 20]),
            _set_key_metadata_field("draft_sink_size", "16"),
            _set_key_metadata_field("draft_window_size", 2048.0),
            _set_key_metadata_field("target_fa_window", False),
            _set_key_metadata_field("format_version", True),
        ],
    )
    def test_malformed_metadata_rejected(self, tmp_path, tamper):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            bad = _write_tampered_snapshot_file(
                tmp_path,
                key,
                [1, 2],
                tamper,
            )

            res = l2.lookup((1, 2), key)

            assert res is None
            assert not bad.exists()
            assert l2.stats()["schema_rejects"] >= 1
        finally:
            l2.shutdown()

    def test_snapshot_serialization_bug_propagates(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            def _bad_serialize(_snapshot):
                raise TypeError("broken serializer")

            monkeypatch.setattr("dflash_mlx.cache.prefix_l2._serialize", _bad_serialize)
            snap = _make_synthetic_snapshot([1, 2, 3], _make_key())

            with pytest.raises(TypeError, match="broken serializer"):
                l2.insert_async(snap)
            assert l2._writer_slots.acquire(blocking=False)
            l2._writer_slots.release()

            with pytest.raises(TypeError, match="broken serializer"):
                l2._write_one(snap)
        finally:
            l2.shutdown()

    def test_key_parser_bug_propagates(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            _write_tampered_snapshot_file(tmp_path, key, [1, 2], lambda _meta: None)

            def _broken_key_parser(_meta):
                raise RuntimeError("broken key parser")

            monkeypatch.setattr(
                "dflash_mlx.cache.prefix_l2._key_from_dict",
                _broken_key_parser,
            )

            with pytest.raises(RuntimeError, match="broken key parser"):
                l2.lookup((1, 2), key)
        finally:
            l2.shutdown()

    def test_finalizer_logs_shutdown_failure(self, caplog):
        class BrokenFinalizerL2(DFlashPrefixL2Cache):
            def shutdown(self, wait: bool = True) -> None:
                raise RuntimeError("broken finalizer")

        l2 = object.__new__(BrokenFinalizerL2)

        with caplog.at_level(logging.DEBUG, logger="dflash_mlx.cache.prefix_l2"):
            l2.__del__()

        assert "L2 finalizer shutdown failed: broken finalizer" in caplog.text

    def test_disk_full_does_not_crash(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            import errno

            def _enospc(*_args, **_kwargs):
                raise OSError(errno.ENOSPC, "disk full (test)")

            monkeypatch.setattr("mlx.core.save_safetensors", _enospc)
            cache = _store(max_entries=1, max_bytes=10**9, l2=l2)
            key = _make_key()
            cache.insert(_make_synthetic_snapshot([1, 2, 3], key))
            cache.insert(_make_synthetic_snapshot([4, 5, 6], key))
            time.sleep(0.5)
            stats = l2.stats()
            assert stats["write_errors"] >= 1
            assert stats["writes"] == 0
        finally:
            l2.shutdown()

class TestEvictionAndStats:
    def test_byte_budget_evicts_oldest(self, tmp_path):
        probe_l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            probe_l2._write_one(_make_synthetic_snapshot([0, 0, 0], _make_key()))
            probe_files = _all_snapshot_files(tmp_path)
            on_disk = probe_files[0].stat().st_size
        finally:
            probe_l2.shutdown()

        for f in _all_snapshot_files(tmp_path):
            f.unlink()
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=int(on_disk * 1.5))
        try:
            for i in range(4):
                tokens = list(range(i * 10 + 1, i * 10 + 4))
                l2._write_one(_make_synthetic_snapshot(tokens, _make_key()))
            files = _all_snapshot_files(tmp_path)
            assert len(files) <= 2
            stats = l2.stats()
            assert stats["evictions"] >= 1
        finally:
            l2.shutdown()

    def test_eviction_recomputes_after_external_parseable_file(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=1)
        try:
            key = _make_key()
            bucket = _bucket_dir(tmp_path, key)
            bucket.mkdir(parents=True, exist_ok=True)
            external = bucket / _format_filename(
                token_len=2,
                token_hash=_token_hash([1, 2]),
                kind="prefill",
                fp_short="0" * 16,
            )
            external.write_bytes(b"x" * 5000)

            l2._write_one(_make_synthetic_snapshot([3, 4, 5], key))

            on_disk = sum(p.stat().st_size for p in _all_snapshot_files(tmp_path))
            stats = l2.stats()
            assert stats["evictions"] >= 1
            assert stats["current_bytes"] == on_disk
        finally:
            l2.shutdown()

    def test_stats_surfaced_through_store(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            cache = _store(max_entries=1, max_bytes=10**9, l2=l2)
            key = _make_key()
            cache.insert(_make_synthetic_snapshot([1, 2, 3], key))
            cache.insert(_make_synthetic_snapshot([4, 5, 6], key))
            _wait_writes(l2, expected=1)
            cache.lookup([1, 2, 3], key)
            stats = cache.stats()
            assert "l2" in stats
            assert stats["l2"]["hits"] >= 1
        finally:
            l2.shutdown()

    def test_clear_removes_l2_files(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            cache = _store(max_entries=1, max_bytes=10**9, l2=l2)
            key = _make_key()
            cache.insert(_make_synthetic_snapshot([1, 2, 3], key))
            cache.insert(_make_synthetic_snapshot([4, 5, 6], key))
            _wait_writes(l2, expected=1)
            cache.clear()
            assert _all_snapshot_files(tmp_path) == []
        finally:
            l2.shutdown()

class TestAtomicity:
    def test_no_partial_files_left_after_normal_write(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            tmp_files = list((tmp_path / L2_LAYOUT_ROOT).rglob("*.tmp.safetensors"))
            assert tmp_files == []
            assert len(_all_snapshot_files(tmp_path)) == 1
        finally:
            l2.shutdown()

class TestConcurrency:
    def test_second_instance_falls_back_to_read_only(self, tmp_path):
        first = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            assert first.writable
            second = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
            try:
                assert not second.writable
                key = _make_key()
                first._write_one(_make_synthetic_snapshot([1, 2, 3], key))
                snap = second.lookup((1, 2, 3), key)
                assert snap is not None
                assert snap.token_ids == (1, 2, 3)
            finally:
                second.shutdown()
        finally:
            first.shutdown()

    def test_second_instance_silently_drops_writes(self, tmp_path):
        first = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            second = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
            try:
                key = _make_key()
                second.insert_async(_make_synthetic_snapshot([9, 9, 9], key))
                time.sleep(0.5)
                assert _all_snapshot_files(tmp_path) == []
            finally:
                second.shutdown()
        finally:
            first.shutdown()

class TestLookupHotPath:
    def test_many_bad_files_zero_loads(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            bucket = _bucket_dir(tmp_path, key)
            bucket.mkdir(parents=True, exist_ok=True)

            import secrets
            for i in range(100):
                name = _format_filename(
                    token_len=3,
                    token_hash=secrets.token_hex(8),
                    kind="prefill",
                    fp_short=secrets.token_hex(8),
                )
                (bucket / name).write_bytes(b"\x00")

            calls = {"n": 0}
            real_load = mx.load

            def _counting_load(*args, **kwargs):
                calls["n"] += 1
                return real_load(*args, **kwargs)

            monkeypatch.setattr("mlx.core.load", _counting_load)

            monkeypatch.setattr("dflash_mlx.cache.prefix_l2.mx.load", _counting_load)

            res = l2.lookup((1, 2, 3), key)
            assert res is None
            assert calls["n"] == 0, (
                f"lookup called mx.load {calls['n']} times; expected 0 — "
                "filename hash filter is broken"
            )
            stats = l2.stats()
            assert stats["lookup_hash_filtered"] >= 100
            assert stats["lookup_loads"] == 0
        finally:
            l2.shutdown()

    def test_single_good_file_one_load(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([5, 6, 7, 8], key))
            bucket = _bucket_dir(tmp_path, key)
            import secrets
            for _ in range(50):
                name = _format_filename(
                    token_len=4,
                    token_hash=secrets.token_hex(8),
                    kind="prefill",
                    fp_short=secrets.token_hex(8),
                )
                (bucket / name).write_bytes(b"\x00")

            calls = {"n": 0}
            real_load = mx.load

            def _counting_load(*args, **kwargs):
                calls["n"] += 1
                return real_load(*args, **kwargs)

            monkeypatch.setattr("dflash_mlx.cache.prefix_l2.mx.load", _counting_load)

            res = l2.lookup((5, 6, 7, 8), key)
            assert res is not None
            assert res.token_ids == (5, 6, 7, 8)
            assert calls["n"] == 1, f"expected exactly 1 mx.load, got {calls['n']}"
        finally:
            l2.shutdown()

def _stop_writer(l2):
    l2._stop.set()
    l2._write_queue.put(None)
    if l2._writer_thread is not None:
        l2._writer_thread.join(timeout=2.0)

class TestEpochInvalidation:
    def test_clear_during_pending_write_drops_payload(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            _stop_writer(l2)
            l2._stop.clear()

            key = _make_key()
            l2.insert_async(_make_synthetic_snapshot([1, 2, 3], key))
            assert l2._write_queue.qsize() == 1

            l2.clear()

            payload = l2._write_queue.get_nowait()
            l2._write_payload(payload)

            assert _all_snapshot_files(tmp_path) == []
            stats = l2.stats()
            assert stats["write_drops_epoch_invalidated"] >= 1
            assert stats["writes"] == 0
        finally:
            l2.shutdown()

    def test_writes_after_clear_use_new_epoch(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            assert len(_all_snapshot_files(tmp_path)) == 1
            l2.clear()
            assert _all_snapshot_files(tmp_path) == []
            l2._write_one(_make_synthetic_snapshot([7, 8, 9], key))
            assert len(_all_snapshot_files(tmp_path)) == 1
        finally:
            l2.shutdown()

class TestQueueBackpressure:
    def test_queue_full_drops_with_stat(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            _stop_writer(l2)

            key = _make_key()
            l2.insert_async(_make_synthetic_snapshot([1, 2, 3], key))
            l2.insert_async(_make_synthetic_snapshot([4, 5, 6], key))
            stats = l2.stats()
            assert stats["write_drops_queue_full"] >= 1
        finally:
            l2.shutdown()

    def test_default_max_in_flight_is_one(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            assert l2._max_in_flight == 1
        finally:
            l2.shutdown()

    def test_drop_does_not_call_mx_eval(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            _stop_writer(l2)
            key = _make_key()

            snap1 = _make_synthetic_snapshot([1, 2, 3], key)
            snap2 = _make_synthetic_snapshot([4, 5, 6], key)

            l2.insert_async(snap1)

            eval_calls = {"n": 0}
            from dflash_mlx.cache import prefix_l2 as _l2mod

            real_eval = _l2mod.mx.eval

            def _spy_eval(*args, **kwargs):
                eval_calls["n"] += 1
                return real_eval(*args, **kwargs)

            monkeypatch.setattr("dflash_mlx.cache.prefix_l2.mx.eval", _spy_eval)

            l2.insert_async(snap2)
            assert eval_calls["n"] == 0, (
                f"insert_async called mx.eval {eval_calls['n']} times "
                "after the slot was full — drop happens AFTER materialize"
            )
            stats = l2.stats()
            assert stats["write_drops_queue_full"] >= 1
        finally:
            l2.shutdown()

class TestReadOnlyInstance:
    def test_read_only_does_not_spawn_writer_thread(self, tmp_path):
        first = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            assert first._writer_thread is not None
            second = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
            try:
                assert not second.writable
                assert second._writer_thread is None
            finally:
                second.shutdown()
        finally:
            first.shutdown()

class TestStatsHotPath:
    def test_stats_does_not_walk_filesystem(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            l2._write_one(_make_synthetic_snapshot([4, 5, 6], key))

            calls = {"n": 0}
            real_walk = l2._walk_snapshots

            def _spy_walk():
                calls["n"] += 1
                return real_walk()

            l2._walk_snapshots = _spy_walk
            for _ in range(10):
                stats = l2.stats()
                assert stats["current_bytes"] > 0
            assert calls["n"] == 0, (
                f"stats() called _walk_snapshots {calls['n']} times — "
                "tracked counter is not used on the hot path"
            )
        finally:
            l2.shutdown()

    def test_tracked_bytes_matches_disk(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            l2._write_one(_make_synthetic_snapshot([4, 5, 6], key))
            l2._write_one(_make_synthetic_snapshot([7, 8, 9], key))
            on_disk = sum(p.stat().st_size for p in _all_snapshot_files(tmp_path))
            assert l2.stats()["current_bytes"] == on_disk
            l2.clear()
            assert l2.stats()["current_bytes"] == 0
        finally:
            l2.shutdown()

    def test_clear_failure_recomputes_tracked_bytes(self, tmp_path, monkeypatch):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            l2._write_one(_make_synthetic_snapshot([4, 5, 6], key))
            before = l2.stats()["current_bytes"]
            assert before > 0
            assert len(_all_snapshot_files(tmp_path)) == 2

            def _fail_rmtree(_path):
                _all_snapshot_files(tmp_path)[0].unlink()
                raise OSError("cannot remove l2 cache")

            monkeypatch.setattr("dflash_mlx.cache.prefix_l2.shutil.rmtree", _fail_rmtree)

            try:
                l2.clear()
            except OSError as exc:
                assert "cannot remove l2 cache" in str(exc)
            else:
                raise AssertionError("clear() should surface failed L2 deletion")

            remaining = _all_snapshot_files(tmp_path)
            assert len(remaining) == 1
            assert l2.stats()["current_bytes"] == sum(p.stat().st_size for p in remaining)
        finally:
            l2.shutdown()

    def test_tracked_bytes_decremented_on_corrupt_unlink(self, tmp_path):
        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            key = _make_key()
            l2._write_one(_make_synthetic_snapshot([1, 2, 3], key))
            initial = l2.stats()["current_bytes"]
            assert initial > 0
            assert initial == sum(p.stat().st_size for p in _all_snapshot_files(tmp_path))

            bucket = _bucket_dir(tmp_path, key)
            corrupt_name = _format_filename(
                token_len=2,
                token_hash=_token_hash([1, 2]),
                kind="prefill",
                fp_short="0" * 16,
            )
            corrupt = bucket / corrupt_name
            corrupt.write_bytes(b"deadbeef" * 100)
            after_inject = l2.stats()["current_bytes"]
            after_inject_disk = sum(p.stat().st_size for p in _all_snapshot_files(tmp_path))
            assert after_inject_disk > after_inject

            l2.lookup((1, 2), key)
            assert not corrupt.exists()
            after = l2.stats()["current_bytes"]

            assert after < after_inject_disk
            assert after == sum(p.stat().st_size for p in _all_snapshot_files(tmp_path))
        finally:
            l2.shutdown()

class TestAsyncWriterSafety:
    def test_async_writer_does_not_call_mx_save_safetensors(self, tmp_path, monkeypatch):
        import mlx.core as _mx

        l2 = DFlashPrefixL2Cache(cache_dir=tmp_path, max_bytes=10**9)
        try:
            _stop_writer(l2)
            l2._stop.clear()

            calls = {"n": 0}
            threads: list[str] = []
            real_save = _mx.save_safetensors

            def _spy(path, arrays, metadata=None):
                calls["n"] += 1
                threads.append(threading.current_thread().name)
                return real_save(path, arrays, metadata=metadata)

            monkeypatch.setattr("mlx.core.save_safetensors", _spy)

            key = _make_key()
            l2.insert_async(_make_synthetic_snapshot([1, 2, 3], key))
            assert l2._write_queue.qsize() == 1
            assert calls["n"] == 1
            assert "dflash-l2-writer" not in threads

            payload = l2._write_queue.get_nowait()
            l2._write_payload(payload)
            assert calls["n"] == 1
            assert len(_all_snapshot_files(tmp_path)) == 1
        finally:
            l2.shutdown()
