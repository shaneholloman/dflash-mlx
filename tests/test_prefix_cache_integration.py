# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from dflash_mlx.cache.codecs import (
    build_snapshot,
    hydrate_target_cache,
    target_cache_is_serializable,
)
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.manager import RuntimeCacheManager
from dflash_mlx.cache.prefix_l1 import DFlashPrefixCache
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache
from dflash_mlx.runtime_context import build_runtime_context, runtime_config_from_profile

def _runtime_context(**overrides):
    values = dict(
        profile="balanced",
        prefix_cache=True,
        prefix_cache_l2=False,
        prefix_cache_l2_dir="/tmp/dflash-prefix-l2-test",
    )
    values.update(overrides)
    return build_runtime_context(runtime_config_from_profile(**values))

def _make_populated_target_cache(n_tokens: int = 8):
    caches = []

    kv = KVCache()
    kv.keys = mx.arange(1 * 2 * n_tokens * 4, dtype=mx.float32).reshape(1, 2, n_tokens, 4)
    kv.values = (mx.arange(1 * 2 * n_tokens * 4, dtype=mx.float32) + 100.0).reshape(1, 2, n_tokens, 4)
    kv.offset = n_tokens
    caches.append(kv)

    gdn = RecurrentRollbackCache(size=3, conv_kernel_size=4)
    gdn.cache[0] = mx.arange(12, dtype=mx.float32).reshape(1, 4, 3)
    gdn.cache[1] = (mx.arange(24, dtype=mx.float32) + 1000.0).reshape(1, 2, 4, 3)
    caches.append(gdn)
    mx.eval(kv.keys, kv.values, gdn.cache[0], gdn.cache[1])
    return caches

def _make_key(**overrides) -> DFlashPrefixKey:
    base = dict(
        target_model_id="test/target",
        draft_model_id="test/draft",
        capture_layer_ids=(10, 20),
        draft_sink_size=64,
        draft_window_size=1024,
    )
    base.update(overrides)
    return DFlashPrefixKey(**base)

def _simulate_serve_insert(
    cache: DFlashPrefixCache,
    key: DFlashPrefixKey,
    prompt_tokens: list[int],
    n_cached_tokens: int,
):
    live_cache = _make_populated_target_cache(n_tokens=n_cached_tokens)
    hidden_dim = 6
    target_hidden = mx.zeros((1, len(prompt_tokens), hidden_dim), dtype=mx.float32)

    target_hidden = mx.arange(1 * len(prompt_tokens) * hidden_dim, dtype=mx.float32).reshape(
        1, len(prompt_tokens), hidden_dim
    )
    last_logits = mx.arange(32, dtype=mx.float32).reshape(1, 32)
    mx.eval(target_hidden, last_logits)

    assert target_cache_is_serializable(live_cache)
    snap = build_snapshot(
        token_ids=prompt_tokens,
        target_cache=live_cache,
        target_hidden=target_hidden,
        last_logits=last_logits,
        key=key,
    )
    cache.insert(snap)
    return snap, live_cache

class TestEndToEnd:
    def test_insert_then_lookup_exact(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        snap, _ = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=8)

        matched, found = cache.lookup(prompt, key)
        assert matched == len(prompt)
        assert found is snap

    def test_insert_then_lookup_prefix(self):

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt_turn1 = [100, 101, 102, 103, 104]
        _simulate_serve_insert(cache, key, prompt_turn1, n_cached_tokens=5)

        prompt_turn2 = prompt_turn1 + [200, 201, 202]
        matched, found = cache.lookup(prompt_turn2, key)
        assert matched == 5
        assert found is not None

    def test_hydrate_after_lookup_preserves_state(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [1, 2, 3, 4]
        snap, live_cache = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=4)

        _, found = cache.lookup(prompt, key)
        assert found is snap

        template = _make_populated_target_cache(n_tokens=1)
        hydrated = hydrate_target_cache(found, template)

        src_k, src_v = live_cache[0].state
        h_k, h_v = hydrated[0].state
        assert mx.all(src_k == h_k).item()
        assert mx.all(src_v == h_v).item()
        assert hydrated[0].offset == live_cache[0].offset

        for a, b in zip(live_cache[1].cache, hydrated[1].cache):
            if a is None:
                assert b is None
            else:
                assert mx.all(a == b).item()

    def test_mutation_of_live_cache_does_not_affect_snapshot(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [7, 8, 9]
        snap, live_cache = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=3)

        live_cache[0].keys = mx.zeros_like(live_cache[0].keys)
        live_cache[1].cache[1] = mx.zeros_like(live_cache[1].cache[1])
        mx.eval(live_cache[0].keys, live_cache[1].cache[1])

        _, found = cache.lookup(prompt, key)
        assert found is not None
        snap_k, _, _ = found.fa_states[0]
        assert not mx.all(snap_k == 0).item()
        assert not mx.all(found.gdn_states[1][1] == 0).item()

    def test_different_key_refuses_match(self):
        cache = DFlashPrefixCache(max_entries=4)
        key_a = _make_key(target_model_id="model-a")
        key_b = _make_key(target_model_id="model-b")
        prompt = [1, 2, 3]
        _simulate_serve_insert(cache, key_a, prompt, n_cached_tokens=3)

        matched, found = cache.lookup(prompt, key_b)
        assert matched == 0
        assert found is None

    def test_cache_handles_monotone_turns_via_pruning(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        _simulate_serve_insert(cache, key, [1, 2], n_cached_tokens=2)
        _simulate_serve_insert(cache, key, [1, 2, 3], n_cached_tokens=3)
        _simulate_serve_insert(cache, key, [1, 2, 3, 4], n_cached_tokens=4)
        stats = cache.stats()

        assert stats["current_entries"] == 1
        assert stats["prefix_prunes"] == 2
        assert stats["evictions"] == 0
        matched, _ = cache.lookup([1, 2, 3, 4], key)
        assert matched == 4

    def test_cache_lru_eviction_on_unrelated_turns(self):
        cache = DFlashPrefixCache(max_entries=2)
        key = _make_key()
        _simulate_serve_insert(cache, key, [1, 2], n_cached_tokens=2)
        _simulate_serve_insert(cache, key, [9, 8], n_cached_tokens=2)
        _simulate_serve_insert(cache, key, [5, 6, 7], n_cached_tokens=3)
        stats = cache.stats()
        assert stats["current_entries"] == 2
        assert stats["evictions"] >= 1
        assert stats["prefix_prunes"] == 0
        matched, _ = cache.lookup([5, 6, 7], key)
        assert matched == 3

class TestServeHelperShapes:

    def test_get_cache_disabled(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        assert cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=False)
        ) is None

    def test_get_cache_enabled_returns_singleton(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        context = _runtime_context(prefix_cache=True)
        first = cache_manager_mod.get_runtime_cache_manager(context)
        second = cache_manager_mod.get_runtime_cache_manager(context)
        assert first is not None
        assert first is second

    def test_build_prefix_key_from_provider(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        class FakeProvider:
            model_key = ("target/x", None, "draft/y")

        class FakeDraft:
            target_layer_ids = [3, 7]

        key = build_prefix_key(FakeProvider(), FakeDraft())
        assert key.target_model_id == "target/x"
        assert key.draft_model_id == "draft/y"
        assert key.capture_layer_ids == (3, 7)
        assert isinstance(key.draft_sink_size, int)
        assert isinstance(key.draft_window_size, int)

    def test_build_prefix_key_uses_runtime_draft_window(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        class FakeProvider:
            model_key = ("target/x", None, "draft/y")

        class FakeDraft:
            target_layer_ids = [3, 7]

        key_default = build_prefix_key(
            FakeProvider(),
            FakeDraft(),
            _runtime_context(draft_sink_size=64, draft_window_size=1024),
        )
        key_windowed = build_prefix_key(
            FakeProvider(),
            FakeDraft(),
            _runtime_context(draft_sink_size=32, draft_window_size=512),
        )

        assert key_default.draft_sink_size == 64
        assert key_default.draft_window_size == 1024
        assert key_windowed.draft_sink_size == 32
        assert key_windowed.draft_window_size == 512
        assert key_default != key_windowed

    def test_build_prefix_key_handles_missing(self):
        from dflash_mlx.server.prefix_cache_manager import build_prefix_key

        class FakeProvider:
            pass

        class FakeDraft:
            target_layer_ids = ()

        key = build_prefix_key(FakeProvider(), FakeDraft())
        assert key.target_model_id == ""
        assert key.draft_model_id == ""
        assert key.capture_layer_ids == ()

class TestContextConfigExposedCorrectly:

    def test_prefix_cache_config_affects_singleton(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        assert cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=False)
        ) is None

        assert cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True)
        ) is not None

    def test_singleton_rebuilds_when_config_changes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        first = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=2)
        )
        second = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=7)
        )

        assert first is not None
        assert second is not None
        assert first is not second
        assert second.stats()["max_entries"] == 7

    def test_singleton_rebuild_shutdowns_previous_cache(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        shutdown_calls: list[DFlashPrefixCache] = []
        original_shutdown = DFlashPrefixCache.shutdown

        def tracked_shutdown(self):
            shutdown_calls.append(self)
            return original_shutdown(self)

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        monkeypatch.setattr(DFlashPrefixCache, "shutdown", tracked_shutdown)

        first = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=2)
        )
        second = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(prefix_cache=True, prefix_cache_max_entries=7)
        )

        assert first is not None
        assert second is not None
        assert first is not second
        assert len(shutdown_calls) == 1

    def test_singleton_rebuild_fails_when_previous_cache_shutdown_fails(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        class BrokenShutdownCache(DFlashPrefixCache):
            def shutdown(self):
                raise RuntimeError("cannot close")

        old_manager = RuntimeCacheManager(BrokenShutdownCache(max_entries=2))
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", old_manager)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", ("old",))

        with pytest.raises(RuntimeError, match="cannot close"):
            cache_manager_mod.get_runtime_cache_manager(_runtime_context(prefix_cache=True))

        assert cache_manager_mod.current_runtime_cache_manager() is old_manager

    def test_shutdown_runtime_cache_manager_shutdowns_current_cache(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        shutdown_calls: list[DFlashPrefixCache] = []
        original_shutdown = DFlashPrefixCache.shutdown

        def tracked_shutdown(self):
            shutdown_calls.append(self)
            return original_shutdown(self)

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        monkeypatch.setattr(DFlashPrefixCache, "shutdown", tracked_shutdown)

        manager = cache_manager_mod.get_runtime_cache_manager(_runtime_context(prefix_cache=True))
        assert manager is not None

        cache_manager_mod.shutdown_runtime_cache_manager()

        assert len(shutdown_calls) == 1
        assert cache_manager_mod.current_runtime_cache_manager() is None

    def test_shutdown_runtime_cache_manager_logs_failure(self, monkeypatch, capsys):
        import dflash_mlx.cache.manager as cache_manager_mod

        class BrokenManager:
            def shutdown(self):
                raise RuntimeError("broken close")

        monkeypatch.setattr(
            cache_manager_mod,
            "_DFLASH_RUNTIME_CACHE_MANAGER",
            BrokenManager(),
        )
        monkeypatch.setattr(
            cache_manager_mod,
            "_DFLASH_RUNTIME_CACHE_CONFIG_KEY",
            ("old",),
        )

        cache_manager_mod.shutdown_runtime_cache_manager()

        assert "runtime cache manager shutdown failed: broken close" in capsys.readouterr().err
        assert cache_manager_mod.current_runtime_cache_manager() is None

    def test_chat_template_marker_ids_returns_none_without_converter(self):
        from dflash_mlx.server.prefix_cache_manager import chat_template_marker_ids

        assert chat_template_marker_ids(object()) == (None, None)

    def test_chat_template_marker_ids_raises_on_tokenizer_failure(self):
        from dflash_mlx.server.prefix_cache_manager import chat_template_marker_ids

        class BrokenTokenizer:
            unk_token_id = -1

            def convert_tokens_to_ids(self, _tokens):
                raise RuntimeError("tokenizer closed")

        with pytest.raises(RuntimeError, match="chat template marker ids"):
            chat_template_marker_ids(BrokenTokenizer())

    def test_budgets_propagate_to_cache(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        manager = cache_manager_mod.get_runtime_cache_manager(
            _runtime_context(
                prefix_cache=True,
                prefix_cache_max_entries=7,
                prefix_cache_max_bytes=1234567,
            )
        )
        assert manager is not None
        stats = manager.stats()
        assert stats["max_entries"] == 7
        assert stats["max_bytes"] == 1234567

    def test_manager_exposes_memory_waterfall_bytes_without_raw_cache(self):
        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [51, 52, 53]
        snap, _ = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=3)
        manager = RuntimeCacheManager(cache)

        memory = manager.memory_waterfall_bytes()

        assert not hasattr(manager, "prefix_cache_for_memory")
        assert memory["l1_snapshot_bytes"] == snap.nbytes
        assert memory["l1_snapshot_target_hidden_bytes"] > 0
        assert memory["l1_snapshot_last_logits_bytes"] > 0

    def test_singleton_rebuilds_when_identity_changes(self, monkeypatch):
        import dflash_mlx.cache.manager as cache_manager_mod

        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_MANAGER", None)
        monkeypatch.setattr(cache_manager_mod, "_DFLASH_RUNTIME_CACHE_CONFIG_KEY", None)
        context = _runtime_context(prefix_cache=True)

        first = cache_manager_mod.get_runtime_cache_manager(context, cache_identity="model-a")
        second = cache_manager_mod.get_runtime_cache_manager(context, cache_identity="model-b")

        assert first is not None
        assert second is not None
        assert first is not second

    def test_prefix_flow_lookup_records_hit(self, monkeypatch):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        class FakeProvider:
            model_key = ("target/x", None, "draft/y")

        class FakeDraft:
            target_layer_ids = [3, 7]

        class FakeTokenizer:
            unk_token_id = -1

            def convert_tokens_to_ids(self, tokens):
                return [-1 for _ in tokens]

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key(
            target_model_id="target/x",
            draft_model_id="draft/y",
            capture_layer_ids=(3, 7),
        )
        prompt = [11, 12, 13, 14]
        snap, _ = _simulate_serve_insert(cache, key, prompt, n_cached_tokens=4)

        monkeypatch.setattr(
            flow_mod,
            "get_runtime_cache_manager",
            lambda _runtime_context, *, cache_identity=None: RuntimeCacheManager(cache),
        )
        monkeypatch.setattr(flow_mod, "build_prefix_key", lambda *_args: key)

        flow = flow_mod.PrefixCacheFlow.for_request(
            model_provider=FakeProvider(),
            draft_model=FakeDraft(),
            tokenizer=FakeTokenizer(),
            prompt=prompt,
            runtime_context=_runtime_context(prefix_cache=True),
        )

        assert flow.cache_active
        assert flow.hit_tokens == len(prompt)
        assert flow.snapshot is snap
        assert flow.stable_prefix_len == len(prompt)

    def test_prefix_flow_inserts_built_snapshot(self):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [21, 22, 23]
        target_hidden = mx.arange(1 * len(prompt) * 6, dtype=mx.float32).reshape(
            1, len(prompt), 6
        )
        snap = build_snapshot(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            key=key,
        )
        flow = flow_mod.PrefixCacheFlow(cache_manager=RuntimeCacheManager(cache), key=key)

        flow.handle_prefill_snapshot(snap)

        matched, found = cache.lookup(prompt, key)
        assert matched == len(prompt)
        assert found is snap
        assert flow.insert_ms >= 0.0

    def test_prefix_flow_inserts_generation_snapshot_as_prefix_only(self):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        prompt = [41, 42, 43]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        snap = build_snapshot(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=None,
            key=key,
            kind="generation",
        )
        flow = flow_mod.PrefixCacheFlow(cache_manager=RuntimeCacheManager(cache), key=key)

        flow.handle_generation_snapshot(snap)

        exact_len, exact = cache.lookup(prompt, key)
        assert exact_len == 0
        assert exact is None

        prefix_len, prefix = cache.lookup(prompt + [99], key)
        assert prefix_len == len(prompt)
        assert prefix is snap

    def test_prefix_flow_raises_on_invalid_snapshot_contract(self):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        cache = DFlashPrefixCache(max_entries=4)
        key = _make_key()
        wrong_key = _make_key(target_model_id="other/target")
        prompt = [31, 32, 33]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        wrong_snap = build_snapshot(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            key=wrong_key,
        )
        flow = flow_mod.PrefixCacheFlow(cache_manager=RuntimeCacheManager(cache), key=key)

        with pytest.raises(TypeError, match="expected DFlashPrefixSnapshot"):
            flow.handle_prefill_snapshot(
                {
                    "target_cache": _make_populated_target_cache(n_tokens=len(prompt)),
                    "target_hidden": target_hidden,
                    "last_logits": mx.zeros((1, 32), dtype=mx.float32),
                    "token_ids": prompt,
                }
            )
        with pytest.raises(ValueError, match="snapshot key"):
            flow.handle_prefill_snapshot(wrong_snap)

        matched, found = cache.lookup(prompt, key)
        assert matched == 0
        assert found is None

    def test_prefix_flow_raises_and_logs_cache_insert_failure(self, capsys):
        import dflash_mlx.server.prefix_cache_flow as flow_mod

        class BrokenInsertCache(DFlashPrefixCache):
            def insert(self, snapshot):
                raise RuntimeError("insert failed")

        cache = BrokenInsertCache(max_entries=4)
        key = _make_key()
        prompt = [51, 52, 53]
        target_hidden = mx.zeros((1, len(prompt), 6), dtype=mx.float32)
        snap = build_snapshot(
            token_ids=prompt,
            target_cache=_make_populated_target_cache(n_tokens=len(prompt)),
            target_hidden=target_hidden,
            last_logits=mx.zeros((1, 32), dtype=mx.float32),
            key=key,
        )
        flow = flow_mod.PrefixCacheFlow(cache_manager=RuntimeCacheManager(cache), key=key)

        with pytest.raises(RuntimeError, match="insert failed"):
            flow.handle_prefill_snapshot(snap)

        assert "prefix cache insert failed: insert failed" in capsys.readouterr().err
