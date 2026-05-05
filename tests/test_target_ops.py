# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache

from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

class _FakeLinearAttn:
    conv_kernel_size = 4

class _FakeGdnLayer:
    is_linear = True
    linear_attn = _FakeLinearAttn()

class _FakeFaLayer:
    is_linear = False
    self_attn = object()

class _FakeEmbed:
    pass

class _FakeTarget:
    def __init__(self, *, model_type: str = "qwen3_5") -> None:
        self.args = SimpleNamespace(model_type=model_type, tie_word_embeddings=True)
        self.model = SimpleNamespace(
            layers=[_FakeFaLayer(), _FakeGdnLayer(), _FakeFaLayer()],
            embed_tokens=_FakeEmbed(),
        )

class _FakePureAttentionTarget:
    def __init__(self, *, model_type: str = "qwen3") -> None:
        self.args = SimpleNamespace(model_type=model_type, tie_word_embeddings=True)
        self.model = SimpleNamespace(
            layers=[_FakeFaLayer(), _FakeFaLayer()],
            embed_tokens=_FakeEmbed(),
        )

def test_resolver_selects_qwen_ops_for_current_target_shape():
    ops = resolve_target_ops(_FakeTarget())

    assert isinstance(ops, QwenGdnTargetOps)
    assert ops.family(_FakeTarget()) == "hybrid_gdn"

def test_resolver_rejects_known_unsupported_family_markers():
    for model_type in ("gemma4_text", "llama", "mistral", "olmo"):
        target = _FakeTarget(model_type=model_type)

        try:
            resolve_target_ops(target)
        except NotImplementedError as exc:
            message = str(exc)
            assert f"model_type={model_type}" in message
            assert "model_class=_FakeTarget" in message
            assert "supported target backends: qwen_gdn" in message
        else:
            raise AssertionError(f"{model_type} marker must not resolve to Qwen ops")

def test_resolver_rejects_unknown_model_type():
    try:
        resolve_target_ops(_FakeTarget(model_type="unknown_arch"))
    except NotImplementedError as exc:
        assert "model_type=unknown_arch" in str(exc)
    else:
        raise AssertionError("unknown non-empty model_type must not resolve to Qwen ops")

def test_resolver_rejects_empty_model_type_even_with_qwen_like_shape():
    try:
        resolve_target_ops(_FakeTarget(model_type=""))
    except NotImplementedError as exc:
        assert "model_type=unknown" in str(exc)
    else:
        raise AssertionError("empty model_type must not resolve to Qwen ops")

def test_capabilities_for_distinguishes_hybrid_and_pure_attention():
    ops = QwenGdnTargetOps()

    hybrid = ops.capabilities_for(_FakeTarget())
    pure = ops.capabilities_for(_FakePureAttentionTarget())

    assert hybrid.supports_recurrent_rollback is True
    assert hybrid.supports_dflash is True
    assert hybrid.supports_prefix_snapshot is True
    assert pure.supports_recurrent_rollback is False
    assert pure.supports_dflash is True
    assert pure.supports_kv_trim is True
    assert pure.supports_rotating_cache_snapshot is False
    assert pure.supports_shared_kv is False

def test_qwen_make_cache_matches_current_hybrid_cache_types(monkeypatch):
    monkeypatch.setattr(QwenGdnTargetOps, "install_speculative_hooks", lambda self, _model: None)
    caches = QwenGdnTargetOps().make_cache(
        _FakeTarget(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )

    assert isinstance(caches[0], KVCache)
    assert isinstance(caches[1], RecurrentRollbackCache)
    assert isinstance(caches[2], KVCache)

def test_qwen_make_cache_quantizes_fa_only(monkeypatch):
    monkeypatch.setattr(
        QwenGdnTargetOps,
        "install_speculative_hooks",
        lambda self, _model: None,
    )
    caches = QwenGdnTargetOps().make_cache(
        _FakeTarget(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=True,
        target_fa_window=0,
    )

    assert isinstance(caches[0], QuantizedKVCache)
    assert isinstance(caches[1], RecurrentRollbackCache)
    assert isinstance(caches[2], QuantizedKVCache)

def test_speculative_gdn_hook_materializes_cached_state():
    root = Path(__file__).resolve().parents[1]
    text = (root / "dflash_mlx/engine/target_qwen_gdn.py").read_text()

    assert (
        "cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :])"
        in text
    )
    assert "cache[1] = mx.contiguous(state)" in text

def test_qwen_make_cache_windows_fa_only(monkeypatch):
    monkeypatch.setattr(QwenGdnTargetOps, "install_speculative_hooks", lambda self, _model: None)
    caches = QwenGdnTargetOps().make_cache(
        _FakeTarget(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=2048,
    )

    assert isinstance(caches[0], RotatingKVCache)
    assert isinstance(caches[1], RecurrentRollbackCache)
    assert isinstance(caches[2], RotatingKVCache)

def test_runtime_and_spec_epoch_do_not_import_deleted_target_modules():
    root = Path(__file__).resolve().parents[1]
    runtime_text = (root / "dflash_mlx/runtime.py").read_text()
    spec_text = (root / "dflash_mlx/engine/spec_epoch.py").read_text()

    for text in (runtime_text, spec_text):
        assert "engine.target_verifier" not in text
        assert "engine.rollback" not in text

def test_model_backend_doc_mentions_workflow():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs/model_backend.md").read_text()

    assert "What is a TargetOps backend?" in text
    assert "register in `TARGET_BACKENDS`" in text
    assert "no backend per model size" in text
