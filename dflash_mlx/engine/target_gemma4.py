# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import time
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models import cache as cache_mod

from dflash_mlx.engine.target_ops import TargetCapabilities


def _model_args(model: Any) -> Any:
    args = getattr(model, "args", None)
    if args is not None:
        return args
    if hasattr(model, "language_model"):
        args = getattr(model.language_model, "args", None)
        if args is not None:
            return args
    return None


def _logit_softcap(logits: mx.array, softcap: Optional[float]) -> mx.array:
    if softcap is None:
        return logits
    cap = float(softcap)
    if cap <= 0.0:
        return logits
    return mx.tanh(logits / cap) * cap


def _trim_recent_cache(cache_entry: Any, n: int) -> int:
    n = max(0, int(n))
    if n <= 0:
        return 0

    offset = int(getattr(cache_entry, "offset", 0) or 0)
    n = min(offset, n)
    if n <= 0:
        return 0

    if isinstance(cache_entry, cache_mod.RotatingKVCache):
        keys = getattr(cache_entry, "keys", None)
        values = getattr(cache_entry, "values", None)
        if keys is not None and values is not None:
            keys = cache_entry._temporal_order(keys)
            values = cache_entry._temporal_order(values)
            keep_len = max(0, int(keys.shape[2]) - n)
            cache_entry.keys = keys[..., :keep_len, :]
            cache_entry.values = values[..., :keep_len, :]
            cache_entry._idx = keep_len
        else:
            cache_entry._idx = max(0, int(getattr(cache_entry, "_idx", 0) or 0) - n)
        cache_entry.offset = offset - n
        return n

    if hasattr(cache_entry, "trim"):
        return int(cache_entry.trim(n))
    if hasattr(cache_entry, "offset"):
        cache_entry.offset = offset - n
        return n
    return 0


class Gemma4TargetOps:
    backend_name = "gemma4"

    def model_type(self, target_model: Any) -> str:
        args = _model_args(target_model)
        value = getattr(args, "model_type", None)
        if value is not None:
            return str(value).lower()
        if hasattr(target_model, "language_model"):
            value = getattr(
                getattr(target_model.language_model, "args", None),
                "model_type",
                None,
            )
            if value is not None:
                return str(value).lower()
        config = getattr(target_model, "config", None)
        if isinstance(config, dict):
            text_config = config.get("text_config", config)
            return str(text_config.get("model_type", config.get("model_type", ""))).lower()
        return ""

    def supports_model(self, target_model: Any) -> bool:
        model_type = self.model_type(target_model)
        if model_type not in ("gemma4", "gemma4_text"):
            return False
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        args = getattr(self.text_wrapper(target_model), "args", None)
        return (
            hasattr(inner, "layers")
            and hasattr(inner, "embed_tokens")
            and hasattr(args, "layer_types")
        )

    def family(self, target_model: Any) -> str:
        return "gemma4_swa"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        args = getattr(self.text_wrapper(target_model), "args", None)
        shared_kv_layers_raw = getattr(args, "num_kv_shared_layers", None)
        if type(shared_kv_layers_raw) is int:
            shared_kv_layers = shared_kv_layers_raw
        else:
            shared_kv_layers = -1
        shared_kv_known = shared_kv_layers >= 0
        shared_kv = bool(shared_kv_known and shared_kv_layers > 0)
        prefix_snapshot_safe = bool(shared_kv_known and shared_kv_layers == 0)
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=False,
            supports_kv_trim=True,
            supports_prefix_snapshot=prefix_snapshot_safe,
            supports_rotating_cache_snapshot=prefix_snapshot_safe,
            supports_shared_kv=shared_kv,
            supports_target_hidden_capture=True,
            supports_verify_linear=True,
            supports_full_context_draft_layers=True,
        )

    def text_wrapper(self, target_model: Any) -> Any:
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        if hasattr(target_model, "model"):
            return target_model
        raise AttributeError(f"Unsupported Gemma4 model wrapper: {type(target_model)!r}")

    def text_model(self, target_model: Any) -> Any:
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "model"):
            return wrapper.model
        raise AttributeError(f"Unsupported Gemma4 text model: {type(wrapper)!r}")

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(self, target_model: Any, hidden_states: mx.array) -> mx.array:
        wrapper = self.text_wrapper(target_model)
        inner = self.text_model(target_model)
        tied = getattr(
            wrapper,
            "tie_word_embeddings",
            getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True),
        )
        if bool(tied):
            logits = inner.embed_tokens.as_linear(hidden_states)
        else:
            logits = wrapper.lm_head(hidden_states)
        softcap = getattr(
            wrapper,
            "final_logit_softcapping",
            getattr(getattr(wrapper, "args", None), "final_logit_softcapping", None),
        )
        return _logit_softcap(
            logits,
            softcap,
        )

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
    ) -> list[Any]:
        if quantize_kv_cache:
            raise ValueError("Gemma4 target KV quantization is not supported yet")
        if target_fa_window is not None and int(target_fa_window) > 0:
            raise ValueError("Gemma4 uses its config-defined SWA/full attention cache")
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "make_cache"):
            return wrapper.make_cache()
        if hasattr(target_model, "make_cache"):
            return target_model.make_cache()
        raise AttributeError("Gemma4 target must expose make_cache()")

    def configure_full_attention_split(
        self,
        target_model: Any,
        *,
        enabled: bool,
        chunk_size: int = 8,
    ) -> None:
        return None

    def install_speculative_hooks(self, target_model: Any) -> None:
        return None

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        inner = self.text_model(target_model)
        if input_embeddings is None:
            input_embeddings = inner.embed_tokens(input_ids)
        h = input_embeddings * getattr(inner, "embed_scale", 1.0)

        per_layer_inputs = [None] * len(inner.layers)
        if getattr(inner, "hidden_size_per_layer_input", 0):
            pli = inner._get_per_layer_inputs(input_ids, input_embeddings)
            pli = inner._project_per_layer_inputs(h, pli)
            per_layer_inputs = [pli[:, :, i, :] for i, _ in enumerate(inner.layers)]

        if cache is None:
            cache = [None] * len(inner.layers)
        else:
            cache = list(cache) + [None] * (len(inner.layers) - len(cache))

        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [h]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: h} if 0 in capture_layer_ids else {}

        masks = inner._make_masks(h, cache)
        intermediates = [(None, None)] * len(inner.layers)
        for idx, (layer, layer_cache, mask, prev_idx, per_layer_input) in enumerate(
            zip(
                inner.layers,
                cache,
                masks,
                inner.previous_kvs,
                per_layer_inputs,
                strict=True,
            )
        ):
            shared_kv, offset = intermediates[prev_idx]
            h, shared_kv, offset = layer(
                h,
                mask,
                layer_cache,
                per_layer_input=per_layer_input,
                shared_kv=shared_kv,
                offset=offset,
            )
            capture_key = idx + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h
            intermediates[idx] = (shared_kv, offset)

        normalized = inner.norm(h)
        if logits_last_only and isinstance(captured, dict):
            captured[-1] = normalized
        logits_hidden = normalized[:, -1:, :] if logits_last_only else normalized
        logits = self.logits_from_hidden(target_model, logits_hidden)
        return logits, captured

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        if int(verify_ids.shape[1]) <= 0:
            raise ValueError("verify block must contain at least one token")
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array] | list[mx.array],
        target_layer_ids: list[int],
    ) -> mx.array:
        selected = [captured_dict[int(layer_id) + 1] for layer_id in target_layer_ids]
        return mx.concatenate(selected, axis=-1)

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        return None

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        replay_ns_total = 0
        for cache_entry in cache_entries:
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset <= target_len:
                continue
            trim_n = offset - int(target_len)
            replay_start_ns = time.perf_counter_ns()
            _trim_recent_cache(cache_entry, trim_n)
            replay_ns_total += time.perf_counter_ns() - replay_start_ns
        return replay_ns_total

    def cleanup_generation_caches(
        self,
        target_cache: list[Any],
        draft_cache: list[Any],
    ) -> None:
        draft_cache.clear()
        target_cache.clear()
