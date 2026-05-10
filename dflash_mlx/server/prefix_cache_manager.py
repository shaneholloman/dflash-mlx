# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any

from dflash_mlx.cache.fingerprints import DFlashPrefixKey

def build_prefix_key(
    model_provider: Any,
    draft_model: Any,
    runtime_context: Any,
) -> DFlashPrefixKey:
    model_key = getattr(model_provider, "model_key", None)
    if not isinstance(model_key, (list, tuple)) or len(model_key) < 3:
        raise ValueError("model provider is missing a loaded model_key")
    target_id = str(model_key[0]) if model_key[0] is not None else ""
    draft_id = str(model_key[2]) if model_key[2] is not None else ""
    if not target_id or not draft_id:
        raise ValueError("prefix cache requires loaded target and draft model ids")

    raw_capture_ids = getattr(draft_model, "target_layer_ids", None)
    if raw_capture_ids is None:
        raise ValueError("draft model is missing target_layer_ids")
    try:
        capture_ids = tuple(int(x) for x in raw_capture_ids)
    except (TypeError, ValueError) as exc:
        raise ValueError("draft model target_layer_ids must be integers") from exc
    if not capture_ids:
        raise ValueError("draft model target_layer_ids must not be empty")

    try:
        runtime_config = runtime_context.runtime
        sink = int(runtime_config.draft_sink_size)
        window = int(runtime_config.draft_window_size)
        target_fa_window = int(runtime_config.target_fa_window)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("runtime context is missing prefix cache identity fields") from exc

    return DFlashPrefixKey(
        target_model_id=target_id,
        draft_model_id=draft_id,
        capture_layer_ids=capture_ids,
        draft_sink_size=int(sink),
        draft_window_size=int(window),
        target_fa_window=int(target_fa_window),
    )

def chat_template_marker_ids(
    tokenizer: Any,
) -> tuple[int | None, int | None]:
    im_start = None
    assistant = None
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return im_start, assistant
    try:
        ids = convert(["<|im_start|>", "assistant"])
    except Exception as exc:
        raise RuntimeError("tokenizer failed to resolve chat template marker ids") from exc
    if not isinstance(ids, (list, tuple)):
        return im_start, assistant
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if ids and ids[0] is not None and ids[0] != unk_token_id:
        im_start = int(ids[0])
    if ids and len(ids) > 1 and ids[1] is not None and ids[1] != unk_token_id:
        assistant = int(ids[1])
    return im_start, assistant
