# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

from dflash_mlx.cache.fingerprints import DFlashPrefixKey

def build_prefix_key(
    model_provider: Any,
    draft_model: Any,
    runtime_context: Optional[Any] = None,
) -> DFlashPrefixKey:
    model_key = getattr(model_provider, "model_key", None) or ("", None, "")
    target_id = str(model_key[0]) if len(model_key) > 0 else ""
    draft_id = (
        str(model_key[2]) if len(model_key) > 2 and model_key[2] is not None else ""
    )
    capture_ids = tuple(
        int(x) for x in getattr(draft_model, "target_layer_ids", ()) or ()
    )
    if runtime_context is not None:
        runtime_config = runtime_context.runtime
        sink = int(getattr(runtime_config, "draft_sink_size", 64))
        window = int(getattr(runtime_config, "draft_window_size", 1024))
    else:
        sink = 64
        window = 1024
    target_fa_window = (
        runtime_context.runtime.target_fa_window
        if runtime_context is not None
        else 0
    )
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
) -> tuple[Optional[int], Optional[int]]:
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
