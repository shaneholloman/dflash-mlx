# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

from dflash_mlx.cache.manager import (
    RuntimeCacheManagerClosed,
    RuntimeCacheManager,
    get_runtime_cache_manager,
)
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
from dflash_mlx.cache.snapshot_service import SnapshotService
from dflash_mlx.server.prefix_cache_manager import (
    build_prefix_key,
    chat_template_marker_ids,
)

def compute_stable_prefix_len(
    tokens: list[int] | tuple[int, ...],
    *,
    im_start_id: Optional[int] = None,
    assistant_id: Optional[int] = None,
) -> int:
    if im_start_id is None or assistant_id is None:
        return len(tokens)
    n = len(tokens)
    if n < 2:
        return n
    for i in range(n - 2, -1, -1):
        if tokens[i] == im_start_id and tokens[i + 1] == assistant_id:
            return i
    return n

@dataclass
class PrefixCacheFlow:
    cache_manager: Optional[RuntimeCacheManager]
    key: Optional[DFlashPrefixKey] = None
    stable_prefix_len: Optional[int] = None
    snapshot: Optional[DFlashPrefixSnapshot] = None
    lookup_ms: float = 0.0
    hit_tokens: int = 0
    snapshot_service: Optional[SnapshotService] = None

    @property
    def cache_active(self) -> bool:
        return self.cache_manager is not None

    @property
    def insert_ms(self) -> float:
        if self.snapshot_service is None:
            return 0.0
        return self.snapshot_service.insert_ms

    def prefix_cache_memory_bytes(self) -> Optional[dict[str, int]]:
        if self.cache_manager is None:
            return None
        try:
            return self.cache_manager.memory_waterfall_bytes()
        except RuntimeCacheManagerClosed:
            return None

    @classmethod
    def for_request(
        cls,
        *,
        model_provider: Any,
        draft_model: Any,
        tokenizer: Any,
        prompt: list[int],
        runtime_context: Optional[Any] = None,
    ) -> "PrefixCacheFlow":
        if runtime_context is None:
            return cls(cache_manager=None)

        runtime_config = runtime_context.runtime
        if runtime_config.target_fa_window > 0 or not runtime_config.prefix_cache:
            get_runtime_cache_manager(runtime_context)
            return cls(cache_manager=None)

        key = build_prefix_key(model_provider, draft_model, runtime_context)
        cache_manager = get_runtime_cache_manager(runtime_context, cache_identity=key)
        if cache_manager is None:
            return cls(cache_manager=None)

        im_start_id, assistant_id = chat_template_marker_ids(tokenizer)
        stable_prefix_len = compute_stable_prefix_len(
            prompt,
            im_start_id=im_start_id,
            assistant_id=assistant_id,
        )
        lookup_tokens = prompt[:stable_prefix_len]
        try:
            lookup = cache_manager.lookup(lookup_tokens, key)
        except RuntimeCacheManagerClosed:
            return cls(cache_manager=None)
        hit_tokens = int(lookup.matched_tokens)
        if lookup.matched_tokens > 0:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix cache hit "
                f"{hit_tokens}/{len(prompt)} tokens (stable prefix {stable_prefix_len})\n"
            )
            sys.stderr.flush()
        try:
            cache_manager.log_stats(label="lookup")
        except RuntimeCacheManagerClosed:
            cache_manager = None
        return cls(
            cache_manager=cache_manager,
            key=key,
            stable_prefix_len=stable_prefix_len,
            snapshot=lookup.snapshot,
            lookup_ms=lookup.elapsed_ms,
            hit_tokens=hit_tokens,
            snapshot_service=(
                SnapshotService.from_request(
                    cache_manager=cache_manager,
                    key=key,
                    draft_model=draft_model,
                    runtime_context=runtime_context,
                )
                if cache_manager is not None
                else None
            ),
        )
