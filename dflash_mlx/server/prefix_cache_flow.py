# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

from dflash_mlx.cache.manager import (
    RuntimeCacheManager,
    get_runtime_cache_manager,
)
from dflash_mlx.cache.codecs import PrefixSnapshotBuilder
from dflash_mlx.cache.fingerprints import DFlashPrefixKey
from dflash_mlx.cache.snapshot import DFlashPrefixSnapshot
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
    insert_ms: float = 0.0
    snapshot_builder: Optional[PrefixSnapshotBuilder] = None

    @property
    def cache_active(self) -> bool:
        return self.cache_manager is not None

    def prefix_cache_memory_bytes(self) -> Optional[dict[str, int]]:
        return (
            None
            if self.cache_manager is None
            else self.cache_manager.memory_waterfall_bytes()
        )

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
        lookup = cache_manager.lookup(lookup_tokens, key)
        hit_tokens = int(lookup.matched_tokens)
        if lookup.matched_tokens > 0:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] prefix cache hit "
                f"{hit_tokens}/{len(prompt)} tokens (stable prefix {stable_prefix_len})\n"
            )
            sys.stderr.flush()
        cache_manager.log_stats(label="lookup")
        return cls(
            cache_manager=cache_manager,
            key=key,
            stable_prefix_len=stable_prefix_len,
            snapshot=lookup.snapshot,
            lookup_ms=lookup.elapsed_ms,
            hit_tokens=hit_tokens,
            snapshot_builder=PrefixSnapshotBuilder(
                key=key,
                draft_model=draft_model,
                draft_sink_size=int(runtime_context.runtime.draft_sink_size),
                draft_window_size=int(runtime_context.runtime.draft_window_size),
            ),
        )

    def handle_prefill_snapshot(self, snapshot: DFlashPrefixSnapshot) -> None:
        self._insert_snapshot(snapshot, kind="prefill", require_logits=True)

    def handle_generation_snapshot(self, snapshot: DFlashPrefixSnapshot) -> None:
        self._insert_snapshot(snapshot, kind="generation", require_logits=False)

    def _insert_snapshot(
        self,
        snapshot: DFlashPrefixSnapshot,
        *,
        kind: str,
        require_logits: bool,
    ) -> None:
        if self.cache_manager is None or self.key is None:
            return
        insert_ms = self.cache_manager.maybe_insert_snapshot(
            snapshot,
            key=self.key,
            kind=kind,
            require_logits=require_logits,
        )
        self.insert_ms += insert_ms
