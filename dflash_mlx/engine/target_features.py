# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from dflash_mlx.engine.prefill import init_target_hidden_from_snapshot


@dataclass
class TargetFeatureStore:
    prompt_len: int
    _current_hidden: mx.array | None = None
    _prefill_hidden_for_snapshot: mx.array | None = None
    _generation_chunks: list[mx.array] = field(default_factory=list)

    @property
    def current_hidden(self) -> mx.array | None:
        return self._current_hidden

    def require_current_hidden(self) -> mx.array:
        if self._current_hidden is None:
            raise RuntimeError("target hidden features are unavailable")
        return self._current_hidden

    @property
    def generation_chunks(self) -> tuple[mx.array, ...]:
        return tuple(self._generation_chunks)

    def hydrate_from_snapshot(
        self,
        prefix_snapshot: Any,
        *,
        snap_prefix_len: int,
    ) -> mx.array:
        self._current_hidden = init_target_hidden_from_snapshot(
            prefix_snapshot,
            snap_prefix_len=int(snap_prefix_len),
            prompt_len=int(self.prompt_len),
        )
        return self._current_hidden

    def write_prompt_slice(
        self,
        *,
        start: int,
        end: int,
        features: mx.array,
    ) -> mx.array:
        if self._current_hidden is None:
            self._current_hidden = mx.zeros(
                (features.shape[0], int(self.prompt_len), features.shape[-1]),
                dtype=features.dtype,
            )
        self._current_hidden[:, int(start):int(end), :] = features
        mx.eval(self._current_hidden)
        return self._current_hidden

    def prefix_view(self, boundary: int) -> mx.array | None:
        if self._current_hidden is None:
            return None
        return self._current_hidden[:, :int(boundary), :]

    def freeze_prefill_for_snapshot(self, *, enabled: bool) -> None:
        self._prefill_hidden_for_snapshot = self._current_hidden if enabled else None

    def commit_generation(
        self,
        committed_hidden: mx.array,
        *,
        collect_snapshot: bool,
    ) -> None:
        self._current_hidden = committed_hidden
        if collect_snapshot:
            self._generation_chunks.append(committed_hidden)

    def generation_snapshot_hidden(self) -> mx.array | None:
        if self._prefill_hidden_for_snapshot is None or not self._generation_chunks:
            return None
        gen_hidden = (
            self._generation_chunks[0]
            if len(self._generation_chunks) == 1
            else mx.concatenate(self._generation_chunks, axis=1)
        )
        hidden = mx.concatenate([self._prefill_hidden_for_snapshot, gen_hidden], axis=1)
        mx.eval(hidden)
        return hidden
