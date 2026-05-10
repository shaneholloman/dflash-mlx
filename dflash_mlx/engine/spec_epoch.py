# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import mlx.core as mx

from dflash_mlx.cache.codecs import hydrate_target_cache
from dflash_mlx.cache.snapshot_service import SnapshotPublication, SnapshotService
from dflash_mlx.cache.snapshot import (
    DFlashPrefixSnapshot,
    validate_prefix_snapshot as _validate_prefix_snapshot,
)
from dflash_mlx.draft_backend import DraftBackend
from dflash_mlx.engine.acceptance import match_acceptance_length as _match_acceptance_length
from dflash_mlx.engine.fallback import stream_baseline_generate
from dflash_mlx.engine.prefill import compute_snapshot_boundary
from dflash_mlx.engine.sampling import (
    build_suppress_token_mask,
    eval_logits_and_captured,
    greedy_tokens_with_mask,
    ns_to_us,
    prepare_prompt_tokens,
)
from dflash_mlx.engine.target_features import TargetFeatureStore
from dflash_mlx.engine.config import (
    _profile_dflash_cycles_enabled,
    resolve_draft_window,
    resolve_speculative_cycle_config,
    verify_token_count_for_block,
)
from dflash_mlx.engine.events import (
    CycleCompleteEvent,
    EngineEvent,
    MemoryWaterfallEvent,
    PrefillCompleteEvent,
    PrefillProgressEvent,
    SnapshotPublishedEvent,
    SummaryEvent,
    TokenEvent,
)
from dflash_mlx.model import DFlashDraftModel
from dflash_mlx.engine.memory_waterfall import (
    collect_memory_waterfall as _collect_memory_waterfall,
    memory_waterfall_enabled as _memory_waterfall_enabled,
    should_sample_cycle as _should_sample_memory_cycle,
)


@dataclass(frozen=True)
class _SessionRequest:
    prompt_tokens: tuple[int, ...]
    max_new_tokens: int
    block_tokens: Optional[int] = None
    stop_token_ids: tuple[int, ...] = ()
    suppress_token_ids: Optional[list[int]] = None
    prefix_snapshot: Optional[DFlashPrefixSnapshot] = None
    snapshot_service: Optional[SnapshotService] = None
    stable_prefix_len: Optional[int] = None
    prefix_cache_active: bool = False
    publish_generation_snapshot: bool = True
    prompt_array: mx.array = field(init=False, repr=False)
    prompt_len: int = field(init=False)
    stop_token_array: Optional[mx.array] = field(init=False, repr=False)

    @classmethod
    def from_tokens(
        cls,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        block_tokens: Optional[int],
        stop_token_ids: Optional[list[int]],
        suppress_token_ids: Optional[list[int]],
        prefix_snapshot: Optional[DFlashPrefixSnapshot],
        snapshot_service: Optional[SnapshotService],
        stable_prefix_len: Optional[int],
        prefix_cache_active: bool,
        publish_generation_snapshot: bool = True,
    ) -> "_SessionRequest":
        return cls(
            prompt_tokens=tuple(int(token) for token in prompt_tokens),
            max_new_tokens=int(max_new_tokens),
            block_tokens=block_tokens,
            stop_token_ids=tuple(int(token) for token in (stop_token_ids or ())),
            suppress_token_ids=(
                [int(token) for token in suppress_token_ids]
                if suppress_token_ids is not None
                else None
            ),
            prefix_snapshot=prefix_snapshot,
            snapshot_service=snapshot_service,
            stable_prefix_len=stable_prefix_len,
            prefix_cache_active=bool(prefix_cache_active),
            publish_generation_snapshot=bool(publish_generation_snapshot),
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_len", len(self.prompt_tokens))
        object.__setattr__(
            self,
            "prompt_array",
            mx.array(self.prompt_tokens, dtype=mx.uint32)[None],
        )
        stop_array = (
            mx.array(self.stop_token_ids, dtype=mx.uint32)
            if self.stop_token_ids
            else None
        )
        object.__setattr__(self, "stop_token_array", stop_array)


@dataclass(frozen=True)
class _PrefillResult:
    feature_store: TargetFeatureStore
    start_ns: int
    prefill_ns: int
    suppress_token_mask: mx.array | None
    supports_prefix_snapshot: bool


@dataclass(frozen=True)
class _DecodeResult:
    effective_block_tokens: int
    verify_len_cap: int
    draft_ns_total: int
    draft_prefill_ns: int
    draft_incremental_ns: int
    verify_ns_total: int
    replay_ns_total: int
    commit_ns_total: int
    cycle_profiles: tuple[CycleCompleteEvent, ...]
    profile_totals_ns: dict[str, int]


@dataclass
class SpeculativeSession:
    target_model: Any
    draft_model: DFlashDraftModel
    target_ops: Any
    target_cache: Any
    draft_cache: list[Any]
    draft_backend: DraftBackend
    runtime_config: Any
    quantize_kv_cache: bool
    snap_prefix_len: int
    supports_prefix_snapshot: bool
    allow_full_context_draft_layers: bool
    draft_sink_size: int
    draft_window_size: int
    target_layer_id_list: list[int]
    capture_layer_ids: set[int]
    profile_cycles: bool
    memory_waterfall: bool
    clear_cache_boundaries: bool
    target_fa_window: int

    @classmethod
    def open(
        cls,
        *,
        target_model: Any,
        draft_model: DFlashDraftModel,
        draft_backend: DraftBackend,
        target_ops: Any,
        supports_prefix_snapshot: bool,
        allow_full_context_draft_layers: bool,
        prompt_tokens: Sequence[int],
        max_new_tokens: int,
        prefix_snapshot: Optional[DFlashPrefixSnapshot],
        quantize_kv_cache: bool,
        target_fa_window: int,
        runtime_context: Any,
    ) -> "SpeculativeSession":
        runtime_config = runtime_context.runtime
        draft_sink_size, draft_window_size = resolve_draft_window(
            runtime_config,
            draft_model,
            context_len=len(prompt_tokens) + max(0, int(max_new_tokens)),
            allow_full_attention_context=allow_full_context_draft_layers,
        )
        snap_prefix_len = _validate_prefix_snapshot(prefix_snapshot, prompt_tokens)
        if not supports_prefix_snapshot:
            snap_prefix_len = 0
        if snap_prefix_len > 0 and (quantize_kv_cache or target_fa_window > 0):
            snap_prefix_len = 0
        if snap_prefix_len > 0:
            template_cache = target_ops.make_cache(
                target_model,
                enable_speculative_linear_cache=True,
                quantize_kv_cache=quantize_kv_cache,
                target_fa_window=target_fa_window,
            )
            try:
                assert prefix_snapshot is not None
                target_cache = hydrate_target_cache(prefix_snapshot, template_cache)
            except (ValueError, TypeError) as exc:
                raise RuntimeError(
                    f"prefix snapshot hydrate failed for {snap_prefix_len} tokens"
                ) from exc
            finally:
                del template_cache
        else:
            target_cache = target_ops.make_cache(
                target_model,
                enable_speculative_linear_cache=True,
                quantize_kv_cache=quantize_kv_cache,
                target_fa_window=target_fa_window,
            )
        draft_cache = draft_backend.make_cache(
            draft_model=draft_model,
            sink_size=draft_sink_size,
            window_size=draft_window_size,
            allow_full_context_layers=allow_full_context_draft_layers,
        )
        diagnostics = runtime_context.diagnostics
        profile_cycles = _profile_dflash_cycles_enabled(diagnostics)
        memory_waterfall = _memory_waterfall_enabled(diagnostics)
        return cls(
            target_model=target_model,
            draft_model=draft_model,
            target_ops=target_ops,
            target_cache=target_cache,
            draft_cache=draft_cache,
            draft_backend=draft_backend,
            runtime_config=runtime_config,
            quantize_kv_cache=bool(quantize_kv_cache),
            snap_prefix_len=snap_prefix_len,
            supports_prefix_snapshot=supports_prefix_snapshot,
            allow_full_context_draft_layers=allow_full_context_draft_layers,
            draft_sink_size=draft_sink_size,
            draft_window_size=draft_window_size,
            target_layer_id_list=list(draft_model.target_layer_ids),
            capture_layer_ids={
                int(layer_id) + 1 for layer_id in draft_model.target_layer_ids
            },
            profile_cycles=profile_cycles,
            memory_waterfall=memory_waterfall,
            clear_cache_boundaries=bool(runtime_config.clear_cache_boundaries),
            target_fa_window=target_fa_window,
        )

    def clear_cache_boundary(self) -> None:
        if self.clear_cache_boundaries and hasattr(mx, "clear_cache"):
            mx.clear_cache()

    def memory_waterfall_event(
        self,
        phase: str,
        *,
        target_hidden_value: Any = None,
        gen_hidden_chunks_value: Any = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[MemoryWaterfallEvent]:
        if not self.memory_waterfall:
            return None
        return MemoryWaterfallEvent(
            fields=_collect_memory_waterfall(
                phase=phase,
                target_cache=self.target_cache,
                draft_cache=self.draft_cache,
                target_hidden=target_hidden_value,
                gen_hidden_chunks=gen_hidden_chunks_value,
                extra=extra,
            ),
        )

    def _run_prefill_events(
        self,
        *,
        request: _SessionRequest,
        state: "_RequestState",
        yield_pause: "_YieldPauseTracker",
    ) -> Generator[EngineEvent, None, _PrefillResult]:
        target_cache = self.target_cache
        target_ops = self.target_ops
        target_model = self.target_model
        draft_model = self.draft_model
        runtime_config = self.runtime_config
        snap_prefix_len = self.snap_prefix_len
        supports_prefix_snapshot = self.supports_prefix_snapshot
        allow_full_context_draft_layers = self.allow_full_context_draft_layers
        target_layer_id_list = self.target_layer_id_list
        capture_layer_ids = self.capture_layer_ids
        profile_cycles = self.profile_cycles
        prompt_tokens = request.prompt_tokens
        prompt_array = request.prompt_array
        prompt_len = request.prompt_len
        suppress_token_ids = request.suppress_token_ids
        prefix_snapshot = request.prefix_snapshot
        snapshot_service = request.snapshot_service
        stable_prefix_len = request.stable_prefix_len
        prefix_cache_active = request.prefix_cache_active

        feature_store = TargetFeatureStore(
            prompt_len=prompt_len,
            project_context=draft_model.project_target_hidden,
        )
        if supports_prefix_snapshot and snapshot_service is None:
            if prefix_cache_active:
                raise ValueError("snapshot_service is required when prefix cache is active")
            supports_prefix_snapshot = False

        start_ns = time.perf_counter_ns()
        evt = self.memory_waterfall_event("after_target_cache_create")
        if evt is not None:
            _pre_yield = yield_pause.mark()
            yield evt
            yield_pause.done(_pre_yield)
        prefill_start_ns = time.perf_counter_ns()
        prefill_step_size = int(runtime_config.prefill_step_size)

        _phase_rebuild_ns = 0
        _phase_cold_ns = 0
        _phase_seam_ns = 0
        _phase_tail_ns = 0

        if snap_prefix_len > 0:
            assert prefix_snapshot is not None
            if profile_cycles:
                _t = time.perf_counter_ns()
            feature_store.hydrate_from_snapshot(
                prefix_snapshot,
                snap_prefix_len=snap_prefix_len,
            )
            if profile_cycles:
                _phase_rebuild_ns += time.perf_counter_ns() - _t
            evt = self.memory_waterfall_event(
                "after_prefix_hydrate",
                target_hidden_value=feature_store.current_hidden,
                extra={"snap_prefix_len": int(snap_prefix_len)},
            )
            if evt is not None:
                _pre_yield = yield_pause.mark()
                yield evt
                yield_pause.done(_pre_yield)

        snapshot_boundary = compute_snapshot_boundary(prompt_len, stable_prefix_len)
        prefill_context_len = max(0, snapshot_boundary - 1)
        chunked_start = min(snap_prefix_len, prefill_context_len)
        for chunk_start in range(chunked_start, prefill_context_len, prefill_step_size):
            if profile_cycles:
                _t = time.perf_counter_ns()
            chunk_end = min(chunk_start + prefill_step_size, prefill_context_len)
            chunk_ids = prompt_array[:, chunk_start:chunk_end]
            state.prefill_logits, prefill_hidden_states = (
                target_ops.forward_with_hidden_capture(
                    target_model,
                    input_ids=chunk_ids,
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                    logits_last_only=True,
                )
            )
            eval_logits_and_captured(state.prefill_logits, prefill_hidden_states)
            feat = target_ops.extract_context_feature(
                prefill_hidden_states,
                target_layer_id_list,
            )
            feature_store.write_prompt_slice(
                start=chunk_start,
                end=chunk_end,
                features=feat,
            )
            del feat, prefill_hidden_states
            if profile_cycles:
                _phase_cold_ns += time.perf_counter_ns() - _t
            self.clear_cache_boundary()
            _pre_yield = yield_pause.mark()
            yield PrefillProgressEvent(
                tokens_processed=int(chunk_end),
                tokens_total=int(prompt_len),
            )
            yield_pause.done(_pre_yield)
            evt = self.memory_waterfall_event(
                "after_prefill_chunk",
                target_hidden_value=feature_store.current_hidden,
                extra={
                    "chunk_start": int(chunk_start),
                    "chunk_end": int(chunk_end),
                },
            )
            if evt is not None:
                _pre_yield = yield_pause.mark()
                yield evt
                yield_pause.done(_pre_yield)

        if (
            snap_prefix_len > 0
            and snap_prefix_len == snapshot_boundary
            and prefix_snapshot is not None
            and prefix_snapshot.last_logits is not None
        ):
            if profile_cycles:
                _t = time.perf_counter_ns()
            last_logits_2d = prefix_snapshot.last_logits
            state.prefill_logits = mx.expand_dims(last_logits_2d, axis=1)
            mx.eval(state.prefill_logits)
            if profile_cycles:
                _phase_seam_ns += time.perf_counter_ns() - _t
        elif snapshot_boundary > 0 and snap_prefix_len < snapshot_boundary:
            if profile_cycles:
                _t = time.perf_counter_ns()
            final_prompt_start = snapshot_boundary - 1
            state.prefill_logits, prefill_hidden_states = (
                target_ops.forward_with_hidden_capture(
                    target_model,
                    input_ids=prompt_array[:, final_prompt_start:snapshot_boundary],
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                    logits_last_only=True,
                )
            )
            eval_logits_and_captured(state.prefill_logits, prefill_hidden_states)
            feat = target_ops.extract_context_feature(
                prefill_hidden_states,
                target_layer_id_list,
            )
            feature_store.write_prompt_slice(
                start=final_prompt_start,
                end=snapshot_boundary,
                features=feat,
            )
            del feat, prefill_hidden_states
            if profile_cycles:
                _phase_seam_ns += time.perf_counter_ns() - _t
        _pre_yield = yield_pause.mark()
        yield PrefillProgressEvent(
            tokens_processed=int(snapshot_boundary),
            tokens_total=int(prompt_len),
        )
        yield_pause.done(_pre_yield)
        self.clear_cache_boundary()

        exact_snapshot_restore = bool(
            snap_prefix_len > 0 and snap_prefix_len == snapshot_boundary
        )
        if (
            exact_snapshot_restore
            and snapshot_service is not None
            and prefix_snapshot is not None
            and prefix_snapshot.last_logits is None
        ):
            raise ValueError("prefill snapshot requires last_logits")
        if supports_prefix_snapshot and not exact_snapshot_restore:
            _snapshot_build = yield_pause.mark()
            snapshot_event = _publish_snapshot_event(
                token_ids=list(prompt_tokens[:snapshot_boundary]),
                target_cache=target_cache,
                target_hidden=feature_store.prefix_view(snapshot_boundary),
                last_logits=(
                    state.prefill_logits[:, -1, :]
                    if state.prefill_logits is not None
                    else None
                ),
                snapshot_service=snapshot_service,
                kind="prefill",
                require_logits=True,
                snapshot_boundary=snapshot_boundary,
                allow_full_context_draft_layers=allow_full_context_draft_layers,
                from_snapshot=bool(snap_prefix_len > 0),
                snap_prefix_len=snap_prefix_len,
            )
            yield_pause.done(_snapshot_build)
            if snapshot_event is not None:
                _pre_yield = yield_pause.mark()
                yield snapshot_event
                yield_pause.done(_pre_yield)
            evt = self.memory_waterfall_event(
                "after_prefill_snapshot_ready",
                target_hidden_value=feature_store.current_hidden,
                extra={"snapshot_boundary": int(snapshot_boundary)},
            )
            if evt is not None:
                _pre_yield = yield_pause.mark()
                yield evt
                yield_pause.done(_pre_yield)

        if snapshot_boundary < prompt_len:
            if profile_cycles:
                _t = time.perf_counter_ns()
            tail_logits, tail_hidden_states = target_ops.forward_with_hidden_capture(
                target_model,
                input_ids=prompt_array[:, snapshot_boundary:prompt_len],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
                logits_last_only=True,
            )
            eval_logits_and_captured(tail_logits, tail_hidden_states)
            tail_feat = target_ops.extract_context_feature(
                tail_hidden_states,
                target_layer_id_list,
            )
            feature_store.write_prompt_slice(
                start=snapshot_boundary,
                end=prompt_len,
                features=tail_feat,
            )
            state.prefill_logits = tail_logits
            del tail_feat, tail_hidden_states
            if profile_cycles:
                _phase_tail_ns += time.perf_counter_ns() - _t
            self.clear_cache_boundary()
            _pre_yield = yield_pause.mark()
            yield PrefillProgressEvent(
                tokens_processed=int(prompt_len),
                tokens_total=int(prompt_len),
            )
            yield_pause.done(_pre_yield)
        evt = self.memory_waterfall_event(
            "after_tail_prefill",
            target_hidden_value=feature_store.current_hidden,
            extra={"prompt_len": int(prompt_len)},
        )
        if evt is not None:
            _pre_yield = yield_pause.mark()
            yield evt
            yield_pause.done(_pre_yield)

        prefill_ns = time.perf_counter_ns() - prefill_start_ns

        feature_store.freeze_prefill_for_snapshot(enabled=supports_prefix_snapshot)

        if state.prefill_logits is None:
            raise RuntimeError("prefill logits unavailable after prefix snapshot restore")
        suppress_token_mask = build_suppress_token_mask(
            int(state.prefill_logits.shape[-1]),
            suppress_token_ids,
        )
        state.staged_first = greedy_tokens_with_mask(
            state.prefill_logits[:, -1, :],
            suppress_token_mask,
        ).reshape(-1)
        prefill_tokens_restored = max(0, min(int(snap_prefix_len), int(prompt_len)))
        prefill_tokens_computed = max(0, int(prompt_len) - prefill_tokens_restored)

        prefill_event = PrefillCompleteEvent(
            prefill_us=prefill_ns / 1_000.0,
            prompt_token_count=int(prompt_len),
            snap_prefix_len=int(snap_prefix_len),
            snapshot_boundary=int(snapshot_boundary),
            logical_ctx_tokens=int(prompt_len),
            physical_prefill_tokens=int(prefill_tokens_computed),
            prefill_tokens_restored=int(prefill_tokens_restored),
            prefill_tokens_computed=int(prefill_tokens_computed),
            phase_rebuild_us=(
                _phase_rebuild_ns / 1_000.0 if profile_cycles else None
            ),
            phase_cold_us=_phase_cold_ns / 1_000.0 if profile_cycles else None,
            phase_seam_us=_phase_seam_ns / 1_000.0 if profile_cycles else None,
            phase_tail_us=_phase_tail_ns / 1_000.0 if profile_cycles else None,
        )
        _pre_yield = yield_pause.mark()
        yield prefill_event
        yield_pause.done(_pre_yield)

        return _PrefillResult(
            feature_store=feature_store,
            start_ns=start_ns,
            prefill_ns=prefill_ns,
            suppress_token_mask=suppress_token_mask,
            supports_prefix_snapshot=supports_prefix_snapshot,
        )

    def _run_generation_snapshot_events(
        self,
        *,
        request: _SessionRequest,
        state: "_RequestState",
        feature_store: TargetFeatureStore,
        supports_prefix_snapshot: bool,
        yield_pause: "_YieldPauseTracker",
    ) -> Iterator[EngineEvent]:
        if not supports_prefix_snapshot or not state.generated_token_ids:
            return
        if not request.publish_generation_snapshot:
            return
        if (
            request.stable_prefix_len is not None
            and 0 < int(request.stable_prefix_len) < int(request.prompt_len)
        ):
            return

        end_target_hidden = feature_store.generation_snapshot_hidden()
        if end_target_hidden is None:
            return

        if state.last_cycle_logits is not None:
            mx.eval(state.last_cycle_logits)
        self.clear_cache_boundary()
        end_total_len = request.prompt_len + len(state.generated_token_ids)
        _snapshot_build = yield_pause.mark()
        snapshot_event = _publish_snapshot_event(
            token_ids=list(request.prompt_tokens) + list(state.generated_token_ids),
            target_cache=self.target_cache,
            target_hidden=end_target_hidden,
            last_logits=state.last_cycle_logits,
            snapshot_service=request.snapshot_service,
            kind="generation",
            require_logits=False,
            snapshot_boundary=end_total_len,
            allow_full_context_draft_layers=self.allow_full_context_draft_layers,
        )
        yield_pause.done(_snapshot_build)
        if snapshot_event is not None:
            _pre_yield = yield_pause.mark()
            yield snapshot_event
            yield_pause.done(_pre_yield)
        evt = self.memory_waterfall_event(
            "after_generation_snapshot_build",
            target_hidden_value=end_target_hidden,
            gen_hidden_chunks_value=feature_store.generation_chunks,
            extra={"snapshot_boundary": int(end_total_len)},
        )
        if evt is not None:
            _pre_yield = yield_pause.mark()
            yield evt
            yield_pause.done(_pre_yield)

    def _run_decode_events(
        self,
        *,
        request: _SessionRequest,
        state: "_RequestState",
        prefill: _PrefillResult,
        yield_pause: "_YieldPauseTracker",
    ) -> Generator[EngineEvent, None, _DecodeResult]:
        target_cache = self.target_cache
        target_ops = self.target_ops
        draft_cache = self.draft_cache
        draft_backend = self.draft_backend
        target_layer_id_list = self.target_layer_id_list
        capture_layer_ids = self.capture_layer_ids
        profile_cycles = self.profile_cycles
        memory_waterfall = self.memory_waterfall
        target_model = self.target_model
        draft_model = self.draft_model
        runtime_config = self.runtime_config
        prompt_len = request.prompt_len
        max_new_tokens = request.max_new_tokens
        block_tokens = request.block_tokens
        stop_token_array = request.stop_token_array
        feature_store = prefill.feature_store
        suppress_token_mask = prefill.suppress_token_mask
        supports_prefix_snapshot = prefill.supports_prefix_snapshot

        def _waterfall_event(
            phase: str,
            *,
            target_hidden_value: Any = None,
            gen_hidden_chunks_value: Any = None,
            extra: Optional[dict[str, Any]] = None,
        ) -> Optional[MemoryWaterfallEvent]:
            return self.memory_waterfall_event(
                phase,
                target_hidden_value=target_hidden_value,
                gen_hidden_chunks_value=gen_hidden_chunks_value,
                extra=extra,
            )

        first_token_yielded = False
        if max_new_tokens > 0:
            first_token_yielded = True
            assert state.staged_first is not None
            _pre_yield = yield_pause.mark()
            yield TokenEvent(
                token_id=int(state.staged_first.item()),
                generated_tokens=1,
                acceptance_ratio=0.0,
                cycles_completed=0,
            )
            yield_pause.done(_pre_yield)

        cycle_config = resolve_speculative_cycle_config(
            runtime_config,
            draft_model,
            block_tokens,
        )
        effective_block_tokens = cycle_config.effective_block_tokens
        block_token_buffer = mx.full(
            (effective_block_tokens,),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        mask_token_tail = mx.full(
            (max(0, effective_block_tokens - 1),),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        verify_len_cap = cycle_config.verify_len_cap
        state.start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        seen_draft_cycle = False
        cycle_profiles: list[CycleCompleteEvent] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
        }

        while len(state.generated_token_ids) < max_new_tokens:
            cycle_start_ns = time.perf_counter_ns() if profile_cycles else 0
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            commit_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            remaining = max_new_tokens - len(state.generated_token_ids)
            block_len = max(1, min(effective_block_tokens, remaining))
            block_token_buffer[:block_len] = int(draft_model.mask_token_id)
            assert state.staged_first is not None
            block_token_buffer[:1] = state.staged_first
            block_token_ids = block_token_buffer[:block_len]
            current_staged_first = state.staged_first
            drafted = None

            if block_len > 1:
                if profile_cycles:
                    draft_start_ns = time.perf_counter_ns()
                    drafted = draft_backend.draft_greedy(
                        target_model=target_model,
                        target_ops=target_ops,
                        draft_model=draft_model,
                        draft_cache=draft_cache,
                        staged_first=current_staged_first,
                        draft_context=feature_store.require_current_hidden(),
                        block_len=block_len,
                        mask_token_tail=mask_token_tail,
                        suppress_token_mask=suppress_token_mask,
                        async_launch=False,
                    )
                    mx.eval(drafted)
                    draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    block_token_ids[1:block_len] = drafted
                else:
                    if (
                        state.prefetched_draft is not None
                        and int(state.prefetched_draft["block_len"]) == block_len
                    ):
                        drafted = state.prefetched_draft["drafted"]
                        current_staged_first = state.prefetched_draft["staged_first"]
                    else:
                        draft_start_ns = time.perf_counter_ns()
                        drafted = draft_backend.draft_greedy(
                            target_model=target_model,
                            target_ops=target_ops,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=current_staged_first,
                            draft_context=feature_store.require_current_hidden(),
                            block_len=block_len,
                            mask_token_tail=mask_token_tail,
                            suppress_token_mask=suppress_token_mask,
                            async_launch=True,
                        )
                        draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    state.prefetched_draft = None
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            verify_token_count = verify_token_count_for_block(block_len, verify_len_cap)
            if profile_cycles or block_len <= 1:
                verify_token_ids = block_token_ids[:verify_token_count]
            elif verify_token_count <= 1:
                verify_token_ids = current_staged_first[:1]
            else:
                verify_token_ids = mx.concatenate(
                    [current_staged_first[:1], drafted[: verify_token_count - 1]],
                    axis=0,
                )
            verify_ids = verify_token_ids[None]
            target_ops.arm_rollback(target_cache, prefix_len=state.start)
            sample_memory_cycle = memory_waterfall and _should_sample_memory_cycle(
                state.cycles_completed + 1
            )
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "before_verify_cycle",
                    target_hidden_value=feature_store.current_hidden,
                    gen_hidden_chunks_value=feature_store.generation_chunks,
                    extra={
                        "cycle": int(state.cycles_completed + 1),
                        "start": int(state.start),
                    },
                )
                if evt is not None:
                    _pre_yield = yield_pause.mark()
                    yield evt
                    yield_pause.done(_pre_yield)
            verify_start_ns = time.perf_counter_ns()
            verify_logits, verify_hidden_states = target_ops.verify_block(
                target_model=target_model,
                verify_ids=verify_ids,
                target_cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            if profile_cycles:
                eval_logits_and_captured(verify_logits, verify_hidden_states)
            verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
            verify_ns_total += verify_cycle_ns
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "after_verify_cycle",
                    target_hidden_value=feature_store.current_hidden,
                    gen_hidden_chunks_value=feature_store.generation_chunks,
                    extra={
                        "cycle": int(state.cycles_completed + 1),
                        "start": int(state.start),
                    },
                )
                if evt is not None:
                    _pre_yield = yield_pause.mark()
                    yield evt
                    yield_pause.done(_pre_yield)

            acceptance_start_ns = time.perf_counter_ns() if profile_cycles else 0
            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
            if not profile_cycles:
                mx.async_eval(posterior)
            acceptance_len = int(
                _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
            )
            state.acceptance_history.append(acceptance_len)
            if profile_cycles:
                acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            hidden_extract_start_ns = time.perf_counter_ns() if profile_cycles else 0
            committed_hidden = target_ops.extract_context_feature(
                verify_hidden_states,
                target_layer_id_list,
            )[:, : (1 + acceptance_len), :]
            if profile_cycles:
                mx.eval(committed_hidden, posterior)
            else:
                mx.async_eval(committed_hidden)
            if profile_cycles:
                hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            commit_start_ns = time.perf_counter_ns()
            state.start += commit_count
            feature_store.commit_generation(
                committed_hidden,
                collect_snapshot=supports_prefix_snapshot,
            )
            state.last_cycle_logits = verify_logits[:, acceptance_len, :]
            replay_cycle_ns = target_ops.restore_after_acceptance(
                target_cache,
                target_len=state.start,
                acceptance_length=acceptance_len,
                drafted_tokens=max(0, verify_token_count - 1),
            )
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "after_rollback",
                    target_hidden_value=feature_store.current_hidden,
                    gen_hidden_chunks_value=feature_store.generation_chunks,
                    extra={
                        "cycle": int(state.cycles_completed + 1),
                        "start": int(state.start),
                        "commit_count": int(commit_count),
                    },
                )
                if evt is not None:
                    _pre_yield = yield_pause.mark()
                    yield evt
                    yield_pause.done(_pre_yield)
            replay_ns_total += replay_cycle_ns
            state.cycles_completed += 1
            commit_wall_ns = time.perf_counter_ns() - commit_start_ns
            commit_ns_total += commit_wall_ns
            commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

            state.accepted_from_draft += acceptance_len
            staged_first_next = posterior[acceptance_len : acceptance_len + 1]
            if not profile_cycles:
                next_remaining = max_new_tokens - len(state.generated_token_ids) - commit_count
                next_block_len = max(1, min(effective_block_tokens, next_remaining))
                if next_remaining > 0 and next_block_len > 1:
                    draft_start_ns = time.perf_counter_ns()
                    next_drafted = draft_backend.draft_greedy(
                        target_model=target_model,
                        target_ops=target_ops,
                        draft_model=draft_model,
                        draft_cache=draft_cache,
                        staged_first=staged_first_next,
                        draft_context=feature_store.require_current_hidden(),
                        block_len=next_block_len,
                        mask_token_tail=mask_token_tail,
                        suppress_token_mask=suppress_token_mask,
                        async_launch=True,
                    )
                    launch_ns = time.perf_counter_ns() - draft_start_ns
                    draft_ns_total += launch_ns
                    draft_incremental_ns += launch_ns
                    state.prefetched_draft = {
                        "block_len": next_block_len,
                        "staged_first": staged_first_next,
                        "drafted": next_drafted,
                    }
                else:
                    state.prefetched_draft = None
            committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
            for token_id in committed_ids:
                if len(state.generated_token_ids) >= max_new_tokens:
                    break
                state.generated_token_ids.append(token_id)
                if first_token_yielded:
                    first_token_yielded = False
                    continue
                _pre_yield = yield_pause.mark()
                yield TokenEvent(
                    token_id=int(token_id),
                    generated_tokens=len(state.generated_token_ids),
                    acceptance_ratio=(
                        state.accepted_from_draft / len(state.generated_token_ids)
                        if state.generated_token_ids
                        else 0.0
                    ),
                    cycles_completed=int(state.cycles_completed),
                )
                yield_pause.done(_pre_yield)

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            if stop_hit:
                break

            state.staged_first = staged_first_next

            if profile_cycles:
                cycle_total_ns = time.perf_counter_ns() - cycle_start_ns
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profile_entry = CycleCompleteEvent(
                    cycle=int(state.cycles_completed),
                    block_len=int(block_len),
                    commit_count=int(commit_count),
                    acceptance_len=int(acceptance_len),
                    draft_us=ns_to_us(draft_cycle_ns),
                    verify_us=ns_to_us(verify_cycle_ns),
                    acceptance_us=ns_to_us(acceptance_cycle_ns),
                    hidden_extraction_us=ns_to_us(hidden_extract_cycle_ns),
                    rollback_us=ns_to_us(replay_cycle_ns),
                    other_us=ns_to_us(other_cycle_ns),
                    cycle_total_us=ns_to_us(cycle_total_ns),
                )
                cycle_profiles.append(cycle_profile_entry)
                _pre_yield = yield_pause.mark()
                yield cycle_profile_entry
                yield_pause.done(_pre_yield)
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns

        return _DecodeResult(
            effective_block_tokens=effective_block_tokens,
            verify_len_cap=verify_len_cap,
            draft_ns_total=draft_ns_total,
            draft_prefill_ns=draft_prefill_ns,
            draft_incremental_ns=draft_incremental_ns,
            verify_ns_total=verify_ns_total,
            replay_ns_total=replay_ns_total,
            commit_ns_total=commit_ns_total,
            cycle_profiles=tuple(cycle_profiles),
            profile_totals_ns=profile_totals_ns,
        )

    def run_events(self, request: _SessionRequest) -> Iterator[EngineEvent]:
        draft_sink_size = self.draft_sink_size
        draft_window_size = self.draft_window_size
        target_fa_window = self.target_fa_window
        profile_cycles = self.profile_cycles
        memory_waterfall = self.memory_waterfall
        clear_cache_boundaries = self.clear_cache_boundaries
        quantize_kv_cache = self.quantize_kv_cache
        prompt_len = request.prompt_len

        yield_pause = _YieldPauseTracker(enabled=bool(profile_cycles or memory_waterfall))
        state = _RequestState()

        try:
            prefill = yield from self._run_prefill_events(
                request=request,
                state=state,
                yield_pause=yield_pause,
            )
            feature_store = prefill.feature_store
            start_ns = prefill.start_ns
            prefill_ns = prefill.prefill_ns
            supports_prefix_snapshot = prefill.supports_prefix_snapshot

            decode = yield from self._run_decode_events(
                request=request,
                state=state,
                prefill=prefill,
                yield_pause=yield_pause,
            )

            yield from self._run_generation_snapshot_events(
                request=request,
                state=state,
                feature_store=feature_store,
                supports_prefix_snapshot=supports_prefix_snapshot,
                yield_pause=yield_pause,
            )

            elapsed_us = (time.perf_counter_ns() - start_ns - yield_pause.pause_ns) / 1_000.0
            first_20 = state.acceptance_history[:20]
            last_20 = state.acceptance_history[-20:]
            summary = SummaryEvent(
                elapsed_us=elapsed_us,
                prompt_token_count=int(prompt_len),
                generated_token_ids=tuple(int(x) for x in state.generated_token_ids),
                generation_tokens=len(state.generated_token_ids),
                accepted_from_draft=int(state.accepted_from_draft),
                acceptance_ratio=(
                    state.accepted_from_draft / len(state.generated_token_ids)
                    if state.generated_token_ids
                    else 0.0
                ),
                block_tokens=int(decode.effective_block_tokens),
                cycles_completed=int(state.cycles_completed),
                phase_timings_us={
                    "prefill": prefill_ns / 1_000.0,
                    "draft": decode.draft_ns_total / 1_000.0,
                    "draft_prefill": decode.draft_prefill_ns / 1_000.0,
                    "draft_incremental": decode.draft_incremental_ns / 1_000.0,
                    "verify": decode.verify_ns_total / 1_000.0,
                    "replay": decode.replay_ns_total / 1_000.0,
                    "commit": decode.commit_ns_total / 1_000.0,
                },
                verify_len_cap=int(decode.verify_len_cap),
                quantize_kv_cache=bool(quantize_kv_cache),
                target_fa_window=int(target_fa_window),
                draft_sink_size=int(draft_sink_size),
                draft_window_size=int(draft_window_size),
                clear_cache_boundaries=bool(clear_cache_boundaries),
                tokens_per_cycle=(
                    len(state.generated_token_ids) / state.cycles_completed
                    if state.cycles_completed > 0
                    else 0.0
                ),
                acceptance_history=tuple(int(x) for x in state.acceptance_history),
                acceptance_first_20_avg=(sum(first_20) / len(first_20)) if first_20 else 0.0,
                acceptance_last_20_avg=(sum(last_20) / len(last_20)) if last_20 else 0.0,
                peak_memory_gb=(
                    float(mx.get_peak_memory()) / 1e9
                    if hasattr(mx, "get_peak_memory")
                    else None
                ),
                cycle_profile_us=(
                    decode.cycle_profiles
                    if profile_cycles
                    else None
                ),
                cycle_profile_totals_us=(
                    {
                        key: ns_to_us(value) for key, value in decode.profile_totals_ns.items()
                    }
                    if profile_cycles
                    else None
                ),
            )
            yield summary
        finally:
            self.close()

    def close(self) -> None:
        self.target_ops.cleanup_generation_caches(self.target_cache, self.draft_cache)
        self.clear_cache_boundary()


@dataclass
class _RequestState:
    prefill_logits: mx.array | None = None
    last_cycle_logits: mx.array | None = None
    generated_token_ids: list[int] = field(default_factory=list)
    accepted_from_draft: int = 0
    cycles_completed: int = 0
    acceptance_history: list[int] = field(default_factory=list)
    start: int = 0
    staged_first: mx.array | None = None
    prefetched_draft: dict[str, Any] | None = None


@dataclass
class _YieldPauseTracker:
    enabled: bool
    pause_ns: int = 0

    def mark(self) -> int:
        return time.perf_counter_ns() if self.enabled else 0

    def done(self, mark: int) -> None:
        if self.enabled:
            self.pause_ns += time.perf_counter_ns() - mark


def _publish_snapshot_event(
    *,
    token_ids: list[int],
    target_cache: Any,
    target_hidden: Optional[mx.array],
    last_logits: Optional[mx.array],
    snapshot_service: Optional[SnapshotService],
    kind: Literal["prefill", "generation"],
    require_logits: bool,
    snapshot_boundary: int,
    allow_full_context_draft_layers: bool,
    from_snapshot: bool = False,
    snap_prefix_len: int = 0,
) -> Optional[SnapshotPublishedEvent]:
    if snapshot_service is None or target_hidden is None:
        return None
    publication = snapshot_service.publish(
        token_ids=token_ids,
        target_cache=target_cache,
        target_hidden=target_hidden,
        last_logits=last_logits,
        kind=kind,
        require_logits=require_logits,
        snapshot_boundary=snapshot_boundary,
        allow_full_attention_context=allow_full_context_draft_layers,
        from_snapshot=from_snapshot,
        snap_prefix_len=snap_prefix_len,
    )
    if publication is None:
        return None
    return _snapshot_published_event(publication)


def _snapshot_published_event(publication: SnapshotPublication) -> SnapshotPublishedEvent:
    return SnapshotPublishedEvent(
        kind=publication.kind,
        snapshot_boundary=int(publication.snapshot_boundary),
        prefix_len=int(publication.prefix_len),
        insert_ms=float(publication.insert_ms),
        admitted=bool(publication.admitted),
        from_snapshot=bool(publication.from_snapshot),
        snap_prefix_len=int(publication.snap_prefix_len),
    )


def stream_dflash_generate_impl(
    *,
    target_model: Any,
    target_ops: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    draft_backend: DraftBackend,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    prefix_snapshot: Optional[DFlashPrefixSnapshot] = None,
    snapshot_service: Optional[SnapshotService] = None,
    stable_prefix_len: Optional[int] = None,
    prefix_cache_active: bool = False,
    publish_generation_snapshot: bool = True,
    runtime_context: Any,
) -> Iterator[EngineEvent]:
    target_capabilities = target_ops.capabilities_for(target_model)
    supports_prefix_snapshot = bool(
        getattr(target_capabilities, "supports_prefix_snapshot", True)
    )
    allow_full_context_draft_layers = bool(
        getattr(target_capabilities, "supports_full_context_draft_layers", False)
    )
    if quantize_kv_cache:
        target_ops.configure_full_attention_split(target_model, enabled=False)

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    fallback_reason: Optional[str] = None

    if runtime_context is None:
        raise ValueError("runtime_context is required")
    runtime_config = runtime_context.runtime
    prompt_len = len(prompt_tokens)
    configured_max_ctx = int(runtime_config.dflash_max_ctx)
    dflash_max_ctx = configured_max_ctx if configured_max_ctx > 0 else sys.maxsize
    target_fa_window = int(runtime_config.target_fa_window)
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        yield from stream_baseline_generate(
            target_model=target_model,
            target_ops=target_ops,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            fallback_reason=fallback_reason,
        )
        return
    request = _SessionRequest.from_tokens(
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        block_tokens=block_tokens,
        stop_token_ids=stop_token_ids,
        suppress_token_ids=suppress_token_ids,
        prefix_snapshot=prefix_snapshot,
        snapshot_service=snapshot_service,
        stable_prefix_len=stable_prefix_len,
        prefix_cache_active=prefix_cache_active,
        publish_generation_snapshot=publish_generation_snapshot,
    )

    session = SpeculativeSession.open(
        target_model=target_model,
        draft_model=draft_model,
        draft_backend=draft_backend,
        target_ops=target_ops,
        supports_prefix_snapshot=supports_prefix_snapshot,
        allow_full_context_draft_layers=allow_full_context_draft_layers,
        prompt_tokens=request.prompt_tokens,
        max_new_tokens=max_new_tokens,
        prefix_snapshot=prefix_snapshot,
        quantize_kv_cache=quantize_kv_cache,
        target_fa_window=target_fa_window,
        runtime_context=runtime_context,
    )
    yield from session.run_events(request)
