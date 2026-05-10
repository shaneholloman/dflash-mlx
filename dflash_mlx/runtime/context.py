# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from dflash_mlx.diagnostics import DiagnosticsConfig
from dflash_mlx.runtime.config import (
    EffectiveRuntimeConfig,
    runtime_config_from_profile_values,
    validate_runtime_config,
)

if TYPE_CHECKING:
    from dflash_mlx.metal_limits import MetalLimitConfig
    from dflash_mlx.runtime import VerifyConfig

@dataclass(frozen=True)
class RuntimeContext:
    runtime: EffectiveRuntimeConfig
    diagnostics: DiagnosticsConfig
    verify: VerifyConfig
    metal_limits: MetalLimitConfig | None = None

def runtime_config_from_profile(
    profile: str = "balanced",
    *,
    prefill_step_size: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    verify_len_cap: int | None = None,
    prefix_cache: bool | None = None,
    prefix_cache_max_entries: int | None = None,
    prefix_cache_max_bytes: int | None = None,
    clear_cache_boundaries: bool | None = None,
    max_snapshot_tokens: int | None = None,
    prefix_cache_l2: bool | None = None,
    prefix_cache_l2_dir: str | None = None,
    prefix_cache_l2_max_bytes: int | None = None,
    target_fa_window: int = 0,
    dflash_max_ctx: int = 0,
    verify_mode: str | None = None,
) -> EffectiveRuntimeConfig:
    return runtime_config_from_profile_values(
        profile=profile,
        prefill_step_size=prefill_step_size,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
        prefix_cache=prefix_cache,
        prefix_cache_max_entries=prefix_cache_max_entries,
        prefix_cache_max_bytes=prefix_cache_max_bytes,
        clear_cache_boundaries=clear_cache_boundaries,
        max_snapshot_tokens=max_snapshot_tokens,
        prefix_cache_l2=prefix_cache_l2,
        prefix_cache_l2_dir=prefix_cache_l2_dir,
        prefix_cache_l2_max_bytes=prefix_cache_l2_max_bytes,
        target_fa_window=target_fa_window,
        dflash_max_ctx=dflash_max_ctx,
        verify_mode=verify_mode,
    )

def build_runtime_context(
    runtime_config: EffectiveRuntimeConfig,
    diagnostics_config: DiagnosticsConfig | None = None,
    metal_limits: MetalLimitConfig | None = None,
) -> RuntimeContext:
    from dflash_mlx.runtime import VerifyConfig

    validated_runtime = validate_runtime_config(runtime_config)
    return RuntimeContext(
        runtime=validated_runtime,
        diagnostics=diagnostics_config or DiagnosticsConfig(),
        verify=VerifyConfig.from_mode(validated_runtime.verify_mode),
        metal_limits=metal_limits,
    )

def build_offline_runtime_config(
    *,
    target_fa_window: int | None = None,
    prefill_step_size: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    verify_len_cap: int | None = None,
    verify_mode: str | None = None,
) -> EffectiveRuntimeConfig:
    runtime_config = runtime_config_from_profile(
        profile="balanced",
        prefix_cache=False,
        prefix_cache_l2=False,
        target_fa_window=0 if target_fa_window is None else int(target_fa_window),
        prefill_step_size=prefill_step_size,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
        verify_mode=verify_mode,
    )
    return validate_runtime_config(runtime_config)

def build_offline_runtime_context(
    *,
    target_fa_window: int | None = None,
    prefill_step_size: int | None = None,
    draft_sink_size: int | None = None,
    draft_window_size: int | None = None,
    verify_len_cap: int | None = None,
    verify_mode: str | None = None,
) -> RuntimeContext:
    runtime_config = build_offline_runtime_config(
        target_fa_window=target_fa_window,
        prefill_step_size=prefill_step_size,
        draft_sink_size=draft_sink_size,
        draft_window_size=draft_window_size,
        verify_len_cap=verify_len_cap,
        verify_mode=verify_mode,
    )
    return build_runtime_context(runtime_config)

def with_metal_limits(
    context: RuntimeContext,
    metal_limits: MetalLimitConfig | None,
) -> RuntimeContext:
    return replace(context, metal_limits=metal_limits)
