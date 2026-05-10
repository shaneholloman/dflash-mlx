# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import Any

GiB = 1024 * 1024 * 1024
RUNTIME_PROFILE_ENV = "DFLASH_RUNTIME_PROFILE"
SURFACE_SERVE_DOCTOR = "serve_doctor"
SURFACE_GENERATE = "generate"
SURFACE_BENCHMARK = "benchmark"

@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    prefill_step_size: int
    draft_sink_size: int
    draft_window_size: int
    verify_len_cap: int
    prefix_cache: bool
    prefix_cache_max_entries: int
    prefix_cache_max_bytes: int
    clear_cache_boundaries: bool
    max_snapshot_tokens: int
    prefix_cache_l2: bool
    prefix_cache_l2_max_bytes: int
    verify_mode: str

@dataclass(frozen=True)
class EffectiveRuntimeConfig:
    profile: str
    prefill_step_size: int
    draft_sink_size: int
    draft_window_size: int
    verify_len_cap: int
    prefix_cache: bool
    prefix_cache_max_entries: int
    prefix_cache_max_bytes: int
    clear_cache_boundaries: bool
    max_snapshot_tokens: int
    prefix_cache_l2: bool
    prefix_cache_l2_dir: str
    prefix_cache_l2_max_bytes: int
    target_fa_window: int
    dflash_max_ctx: int
    verify_mode: str

@dataclass(frozen=True)
class RuntimeConfigFieldSpec:
    field: str
    flags: tuple[str, ...]
    env: str | None
    source_default: str | None
    help: str
    value_type: Any | None = None
    action: Any | None = None
    choices: tuple[str, ...] = ()
    metavar: str | None = None
    doc: str | None = None
    doc_group: str | None = None
    surfaces: tuple[str, ...] = (SURFACE_SERVE_DOCTOR,)

    @property
    def primary_flag(self) -> str:
        return self.flags[0]

    @property
    def doc_text(self) -> str:
        return self.doc or self.help


@dataclass(frozen=True)
class RuntimeConfigSpec:
    fields: tuple[RuntimeConfigFieldSpec, ...]

    def __post_init__(self) -> None:
        names = [field.field for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("runtime config field names must be unique")

    @property
    def by_field(self) -> dict[str, RuntimeConfigFieldSpec]:
        return {field.field: field for field in self.fields}

    def require_field(self, field: str) -> RuntimeConfigFieldSpec:
        try:
            return self.by_field[field]
        except KeyError as exc:
            raise KeyError(f"unknown runtime config field: {field}") from exc

    def select(self, fields: tuple[str, ...]) -> tuple[RuntimeConfigFieldSpec, ...]:
        by_field = self.by_field
        return tuple(by_field[field] for field in fields)

    def full_field_names(self) -> tuple[str, ...]:
        return tuple(field.field for field in self.fields)

    def surface_field_names(
        self,
        surface: str,
        *,
        order: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        return self._ordered_field_names(
            tuple(field.field for field in self.fields if surface in field.surfaces),
            order=order,
            label=f"runtime surface {surface}",
        )

    def doc_group_field_names(
        self,
        group: str,
        *,
        order: tuple[str, ...] | None = None,
    ) -> tuple[str, ...]:
        return self._ordered_field_names(
            tuple(field.field for field in self.fields if field.doc_group == group),
            order=order,
            label=f"runtime docs group {group}",
        )

    @staticmethod
    def _ordered_field_names(
        fields: tuple[str, ...],
        *,
        order: tuple[str, ...] | None,
        label: str,
    ) -> tuple[str, ...]:
        if order is None:
            return fields
        field_set = set(fields)
        order_set = set(order)
        if field_set != order_set:
            missing = sorted(field_set - order_set)
            extra = sorted(order_set - field_set)
            raise ValueError(f"{label} order mismatch: missing={missing}, extra={extra}")
        return order

PROFILES: dict[str, RuntimeProfile] = {
    "balanced": RuntimeProfile(
        name="balanced",
        prefill_step_size=4096,
        draft_sink_size=64,
        draft_window_size=1024,
        verify_len_cap=0,
        prefix_cache=True,
        prefix_cache_max_entries=4,
        prefix_cache_max_bytes=8 * GiB,
        clear_cache_boundaries=False,
        max_snapshot_tokens=32000,
        prefix_cache_l2=False,
        prefix_cache_l2_max_bytes=50 * GiB,
        verify_mode="auto",
    ),
    "fast": RuntimeProfile(
        name="fast",
        prefill_step_size=8192,
        draft_sink_size=64,
        draft_window_size=1024,
        verify_len_cap=0,
        prefix_cache=True,
        prefix_cache_max_entries=4,
        prefix_cache_max_bytes=16 * GiB,
        clear_cache_boundaries=False,
        max_snapshot_tokens=0,
        prefix_cache_l2=False,
        prefix_cache_l2_max_bytes=50 * GiB,
        verify_mode="auto",
    ),
    "low-memory": RuntimeProfile(
        name="low-memory",
        prefill_step_size=1024,
        draft_sink_size=64,
        draft_window_size=1024,
        verify_len_cap=0,
        prefix_cache=True,
        prefix_cache_max_entries=2,
        prefix_cache_max_bytes=2 * GiB,
        clear_cache_boundaries=False,
        max_snapshot_tokens=8000,
        prefix_cache_l2=False,
        prefix_cache_l2_max_bytes=50 * GiB,
        verify_mode="auto",
    ),
    "long-session": RuntimeProfile(
        name="long-session",
        prefill_step_size=4096,
        draft_sink_size=64,
        draft_window_size=1024,
        verify_len_cap=0,
        prefix_cache=True,
        prefix_cache_max_entries=8,
        prefix_cache_max_bytes=8 * GiB,
        clear_cache_boundaries=False,
        max_snapshot_tokens=32000,
        prefix_cache_l2=True,
        prefix_cache_l2_max_bytes=50 * GiB,
        verify_mode="auto",
    ),
}

RUNTIME_CONFIG_FIELDS: tuple[RuntimeConfigFieldSpec, ...] = (
    RuntimeConfigFieldSpec(
        field="profile",
        flags=("--profile",),
        env=RUNTIME_PROFILE_ENV,
        source_default="balanced",
        choices=tuple(PROFILES),
        help="Runtime preset. Explicit CLI flags override profile values.",
        doc="preset defaults",
        doc_group="core",
    ),
    RuntimeConfigFieldSpec(
        field="prefill_step_size",
        flags=("--prefill-step-size",),
        env="DFLASH_PREFILL_STEP_SIZE",
        source_default=None,
        value_type=int,
        help="Prompt prefill chunk size.",
        metavar="INT",
        doc="target prefill chunk size",
        doc_group="runtime",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE),
    ),
    RuntimeConfigFieldSpec(
        field="draft_sink_size",
        flags=("--draft-sink-size",),
        env="DFLASH_DRAFT_SINK_SIZE",
        source_default=None,
        value_type=int,
        help="Draft context cache sink tokens kept before the rolling window.",
        metavar="INT",
        doc="draft cache sink tokens",
        doc_group="runtime",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="draft_window_size",
        flags=("--draft-window-size",),
        env="DFLASH_DRAFT_WINDOW_SIZE",
        source_default=None,
        value_type=int,
        help="Draft context cache rolling window tokens.",
        metavar="INT",
        doc="draft cache rolling window tokens",
        doc_group="runtime",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="verify_len_cap",
        flags=("--verify-len-cap",),
        env="DFLASH_VERIFY_LEN_CAP",
        source_default="0",
        value_type=int,
        help="Max tokens verified per target forward.",
        metavar="INT",
        doc="max tokens per verify forward, `0` means block size",
        doc_group="runtime",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="clear_cache_boundaries",
        flags=("--clear-cache-boundaries",),
        env="DFLASH_CLEAR_CACHE_BOUNDARIES",
        source_default=None,
        action=argparse.BooleanOptionalAction,
        help="Clear the MLX cache at safe request boundaries.",
        doc="clear the MLX cache at safe request boundaries",
        doc_group="runtime",
    ),
    RuntimeConfigFieldSpec(
        field="verify_mode",
        flags=("--verify-mode",),
        env="DFLASH_VERIFY_MODE",
        source_default=None,
        choices=("auto", "off"),
        help="Verify path mode. Use off only for debug/parity.",
        doc="verifier path mode; `off` is debug/parity only",
        doc_group="runtime",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE),
    ),
    RuntimeConfigFieldSpec(
        field="max_snapshot_tokens",
        flags=("--max-snapshot-tokens",),
        env="DFLASH_MAX_SNAPSHOT_TOKENS",
        source_default=None,
        value_type=int,
        help="Skip prefix-cache snapshot inserts above this token count; 0 disables the cap.",
        metavar="INT",
        doc="snapshot insert token cap; `0` disables the cap",
        doc_group="prefix_cache",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_l2",
        flags=("--prefix-cache-l2",),
        env="DFLASH_PREFIX_CACHE_L2_ENABLED",
        source_default=None,
        action=argparse.BooleanOptionalAction,
        help="Enable SSD L2 for persistent and spilled prefix snapshots.",
        doc="enable/disable SSD L2 for persisted/spilled snapshots",
        doc_group="prefix_cache",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_l2_dir",
        flags=("--prefix-cache-l2-dir",),
        env="DFLASH_PREFIX_CACHE_L2_DIR",
        source_default="default",
        value_type=str,
        help="Directory for prefix-cache L2 files.",
        metavar="PATH",
        doc="L2 root directory",
        doc_group="prefix_cache",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_l2_max_bytes",
        flags=("--prefix-cache-l2-max-bytes",),
        env="DFLASH_PREFIX_CACHE_L2_MAX_BYTES",
        source_default=None,
        value_type=int,
        help="Byte budget for prefix-cache L2.",
        metavar="BYTES",
        doc="L2 disk budget",
        doc_group="prefix_cache",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache",
        flags=("--prefix-cache",),
        env="DFLASH_PREFIX_CACHE",
        source_default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable the DFlash prefix cache that reuses cross-turn KV state. "
            "Default: enabled. Big win on multi-turn agentic workloads, "
            "~neutral on single-turn."
        ),
        doc="enable/disable DFlash prefix snapshots",
        doc_group="prefix_cache",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_max_entries",
        flags=("--prefix-cache-max-entries",),
        env="DFLASH_PREFIX_CACHE_MAX_ENTRIES",
        source_default=None,
        value_type=int,
        help="Maximum number of cached prefix snapshots.",
        metavar="INT",
        doc="L1 snapshot entry budget",
        doc_group="prefix_cache",
    ),
    RuntimeConfigFieldSpec(
        field="prefix_cache_max_bytes",
        flags=("--prefix-cache-max-bytes",),
        env="DFLASH_PREFIX_CACHE_MAX_BYTES",
        source_default=None,
        value_type=int,
        help="Maximum total bytes the prefix cache may hold.",
        metavar="BYTES",
        doc="L1 snapshot byte budget",
        doc_group="prefix_cache",
    ),
    RuntimeConfigFieldSpec(
        field="target_fa_window",
        flags=("--target-fa-window",),
        env="DFLASH_TARGET_FA_WINDOW",
        source_default="0",
        value_type=int,
        help=(
            "Experimental target verifier full-attention KV window. "
            "N>0 uses a rotating KV cache of N tokens for target full-attention layers only."
        ),
        metavar="INT",
        doc="experimental target FA rotating window; `0` means full KV",
        doc_group="runtime",
        surfaces=(SURFACE_SERVE_DOCTOR, SURFACE_GENERATE, SURFACE_BENCHMARK),
    ),
    RuntimeConfigFieldSpec(
        field="dflash_max_ctx",
        flags=("--dflash-max-ctx",),
        env="DFLASH_MAX_CTX",
        source_default="0",
        value_type=int,
        help="Hard cap on runtime context length.",
        metavar="INT",
        doc="DFlash runtime context cap; `0` means no cap",
        doc_group="runtime",
    ),
)

RUNTIME_CONFIG_SPEC = RuntimeConfigSpec(RUNTIME_CONFIG_FIELDS)

def profile_names() -> tuple[str, ...]:
    return tuple(PROFILES)

def _add_runtime_config_argument(
    parser: argparse.ArgumentParser,
    option: RuntimeConfigFieldSpec,
    *,
    default: Any,
    help_text: str | None = None,
) -> None:
    kwargs: dict[str, Any] = {"default": default, "help": help_text or option.help}
    if option.value_type is not None:
        kwargs["type"] = option.value_type
    if option.action is not None:
        kwargs["action"] = option.action
    if option.choices:
        kwargs["choices"] = option.choices
    if option.metavar is not None:
        kwargs["metavar"] = option.metavar
    parser.add_argument(*option.flags, **kwargs)


def add_runtime_config_field_arguments(
    parser: argparse.ArgumentParser,
    fields: tuple[str, ...],
    *,
    defaults: dict[str, Any] | None = None,
    help_suffixes: dict[str, str] | None = None,
) -> None:
    defaults = defaults or {}
    help_suffixes = help_suffixes or {}
    for field in fields:
        option = RUNTIME_CONFIG_SPEC.require_field(field)
        help_text = option.help
        suffix = help_suffixes.get(field)
        if suffix:
            help_text = f"{help_text} {suffix}"
        _add_runtime_config_argument(
            parser,
            option,
            default=defaults.get(field),
            help_text=help_text,
        )


def add_runtime_config_arguments(parser: argparse.ArgumentParser) -> None:
    add_runtime_config_field_arguments(
        parser,
        RUNTIME_CONFIG_SPEC.full_field_names(),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List runtime profiles and exit.",
    )


def runtime_config_field_defaults(
    fields: tuple[str, ...],
    *,
    profile: str = "balanced",
) -> dict[str, Any]:
    cfg = runtime_config_from_profile_values(profile=profile)
    return {field: getattr(cfg, field) for field in fields}


_GENERATE_RUNTIME_FIELD_ORDER = (
    "verify_mode",
    "prefill_step_size",
    "target_fa_window",
    "draft_sink_size",
    "draft_window_size",
    "verify_len_cap",
)

_BENCHMARK_RUNTIME_FIELD_ORDER = (
    "target_fa_window",
    "draft_sink_size",
    "draft_window_size",
    "verify_len_cap",
)

_SERVE_RUNTIME_DOC_FIELD_ORDER = (
    "prefill_step_size",
    "draft_sink_size",
    "draft_window_size",
    "verify_len_cap",
    "verify_mode",
    "dflash_max_ctx",
    "target_fa_window",
    "clear_cache_boundaries",
)

_PREFIX_CACHE_DOC_FIELD_ORDER = (
    "prefix_cache",
    "prefix_cache_max_entries",
    "prefix_cache_max_bytes",
    "max_snapshot_tokens",
    "prefix_cache_l2",
    "prefix_cache_l2_dir",
    "prefix_cache_l2_max_bytes",
)

GENERATE_RUNTIME_FIELDS = RUNTIME_CONFIG_SPEC.surface_field_names(
    SURFACE_GENERATE,
    order=_GENERATE_RUNTIME_FIELD_ORDER,
)
BENCHMARK_RUNTIME_FIELDS = RUNTIME_CONFIG_SPEC.surface_field_names(
    SURFACE_BENCHMARK,
    order=_BENCHMARK_RUNTIME_FIELD_ORDER,
)
SERVE_RUNTIME_DOC_FIELDS = RUNTIME_CONFIG_SPEC.doc_group_field_names(
    "runtime",
    order=_SERVE_RUNTIME_DOC_FIELD_ORDER,
)
PREFIX_CACHE_DOC_FIELDS = RUNTIME_CONFIG_SPEC.doc_group_field_names(
    "prefix_cache",
    order=_PREFIX_CACHE_DOC_FIELD_ORDER,
)


def runtime_config_markdown_sections() -> dict[str, str]:
    return {
        "profiles": _profiles_markdown_table(),
        "serve-runtime": runtime_config_flags_markdown_table(SERVE_RUNTIME_DOC_FIELDS),
        "prefix-cache": runtime_config_flags_markdown_table(PREFIX_CACHE_DOC_FIELDS),
        "generate-runtime": runtime_config_flags_markdown_table(GENERATE_RUNTIME_FIELDS),
        "benchmark-runtime": runtime_config_flags_markdown_table(BENCHMARK_RUNTIME_FIELDS),
        "env": runtime_config_env_markdown_table(),
    }


def runtime_config_flags_markdown_table(fields: tuple[str, ...]) -> str:
    rows = ["| Flag | Meaning |", "| --- | --- |"]
    for option in RUNTIME_CONFIG_SPEC.select(fields):
        rows.append(f"| {_markdown_flag(option)} | {option.doc_text} |")
    return "\n".join(rows)


def runtime_config_env_markdown_table() -> str:
    rows = ["| Env var | Matching config |", "| --- | --- |"]
    for option in RUNTIME_CONFIG_SPEC.fields:
        if option.env is None:
            continue
        rows.append(f"| `{option.env}` | {_markdown_flag(option)} |")
    return "\n".join(rows)


def _profiles_markdown_table() -> str:
    rows = [
        "| Profile | Prefill | Draft window | Prefix cache | L1 entries / byte budget | L2 | Intent |",
        "| --- | ---: | --- | --- | --- | --- | --- |",
    ]
    notes = {
        "balanced": "default normal coding",
        "fast": "throughput first",
        "low-memory": "lower pressure, slower prefill",
        "long-session": "revisit-oriented long sessions",
    }
    for profile in PROFILES.values():
        rows.append(
            " | ".join(
                [
                    f"| `{profile.name}`",
                    str(profile.prefill_step_size),
                    f"`{profile.draft_sink_size}+{profile.draft_window_size}`",
                    _on_off(profile.prefix_cache),
                    f"`{profile.prefix_cache_max_entries} / {_format_gib(profile.prefix_cache_max_bytes)}`",
                    _l2_markdown(profile),
                    f"{notes[profile.name]} |",
                ]
            )
        )
    return "\n".join(rows)


def _markdown_flag(option: RuntimeConfigFieldSpec) -> str:
    primary = option.primary_flag
    if option.action is argparse.BooleanOptionalAction:
        return f"`{primary}`, `--no-{primary[2:]}`"
    if option.choices:
        return f"`{primary} {{{','.join(option.choices)}}}`"
    if option.metavar:
        return f"`{primary} {option.metavar}`"
    return f"`{primary}`"


def _on_off(value: bool) -> str:
    return "on" if value else "off"


def _format_gib(value: int) -> str:
    if value % GiB == 0:
        return f"{value // GiB}GiB"
    return f"{value / GiB:.1f}GiB"


def _l2_markdown(profile: RuntimeProfile) -> str:
    if not profile.prefix_cache_l2:
        return "off"
    return f"on / `{_format_gib(profile.prefix_cache_l2_max_bytes)}`"


def runtime_config_from_profile_values(
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
    runtime_profile = PROFILES[profile]
    return validate_runtime_config(
        EffectiveRuntimeConfig(
            profile=profile,
            prefill_step_size=(
                runtime_profile.prefill_step_size
                if prefill_step_size is None
                else int(prefill_step_size)
            ),
            draft_sink_size=(
                runtime_profile.draft_sink_size
                if draft_sink_size is None
                else int(draft_sink_size)
            ),
            draft_window_size=(
                runtime_profile.draft_window_size
                if draft_window_size is None
                else int(draft_window_size)
            ),
            verify_len_cap=(
                runtime_profile.verify_len_cap
                if verify_len_cap is None
                else int(verify_len_cap)
            ),
            prefix_cache=(
                runtime_profile.prefix_cache if prefix_cache is None else bool(prefix_cache)
            ),
            prefix_cache_max_entries=(
                runtime_profile.prefix_cache_max_entries
                if prefix_cache_max_entries is None
                else int(prefix_cache_max_entries)
            ),
            prefix_cache_max_bytes=(
                runtime_profile.prefix_cache_max_bytes
                if prefix_cache_max_bytes is None
                else int(prefix_cache_max_bytes)
            ),
            clear_cache_boundaries=(
                runtime_profile.clear_cache_boundaries
                if clear_cache_boundaries is None
                else bool(clear_cache_boundaries)
            ),
            max_snapshot_tokens=(
                runtime_profile.max_snapshot_tokens
                if max_snapshot_tokens is None
                else int(max_snapshot_tokens)
            ),
            prefix_cache_l2=(
                runtime_profile.prefix_cache_l2 if prefix_cache_l2 is None else bool(prefix_cache_l2)
            ),
            prefix_cache_l2_dir=(
                os.path.expanduser("~/.cache/dflash/prefix_l2")
                if prefix_cache_l2_dir is None
                else str(prefix_cache_l2_dir)
            ),
            prefix_cache_l2_max_bytes=(
                runtime_profile.prefix_cache_l2_max_bytes
                if prefix_cache_l2_max_bytes is None
                else int(prefix_cache_l2_max_bytes)
            ),
            target_fa_window=int(target_fa_window),
            dflash_max_ctx=int(dflash_max_ctx),
            verify_mode=runtime_profile.verify_mode if verify_mode is None else str(verify_mode),
        )
    )

def resolve_runtime_config(args: Any) -> EffectiveRuntimeConfig:
    profile_name = _resolve_profile_name(getattr(args, "profile", None))
    profile = PROFILES[profile_name]
    return runtime_config_from_profile_values(
        profile=profile_name,
        prefill_step_size=_resolve_int(
            getattr(args, "prefill_step_size", None),
            _runtime_env("prefill_step_size"),
            profile.prefill_step_size,
        ),
        draft_sink_size=_resolve_int(
            getattr(args, "draft_sink_size", None),
            _runtime_env("draft_sink_size"),
            profile.draft_sink_size,
        ),
        draft_window_size=_resolve_int(
            getattr(args, "draft_window_size", None),
            _runtime_env("draft_window_size"),
            profile.draft_window_size,
        ),
        verify_len_cap=_resolve_int(
            getattr(args, "verify_len_cap", None),
            _runtime_env("verify_len_cap"),
            profile.verify_len_cap,
        ),
        prefix_cache=_resolve_bool(
            getattr(args, "prefix_cache", None),
            _runtime_env("prefix_cache"),
            profile.prefix_cache,
        ),
        prefix_cache_max_entries=_resolve_int(
            getattr(args, "prefix_cache_max_entries", None),
            _runtime_env("prefix_cache_max_entries"),
            profile.prefix_cache_max_entries,
        ),
        prefix_cache_max_bytes=_resolve_int(
            getattr(args, "prefix_cache_max_bytes", None),
            _runtime_env("prefix_cache_max_bytes"),
            profile.prefix_cache_max_bytes,
        ),
        clear_cache_boundaries=_resolve_bool(
            getattr(args, "clear_cache_boundaries", None),
            _runtime_env("clear_cache_boundaries"),
            profile.clear_cache_boundaries,
        ),
        max_snapshot_tokens=_resolve_int(
            getattr(args, "max_snapshot_tokens", None),
            _runtime_env("max_snapshot_tokens"),
            profile.max_snapshot_tokens,
        ),
        prefix_cache_l2=_resolve_bool(
            getattr(args, "prefix_cache_l2", None),
            _runtime_env("prefix_cache_l2"),
            profile.prefix_cache_l2,
        ),
        prefix_cache_l2_dir=_resolve_str(
            getattr(args, "prefix_cache_l2_dir", None),
            _runtime_env("prefix_cache_l2_dir"),
            os.path.expanduser("~/.cache/dflash/prefix_l2"),
        ),
        prefix_cache_l2_max_bytes=_resolve_int(
            getattr(args, "prefix_cache_l2_max_bytes", None),
            _runtime_env("prefix_cache_l2_max_bytes"),
            profile.prefix_cache_l2_max_bytes,
        ),
        target_fa_window=_resolve_int(
            getattr(args, "target_fa_window", None),
            _runtime_env("target_fa_window"),
            0,
        ),
        dflash_max_ctx=_resolve_int(
            getattr(args, "dflash_max_ctx", None),
            _runtime_env("dflash_max_ctx"),
            0,
        ),
        verify_mode=_resolve_verify_mode(getattr(args, "verify_mode", None), profile.verify_mode),
    )

def runtime_config_sources(args: Any, cfg: EffectiveRuntimeConfig) -> dict[str, str]:
    return {
        field: _source_for_runtime_field(args, field, cfg.profile)
        for field in cfg.__dataclass_fields__
    }

def validate_runtime_config(cfg: EffectiveRuntimeConfig) -> EffectiveRuntimeConfig:
    if cfg.prefill_step_size <= 0:
        raise ValueError("--prefill-step-size / prefill_step_size must be > 0")
    if cfg.draft_sink_size < 0:
        raise ValueError("--draft-sink-size / draft_sink_size must be >= 0")
    if cfg.draft_window_size <= 0:
        raise ValueError("--draft-window-size / draft_window_size must be > 0")
    if cfg.verify_len_cap < 0:
        raise ValueError("--verify-len-cap / verify_len_cap must be >= 0")
    if cfg.prefix_cache_max_entries <= 0:
        raise ValueError(
            "--prefix-cache-max-entries / prefix_cache_max_entries must be > 0"
        )
    if cfg.prefix_cache_max_bytes < 0:
        raise ValueError(
            "--prefix-cache-max-bytes / prefix_cache_max_bytes must be >= 0"
        )
    if cfg.max_snapshot_tokens < 0:
        raise ValueError("--max-snapshot-tokens / max_snapshot_tokens must be >= 0")
    if cfg.prefix_cache_l2 and not cfg.prefix_cache_l2_dir.strip():
        raise ValueError(
            "--prefix-cache-l2-dir / prefix_cache_l2_dir must not be empty when L2 is enabled"
        )
    if cfg.prefix_cache_l2 and cfg.prefix_cache_l2_max_bytes < 0:
        raise ValueError(
            "--prefix-cache-l2-max-bytes / prefix_cache_l2_max_bytes must be >= 0"
        )
    if cfg.target_fa_window < 0:
        raise ValueError("--target-fa-window / target_fa_window must be >= 0")
    if cfg.dflash_max_ctx < 0:
        raise ValueError("--dflash-max-ctx / dflash_max_ctx must be >= 0")
    if cfg.verify_mode not in ("auto", "off"):
        raise ValueError("--verify-mode / verify_mode must be auto or off")
    if not cfg.prefix_cache and cfg.prefix_cache_l2:
        return replace(cfg, prefix_cache_l2=False)
    if cfg.target_fa_window > 0 and cfg.prefix_cache:
        return replace(cfg, prefix_cache=False, prefix_cache_l2=False)
    if cfg.target_fa_window > 0 and cfg.prefix_cache_l2:
        return replace(cfg, prefix_cache_l2=False)
    return cfg

def _runtime_env(field: str) -> str:
    env_key = RUNTIME_CONFIG_SPEC.require_field(field).env
    if env_key is None:
        raise KeyError(f"runtime field {field!r} has no environment key")
    return env_key

def _resolve_profile_name(cli_value: str | None) -> str:
    raw = cli_value or os.environ.get(_runtime_env("profile"), "").strip() or "balanced"
    if raw not in PROFILES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILES)}")
    return raw

def _resolve_int(cli_value: int | None, env_key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    raw = os.environ.get(env_key, "").strip()
    if raw:
        return int(raw)
    return int(default)

def _resolve_bool(cli_value: bool | None, env_key: str, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    raw = os.environ.get(env_key, "").strip().lower()
    if raw:
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        raise ValueError(
            f"{env_key} must be a boolean: use 1/0, true/false, yes/no, or on/off"
        )
    return bool(default)

def _resolve_str(cli_value: str | None, env_key: str, default: str) -> str:
    if cli_value is not None:
        return str(cli_value)
    raw = os.environ.get(env_key)
    if raw is not None and raw.strip():
        return str(raw)
    return str(default)

def _resolve_verify_mode(cli_value: str | None, default: str) -> str:
    if cli_value is not None:
        return str(cli_value)
    raw = os.environ.get(_runtime_env("verify_mode"), "").strip()
    if raw:
        return raw
    return default

def _source_for_runtime_field(args: Any, field: str, profile: str) -> str:
    if getattr(args, field, None) is not None:
        return "cli"
    option = RUNTIME_CONFIG_SPEC.require_field(field)
    env_key = option.env
    if env_key is not None and os.environ.get(env_key, "").strip():
        return "env"
    if field == "profile" and profile == "balanced":
        return "default"
    if option.source_default is not None:
        return "default"
    return "profile"


_OFFLINE_RUNTIME_FIELDS = tuple(dict.fromkeys((*GENERATE_RUNTIME_FIELDS, *BENCHMARK_RUNTIME_FIELDS)))
_OFFLINE_HELP_SUFFIXES = {
    "prefill_step_size": lambda value: f"Default: profile balanced value, {value}.",
    "draft_sink_size": lambda value: f"Default: {value}.",
    "draft_window_size": lambda value: f"Default: {value}.",
    "verify_len_cap": lambda value: "Default: 0 = block size." if int(value) == 0 else f"Default: {value}.",
    "target_fa_window": lambda value: "Default: 0 = full KV." if int(value) == 0 else f"Default: {value}.",
}


def _offline_help_overrides(fields: tuple[str, ...], defaults: dict[str, Any]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for field in fields:
        suffix = _OFFLINE_HELP_SUFFIXES.get(field)
        if suffix is None:
            continue
        overrides[field] = suffix(defaults[field])
    return overrides


def add_offline_runtime_arguments(
    parser: argparse.ArgumentParser,
    fields: tuple[str, ...],
) -> None:
    defaults = runtime_config_field_defaults(fields)
    add_runtime_config_field_arguments(
        parser,
        fields,
        defaults=defaults,
        help_suffixes=_offline_help_overrides(fields, defaults),
    )


def offline_runtime_kwargs(args: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: getattr(args, field) for field in fields}


def offline_runtime_error_message(message: str) -> str:
    cleaned = str(message)
    for field in _OFFLINE_RUNTIME_FIELDS:
        cleaned = cleaned.replace(f" / {field}", "")
    return cleaned
