# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from typing import Any

_ARCH_RE = re.compile(r"applegpu_g(?P<gen>[0-9]+)(?P<suffix>[a-z])?$", re.IGNORECASE)
_FAMILY_BY_GEN = {13: "M1", 14: "M2", 15: "M3", 16: "M4", 17: "M5"}
_TIER_BY_SUFFIX = {
    "g": "base_or_pro",
    "s": "max",
    "d": "ultra",
    "p": "phone",
}


class ChipDetectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChipProfile:
    metal_available: bool
    arch_string: str
    arch_gen: int
    suffix: str
    family: str
    tier: str
    bf16_native: bool
    nax_capable: bool
    macos_version: str
    macos_major: int
    macos_minor: int

    @property
    def bf16_emulated(self) -> bool:
        return self.metal_available and self.arch_gen in (13, 14)

    @property
    def pre_m5_gpu(self) -> bool:
        return self.metal_available and 0 < self.arch_gen < 17


def detect_chip() -> ChipProfile:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise ChipDetectionError("failed to import mlx.core for chip detection") from exc

    try:
        metal_available = bool(mx.metal.is_available())
    except Exception as exc:
        raise ChipDetectionError("failed to query MLX Metal availability") from exc
    try:
        device_info = dict(mx.device_info()) if metal_available else {}
    except Exception as exc:
        raise ChipDetectionError("failed to query mx.device_info() for chip detection") from exc
    return chip_profile_from_device_info(
        device_info,
        metal_available=metal_available,
        macos_version=platform.mac_ver()[0],
    )


def chip_profile_from_device_info(
    device_info: dict[str, Any] | None,
    *,
    metal_available: bool = True,
    macos_version: str = "",
) -> ChipProfile:
    info = device_info or {}
    arch = str(info.get("architecture", "") or "")
    match = _ARCH_RE.search(arch)
    arch_gen = int(match.group("gen")) if match else 0
    suffix = (match.group("suffix") or "").lower() if match else ""
    family = _FAMILY_BY_GEN.get(arch_gen, f"unknown_gen{arch_gen}" if arch_gen else "unknown")
    tier = _TIER_BY_SUFFIX.get(suffix, "unknown")
    major, minor = _parse_macos_version(macos_version)
    bf16_native = bool(metal_available and arch_gen >= 15)
    nax_capable = bool(
        metal_available
        and arch_gen >= (18 if suffix == "p" else 17)
        and (major > 26 or (major == 26 and minor >= 2))
    )
    return ChipProfile(
        metal_available=bool(metal_available),
        arch_string=arch,
        arch_gen=arch_gen,
        suffix=suffix,
        family=family,
        tier=tier,
        bf16_native=bf16_native,
        nax_capable=nax_capable,
        macos_version=macos_version,
        macos_major=major,
        macos_minor=minor,
    )


def _parse_macos_version(version: str) -> tuple[int, int]:
    parts = str(version or "").split(".")
    try:
        major = int(parts[0]) if parts and parts[0] else 0
    except ValueError:
        major = 0
    try:
        minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    except ValueError:
        minor = 0
    return major, minor
