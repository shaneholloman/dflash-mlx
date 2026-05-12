# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import builtins
import ctypes
import os
import resource
import subprocess
import sys
from typing import Any

import mlx.core as mx

GB = 1_000_000_000.0
_TASK_VM_INFO = 22
_DARWIN_TASK_VM_BYTE_FIELDS = (
    "virtual_size",
    "resident_size",
    "resident_size_peak",
    "device",
    "device_peak",
    "internal",
    "internal_peak",
    "external",
    "external_peak",
    "reusable",
    "reusable_peak",
    "purgeable_volatile_pmap",
    "purgeable_volatile_resident",
    "purgeable_volatile_virtual",
    "compressed",
    "compressed_peak",
    "compressed_lifetime",
    "phys_footprint",
)


class _DarwinTaskVmInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_int32),
        ("page_size", ctypes.c_int32),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_peak", ctypes.c_uint64),
        ("device", ctypes.c_uint64),
        ("device_peak", ctypes.c_uint64),
        ("internal", ctypes.c_uint64),
        ("internal_peak", ctypes.c_uint64),
        ("external", ctypes.c_uint64),
        ("external_peak", ctypes.c_uint64),
        ("reusable", ctypes.c_uint64),
        ("reusable_peak", ctypes.c_uint64),
        ("purgeable_volatile_pmap", ctypes.c_uint64),
        ("purgeable_volatile_resident", ctypes.c_uint64),
        ("purgeable_volatile_virtual", ctypes.c_uint64),
        ("compressed", ctypes.c_uint64),
        ("compressed_peak", ctypes.c_uint64),
        ("compressed_lifetime", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),
    ]


def process_memory_snapshot(
    *,
    include_system_wired: bool = True,
) -> dict[str, int | None]:
    rss = current_rss_bytes()
    mlx_active = mlx_memory_bytes("get_active_memory")
    mlx_cache = mlx_memory_bytes("get_cache_memory")
    task_vm = darwin_task_vm_info_bytes()
    return {
        "rss_bytes": rss,
        "phys_footprint_bytes": _task_vm_value(task_vm, "phys_footprint"),
        "darwin_resident_bytes": _task_vm_value(task_vm, "resident_size"),
        "darwin_resident_peak_bytes": _task_vm_value(task_vm, "resident_size_peak"),
        "darwin_device_bytes": _task_vm_value(task_vm, "device"),
        "darwin_device_peak_bytes": _task_vm_value(task_vm, "device_peak"),
        "darwin_internal_bytes": _task_vm_value(task_vm, "internal"),
        "darwin_internal_peak_bytes": _task_vm_value(task_vm, "internal_peak"),
        "darwin_external_bytes": _task_vm_value(task_vm, "external"),
        "darwin_reusable_bytes": _task_vm_value(task_vm, "reusable"),
        "darwin_compressed_bytes": _task_vm_value(task_vm, "compressed"),
        "system_wired_bytes": system_wired_bytes() if include_system_wired else None,
        "mlx_active_bytes": mlx_active,
        "mlx_cache_bytes": mlx_cache,
        "mlx_peak_bytes": mlx_memory_bytes("get_peak_memory"),
        "untracked_bytes": _untracked_bytes(rss, mlx_active, mlx_cache),
    }


def process_memory_bytes() -> dict[str, int | None]:
    return dict(process_memory_snapshot())


def live_memory_payload(*, wired_limit_bytes: int | None = None) -> dict[str, Any]:
    snapshot = process_memory_snapshot(include_system_wired=False)
    return {
        "rss_gb": _gb_or_none(snapshot["rss_bytes"]),
        "rss_peak_gb": _gb_or_none(rss_peak_bytes()),
        "phys_footprint_gb": _gb_or_none(snapshot["phys_footprint_bytes"]),
        "mlx_active_gb": _gb_or_none(snapshot["mlx_active_bytes"]),
        "mlx_cache_gb": _gb_or_none(snapshot["mlx_cache_bytes"]),
        "mlx_peak_gb": _gb_or_none(snapshot["mlx_peak_bytes"]),
        "wired_gb": _gb_or_none(snapshot["system_wired_bytes"]),
        "wired_limit_gb": _gb_or_none(wired_limit_bytes),
    }


def current_rss_bytes() -> int | None:
    if sys.platform == "darwin":
        return (
            darwin_proc_resident_size_bytes()
            or darwin_task_resident_size_bytes()
        )
    return linux_proc_rss_bytes()


def linux_proc_rss_bytes() -> int | None:
    try:
        with builtins.open("/proc/self/statm") as fp:
            fields = fp.read().split()
        if len(fields) < 2:
            return None
        return int(fields[1]) * int(resource.getpagesize())
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def darwin_proc_resident_size_bytes() -> int | None:
    class ProcTaskInfo(ctypes.Structure):
        _fields_ = [
            ("pti_virtual_size", ctypes.c_uint64),
            ("pti_resident_size", ctypes.c_uint64),
            ("pti_total_user", ctypes.c_uint64),
            ("pti_total_system", ctypes.c_uint64),
            ("pti_threads_user", ctypes.c_uint64),
            ("pti_threads_system", ctypes.c_uint64),
            ("pti_policy", ctypes.c_int32),
            ("pti_faults", ctypes.c_int32),
            ("pti_pageins", ctypes.c_int32),
            ("pti_cow_faults", ctypes.c_int32),
            ("pti_messages_sent", ctypes.c_int32),
            ("pti_messages_received", ctypes.c_int32),
            ("pti_syscalls_mach", ctypes.c_int32),
            ("pti_csw", ctypes.c_int32),
            ("pti_threadnum", ctypes.c_int32),
            ("pti_numrunning", ctypes.c_int32),
            ("pti_priority", ctypes.c_int32),
        ]

    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        info = ProcTaskInfo()
        result = proc_pidinfo(
            os.getpid(),
            4,  # PROC_PIDTASKINFO
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if result < ctypes.sizeof(info) or info.pti_resident_size <= 0:
            return None
        return int(info.pti_resident_size)
    except (AttributeError, OSError, ValueError):
        return None


def darwin_task_resident_size_bytes() -> int | None:
    class TimeValue(ctypes.Structure):
        _fields_ = [
            ("seconds", ctypes.c_int32),
            ("microseconds", ctypes.c_int32),
        ]

    class TaskBasicInfo64(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time", TimeValue),
            ("system_time", TimeValue),
            ("policy", ctypes.c_int32),
            ("suspend_count", ctypes.c_int32),
        ]

    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        task_info = libc.task_info
        task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        task_info.restype = ctypes.c_int32
        libc.mach_task_self.restype = ctypes.c_uint32
        info = TaskBasicInfo64()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int32))
        result = task_info(
            libc.mach_task_self(),
            5,  # TASK_BASIC_INFO_64
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if result != 0 or info.resident_size <= 0 or info.resident_size == 0xFFFFFFFF:
            return None
        return int(info.resident_size)
    except (AttributeError, OSError, ValueError):
        return None


def darwin_task_vm_info_bytes() -> dict[str, int] | None:
    if sys.platform != "darwin":
        return None

    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        task_info = libc.task_info
        task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        task_info.restype = ctypes.c_int32
        libc.mach_task_self.restype = ctypes.c_uint32
        info = _DarwinTaskVmInfo()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_int32))
        result = task_info(
            libc.mach_task_self(),
            _TASK_VM_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if result != 0 or info.phys_footprint <= 0:
            return None
        return {
            field: int(getattr(info, field))
            for field in _DARWIN_TASK_VM_BYTE_FIELDS
        }
    except (AttributeError, OSError, ValueError):
        return None


def system_wired_bytes() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.check_output(
            ["vm_stat"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None
    page_size = 4096
    wired_pages = 0
    for line in out.splitlines():
        if "page size of" in line:
            parts = [part for part in line.split() if part.isdigit()]
            if parts:
                page_size = int(parts[0])
        if line.startswith("Pages wired down:"):
            raw = line.split(":", 1)[1].strip().rstrip(".").replace(",", "")
            try:
                wired_pages = int(raw)
            except ValueError:
                wired_pages = 0
    return int(wired_pages * page_size)


def resource_rss_bytes() -> int | None:
    try:
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):
        return None
    if sys.platform == "darwin":
        return raw
    return raw * 1024


def rss_peak_bytes() -> int | None:
    return resource_rss_bytes()


def mlx_memory_bytes(name: str) -> int | None:
    fn = getattr(mx, name, None)
    if fn is None:
        return None
    try:
        return int(fn())
    except RuntimeError:
        return None


def _untracked_bytes(
    rss: int | None,
    mlx_active: int | None,
    mlx_cache: int | None,
) -> int | None:
    if rss is None or mlx_active is None or mlx_cache is None:
        return None
    return int(max(0, rss - mlx_active - mlx_cache))


def _gb_or_none(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / GB


def _task_vm_value(task_vm: dict[str, int] | None, key: str) -> int | None:
    if task_vm is None:
        return None
    value = task_vm.get(key)
    if value is None:
        return None
    return int(value)
