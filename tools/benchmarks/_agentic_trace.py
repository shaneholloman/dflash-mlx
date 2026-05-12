# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dflash_mlx.artifacts import create_run_dir, write_manifest
from tools.benchmarks._agentic_proxy import _parse_sse_event
from tools.benchmarks._agentic_session import DEFAULT_TASK

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROXY_PORT = 9788
DEFAULT_DFLASH_PORT = 8090
DEFAULT_MLXLM_PORT = 8091
OPENCODE_CONFIG = Path.home() / ".config/opencode/opencode.jsonc"
PI_CONFIG = Path.home() / ".pi/agent/models.json"
TRACE_PROVIDER_ID = "trace"
PI_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")
DEFAULT_WORKSPACE_EXCLUDES = (
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".artifacts",
    "benchmark/results",
    "build",
    "dist",
    "target",
    ".DS_Store",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
)

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"

def _wait_health(url: str, timeout_s: float, label: str) -> bool:
    deadline = time.time() + timeout_s
    last_error: BaseException | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except (OSError, TimeoutError, urllib.error.URLError) as e:
            last_error = e
        time.sleep(2)
    suffix = f" last_error={last_error!r}" if last_error is not None else ""
    sys.stderr.write(f"[orch] {label} health timeout on {url}{suffix}\n")
    return False

def _ensure_port_available(host: str, port: int, label: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as e:
            raise SystemExit(f"{label} port {host}:{port} is already in use") from e

def _server_exited_message(proc: subprocess.Popen, label: str, stderr_path: Path) -> str | None:
    rc = proc.poll()
    if rc is None:
        return None
    tail = ""
    try:
        lines = stderr_path.read_text(errors="replace").splitlines()
        tail = "\n".join(lines[-20:])
    except OSError as e:
        tail = f"<could not read stderr tail: {e!r}>"
    suffix = f"\nstderr tail:\n{tail}" if tail else ""
    return f"{label} exited before replay (code {rc}){suffix}"

def _health_model_ids(url: str) -> set[str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(f"could not read server model identity from {url}: {e!r}") from e
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return set()
    ids: set[str] = set()
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.add(item["id"])
    return ids

def _verify_ready_model(url: str, expected_model: str) -> None:
    model_ids = _health_model_ids(url)
    if model_ids and expected_model not in model_ids:
        rendered = ", ".join(sorted(model_ids))
        raise SystemExit(f"server identity mismatch: expected {expected_model!r}, got {rendered!r}")

def _validate_l2_flags(args) -> None:
    has_l2_options = args.prefix_cache_l2_dir is not None or args.prefix_cache_l2_max_bytes is not None
    if has_l2_options and args.prefix_cache_l2 is False:
        raise SystemExit("--prefix-cache-l2-dir/--prefix-cache-l2-max-bytes conflict with --no-prefix-cache-l2")

def _append_dflash_cache_flags(server_cmd: list[str], args) -> None:
    if args.prefix_cache is not None:
        server_cmd.append("--prefix-cache" if args.prefix_cache else "--no-prefix-cache")
    if args.prefix_cache_l2 is not None:
        server_cmd.append("--prefix-cache-l2" if args.prefix_cache_l2 else "--no-prefix-cache-l2")
    if args.prefix_cache_l2_dir is not None:
        Path(args.prefix_cache_l2_dir).mkdir(parents=True, exist_ok=True)
        server_cmd.extend(["--prefix-cache-l2-dir", str(args.prefix_cache_l2_dir)])
    if args.prefix_cache_l2_max_bytes is not None:
        server_cmd.extend(["--prefix-cache-l2-max-bytes", str(int(args.prefix_cache_l2_max_bytes))])

def _patch_opencode_config(target: str, proxy_port: int) -> dict[str, Any]:
    raw = OPENCODE_CONFIG.read_text()

    no_comments = re.sub(r"^\s*//.*$", "", raw, flags=re.MULTILINE)
    config = json.loads(no_comments)
    config.setdefault("provider", {})
    config["provider"][TRACE_PROVIDER_ID] = {
        "name": "Trace",
        "npm": "@ai-sdk/openai-compatible",
        "models": {
            target: {
                "name": target,
                "limit": {"context": 131072, "output": 40000},
            }
        },
        "options": {"baseURL": f"http://127.0.0.1:{proxy_port}/v1"},
    }
    OPENCODE_CONFIG.write_text(json.dumps(config, indent=2))
    return config

def _restore_opencode_config(snapshot_text: str) -> None:
    OPENCODE_CONFIG.write_text(snapshot_text)

def _patch_pi_config(target: str, proxy_port: int) -> dict[str, Any]:
    raw = PI_CONFIG.read_text()
    config = json.loads(raw)
    config.setdefault("providers", {})
    config["providers"][TRACE_PROVIDER_ID] = {
        "baseUrl": f"http://127.0.0.1:{proxy_port}/v1",
        "api": "openai-completions",
        "apiKey": TRACE_PROVIDER_ID,
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
        },
        "models": [
            {
                "id": target,
                "name": f"Trace {target}",
                "reasoning": False,
                "input": ["text"],
                "contextWindow": 65536,
                "maxTokens": 8192,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            }
        ],
    }
    PI_CONFIG.write_text(json.dumps(config, indent=2))
    return config

def _restore_pi_config(snapshot_text: str) -> None:
    PI_CONFIG.write_text(snapshot_text)

def _spawn(cmd: list[str], stdout_path: Path, stderr_path: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    stdout_f = stdout_path.open("w")
    stderr_f = stderr_path.open("w")
    return subprocess.Popen(
        cmd,
        stdout=stdout_f,
        stderr=stderr_f,
        env={**os.environ, **(env or {})},
        cwd=REPO_ROOT,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

def _run_capture(cmd: list[str], *, timeout_s: float = 2.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"cmd": cmd, "error": repr(exc)}
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

def _parse_ps_resource(raw: str) -> dict[str, Any] | None:
    parts = raw.strip().split()
    if len(parts) < 4:
        return None
    try:
        rss_kb = int(parts[0])
        vsz_kb = int(parts[1])
        cpu_pct = float(parts[2])
        mem_pct = float(parts[3])
    except ValueError:
        return None
    return {
        "rss_gb": rss_kb * 1024 / 1e9,
        "vsz_gb": vsz_kb * 1024 / 1e9,
        "cpu_pct": cpu_pct,
        "mem_pct": mem_pct,
    }

def _parse_vm_stat(raw: str) -> dict[str, Any]:
    page_size = 16384
    size_match = re.search(r"page size of (\d+) bytes", raw)
    if size_match:
        page_size = int(size_match.group(1))
    pages: dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value_match = re.search(r"(-?\d+)", value.replace(".", ""))
        if not value_match:
            continue
        normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        pages[normalized] = int(value_match.group(1))

    def gb(name: str) -> float | None:
        value = pages.get(name)
        if value is None:
            return None
        return value * page_size / 1e9

    return {
        "page_size": page_size,
        "pages": pages,
        "free_gb": gb("pages_free"),
        "active_gb": gb("pages_active"),
        "inactive_gb": gb("pages_inactive"),
        "speculative_gb": gb("pages_speculative"),
        "wired_gb": gb("pages_wired_down"),
        "compressor_gb": gb("pages_used_by_compressor"),
    }

def _thermal_flags(raw: str) -> dict[str, bool]:
    lowered = raw.lower()
    return {
        "thermal_warning_recorded": "no thermal warning" not in lowered,
        "performance_warning_recorded": "no performance warning" not in lowered,
        "cpu_power_status_recorded": "no cpu power status" not in lowered,
    }

def _system_sample(*, server_pid: int, start_perf: float) -> dict[str, Any]:
    ps_raw = _run_capture(["ps", "-p", str(int(server_pid)), "-o", "rss=,vsz=,pcpu=,pmem="])
    vm_raw = _run_capture(["vm_stat"])
    therm_raw = _run_capture(["pmset", "-g", "therm"])
    sample: dict[str, Any] = {
        "ts": _iso_now(),
        "mono_s": time.perf_counter() - start_perf,
        "server_pid": int(server_pid),
        "server_ps": _parse_ps_resource(str(ps_raw.get("stdout") or "")),
        "server_ps_raw": ps_raw,
        "vm_stat_raw": vm_raw,
        "thermal_raw": therm_raw,
    }
    if isinstance(vm_raw.get("stdout"), str):
        sample["vm_stat"] = _parse_vm_stat(vm_raw["stdout"])
    if isinstance(therm_raw.get("stdout"), str):
        sample["thermal"] = _thermal_flags(therm_raw["stdout"])
    return sample

class _SystemSampler:
    def __init__(self, *, run_dir: Path, server_pid: int, interval_s: float) -> None:
        self.path = run_dir / "system_samples.jsonl"
        self.server_pid = int(server_pid)
        self.interval_s = float(interval_s)
        self._start_perf = time.perf_counter()
        self._stop = threading.Event()
        self._write_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="dflash-system-sampler", daemon=True)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s + 1.0))
        if not self._thread.is_alive():
            self._write_sample()

    def _run(self) -> None:
        self._write_sample()
        while not self._stop.wait(self.interval_s):
            self._write_sample()

    def _write_sample(self) -> None:
        with self._write_lock:
            sample = _system_sample(server_pid=self.server_pid, start_perf=self._start_perf)
            with self.path.open("a", buffering=1) as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def _start_system_sampler(
    *, run_dir: Path, server_pid: int, interval_s: float
) -> _SystemSampler | None:
    if interval_s <= 0:
        return None
    sampler = _SystemSampler(run_dir=run_dir, server_pid=server_pid, interval_s=interval_s)
    sampler.start()
    return sampler

def _stop_system_sampler(sampler: _SystemSampler | None) -> dict[str, Any] | None:
    if sampler is None:
        return None
    try:
        sampler.stop()
    except Exception as exc:
        return {
            "ts": _iso_now(),
            "error": repr(exc),
        }
    return None

def _terminate(proc: subprocess.Popen, label: str, term_grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except OSError as e:
        sys.stderr.write(f"[orch] {label} term err: {e!r}\n")
    try:
        proc.wait(timeout=term_grace_s)
    except subprocess.TimeoutExpired:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except OSError as e:
            sys.stderr.write(f"[orch] {label} kill err: {e!r}\n")
        proc.wait(timeout=5)

def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"malformed JSONL at {path}:{lineno}: {e.msg}") from e
        if not isinstance(obj, dict):
            raise RuntimeError(f"malformed JSONL at {path}:{lineno}: expected object")
        rows.append(obj)
    return rows

def read_dflash_events(events_dir: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    posts: list[dict[str, Any]] = []
    cycles_by_req: dict[int, list[dict[str, Any]]] = {}
    cache: list[dict[str, Any]] = []

    pe = events_dir / "post_events.jsonl"
    posts.extend(_read_jsonl_objects(pe))

    ce = events_dir / "cycle_events.jsonl"
    for ev in _read_jsonl_objects(ce):
        rid = ev.get("request_id")
        if rid is not None:
            cycles_by_req.setdefault(rid, []).append(ev)

    xe = events_dir / "cache_events.jsonl"
    cache.extend(_read_jsonl_objects(xe))

    return posts, cycles_by_req, cache

def summarize_cycles(cycles: list[dict[str, Any]]) -> dict[str, Any] | None:
    cycles = [cycle for cycle in cycles if _is_engine_cycle_event(cycle)]
    if not cycles:
        return None
    n = len(cycles)
    total_commits = sum(c.get("commit_count", 0) for c in cycles)
    sorted_verify = sorted(c.get("verify_us", 0.0) for c in cycles)
    sorted_block = sorted(c.get("block_len", 0) for c in cycles)
    sorted_commit = sorted(c.get("commit_count", 0) for c in cycles)
    sorted_accept = sorted(c.get("acceptance_len", 0) for c in cycles)
    return {
        "n_cycles": n,
        "total_commits": total_commits,
        "tokens_per_cycle": (total_commits / n) if n else None,
        "mean_acceptance_len": (sum(sorted_accept) / n) if n else None,
        "mean_block_len": (sum(sorted_block) / n) if n else None,
        "mean_commit_count": (sum(sorted_commit) / n) if n else None,
        "verify_us_p50": sorted_verify[n // 2] if n else None,
        "verify_us_p99": sorted_verify[min(n - 1, max(0, int(n * 0.99) - 1))] if n else None,
    }


def _is_engine_cycle_event(event: dict[str, Any]) -> bool:
    return any(
        key in event
        for key in (
            "commit_count",
            "acceptance_len",
            "block_len",
            "verify_us",
        )
    )


def post_event_to_server_metric(
    pe: dict[str, Any],
    cycles_summary: dict[str, Any] | None,
    cache_lookup_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    wall_ms = pe.get("wall_ms") or 0.0
    gen = pe.get("generated_tokens") or 0
    tps = (gen / (wall_ms / 1000.0)) if wall_ms > 0 else None
    tokens_per_cycle = pe.get("tokens_per_cycle")
    if tokens_per_cycle is None and cycles_summary:
        tokens_per_cycle = cycles_summary.get("tokens_per_cycle")
    return {
        "tps": tps,
        "accept": pe.get("acceptance_ratio"),
        "tokens": gen,
        "wall_s": wall_ms / 1000.0 if wall_ms else None,
        "prompt_tokens": pe.get("prompt_tokens"),
        "cache_hit_tokens": pe.get("cache_hit_tokens"),
        "tokens_per_cycle": tokens_per_cycle,
        "adaptive_block_reductions": pe.get("adaptive_block_reductions"),
        "adaptive_block_cycles": pe.get("adaptive_block_cycles"),
        "adaptive_block_min": pe.get("adaptive_block_min"),

        "ttft_ms_server": pe.get("ttft_ms"),
        "prefill_ms_server": pe.get("prefill_ms"),
        "decode_ms_server": pe.get("decode_ms"),
        "decode_tok_s": pe.get("decode_tok_s"),
        "prefill_tok_s": pe.get("prefill_tok_s"),
        "prefill_tok_s_physical": pe.get("prefill_tok_s_physical"),
        "prefill_tok_s_apparent": pe.get("prefill_tok_s_apparent"),
        "logical_ctx_tokens": pe.get("logical_ctx_tokens"),
        "physical_prefill_tokens": pe.get("physical_prefill_tokens"),
        "prefill_tokens_restored": pe.get("prefill_tokens_restored"),
        "prefill_tokens_computed": pe.get("prefill_tokens_computed"),
        "cycles_completed": pe.get("cycles_completed"),
        "finish_reason_server": pe.get("finish_reason"),
        "cache_lookup_ms": pe.get("cache_lookup_ms"),
        "cache_insert_ms": pe.get("cache_insert_ms"),
        "mode_used": pe.get("mode_used"),
        "prompt_regime": pe.get("prompt_regime") or {},
        "memory_waterfall_peak": pe.get("memory_waterfall_peak") or {},
        "memory_waterfall_start": pe.get("memory_waterfall_start") or {},
        "memory_waterfall_end": pe.get("memory_waterfall_end") or {},
        "memory_boundary_start": pe.get("memory_boundary_start") or {},
        "memory_boundary_end": pe.get("memory_boundary_end") or {},
        "cache_hit_source": _cache_hit_source(cache_lookup_event),
        "cache_lookup_result": (
            cache_lookup_event.get("result") if cache_lookup_event else None
        ),
        "request_id": pe.get("request_id"),
        "cycles_summary": cycles_summary,
        "_source": "events",
    }

def summarize_cache_events(cache: list[dict[str, Any]]) -> dict[str, Any]:
    lookups = [e for e in cache if e.get("op") == "lookup"]

    hits = [e for e in lookups if e.get("result") and e.get("result") != "miss"]
    inserts = [e for e in cache if e.get("op") == "insert"]
    fingerprint_reject = sum(1 for e in lookups if e.get("fingerprint_reject"))
    miss_reasons: dict[str, int] = {}
    for event in lookups:
        if event.get("result") == "miss":
            reason = str(event.get("miss_reason") or "unknown")
            miss_reasons[reason] = miss_reasons.get(reason, 0) + 1
    return {
        "n_lookups": len(lookups),
        "n_hits": len(hits),
        "hit_rate": (len(hits) / len(lookups)) if lookups else None,
        "n_inserts": len(inserts),
        "fingerprint_rejects": fingerprint_reject,
        "total_matched_tokens": sum(e.get("matched_len", 0) for e in hits),
        "miss_reasons": miss_reasons,
        "deeper_hit_gate": summarize_deeper_hit_gate(lookups),
    }


def summarize_deeper_hit_gate(lookups: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        [event for event in lookups if isinstance(event.get("request_id"), int)],
        key=lambda event: int(event["request_id"]),
    )
    if not ordered:
        ordered = list(lookups)

    previous_matched: int | None = None
    advances = 0
    stalled: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    resets: list[dict[str, Any]] = []

    for index, event in enumerate(ordered, start=1):
        request_id = event.get("request_id", index)
        matched_len = _to_int(event.get("matched_len"), default=0)
        if matched_len <= 0:
            if previous_matched:
                resets.append(
                    {
                        "request_id": request_id,
                        "miss_reason": event.get("miss_reason"),
                        "first_divergence_pos": event.get("first_divergence_pos"),
                        "previous_matched_len": previous_matched,
                    }
                )
                previous_matched = None
            continue
        if previous_matched is None or matched_len > previous_matched:
            advances += 1
        elif matched_len == previous_matched:
            stalled.append(
                {
                    "request_id": request_id,
                    "matched_len": matched_len,
                    "previous_matched_len": previous_matched,
                }
            )
        else:
            regressions.append(
                {
                    "request_id": request_id,
                    "matched_len": matched_len,
                    "previous_matched_len": previous_matched,
                }
            )
        previous_matched = matched_len

    return {
        "pass": not stalled and not regressions,
        "advancing_hits": advances,
        "stalled_hits": stalled,
        "regressions": regressions,
        "resets": resets,
    }


def _to_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def cache_lookup_events_by_request(
    post_events: list[dict[str, Any]],
    cache_events: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    lookups = [event for event in cache_events if event.get("op") == "lookup"]
    request_tagged = [
        event for event in lookups if isinstance(event.get("request_id"), int)
    ]
    if request_tagged:
        return {int(event["request_id"]): event for event in request_tagged}
    if len(lookups) != len(post_events):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for post_event, lookup_event in zip(
        sorted(post_events, key=lambda e: e.get("request_id", 0)),
        lookups,
    ):
        request_id = post_event.get("request_id")
        if isinstance(request_id, int):
            joined = dict(lookup_event)
            joined["_join_status"] = "ordinal_unverified"
            out[int(request_id)] = joined
    return out

def _cache_hit_source(cache_lookup_event: dict[str, Any] | None) -> str | None:
    if not cache_lookup_event:
        return None
    if cache_lookup_event.get("_join_status") == "ordinal_unverified":
        return "unknown"
    result = str(cache_lookup_event.get("result") or "")
    if result in ("exact_hit", "prefix_hit"):
        return "L1"
    if result == "l2_hit":
        return "L2"
    if "disk" in result:
        return "disk"
    return None

_DFLASH_TPS_RE = re.compile(
    r"\[dflash\]\s+([\d.]+)\s+tok/s\s+\|\s+([\d.]+)%\s+accepted\s+\|\s+(\d+)\s+tokens\s+\|\s+([\d.]+)s\s+\|\s+prompt:\s+(\d+)\s+tokens"
)
_DFLASH_HIT_RE = re.compile(
    r"\[dflash\]\s+prefix\s+cache\s+hit\s+(\d+)/(\d+)\s+tokens"
)
_DFLASH_STATS_RE = re.compile(
    r"\[dflash\]\s+prefix-cache-stats.*?prefill_tokens_saved=(\d+)"
)
_PROXY_POST_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) req#(?P<idx>\d+) "
    r"/v1/chat/completions .*?body_bytes=(?P<body_bytes>\d+)"
)
_PROXY_DONE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) req#(?P<idx>\d+) "
    r"done t_total_ms=(?P<total_ms>[\d.]+)"
)

def parse_dflash_stderr(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _DFLASH_TPS_RE.search(line)
        if m:
            events.append({
                "kind": "tps",
                "raw": line,
                "tps": float(m.group(1)),
                "accept": float(m.group(2)) / 100.0,
                "tokens": int(m.group(3)),
                "wall_s": float(m.group(4)),
                "prompt_tokens": int(m.group(5)),
            })
            continue
        m = _DFLASH_HIT_RE.search(line)
        if m:
            events.append({
                "kind": "hit",
                "raw": line,
                "hit": int(m.group(1)),
                "stable": int(m.group(2)),
            })
            continue
        m = _DFLASH_STATS_RE.search(line)
        if m:
            events.append({
                "kind": "stats",
                "raw": line,
                "prefill_tokens_saved": int(m.group(1)),
            })
    return events

def _apply_dflash_stderr_totals(totals: dict[str, Any], server_stderr_text: str) -> None:
    prefill_saved = [
        ev["prefill_tokens_saved"]
        for ev in parse_dflash_stderr(server_stderr_text)
        if ev.get("kind") == "stats" and isinstance(ev.get("prefill_tokens_saved"), int)
    ]
    if prefill_saved:
        totals["prefill_tokens_saved_cumulative"] = max(prefill_saved)

def _delta_text(delta: dict[str, Any]) -> str:
    if not isinstance(delta, dict):
        return ""
    out = ""
    if isinstance(delta.get("content"), str):
        out += delta["content"]
    return out

def _delta_reasoning(delta: dict[str, Any]) -> str:
    if not isinstance(delta, dict):
        return ""
    for k in ("reasoning_content", "reasoning"):
        v = delta.get(k)
        if isinstance(v, str):
            return v
    return ""

def _delta_tool_calls(delta: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not isinstance(delta, dict):
        return None
    tc = delta.get("tool_calls")
    if isinstance(tc, list) and tc:
        return tc
    return None

def derive_post_landmarks(sse_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "first_byte_ms": None,
        "first_content_token_ms": None,
        "first_reasoning_ms": None,
        "first_tool_call_sent_ms": None,
        "tool_call_complete_ms": None,
        "finish_reason": None,
        "n_chunks": 0,
        "total_content_chars": 0,
        "total_reasoning_chars": 0,
        "saw_think_open_ms": None,
        "saw_think_close_ms": None,
        "end_t_ms": None,
        "tool_calls": [],
    }
    tool_call_indices = set()
    in_think = False

    with sse_path.open() as f:
        for lineno, raw_line in enumerate(f, start=1):
            try:
                ev = json.loads(raw_line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"malformed SSE trace JSONL at {sse_path}:{lineno}: {e.msg}") from e
            if not isinstance(ev, dict):
                raise RuntimeError(f"malformed SSE trace JSONL at {sse_path}:{lineno}: expected object")
            if ev.get("type") == "first_byte":
                out["first_byte_ms"] = ev["t_ms"]
                continue
            if ev.get("type") == "end":
                out["end_t_ms"] = ev["t_ms"]
                continue
            if ev.get("type") not in ("event", "event_tail"):
                continue
            payload = ev.get("payload") or {}
            data = payload.get("data")
            if data is None:

                continue
            t_ms = ev["t_ms"]
            out["n_chunks"] += 1
            choices = data.get("choices") or []
            for ch in choices:
                delta = ch.get("delta") or {}
                fr = ch.get("finish_reason")
                if fr and out["finish_reason"] is None:
                    out["finish_reason"] = fr
                txt = _delta_text(delta)
                rsn = _delta_reasoning(delta)
                tcs = _delta_tool_calls(delta)
                if rsn:
                    out["total_reasoning_chars"] += len(rsn)
                    if out["first_reasoning_ms"] is None:
                        out["first_reasoning_ms"] = t_ms
                if txt:
                    if not in_think and "<think>" in txt and out["saw_think_open_ms"] is None:
                        out["saw_think_open_ms"] = t_ms
                        in_think = True
                    if in_think and "</think>" in txt and out["saw_think_close_ms"] is None:
                        out["saw_think_close_ms"] = t_ms
                        in_think = False
                    out["total_content_chars"] += len(txt)
                    if out["first_content_token_ms"] is None:
                        out["first_content_token_ms"] = t_ms
                if tcs:
                    if out["first_tool_call_sent_ms"] is None:
                        out["first_tool_call_sent_ms"] = t_ms
                    for tc in tcs:
                        if tc.get("index") is not None:
                            tool_call_indices.add(tc.get("index"))
                        out["tool_calls"].append({"t_ms": t_ms, "delta": tc})
                    if fr in ("tool_calls", "function_call") and out["tool_call_complete_ms"] is None:
                        out["tool_call_complete_ms"] = t_ms
            if data.get("usage"):
                out["usage"] = data["usage"]
    if out["tool_calls"] and out["tool_call_complete_ms"] is None:
        out["tool_call_complete_ms"] = out["tool_calls"][-1]["t_ms"]
    out["tool_call_delta_count"] = len(out["tool_calls"])
    out["tool_call_count"] = len(tool_call_indices) if tool_call_indices else len(out["tool_calls"])
    return out

def derive_request_summary(req_path: Path) -> dict[str, Any]:
    obj = json.loads(req_path.read_text())
    body = obj.get("body") or {}
    msgs = body.get("messages") or []
    return {
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
        "stream": body.get("stream"),
        "n_messages": len(msgs),
        "last_role": msgs[-1].get("role") if msgs else None,
        "tools_count": len(body.get("tools") or []),
        "tool_choice": body.get("tool_choice"),
        "stream_options": body.get("stream_options"),
        "response_format": body.get("response_format"),
        "has_tool_choice": "tool_choice" in body,
        "first_message_chars": sum(len(str(m.get("content", ""))) for m in msgs[:1]),
        "total_message_chars": sum(len(str(m.get("content", ""))) for m in msgs),
    }


def summarize_prompt_transitions(
    request_files: Sequence[Path],
    cache_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bodies = [_load_request_body(path) for path in request_files]
    transitions = [
        _classify_prompt_transition(bodies[i - 1], bodies[i], request_id=i + 1)
        for i in range(1, len(bodies))
    ]
    kinds: dict[str, int] = {}
    for transition in transitions:
        kind = str(transition.get("kind") or "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1

    reset_by_request: dict[int, dict[str, Any]] = {}
    gate = (cache_summary or {}).get("deeper_hit_gate")
    if isinstance(gate, dict):
        for reset in gate.get("resets") or []:
            request_id = reset.get("request_id")
            if isinstance(request_id, int):
                reset_by_request[request_id] = reset

    reset_annotations = []
    for transition in transitions:
        request_id = transition.get("request_id")
        if not isinstance(request_id, int) or request_id not in reset_by_request:
            continue
        reset = reset_by_request[request_id]
        reset_annotations.append(
            {
                "request_id": request_id,
                "miss_reason": reset.get("miss_reason"),
                "first_divergence_pos": reset.get("first_divergence_pos"),
                "prompt_change_kind": transition.get("kind"),
                "common_message_prefix": transition.get("common_message_prefix"),
                "first_diff_index": transition.get("first_diff_index"),
                "first_diff_roles": transition.get("first_diff_roles"),
            }
        )

    return {
        "n_transitions": len(transitions),
        "change_kinds": kinds,
        "cache_reset_annotations": reset_annotations,
        "transitions": transitions,
    }


def _load_request_body(req_path: Path) -> dict[str, Any]:
    obj = json.loads(req_path.read_text())
    body = obj.get("body")
    return body if isinstance(body, dict) else {}


def _classify_prompt_transition(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    request_id: int,
) -> dict[str, Any]:
    previous_messages = _request_messages(previous)
    current_messages = _request_messages(current)
    common_messages = _common_json_prefix_len(previous_messages, current_messages)
    common_suffix = _common_json_suffix_len(
        previous_messages,
        current_messages,
        common_prefix_len=common_messages,
    )
    first_diff_index = (
        common_messages
        if common_messages < min(len(previous_messages), len(current_messages))
        else None
    )
    first_diff_roles = None
    if first_diff_index is not None:
        first_diff_roles = (
            f"{_message_role(previous_messages[first_diff_index])}"
            f"->{_message_role(current_messages[first_diff_index])}"
        )

    added_roles = [_message_role(message) for message in current_messages[common_messages:]]
    removed_roles = [_message_role(message) for message in previous_messages[common_messages:]]
    system_changed = _system_message_changed(previous_messages, current_messages)
    tools_changed = _canonical_json(previous.get("tools")) != _canonical_json(current.get("tools"))
    option_changes = _request_option_changes(previous, current)
    prompt_shrunk = len(current_messages) < len(previous_messages)

    if system_changed:
        kind = "system_prompt_mutation"
    elif tools_changed:
        kind = "tool_schema_mutation"
    elif common_messages == len(previous_messages) and len(current_messages) > len(previous_messages):
        kind = "tool_result_append" if "tool" in added_roles else "append_only"
    elif prompt_shrunk and (common_messages == len(current_messages) or common_suffix > 0):
        kind = "prompt_truncation"
    elif first_diff_index is None and option_changes:
        kind = "request_config_change"
    elif first_diff_index is None:
        kind = "unchanged"
    elif common_messages >= max(0, min(len(previous_messages), len(current_messages)) - 2):
        kind = "tail_rewrite"
    else:
        kind = "transcript_rewrite"

    return {
        "request_id": request_id,
        "kind": kind,
        "common_message_prefix": common_messages,
        "common_message_suffix": common_suffix,
        "previous_messages": len(previous_messages),
        "current_messages": len(current_messages),
        "previous_last_role": _message_role(previous_messages[-1]) if previous_messages else None,
        "current_last_role": _message_role(current_messages[-1]) if current_messages else None,
        "first_diff_index": first_diff_index,
        "first_diff_roles": first_diff_roles,
        "added_roles": added_roles,
        "removed_roles": removed_roles,
        "tools_changed": tools_changed,
        "option_changes": option_changes,
    }


def _request_messages(body: dict[str, Any]) -> list[Any]:
    messages = body.get("messages")
    return messages if isinstance(messages, list) else []


def _common_json_prefix_len(left: Sequence[Any], right: Sequence[Any]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if _canonical_json(left_item) != _canonical_json(right_item):
            break
        count += 1
    return count


def _common_json_suffix_len(
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    common_prefix_len: int,
) -> int:
    count = 0
    max_count = max(0, min(len(left), len(right)) - common_prefix_len)
    while count < max_count:
        if _canonical_json(left[-1 - count]) != _canonical_json(right[-1 - count]):
            break
        count += 1
    return count


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        role = message.get("role")
        return str(role) if role is not None else None
    return None


def _system_message_changed(left: Sequence[Any], right: Sequence[Any]) -> bool:
    if not left or not right:
        return False
    if _message_role(left[0]) != "system" or _message_role(right[0]) != "system":
        return False
    return _canonical_json(left[0]) != _canonical_json(right[0])


def _request_option_changes(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    ignored = {"messages", "tools"}
    return [
        key
        for key in sorted((set(left) | set(right)) - ignored)
        if _canonical_json(left.get(key)) != _canonical_json(right.get(key))
    ]


def _build_server_cmd(args) -> tuple[list[str], int, str]:
    if args.backend == "dflash":
        port = args.dflash_port
        cmd = [
            sys.executable,
            "-m",
            "dflash_mlx.cli",
            "serve",
            "--model",
            args.target,
            "--draft",
            args.draft,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--chat-template-args",
            args.chat_template_args,
        ]
        if args.draft_quant:
            cmd.extend(["--draft-quant", args.draft_quant])
        if args.wired_limit:
            cmd.extend(["--wired-limit", args.wired_limit])
        if args.cache_limit:
            cmd.extend(["--cache-limit", args.cache_limit])
        if args.prefill_step_size is not None:
            cmd.extend(["--prefill-step-size", str(int(args.prefill_step_size))])
        if args.draft_sink_size is not None:
            cmd.extend(["--draft-sink-size", str(int(args.draft_sink_size))])
        if args.draft_window_size is not None:
            cmd.extend(["--draft-window-size", str(int(args.draft_window_size))])
        fastpath_max_tokens = (
            0 if args.fastpath_max_tokens is None else int(args.fastpath_max_tokens)
        )
        cmd.extend(["--fastpath-max-tokens", str(fastpath_max_tokens)])
        if args.verify_len_cap is not None:
            cmd.extend(["--verify-len-cap", str(int(args.verify_len_cap))])
        if args.max_snapshot_tokens is not None:
            cmd.extend(["--max-snapshot-tokens", str(int(args.max_snapshot_tokens))])
        if args.verify_mode:
            cmd.extend(["--verify-mode", str(args.verify_mode)])
        if args.clear_cache_boundaries is not None:
            cmd.append(
                "--clear-cache-boundaries"
                if args.clear_cache_boundaries
                else "--no-clear-cache-boundaries"
            )
        if int(args.target_fa_window) > 0:
            cmd.extend(["--target-fa-window", str(int(args.target_fa_window))])
        return cmd, port, f"http://127.0.0.1:{port}"
    if args.backend == "mlxlm":
        port = args.mlxlm_port
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            args.target,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--chat-template-args",
            args.chat_template_args,
        ]
        return cmd, port, f"http://127.0.0.1:{port}"
    raise SystemExit(f"unknown backend {args.backend}")

def _build_proxy_cmd(args, run_dir: Path, upstream_url: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.benchmarks.agentic_trace",
        "proxy",
        "--listen-host",
        "127.0.0.1",
        "--listen-port",
        str(args.proxy_port),
        "--upstream-url",
        upstream_url,
        "--out-dir",
        str(run_dir),
    ]

def _write_sse_event(path, event: dict[str, Any]) -> None:
    path.write(json.dumps(event, ensure_ascii=False) + "\n")

def _replay_body_bytes(request_obj: dict[str, Any], target: str) -> bytes:
    body = dict(request_obj.get("body") or {})
    body["model"] = target
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

def _body_sha256(body_bytes: bytes) -> str:
    return hashlib.sha256(body_bytes).hexdigest()

def _replay_headers(request_obj: dict[str, Any], body_bytes: bytes) -> dict[str, str]:
    captured = request_obj.get("headers") or {}
    headers = {
        str(k): str(v)
        for k, v in captured.items()
        if str(k).lower()
        not in {
            "authorization",
            "connection",
            "content-length",
            "host",
            "x-session-affinity",
        }
    }
    headers.setdefault("Content-Type", "application/json")
    headers.setdefault("Accept", "text/event-stream")
    headers["Content-Length"] = str(len(body_bytes))
    return headers

def _post_replay_request(
    *,
    idx: int,
    request_path: Path,
    out_dir: Path,
    upstream_url: str,
    target: str,
    timeout_s: float,
) -> None:
    request_obj = json.loads(request_path.read_text())
    body_bytes = _replay_body_bytes(request_obj, target)
    body = json.loads(body_bytes.decode("utf-8"))
    path = str(request_obj.get("path") or "/v1/chat/completions")
    is_stream = bool(body.get("stream"))
    out_request = out_dir / "requests" / f"{idx:03d}.json"
    out_sse = out_dir / "sse" / f"{idx:03d}.jsonl"
    out_request.parent.mkdir(parents=True, exist_ok=True)
    out_sse.parent.mkdir(parents=True, exist_ok=True)
    out_request.write_text(
        json.dumps(
            {
                "idx": idx,
                "method": "POST",
                "path": path,
                "wall_ts": time.time(),
                "stream": is_stream,
                "headers": _replay_headers(request_obj, body_bytes),
                "body": body,
                "body_bytes": len(body_bytes),
                "body_sha256": _body_sha256(body_bytes),
                "source_body_sha256": request_obj.get("body_sha256"),
                "source_body_bytes": request_obj.get("body_bytes"),
                "source_request": str(request_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    url = upstream_url.rstrip("/") + path
    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers=_replay_headers(request_obj, body_bytes),
        method="POST",
    )
    t0 = time.perf_counter()
    status = None
    first_byte_logged = False
    with out_sse.open("w", buffering=1) as sse_f:
        try:
            resp = urllib.request.urlopen(req, timeout=timeout_s)
        except urllib.error.HTTPError as exc:
            status = exc.code
            payload = exc.read()
            _write_sse_event(
                sse_f,
                {
                    "type": "meta",
                    "t_ms": 0.0,
                    "status": status,
                    "headers": list(exc.headers.items()),
                },
            )
            _write_sse_event(
                sse_f,
                {
                    "type": "non_stream_body",
                    "t_ms": (time.perf_counter() - t0) * 1000.0,
                    "payload": {
                        "bytes": len(payload),
                        "preview": payload[:512].decode("utf-8", "replace"),
                    },
                },
            )
            _write_sse_event(sse_f, {"type": "end", "t_ms": (time.perf_counter() - t0) * 1000.0})
            raise RuntimeError(f"replay request {idx} failed with HTTP {status}") from exc

        with resp:
            status = resp.status
            _write_sse_event(
                sse_f,
                {
                    "type": "meta",
                    "t_ms": 0.0,
                    "status": status,
                    "headers": list(resp.headers.items()),
                },
            )
            if is_stream:
                buf: list[bytes] = []
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    if not first_byte_logged:
                        _write_sse_event(
                            sse_f,
                            {"type": "first_byte", "t_ms": (time.perf_counter() - t0) * 1000.0},
                        )
                        first_byte_logged = True
                    if line in (b"\n", b"\r\n"):
                        if buf:
                            _write_sse_event(
                                sse_f,
                                {
                                    "type": "event",
                                    "t_ms": (time.perf_counter() - t0) * 1000.0,
                                    "payload": _parse_sse_event(b"".join(buf)),
                                },
                            )
                        buf = []
                    else:
                        buf.append(line)
                if buf:
                    _write_sse_event(
                        sse_f,
                        {
                            "type": "event_tail",
                            "t_ms": (time.perf_counter() - t0) * 1000.0,
                            "payload": _parse_sse_event(b"".join(buf)),
                        },
                    )
            else:
                data = resp.read()
                if not first_byte_logged and data:
                    _write_sse_event(
                        sse_f,
                        {"type": "first_byte", "t_ms": (time.perf_counter() - t0) * 1000.0},
                    )
                _write_sse_event(
                    sse_f,
                    {
                        "type": "non_stream_body",
                        "t_ms": (time.perf_counter() - t0) * 1000.0,
                        "payload": {
                            "bytes": len(data),
                            "preview": data[:512].decode("utf-8", "replace"),
                        },
                    },
                )
            _write_sse_event(sse_f, {"type": "end", "t_ms": (time.perf_counter() - t0) * 1000.0})
    if status is not None and status >= 400:
        raise RuntimeError(f"replay request {idx} failed with HTTP {status}")

def _run_replay_requests(
    *,
    source_trace: Path,
    run_dir: Path,
    upstream_url: str,
    target: str,
    request_limit: int,
    request_timeout_s: float,
    pause_before_request: dict[int, float] | None = None,
) -> tuple[int, float]:
    source_requests = sorted((source_trace / "requests").glob("*.json"))
    if request_limit > 0:
        source_requests = source_requests[:request_limit]
    if not source_requests:
        raise SystemExit(f"no captured requests found under {source_trace / 'requests'}")
    pauses = pause_before_request or {}
    start = time.perf_counter()
    for idx, request_path in enumerate(source_requests, start=1):
        pause_s = pauses.get(idx)
        if pause_s:
            sys.stderr.write(f"[replay] pause {pause_s:.1f}s before POST {idx}\n")
            time.sleep(pause_s)
        sys.stderr.write(f"[replay] POST {idx}/{len(source_requests)} from {request_path.name}\n")
        _post_replay_request(
            idx=idx,
            request_path=request_path,
            out_dir=run_dir,
            upstream_url=upstream_url,
            target=target,
            timeout_s=request_timeout_s,
        )
    return len(source_requests), time.perf_counter() - start

def _parse_pause_before_request(raw_values: Sequence[str]) -> dict[int, float]:
    pauses: dict[int, float] = {}
    for raw in raw_values:
        try:
            request_raw, seconds_raw = raw.split(":", 1)
            request_idx = int(request_raw)
            pause_s = float(seconds_raw)
        except ValueError as e:
            raise SystemExit("--pause-before-request must be IDX:SECONDS") from e
        if request_idx <= 0:
            raise SystemExit("--pause-before-request IDX must be >= 1")
        if pause_s < 0:
            raise SystemExit("--pause-before-request SECONDS must be >= 0")
        pauses[request_idx] = pause_s
    return pauses

def _workspace_excluded(
    *,
    source_root: Path,
    path: Path,
    name: str,
    patterns: Sequence[str],
) -> bool:
    if path.is_symlink():
        return True
    rel = path.relative_to(source_root).as_posix()
    for pattern in patterns:
        if (
            name == pattern
            or rel == pattern
            or fnmatch.fnmatch(name, pattern)
            or fnmatch.fnmatch(rel, pattern)
            or rel.startswith(f"{pattern.rstrip('/')}/")
        ):
            return True
    return False

def _workspace_ignore(source_root: Path, patterns: Sequence[str]):
    def ignore(current_dir: str, names: list[str]) -> set[str]:
        current = Path(current_dir)
        ignored: set[str] = set()
        for name in names:
            path = current / name
            if _workspace_excluded(
                source_root=source_root,
                path=path,
                name=name,
                patterns=patterns,
            ):
                ignored.add(name)
        return ignored

    return ignore

def _workspace_file_summary(workspace: Path) -> dict[str, Any]:
    file_count = 0
    byte_count = 0
    for path in workspace.rglob("*"):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        try:
            byte_count += path.stat().st_size
        except OSError:
            continue
        file_count += 1
    return {"file_count": file_count, "bytes": byte_count}

def _prepare_workspace(
    *,
    workspace: Path,
    workspace_source: str | None,
    workspace_excludes: Sequence[str],
) -> dict[str, Any]:
    patterns = [*DEFAULT_WORKSPACE_EXCLUDES, *workspace_excludes]
    if workspace_source is None:
        return {
            "source": None,
            "copied": False,
            "exclude_patterns": patterns,
            "initial": _workspace_file_summary(workspace),
        }

    source = Path(workspace_source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"--workspace-source does not exist: {source}")
    if not source.is_dir():
        raise SystemExit(f"--workspace-source must be a directory: {source}")
    workspace_resolved = workspace.resolve()
    if source == workspace_resolved or source in workspace_resolved.parents:
        rel = workspace_resolved.relative_to(source)
        recursive_root = rel.parts[0] if rel.parts else "."
        recursive_root_path = source / recursive_root
        if not _workspace_excluded(
            source_root=source,
            path=recursive_root_path,
            name=recursive_root,
            patterns=patterns,
        ):
            raise SystemExit(
                "--workspace-source would copy the run output into itself; "
                f"use an out-root outside {source} or pass "
                f"--workspace-exclude {recursive_root!r}"
            )

    shutil.copytree(
        source,
        workspace,
        dirs_exist_ok=True,
        ignore=_workspace_ignore(source, patterns),
        symlinks=True,
    )
    return {
        "source": str(source),
        "copied": True,
        "exclude_patterns": patterns,
        "initial": _workspace_file_summary(workspace),
    }

def _build_opencode_cmd(args, workspace: Path, task: str, label: str) -> list[str]:
    cmd = [
        args.opencode_bin,
        "run",
        "--model",
        f"{TRACE_PROVIDER_ID}/{args.target}",
        "--dir",
        str(workspace.resolve()),
        "--format",
        "json",
        "--title",
        label,
        "--print-logs",
        "--log-level",
        "INFO",
    ]
    if args.thinking:
        cmd.append("--thinking")
    if args.dangerously_skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    cmd.append(task)
    return cmd

def _build_pi_cmd(args, task: str) -> list[str]:
    cmd = [
        args.pi_bin,
        "-p",
        "--model",
        f"{TRACE_PROVIDER_ID}/{args.target}",
        "--mode",
        "json",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-context-files",
    ]
    if args.pi_thinking != "off":
        cmd += ["--thinking", args.pi_thinking]
    cmd.append(task)
    return cmd

def replay_main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Replay captured OpenAI chat POST bodies against a fresh server."
    )
    p.add_argument("--source-trace", required=True, help="Trace run dir containing requests/*.json.")
    p.add_argument("--backend", choices=["dflash", "mlxlm"], required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--draft", default=None, help="required for --backend dflash")
    p.add_argument("--label", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--dflash-port", type=int, default=DEFAULT_DFLASH_PORT)
    p.add_argument("--mlxlm-port", type=int, default=DEFAULT_MLXLM_PORT)
    p.add_argument("--server-ready-timeout-s", type=float, default=300.0)
    p.add_argument("--request-timeout-s", type=float, default=900.0)
    p.add_argument("--request-limit", type=int, default=0, help="0 means replay all captured POSTs.")
    p.add_argument(
        "--system-sample-interval-s",
        type=float,
        default=0.0,
        help="Replay only: write low-overhead ps/vm_stat/thermal samples every N seconds.",
    )
    p.add_argument(
        "--pause-before-request",
        action="append",
        default=[],
        metavar="IDX:SECONDS",
        help="Replay only: keep the server alive and sleep before request IDX.",
    )
    p.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prefix-cache-l2", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prefix-cache-l2-max-bytes", type=int, default=None)
    p.add_argument("--prefix-cache-l2-dir", default=None)
    p.add_argument("--diagnostics", choices=("off", "basic", "full"), default=None)
    p.add_argument("--draft-quant", default=None)
    p.add_argument("--wired-limit", default=None, help="dflash only: forward --wired-limit to serve.")
    p.add_argument("--cache-limit", default=None, help="dflash only: forward --cache-limit to serve.")
    p.add_argument("--prefill-step-size", type=int, default=None)
    p.add_argument("--draft-sink-size", type=int, default=None)
    p.add_argument("--draft-window-size", type=int, default=None)
    p.add_argument("--fastpath-max-tokens", type=int, default=None)
    p.add_argument("--verify-len-cap", type=int, default=None)
    p.add_argument("--max-snapshot-tokens", type=int, default=None)
    p.add_argument("--verify-mode", choices=("auto", "adaptive", "off"), default=None)
    p.add_argument("--clear-cache-boundaries", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--target-fa-window", type=int, default=None)
    p.add_argument("--chat-template-args", default='{"enable_thinking":true}')
    p.add_argument("--compare-to", default=None)
    args = p.parse_args(list(argv) if argv is not None else None)
    pause_before_request = _parse_pause_before_request(args.pause_before_request)

    source_trace = Path(args.source_trace)
    if not (source_trace / "requests").is_dir():
        raise SystemExit(f"--source-trace must contain requests/*.json: {source_trace}")
    if args.backend == "dflash" and not args.draft:
        raise SystemExit("--draft is required when --backend=dflash")
    if args.backend != "dflash":
        dflash_only = {
            "--prefix-cache": args.prefix_cache,
            "--prefix-cache-l2": args.prefix_cache_l2,
            "--prefix-cache-l2-max-bytes": args.prefix_cache_l2_max_bytes,
            "--prefix-cache-l2-dir": args.prefix_cache_l2_dir,
            "--diagnostics": args.diagnostics,
            "--draft-quant": args.draft_quant,
            "--wired-limit": args.wired_limit,
            "--cache-limit": args.cache_limit,
            "--prefill-step-size": args.prefill_step_size,
            "--draft-sink-size": args.draft_sink_size,
            "--draft-window-size": args.draft_window_size,
            "--fastpath-max-tokens": args.fastpath_max_tokens,
            "--verify-len-cap": args.verify_len_cap,
            "--max-snapshot-tokens": args.max_snapshot_tokens,
            "--verify-mode": args.verify_mode,
            "--clear-cache-boundaries": args.clear_cache_boundaries,
            "--target-fa-window": args.target_fa_window,
        }
        used = [flag for flag, value in dflash_only.items() if value is not None]
        if used:
            raise SystemExit(f"{', '.join(used)} require --backend dflash")
    for flag_name in (
        "prefill_step_size",
        "draft_sink_size",
        "draft_window_size",
        "fastpath_max_tokens",
        "verify_len_cap",
        "max_snapshot_tokens",
        "request_limit",
    ):
        value = getattr(args, flag_name)
        if value is not None and int(value) < 0:
            raise SystemExit(f"--{flag_name.replace('_', '-')} must be >= 0")
    if args.system_sample_interval_s < 0:
        raise SystemExit("--system-sample-interval-s must be >= 0")
    if args.target_fa_window is not None and args.target_fa_window < 0:
        raise SystemExit("--target-fa-window must be >= 0")
    if args.prefix_cache_l2_max_bytes is not None and int(args.prefix_cache_l2_max_bytes) < 0:
        raise SystemExit("--prefix-cache-l2-max-bytes must be >= 0")
    if args.backend == "dflash":
        _validate_l2_flags(args)
    if args.backend == "dflash":
        args.prefix_cache = None if args.prefix_cache is None else bool(args.prefix_cache)
        args.prefix_cache_l2 = None if args.prefix_cache_l2 is None else bool(args.prefix_cache_l2)
        args.prefix_cache_l2_max_bytes = (
            None if args.prefix_cache_l2_max_bytes is None else int(args.prefix_cache_l2_max_bytes)
        )
        args.diagnostics = "basic" if args.diagnostics is None else args.diagnostics
        args.fastpath_max_tokens = (
            0 if args.fastpath_max_tokens is None else int(args.fastpath_max_tokens)
        )
        args.target_fa_window = 0 if args.target_fa_window is None else int(args.target_fa_window)

    label = args.label or f"{args.backend}-{source_trace.name}"
    stamp = _now_stamp()
    if args.out_root is None:
        run_dir = create_run_dir("trace", f"replay-{label}")
    else:
        run_dir = Path(args.out_root) / f"{stamp}-replay-{label}"
        run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "server").mkdir()
    (run_dir / "requests").mkdir()
    (run_dir / "sse").mkdir()

    server_cmd, server_port, upstream_url = _build_server_cmd(args)
    _ensure_port_available("127.0.0.1", server_port, "server")
    if args.backend == "dflash":
        if args.diagnostics != "off":
            events_dir = run_dir / "events"
            events_dir.mkdir(exist_ok=True)
            server_cmd.extend(
                [
                    "--diagnostics",
                    args.diagnostics,
                    "--diagnostics-dir",
                    str(events_dir),
                ]
            )
        _append_dflash_cache_flags(server_cmd, args)

    metadata = {
        "started_at": _iso_now(),
        "label": label,
        "backend": args.backend,
        "client": "replay",
        "source_trace": str(source_trace),
        "target": args.target,
        "draft": args.draft,
        "prompt_regime": {
            "harness": "replay",
            "protocol": "openai_chat_completions",
            "streaming": True,
            "dflash_runtime_input": "prompt_tokens_override"
            if args.backend == "dflash"
            else None,
        },
        "git": {
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _git(["rev-parse", "HEAD"]),
        },
        "host": platform.platform(),
        "python": sys.version,
        "server_cmd": server_cmd,
        "server_port": server_port,
        "request_limit": args.request_limit,
        "pause_before_request": pause_before_request,
        "system_sample_interval_s": args.system_sample_interval_s,
    }
    if args.backend == "dflash":
        metadata["dflash_runtime_overrides"] = {
            "draft_quant": args.draft_quant,
            "wired_limit": args.wired_limit,
            "cache_limit": args.cache_limit,
            "prefill_step_size": args.prefill_step_size,
            "draft_sink_size": args.draft_sink_size,
            "draft_window_size": args.draft_window_size,
            "fastpath_max_tokens": args.fastpath_max_tokens,
            "verify_len_cap": args.verify_len_cap,
            "max_snapshot_tokens": args.max_snapshot_tokens,
            "verify_mode": args.verify_mode,
            "clear_cache_boundaries": args.clear_cache_boundaries,
            "target_fa_window": args.target_fa_window,
            "prefix_cache": args.prefix_cache,
            "prefix_cache_l2": args.prefix_cache_l2,
            "prefix_cache_l2_dir": args.prefix_cache_l2_dir,
            "prefix_cache_l2_max_bytes": args.prefix_cache_l2_max_bytes,
            "diagnostics": args.diagnostics,
            "chat_template_args": args.chat_template_args,
        }
    write_manifest(
        run_dir,
        kind="trace",
        label=f"replay-{label}",
        argv=list(sys.argv),
        model=args.target,
        draft=args.draft,
        effective_config=metadata,
    )
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (run_dir / "server" / "cmd.txt").write_text(shlex.join(server_cmd) + "\n")

    server_proc = None
    system_sampler = None
    system_sampler_error = None
    request_count = 0
    replay_wall_s = None
    try:
        sys.stderr.write(f"[replay] starting server: {' '.join(server_cmd)}\n")
        stderr_path = run_dir / "server" / "stderr.log"
        server_proc = _spawn(server_cmd, run_dir / "server" / "stdout.log", stderr_path)
        exited = _server_exited_message(server_proc, "server", stderr_path)
        if exited:
            raise SystemExit(exited)
        if not _wait_health(f"{upstream_url}/v1/models", args.server_ready_timeout_s, "server"):
            raise SystemExit("server not ready")
        exited = _server_exited_message(server_proc, "server", stderr_path)
        if exited:
            raise SystemExit(exited)
        _verify_ready_model(f"{upstream_url}/v1/models", args.target)
        system_sampler = _start_system_sampler(
            run_dir=run_dir,
            server_pid=int(server_proc.pid),
            interval_s=float(args.system_sample_interval_s),
        )
        request_count, replay_wall_s = _run_replay_requests(
            source_trace=source_trace,
            run_dir=run_dir,
            upstream_url=upstream_url,
            target=args.target,
            request_limit=int(args.request_limit),
            request_timeout_s=float(args.request_timeout_s),
            pause_before_request=pause_before_request,
        )
    finally:
        system_sampler_error = _stop_system_sampler(system_sampler)
        if server_proc is not None:
            _terminate(server_proc, "server")

    cache_summary: dict[str, Any] = {}
    per_post_metrics: list[dict[str, Any]] = []
    if args.backend == "dflash":
        events_dir = run_dir / "events"
        post_evts, cycles_by_req, cache_evts = read_dflash_events(events_dir)
        cache_lookups_by_request = cache_lookup_events_by_request(post_evts, cache_evts)
        for pe in sorted(post_evts, key=lambda e: e.get("request_id", 0)):
            rid = pe.get("request_id")
            cycles_summary = summarize_cycles(cycles_by_req.get(rid, []))
            per_post_metrics.append(
                post_event_to_server_metric(
                    pe,
                    cycles_summary,
                    cache_lookups_by_request.get(rid),
                )
            )
        cache_summary = summarize_cache_events(cache_evts)
        (run_dir / "server" / "metrics.jsonl").write_text(
            "\n".join(json.dumps(m) for m in per_post_metrics)
            + ("\n" if per_post_metrics else "")
        )
    else:
        (run_dir / "server" / "metrics.jsonl").write_text("")

    request_files = sorted((run_dir / "requests").glob("*.json"))
    prompt_transitions = summarize_prompt_transitions(request_files, cache_summary)
    posts: list[dict[str, Any]] = []
    for i, req_path in enumerate(request_files, start=1):
        sse_path = run_dir / "sse" / req_path.name.replace(".json", ".jsonl")
        req_summary = derive_request_summary(req_path)
        landmarks = derive_post_landmarks(sse_path) if sse_path.exists() else {}
        server_metric = per_post_metrics[i - 1] if (i - 1) < len(per_post_metrics) else None
        effective_finish_reason = landmarks.get("finish_reason")
        if effective_finish_reason is None and server_metric:
            effective_finish_reason = server_metric.get("finish_reason_server")
        posts.append(
            {
                "idx": i,
                "request": req_summary,
                "landmarks": landmarks,
                "server_metric": server_metric,
                "effective_finish_reason": effective_finish_reason,
            }
        )

    _ensure_replay_outputs(posts)
    totals = _aggregate(posts, replay_wall_s)
    if args.backend == "dflash":
        _apply_dflash_stderr_totals(totals, (run_dir / "server" / "stderr.log").read_text())
    summary = {
        "metadata": metadata,
        "finished_at": _iso_now(),
        "client": "replay",
        "client_exit_code": 0,
        "client_wall_s": replay_wall_s,
        "post_count": request_count,
        "posts": posts,
        "workspace_files": [],
        "totals": totals,
        "cache_summary": cache_summary if args.backend == "dflash" else None,
        "prompt_transitions": prompt_transitions,
    }
    if system_sampler_error is not None:
        summary["system_sampler_error"] = system_sampler_error
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _write_post_rows(run_dir, summary)

    peer_summary = None
    if args.compare_to:
        peer_path = Path(args.compare_to)
        peer_json = peer_path / "summary.json" if peer_path.is_dir() else peer_path
        try:
            peer_summary = json.loads(peer_json.read_text())
        except (OSError, json.JSONDecodeError) as e:
            sys.stderr.write(f"[replay] could not load peer summary {peer_json}: {e!r}\n")
    (run_dir / "compare.md").write_text(_render_compare(summary, peer=peer_summary))

    print(f"Run directory: {run_dir}")
    print("replay exit: 0")
    print(f"Wall       : {replay_wall_s:.2f}s" if replay_wall_s is not None else "Wall       : —")
    print(f"POSTs      : {request_count}")
    print(f"Summary    : {run_dir / 'summary.json'}")
    print(f"Compare    : {run_dir / 'compare.md'}")
    return 0

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["dflash", "mlxlm"], required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--draft", default=None, help="required for --backend dflash")
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="Inline task string (used when --task-file is omitted).")
    p.add_argument("--task-file", default=None,
                   help="Path to a file holding the task prompt (overrides --task).")
    p.add_argument("--label", default=None)
    p.add_argument("--out-root", default=None, help="Output root directory (default: .artifacts/dflash/traces).")
    p.add_argument(
        "--workspace-source",
        default=None,
        help="Copy this real project directory into the run workspace before starting the client.",
    )
    p.add_argument(
        "--workspace-exclude",
        action="append",
        default=[],
        help="Extra glob/path pattern excluded when copying --workspace-source.",
    )
    p.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    p.add_argument("--dflash-port", type=int, default=DEFAULT_DFLASH_PORT)
    p.add_argument("--mlxlm-port", type=int, default=DEFAULT_MLXLM_PORT)
    p.add_argument("--server-ready-timeout-s", type=float, default=300.0)
    p.add_argument("--proxy-ready-timeout-s", type=float, default=30.0)
    p.add_argument(
        "--system-sample-interval-s",
        type=float,
        default=0.0,
        help="Write low-overhead ps/vm_stat/thermal samples every N seconds.",
    )
    p.add_argument("--client", choices=["opencode", "pi"], default="opencode",
                   help="agentic client to drive through the proxy")
    p.add_argument("--client-timeout-s", type=float, default=1800.0,
                   help="agentic client subprocess wall timeout")
    p.add_argument("--opencode-bin", default=shutil.which("opencode") or "opencode")
    p.add_argument("--pi-bin", default=shutil.which("pi") or "pi")
    p.add_argument("--pi-thinking", choices=PI_THINKING_LEVELS, default="high",
                   help="pi --thinking level (only used when --client=pi)")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True,
                   help="opencode boolean --thinking (only used when --client=opencode)")
    p.add_argument(
        "--dangerously-skip-permissions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=None,
                   help="dflash only: pass --prefix-cache/--no-prefix-cache to dflash serve")
    p.add_argument(
        "--prefix-cache-l2",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="dflash only: also pass --prefix-cache-l2 to dflash serve.",
    )
    p.add_argument(
        "--prefix-cache-l2-max-bytes",
        type=int,
        default=None,
        help="L2 cache budget in bytes (default 50 GiB).",
    )
    p.add_argument(
        "--prefix-cache-l2-dir",
        default=None,
        help="L2 cache dir (default: <run_dir>/l2_cache).",
    )
    p.add_argument(
        "--diagnostics",
        choices=("off", "basic", "full"),
        default=None,
        help="dflash only: pass --diagnostics to dflash serve (default: basic).",
    )
    p.add_argument(
        "--draft-quant",
        default=None,
        help="dflash only: pass --draft-quant to dflash serve.",
    )
    p.add_argument(
        "--wired-limit",
        default=None,
        help="dflash only: pass --wired-limit to dflash serve.",
    )
    p.add_argument(
        "--cache-limit",
        default=None,
        help="dflash only: pass --cache-limit to dflash serve.",
    )
    p.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="dflash only: pass --prefill-step-size to dflash serve.",
    )
    p.add_argument(
        "--draft-sink-size",
        type=int,
        default=None,
        help="dflash only: pass --draft-sink-size to dflash serve.",
    )
    p.add_argument(
        "--draft-window-size",
        type=int,
        default=None,
        help="dflash only: pass --draft-window-size to dflash serve.",
    )
    p.add_argument(
        "--fastpath-max-tokens",
        type=int,
        default=None,
        help="dflash only: pass --fastpath-max-tokens to dflash serve.",
    )
    p.add_argument(
        "--verify-len-cap",
        type=int,
        default=None,
        help="dflash only: pass --verify-len-cap to dflash serve.",
    )
    p.add_argument(
        "--max-snapshot-tokens",
        type=int,
        default=None,
        help="dflash only: pass --max-snapshot-tokens to dflash serve.",
    )
    p.add_argument(
        "--verify-mode",
        choices=("auto", "adaptive", "off"),
        default=None,
        help="dflash only: pass --verify-mode to dflash serve.",
    )
    p.add_argument(
        "--clear-cache-boundaries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="dflash only: pass --clear-cache-boundaries/--no-clear-cache-boundaries to dflash serve.",
    )
    p.add_argument(
        "--target-fa-window",
        type=int,
        default=None,
        help="dflash only: pass --target-fa-window to dflash serve.",
    )
    p.add_argument(
        "--chat-template-args",
        default='{"enable_thinking":true}',
        help="dflash/mlxlm: JSON passed to server --chat-template-args.",
    )
    p.add_argument(
        "--compare-to",
        default=None,
        help=(
            "Path to a prior agentic-trace run dir; emits trajectory-invariant "
            "metrics and per-POST timing only when trajectories align."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.backend == "dflash" and not args.draft:
        raise SystemExit("--draft is required when --backend=dflash")
    if args.prefix_cache_l2 and args.backend != "dflash":
        raise SystemExit("--prefix-cache-l2 requires --backend dflash")
    if args.backend != "dflash":
        dflash_only = {
            "--prefix-cache": args.prefix_cache,
            "--prefix-cache-l2": args.prefix_cache_l2,
            "--prefix-cache-l2-max-bytes": args.prefix_cache_l2_max_bytes,
            "--prefix-cache-l2-dir": args.prefix_cache_l2_dir,
            "--diagnostics": args.diagnostics,
            "--draft-quant": args.draft_quant,
            "--wired-limit": args.wired_limit,
            "--cache-limit": args.cache_limit,
            "--prefill-step-size": args.prefill_step_size,
            "--draft-sink-size": args.draft_sink_size,
            "--draft-window-size": args.draft_window_size,
            "--fastpath-max-tokens": args.fastpath_max_tokens,
            "--verify-len-cap": args.verify_len_cap,
            "--max-snapshot-tokens": args.max_snapshot_tokens,
            "--verify-mode": args.verify_mode,
            "--clear-cache-boundaries": args.clear_cache_boundaries,
            "--target-fa-window": args.target_fa_window,
        }
        used = [flag for flag, value in dflash_only.items() if value is not None]
        if used:
            raise SystemExit(f"{', '.join(used)} require --backend dflash")
    for flag_name in (
        "prefill_step_size",
        "draft_sink_size",
        "draft_window_size",
        "fastpath_max_tokens",
        "verify_len_cap",
        "max_snapshot_tokens",
    ):
        value = getattr(args, flag_name)
        if value is not None and int(value) < 0:
            raise SystemExit(f"--{flag_name.replace('_', '-')} must be >= 0")
    if args.target_fa_window is not None and args.target_fa_window < 0:
        raise SystemExit("--target-fa-window must be >= 0")
    if args.prefix_cache_l2_max_bytes is not None and int(args.prefix_cache_l2_max_bytes) < 0:
        raise SystemExit("--prefix-cache-l2-max-bytes must be >= 0")
    if args.system_sample_interval_s < 0:
        raise SystemExit("--system-sample-interval-s must be >= 0")
    if args.backend == "dflash":
        _validate_l2_flags(args)
    if args.backend == "dflash":
        args.prefix_cache = None if args.prefix_cache is None else bool(args.prefix_cache)
        args.prefix_cache_l2 = None if args.prefix_cache_l2 is None else bool(args.prefix_cache_l2)
        args.prefix_cache_l2_max_bytes = (
            None if args.prefix_cache_l2_max_bytes is None else int(args.prefix_cache_l2_max_bytes)
        )
        args.diagnostics = "basic" if args.diagnostics is None else args.diagnostics
        args.fastpath_max_tokens = (
            0 if args.fastpath_max_tokens is None else int(args.fastpath_max_tokens)
        )
        args.target_fa_window = 0 if args.target_fa_window is None else int(args.target_fa_window)

    client_timeout_s = args.client_timeout_s
    client_subdir = args.client

    label = args.label or f"{args.backend}_{Path(args.target).name}"
    stamp = _now_stamp()
    if args.out_root is None:
        run_dir = create_run_dir("trace", f"{args.client}-{label}")
    else:
        run_dir = Path(args.out_root) / f"{stamp}-{args.client}-{label}"
        run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "server").mkdir()
    (run_dir / "proxy").mkdir()
    (run_dir / client_subdir).mkdir()
    workspace = run_dir / "workspace"
    workspace.mkdir()
    workspace_source_summary = _prepare_workspace(
        workspace=workspace,
        workspace_source=args.workspace_source,
        workspace_excludes=args.workspace_exclude,
    )

    task = Path(args.task_file).read_text() if args.task_file else args.task

    server_cmd, server_port, upstream_url = _build_server_cmd(args)
    server_health_url = f"{upstream_url}/v1/models"
    proxy_cmd = _build_proxy_cmd(args, run_dir, upstream_url)
    proxy_health_url = f"http://127.0.0.1:{args.proxy_port}/v1/models"
    if args.client == "opencode":
        client_cmd = _build_opencode_cmd(args, workspace, task, label)
    elif args.client == "pi":
        client_cmd = _build_pi_cmd(args, task)
    else:
        raise SystemExit(f"unknown client {args.client}")

    if args.backend == "dflash":
        if args.diagnostics != "off":
            events_dir = run_dir / "events"
            events_dir.mkdir(exist_ok=True)
            server_cmd.extend([
                "--diagnostics",
                args.diagnostics,
                "--diagnostics-dir",
                str(events_dir),
            ])
        _append_dflash_cache_flags(server_cmd, args)

    (run_dir / "server" / "cmd.txt").write_text(shlex.join(server_cmd) + "\n")
    (run_dir / "proxy" / "cmd.txt").write_text(shlex.join(proxy_cmd) + "\n")
    (run_dir / client_subdir / "cmd.txt").write_text(shlex.join(client_cmd) + "\n")
    (run_dir / "task.txt").write_text(task)

    if args.client == "opencode":
        config_text_before = OPENCODE_CONFIG.read_text()
    else:
        config_text_before = PI_CONFIG.read_text()
    (run_dir / "config_snapshot.json").write_text(config_text_before)

    metadata = {
        "started_at": _iso_now(),
        "label": label,
        "backend": args.backend,
        "client": args.client,
        "target": args.target,
        "draft": args.draft,
        "prompt_regime": {
            "harness": args.client,
            "protocol": "openai_chat_completions",
            "streaming": True,
            "opencode_thinking": bool(args.thinking) if args.client == "opencode" else None,
            "pi_thinking": args.pi_thinking if args.client == "pi" else None,
            "dflash_runtime_input": "prompt_tokens_override"
            if args.backend == "dflash"
            else None,
        },
        "git": {
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "commit": _git(["rev-parse", "HEAD"]),
        },
        "host": platform.platform(),
        "python": sys.version,
        "server_cmd": server_cmd,
        "proxy_port": args.proxy_port,
        "server_port": server_port,
        "workspace_source": workspace_source_summary,
        "system_sample_interval_s": args.system_sample_interval_s,
    }
    if args.backend == "dflash":
        metadata["dflash_runtime_overrides"] = {
            "draft_quant": args.draft_quant,
            "wired_limit": args.wired_limit,
            "cache_limit": args.cache_limit,
            "prefill_step_size": args.prefill_step_size,
            "draft_sink_size": args.draft_sink_size,
            "draft_window_size": args.draft_window_size,
            "fastpath_max_tokens": args.fastpath_max_tokens,
            "verify_len_cap": args.verify_len_cap,
            "max_snapshot_tokens": args.max_snapshot_tokens,
            "verify_mode": args.verify_mode,
            "clear_cache_boundaries": args.clear_cache_boundaries,
            "target_fa_window": args.target_fa_window,
            "prefix_cache": args.prefix_cache,
            "prefix_cache_l2": args.prefix_cache_l2,
            "prefix_cache_l2_dir": args.prefix_cache_l2_dir,
            "prefix_cache_l2_max_bytes": args.prefix_cache_l2_max_bytes,
            "diagnostics": args.diagnostics,
            "chat_template_args": args.chat_template_args,
        }
    write_manifest(
        run_dir,
        kind="trace",
        label=f"{args.client}-{label}",
        argv=list(sys.argv),
        model=args.target,
        draft=args.draft,
        effective_config=metadata,
    )
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    server_proc = None
    proxy_proc = None
    client_proc = None
    system_sampler = None
    system_sampler_error = None
    client_returncode = None
    client_wall_s = None

    try:

        sys.stderr.write(f"[orch] starting server: {' '.join(server_cmd)}\n")
        server_proc = _spawn(server_cmd, run_dir / "server" / "stdout.log", run_dir / "server" / "stderr.log")
        if not _wait_health(server_health_url, args.server_ready_timeout_s, "server"):
            raise SystemExit("server not ready")
        system_sampler = _start_system_sampler(
            run_dir=run_dir,
            server_pid=int(server_proc.pid),
            interval_s=float(args.system_sample_interval_s),
        )

        sys.stderr.write(f"[orch] starting proxy: {' '.join(proxy_cmd)}\n")
        proxy_proc = _spawn(proxy_cmd, run_dir / "proxy" / "stdout.log", run_dir / "proxy" / "stderr.log")
        if not _wait_health(proxy_health_url, args.proxy_ready_timeout_s, "proxy"):
            raise SystemExit("proxy not ready")

        if args.client == "opencode":
            _patch_opencode_config(args.target, args.proxy_port)
        else:
            _patch_pi_config(args.target, args.proxy_port)

        sys.stderr.write(f"[orch] starting {args.client}: {' '.join(client_cmd)}\n")
        client_t0 = time.perf_counter()
        client_proc = subprocess.Popen(
            client_cmd,
            cwd=workspace,
            stdout=(run_dir / client_subdir / "stdout.jsonl").open("w"),
            stderr=(run_dir / client_subdir / "stderr.log").open("w"),
            env=os.environ.copy(),
        )
        try:
            client_returncode = client_proc.wait(timeout=client_timeout_s)
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"[orch] {args.client} timeout, killing\n")
            client_proc.kill()
            client_returncode = -9
        client_wall_s = time.perf_counter() - client_t0
    finally:

        try:
            if args.client == "opencode":
                _restore_opencode_config(config_text_before)
            else:
                _restore_pi_config(config_text_before)
        except OSError as e:
            sys.stderr.write(f"[orch] restore config err: {e!r}\n")
        if proxy_proc is not None:
            _terminate(proxy_proc, "proxy")
        system_sampler_error = _stop_system_sampler(system_sampler)
        if server_proc is not None:
            _terminate(server_proc, "server")

    server_stderr_text = (run_dir / "server" / "stderr.log").read_text()
    cache_summary: dict[str, Any] = {}
    per_post_metrics: list[dict[str, Any]] = []
    if args.backend == "dflash":
        events_dir = run_dir / "events"
        post_evts, cycles_by_req, cache_evts = read_dflash_events(events_dir)
        if post_evts:
            cache_lookups_by_request = cache_lookup_events_by_request(
                post_evts,
                cache_evts,
            )
            for pe in sorted(post_evts, key=lambda e: e.get("request_id", 0)):
                rid = pe.get("request_id")
                cycles_summary = summarize_cycles(cycles_by_req.get(rid, []))
                per_post_metrics.append(
                    post_event_to_server_metric(
                        pe,
                        cycles_summary,
                        cache_lookups_by_request.get(rid),
                    )
                )
            cache_summary = summarize_cache_events(cache_evts)
        else:
            per_post_metrics = []
        (run_dir / "server" / "metrics.jsonl").write_text(
            "\n".join(json.dumps(m) for m in per_post_metrics) + ("\n" if per_post_metrics else "")
        )
    else:
        (run_dir / "server" / "metrics.jsonl").write_text("")

    request_files = sorted((run_dir / "requests").glob("*.json"))
    prompt_transitions = summarize_prompt_transitions(request_files, cache_summary)

    posts: list[dict[str, Any]] = []
    for i, req_path in enumerate(request_files, start=1):
        sse_path = run_dir / "sse" / req_path.name.replace(".json", ".jsonl")
        req_summary = derive_request_summary(req_path)
        landmarks = derive_post_landmarks(sse_path) if sse_path.exists() else {}
        server_metric = per_post_metrics[i - 1] if (i - 1) < len(per_post_metrics) else None
        effective_finish_reason = landmarks.get("finish_reason")
        if effective_finish_reason is None and server_metric:
            effective_finish_reason = server_metric.get("finish_reason_server")
        posts.append({
            "idx": i,
            "request": req_summary,
            "landmarks": landmarks,
            "server_metric": server_metric,
            "effective_finish_reason": effective_finish_reason,
        })

    workspace_files = sorted([
        {"path": str(p.relative_to(workspace)), "bytes": p.stat().st_size}
        for p in workspace.rglob("*") if p.is_file() and ".ruff_cache" not in p.parts
    ], key=lambda d: d["path"])

    totals = _aggregate(posts, client_wall_s)
    _apply_dflash_stderr_totals(totals, server_stderr_text)

    summary = {
        "metadata": metadata,
        "finished_at": _iso_now(),
        "client": args.client,
        "client_exit_code": client_returncode,
        "client_wall_s": client_wall_s,
        "post_count": len(posts),
        "posts": posts,
        "workspace_source": workspace_source_summary,
        "workspace_files": workspace_files,
        "totals": totals,
        "cache_summary": cache_summary if args.backend == "dflash" else None,
        "prompt_transitions": prompt_transitions,
    }
    if system_sampler_error is not None:
        summary["system_sampler_error"] = system_sampler_error
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    _write_post_rows(run_dir, summary)

    peer_summary = None
    if args.compare_to:
        peer_path = Path(args.compare_to)
        peer_json = peer_path / "summary.json" if peer_path.is_dir() else peer_path
        try:
            peer_summary = json.loads(peer_json.read_text())
        except (OSError, json.JSONDecodeError) as e:
            sys.stderr.write(f"[orch] could not load peer summary {peer_json}: {e!r}\n")
    (run_dir / "compare.md").write_text(_render_compare(summary, peer=peer_summary))

    print(f"Run directory: {run_dir}")
    print(f"{args.client} exit: {client_returncode}")
    print(f"Wall         : {client_wall_s:.2f}s")
    print(f"POSTs        : {len(posts)}")
    print(f"Summary      : {run_dir / 'summary.json'}")
    print(f"Compare      : {run_dir / 'compare.md'}")
    return 0 if client_returncode == 0 else 1

def _post_view(p: dict[str, Any]) -> dict[str, Any]:
    sm = p.get("server_metric") or {}
    lm = p.get("landmarks") or {}
    usage = lm.get("usage") or {}
    fb = lm.get("first_byte_ms")
    end = lm.get("end_t_ms")
    decode_wall_s_est = ((end - fb) / 1000.0) if (fb is not None and end is not None and end > fb) else None
    prompt_tokens = sm.get("prompt_tokens") if sm.get("prompt_tokens") is not None else usage.get("prompt_tokens")
    decode_tokens = sm.get("tokens") if sm.get("tokens") is not None else usage.get("completion_tokens")
    usage_cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cache_hit_tokens = sm.get("cache_hit_tokens")
    if cache_hit_tokens is None and isinstance(usage_cached, int):
        cache_hit_tokens = usage_cached
    wall_s = sm.get("wall_s") if sm.get("wall_s") is not None else decode_wall_s_est
    tps = sm.get("tps")
    if tps is None and decode_tokens and wall_s and wall_s > 0:
        tps = decode_tokens / wall_s
    return {
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "wall_s": wall_s,
        "tps": tps,
        "accept": sm.get("accept"),
        "tokens_per_cycle": sm.get("tokens_per_cycle"),
        "cache_hit_tokens": cache_hit_tokens,
        "prefill_tokens_saved_cumulative": sm.get("prefill_tokens_saved_cumulative"),
        "tool_call_count": lm.get("tool_call_count"),
        "finish_reason": p.get("effective_finish_reason")
        or lm.get("finish_reason")
        or sm.get("finish_reason_server"),
        "source": "server" if sm else ("usage" if usage else "none"),
    }

def _int_value(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None

def _float_value(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None

def _memory_gb(memory: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = memory.get(key)
        if isinstance(value, (int, float)):
            return float(value) / 1_000_000_000.0 if key.endswith("_bytes") else float(value)
    return None

def _cache_class(cached_tok: int, cache_hit_source: str | None) -> str:
    if cached_tok <= 0:
        return "cold"
    if cache_hit_source in ("L2", "disk"):
        return "warm-l2"
    return "warm"

def _normalized_post_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    meta = summary.get("metadata") or {}
    rows: list[dict[str, Any]] = []
    for post in summary.get("posts") or []:
        view = _post_view(post)
        sm = post.get("server_metric") or {}
        lm = post.get("landmarks") or {}
        prompt_tok = _int_value(view.get("prompt_tokens"))
        cached_tok = _int_value(view.get("cache_hit_tokens")) or 0
        computed_tok = _int_value(sm.get("prefill_tokens_computed"))
        if computed_tok is None and prompt_tok is not None:
            computed_tok = max(0, prompt_tok - cached_tok)
        ctx_tok = _int_value(sm.get("logical_ctx_tokens")) or prompt_tok
        cache_hit = (
            float(cached_tok) / float(prompt_tok)
            if prompt_tok is not None and prompt_tok > 0
            else None
        )
        memory = sm.get("memory_waterfall_peak") or {}
        if not isinstance(memory, dict):
            memory = {}
        memory_start = sm.get("memory_boundary_start") or sm.get("memory_waterfall_start") or {}
        if not isinstance(memory_start, dict):
            memory_start = {}
        memory_end = sm.get("memory_boundary_end") or sm.get("memory_waterfall_end") or {}
        if not isinstance(memory_end, dict):
            memory_end = {}
        phys_start = _memory_gb(
            memory_start,
            "phys_footprint_gb",
            "phys_footprint_bytes",
        )
        phys_end = _memory_gb(memory_end, "phys_footprint_gb", "phys_footprint_bytes")
        phys_delta = (
            phys_end - phys_start
            if phys_start is not None and phys_end is not None
            else None
        )
        cache_hit_source = sm.get("cache_hit_source")
        decode_ms = _float_value(sm.get("decode_ms_server"))
        if decode_ms is None and isinstance(view.get("wall_s"), (int, float)):
            decode_ms = float(view["wall_s"]) * 1000.0
        rows.append(
            {
                "run": meta.get("label"),
                "backend": meta.get("backend"),
                "mode": summary.get("client") or meta.get("client"),
                "turn_post": post.get("idx"),
                "cache": _cache_class(cached_tok, cache_hit_source),
                "cache_hit_source": cache_hit_source,
                "prompt_tok": prompt_tok,
                "ctx_tok": ctx_tok,
                "cached_tok": cached_tok,
                "computed_tok": computed_tok,
                "cache_hit": cache_hit,
                "ttft_ms": _float_value(sm.get("ttft_ms_server"))
                or _float_value(lm.get("first_content_token_ms"))
                or _float_value(lm.get("first_tool_call_sent_ms"))
                or _float_value(lm.get("first_byte_ms")),
                "prefill_ms": _float_value(sm.get("prefill_ms_server")),
                "decode_ms": decode_ms,
                "decode_tok_s": _float_value(sm.get("decode_tok_s"))
                or _float_value(view.get("tps")),
                "out_tok": _int_value(view.get("decode_tokens")),
                "wall_s": _float_value(view.get("wall_s")),
                "acceptance": _float_value(view.get("accept")),
                "tokens_per_cycle": _float_value(view.get("tokens_per_cycle")),
                "cycles": _int_value(sm.get("cycles_completed")),
                "adaptive_block_reductions": _int_value(
                    sm.get("adaptive_block_reductions")
                ),
                "adaptive_block_cycles": _int_value(sm.get("adaptive_block_cycles")),
                "adaptive_block_min": _int_value(sm.get("adaptive_block_min")),
                "phys_footprint_peak_gb": _memory_gb(
                    memory,
                    "phys_footprint_gb",
                    "phys_footprint_peak_gb",
                    "phys_footprint_bytes",
                ),
                "phys_footprint_start_gb": phys_start,
                "phys_footprint_end_gb": phys_end,
                "phys_footprint_delta_gb": phys_delta,
                "rss_peak_gb": _memory_gb(memory, "rss_peak_gb", "rss_gb"),
                "mlx_active_peak_gb": _memory_gb(memory, "mlx_active_gb"),
                "mlx_cache_peak_gb": _memory_gb(memory, "mlx_cache_gb"),
                "mlx_peak_gb": _memory_gb(memory, "mlx_peak_gb"),
                "l1_snapshot_gb": _memory_gb(memory, "l1_snapshot_gb"),
                "l2_disk_gb": _memory_gb(memory, "l2_disk_gb"),
                "finish_reason": view.get("finish_reason"),
                "tool_calls": _int_value(view.get("tool_call_count")) or 0,
                "source": view.get("source"),
            }
        )
    return rows

_ROWS_MD_COLUMNS = (
    ("turn_post", "#"),
    ("cache", "cache"),
    ("cache_hit_source", "src"),
    ("prompt_tok", "prompt"),
    ("cached_tok", "cached"),
    ("computed_tok", "computed"),
    ("cache_hit", "hit"),
    ("ttft_ms", "TTFT ms"),
    ("prefill_ms", "prefill ms"),
    ("decode_ms", "decode ms"),
    ("decode_tok_s", "decode tok/s"),
    ("out_tok", "out"),
    ("wall_s", "wall s"),
    ("acceptance", "accept"),
    ("tokens_per_cycle", "tpc"),
    ("cycles", "cycles"),
    ("phys_footprint_peak_gb", "foot GB"),
    ("mlx_cache_peak_gb", "cache GB"),
    ("finish_reason", "finish"),
    ("tool_calls", "tools"),
)

def _rows_md_value(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if key == "cache_hit":
        return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else str(value)
    if key == "acceptance":
        return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else str(value)
    if key.endswith("_gb") or key in ("wall_s", "decode_tok_s", "tokens_per_cycle"):
        return f"{float(value):.2f}" if isinstance(value, (int, float)) else str(value)
    if key.endswith("_ms"):
        return f"{float(value):.0f}" if isinstance(value, (int, float)) else str(value)
    return str(value)

def _render_rows_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    meta = summary.get("metadata") or {}
    md: list[str] = []
    md.append(f"# Agentic rows — {meta.get('label', 'trace')}")
    md.append("")
    md.append(f"- backend: `{meta.get('backend')}`")
    md.append(f"- mode: `{summary.get('client') or meta.get('client')}`")
    md.append(f"- target: `{meta.get('target')}`")
    md.append(f"- draft: `{meta.get('draft')}`")
    md.append("")
    headers = [label for _, label in _ROWS_MD_COLUMNS]
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        md.append(
            "| "
            + " | ".join(_rows_md_value(key, row.get(key)) for key, _ in _ROWS_MD_COLUMNS)
            + " |"
        )
    return "\n".join(md) + "\n"

def _write_post_rows(run_dir: Path, summary: dict[str, Any]) -> None:
    rows = _normalized_post_rows(summary)
    (run_dir / "rows.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else "")
    )
    (run_dir / "rows.md").write_text(_render_rows_markdown(summary, rows))
    _write_timeline_artifacts(run_dir, summary, rows)

def _content_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False))

def _read_request_anatomy(req_path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(req_path.read_text())
    except OSError as e:
        raise RuntimeError(f"could not read request anatomy from {req_path}: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"malformed request JSON at {req_path}: {e.msg}") from e
    body = obj.get("body") if isinstance(obj, dict) else None
    if not isinstance(body, dict):
        return {}
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    role_counts: dict[str, int] = {}
    roles: list[str] = []
    tool_output_chars = 0
    reasoning_chars = 0
    prompt_tool_calls = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        roles.append(role)
        role_counts[role] = role_counts.get(role, 0) + 1
        if role == "tool":
            tool_output_chars += _content_len(message.get("content"))
        reasoning_chars += _content_len(message.get("reasoning_content"))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            prompt_tool_calls += len(tool_calls)
    tools = body.get("tools") or []
    if not isinstance(tools, list):
        tools = []
    body_bytes = obj.get("body_bytes")
    if not isinstance(body_bytes, int):
        body_bytes = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    body_sha256 = obj.get("body_sha256")
    if not isinstance(body_sha256, str):
        body_sha256 = None
    source_body_sha256 = obj.get("source_body_sha256")
    if not isinstance(source_body_sha256, str):
        source_body_sha256 = None
    return {
        "body_bytes": body_bytes,
        "body_sha256": body_sha256,
        "source_body_sha256": source_body_sha256,
        "message_count": len(messages),
        "role_counts": role_counts,
        "role_ladder_tail": roles[-12:],
        "tools_count": len(tools),
        "tool_output_chars": tool_output_chars,
        "assistant_reasoning_chars": reasoning_chars,
        "prompt_tool_call_count": prompt_tool_calls,
        "total_message_chars": sum(
            _content_len(message.get("content"))
            for message in messages
            if isinstance(message, dict)
        ),
    }

def _request_anatomy_by_post(run_dir: Path) -> dict[int, dict[str, Any]]:
    requests_dir = run_dir / "requests"
    out: dict[int, dict[str, Any]] = {}
    if not requests_dir.is_dir():
        return out
    for idx, req_path in enumerate(sorted(requests_dir.glob("*.json")), start=1):
        out[idx] = _read_request_anatomy(req_path)
    return out

def _parse_proxy_http_timeline(run_dir: Path) -> dict[int, dict[str, Any]]:
    proxy_log = run_dir / "proxy.log"
    if not proxy_log.exists():
        return {}
    starts: dict[int, datetime] = {}
    body_bytes: dict[int, int] = {}
    durations: dict[int, float] = {}
    for line in proxy_log.read_text(errors="replace").splitlines():
        start_match = _PROXY_POST_RE.match(line)
        if start_match:
            idx = int(start_match.group("idx"))
            starts[idx] = datetime.strptime(start_match.group("ts"), "%Y-%m-%d %H:%M:%S")
            body_bytes[idx] = int(start_match.group("body_bytes"))
            continue
        done_match = _PROXY_DONE_RE.match(line)
        if done_match:
            durations[int(done_match.group("idx"))] = float(done_match.group("total_ms")) / 1000.0
    out: dict[int, dict[str, Any]] = {}
    for idx in sorted(starts):
        start = starts[idx]
        duration_s = durations.get(idx)
        out[idx] = {
            "proxy_http_start": start.isoformat(sep=" "),
            "proxy_http_duration_s": duration_s,
            "proxy_http_gap_before_s": None,
            "proxy_body_bytes": body_bytes.get(idx),
            "proxy_gap_precision": "second_rounded",
        }
    previous_done = None
    for idx in sorted(starts):
        start = starts[idx]
        duration_s = durations.get(idx)
        if idx in out:
            out[idx]["proxy_http_gap_before_s"] = (
                (start - previous_done).total_seconds()
                if previous_done is not None
                else None
            )
        previous_done = start + timedelta(seconds=duration_s) if duration_s is not None else None
    return out

def _parse_opencode_step_timeline(run_dir: Path) -> dict[int, dict[str, Any]]:
    stdout_jsonl = run_dir / "opencode" / "stdout.jsonl"
    if not stdout_jsonl.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    message_to_step: dict[str, int] = {}
    step_start_order = 0
    step_by_finish_order = 0
    for lineno, line in enumerate(stdout_jsonl.read_text(errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"malformed OpenCode stdout JSONL at {stdout_jsonl}:{lineno}: {e.msg}"
            ) from e
        event_type = event.get("type")
        timestamp = event.get("timestamp")
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        message_id = part.get("messageID")
        if event_type == "step_start" and isinstance(timestamp, int):
            step_start_order += 1
            idx = step_start_order
            row = out.setdefault(idx, {})
            row["opencode_step_start_ts_ms"] = timestamp
            if isinstance(message_id, str):
                row["opencode_message_id"] = message_id
                message_to_step[message_id] = idx
        elif event_type == "step_finish" and isinstance(timestamp, int):
            idx = message_to_step.get(message_id) if isinstance(message_id, str) else None
            if idx is None:
                step_by_finish_order += 1
                idx = step_by_finish_order
            row = out.setdefault(idx, {})
            row["opencode_step_finish_ts_ms"] = timestamp
            tokens = part.get("tokens")
            if isinstance(tokens, dict):
                row["opencode_input_tokens"] = tokens.get("input")
                row["opencode_output_tokens"] = tokens.get("output")
                cache = tokens.get("cache")
                if isinstance(cache, dict):
                    row["opencode_cache_read_tokens"] = cache.get("read")
                    row["opencode_cache_write_tokens"] = cache.get("write")
        elif event_type == "tool_use" and isinstance(timestamp, int):
            idx = message_to_step.get(message_id) if isinstance(message_id, str) else None
            if idx is None:
                continue
            row = out.setdefault(idx, {})
            tools = row.setdefault("opencode_tool_names", [])
            tool_name = part.get("tool")
            if isinstance(tool_name, str):
                tools.append(tool_name)
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            timing = state.get("time") if isinstance(state.get("time"), dict) else {}
            start_ms = timing.get("start")
            end_ms = timing.get("end")
            if isinstance(start_ms, int) and isinstance(end_ms, int) and end_ms >= start_ms:
                row["opencode_tool_exec_ms"] = row.get("opencode_tool_exec_ms", 0) + (end_ms - start_ms)
    previous_finish: int | None = None
    for idx in sorted(out):
        row = out[idx]
        start_ms = row.get("opencode_step_start_ts_ms")
        finish_ms = row.get("opencode_step_finish_ts_ms")
        if isinstance(start_ms, int) and isinstance(finish_ms, int) and finish_ms >= start_ms:
            row["opencode_step_duration_s"] = (finish_ms - start_ms) / 1000.0
        if isinstance(start_ms, int) and previous_finish is not None:
            row["opencode_step_gap_before_s"] = (start_ms - previous_finish) / 1000.0
        if isinstance(finish_ms, int):
            previous_finish = finish_ms
    return out

def _build_timeline_rows(
    run_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    meta = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    pauses = meta.get("pause_before_request") if isinstance(meta.get("pause_before_request"), dict) else {}
    anatomy = _request_anatomy_by_post(run_dir)
    proxy = _parse_proxy_http_timeline(run_dir)
    opencode = _parse_opencode_step_timeline(run_dir)
    posts_by_idx = {
        int(post["idx"]): post
        for post in summary.get("posts", [])
        if isinstance(post, dict) and isinstance(post.get("idx"), int)
    }
    timeline: list[dict[str, Any]] = []
    for row in rows:
        idx = row.get("turn_post")
        if not isinstance(idx, int):
            continue
        post = posts_by_idx.get(idx, {})
        landmarks = post.get("landmarks") if isinstance(post.get("landmarks"), dict) else {}
        request = post.get("request") if isinstance(post.get("request"), dict) else {}
        replay_pause = pauses[str(idx)] if str(idx) in pauses else pauses.get(idx)
        proxy_gap = proxy.get(idx, {}).get("proxy_http_gap_before_s")
        proxy_gap_excluding_replay_pause = (
            max(
                0.0,
                float(proxy_gap)
                - (float(replay_pause) if isinstance(replay_pause, (int, float)) else 0.0),
            )
            if isinstance(proxy_gap, (int, float))
            else None
        )
        timeline_row = {
            **row,
            "replay_pause_before_s": replay_pause,
            "body_bytes": anatomy.get(idx, {}).get("body_bytes") or proxy.get(idx, {}).get("proxy_body_bytes"),
            "body_sha256": anatomy.get(idx, {}).get("body_sha256"),
            "source_body_sha256": anatomy.get(idx, {}).get("source_body_sha256"),
            "message_count": anatomy.get(idx, {}).get("message_count") or request.get("n_messages"),
            "role_counts": anatomy.get(idx, {}).get("role_counts"),
            "role_ladder_tail": anatomy.get(idx, {}).get("role_ladder_tail"),
            "tools_count": anatomy.get(idx, {}).get("tools_count") or request.get("tools_count"),
            "tool_output_chars": anatomy.get(idx, {}).get("tool_output_chars"),
            "assistant_reasoning_chars": anatomy.get(idx, {}).get("assistant_reasoning_chars"),
            "prompt_tool_call_count": anatomy.get(idx, {}).get("prompt_tool_call_count"),
            "total_message_chars": anatomy.get(idx, {}).get("total_message_chars") or request.get("total_message_chars"),
            "first_byte_ms": _float_value(landmarks.get("first_byte_ms")),
            "first_content_token_ms": _float_value(landmarks.get("first_content_token_ms")),
            "first_reasoning_ms": _float_value(landmarks.get("first_reasoning_ms")),
            "first_tool_call_sent_ms": _float_value(landmarks.get("first_tool_call_sent_ms")),
            "sse_end_ms": _float_value(landmarks.get("end_t_ms")),
            "cache_lookup_result": (post.get("server_metric") or {}).get("cache_lookup_result"),
            "cache_lookup_ms": (post.get("server_metric") or {}).get("cache_lookup_ms"),
            "cache_insert_ms": (post.get("server_metric") or {}).get("cache_insert_ms"),
            **proxy.get(idx, {}),
            **opencode.get(idx, {}),
        }
        timeline_row["proxy_http_gap_excluding_replay_pause_s"] = proxy_gap_excluding_replay_pause
        timeline.append(timeline_row)
    return timeline

_TIMELINE_MD_COLUMNS = (
    ("turn_post", "#"),
    ("prompt_tok", "prompt"),
    ("cached_tok", "cached"),
    ("computed_tok", "computed"),
    ("cache", "cache"),
    ("decode_tok_s", "tok/s"),
    ("out_tok", "out"),
    ("acceptance", "accept"),
    ("body_bytes", "body KB"),
    ("message_count", "msgs"),
    ("tools_count", "tools"),
    ("replay_pause_before_s", "replay sleep"),
    ("proxy_http_gap_excluding_replay_pause_s", "http gap"),
    ("opencode_step_gap_before_s", "step gap"),
    ("opencode_tool_exec_ms", "tool ms"),
    ("first_tool_call_sent_ms", "first tool"),
    ("phys_footprint_peak_gb", "foot GB"),
)

def _timeline_md_value(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if key == "acceptance":
        return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else str(value)
    if key == "body_bytes":
        return f"{float(value) / 1000.0:.1f}" if isinstance(value, (int, float)) else str(value)
    if key.endswith("_gb") or key.endswith("_s") or key == "decode_tok_s":
        return f"{float(value):.2f}" if isinstance(value, (int, float)) else str(value)
    if key.endswith("_ms"):
        return f"{float(value):.0f}" if isinstance(value, (int, float)) else str(value)
    return str(value)

def _render_timeline_markdown(summary: dict[str, Any], timeline: list[dict[str, Any]]) -> str:
    meta = summary.get("metadata") or {}
    md = [f"# Agentic timeline — {meta.get('label', 'trace')}", ""]
    md.append("Joined per-POST view: request anatomy, SSE landmarks, server rows, proxy gaps, and OpenCode step gaps when present.")
    md.append("")
    headers = [label for _, label in _TIMELINE_MD_COLUMNS]
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in timeline:
        md.append(
            "| "
            + " | ".join(_timeline_md_value(key, row.get(key)) for key, _ in _TIMELINE_MD_COLUMNS)
            + " |"
        )
    return "\n".join(md) + "\n"

def _write_timeline_artifacts(
    run_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]] | None = None,
) -> None:
    normalized = rows if rows is not None else _normalized_post_rows(summary)
    timeline = _build_timeline_rows(run_dir, summary, normalized)
    (run_dir / "timeline.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in timeline)
        + ("\n" if timeline else "")
    )
    (run_dir / "rows_timeline.md").write_text(_render_timeline_markdown(summary, timeline))

def _aggregate(posts: list[dict[str, Any]], wall_s: float | None) -> dict[str, Any]:
    total_decode_tokens = 0
    total_decode_wall_s = 0.0
    total_prompt_tokens = 0
    total_cache_hit = 0
    total_cycles = 0
    total_commits = 0
    total_tool_calls = 0
    prefill_saved_values: list[int] = []
    accept_weighted_num = 0.0
    accept_weighted_den = 0.0
    first_tool_call_ms_list: list[float] = []
    for p in posts:
        v = _post_view(p)
        if v["decode_tokens"]:
            total_decode_tokens += v["decode_tokens"]
        if v["wall_s"]:
            total_decode_wall_s += v["wall_s"]
        if v["prompt_tokens"]:
            total_prompt_tokens += v["prompt_tokens"]
        if v["cache_hit_tokens"]:
            total_cache_hit += v["cache_hit_tokens"]
        if v["tool_call_count"]:
            total_tool_calls += v["tool_call_count"]
        if isinstance(v.get("prefill_tokens_saved_cumulative"), int):
            prefill_saved_values.append(v["prefill_tokens_saved_cumulative"])
        sm = p.get("server_metric") or {}
        cyc = sm.get("cycles_summary") or {}
        if cyc:
            total_cycles += int(cyc.get("n_cycles") or 0)
            total_commits += int(cyc.get("total_commits") or 0)
        elif isinstance(sm.get("cycles_completed"), int):
            total_cycles += int(sm.get("cycles_completed") or 0)
            if isinstance(sm.get("tokens"), int):
                total_commits += int(sm.get("tokens") or 0)
        if v["accept"] is not None and v["decode_tokens"]:
            accept_weighted_num += v["accept"] * v["decode_tokens"]
            accept_weighted_den += v["decode_tokens"]
        ftc = (p.get("landmarks") or {}).get("first_tool_call_sent_ms")
        if isinstance(ftc, (int, float)):
            first_tool_call_ms_list.append(float(ftc))
    return {
        "wall_s": wall_s,
        "post_count": len(posts),
        "total_prompt_tokens": total_prompt_tokens,
        "total_decode_tokens": total_decode_tokens,
        "total_decode_wall_s": total_decode_wall_s,
        "decode_tps_avg": (total_decode_tokens / total_decode_wall_s) if total_decode_wall_s > 0 else None,
        "total_cache_hit_tokens": total_cache_hit,
        "prefill_tokens_saved_cumulative": max(prefill_saved_values) if prefill_saved_values else None,
        "total_cycles": total_cycles,
        "total_cycle_commits": total_commits,
        "avg_tokens_per_cycle": (total_commits / total_cycles) if total_cycles else None,
        "total_tool_calls": total_tool_calls,
        "weighted_acceptance": (accept_weighted_num / accept_weighted_den) if accept_weighted_den else None,
        "first_tool_call_ms_per_post": first_tool_call_ms_list,
        "first_tool_call_ms_sum": sum(first_tool_call_ms_list) if first_tool_call_ms_list else None,
        "first_tool_call_ms_avg": (sum(first_tool_call_ms_list) / len(first_tool_call_ms_list)) if first_tool_call_ms_list else None,
    }

def _post_has_model_output(post: dict[str, Any]) -> bool:
    landmarks = post.get("landmarks") or {}
    server_metric = post.get("server_metric") or {}
    if int(server_metric.get("tokens") or 0) > 0:
        return True
    usage = landmarks.get("usage") or {}
    if isinstance(usage, dict) and int(usage.get("completion_tokens") or 0) > 0:
        return True
    return bool(
        landmarks.get("first_content_token_ms") is not None
        or landmarks.get("first_tool_call_sent_ms") is not None
        or int(landmarks.get("tool_call_count") or 0) > 0
        or int(landmarks.get("tool_call_delta_count") or 0) > 0
    )

def _ensure_replay_outputs(posts: list[dict[str, Any]]) -> None:
    if posts and not any(_post_has_model_output(post) for post in posts):
        raise SystemExit(
            "replay completed without model output; inspect server/stderr.log for handler errors"
        )

def _ms(v):
    return f"{v:.0f}" if isinstance(v, (int, float)) else "—"

def _render_compare(summary: dict[str, Any], peer: dict[str, Any] | None = None) -> str:
    md = []
    meta = summary["metadata"]
    tot = summary["totals"]
    md.append(f"# Agentic trace — {meta['label']}")
    md.append("")
    md.append(f"- backend: `{meta['backend']}`")
    md.append(f"- target: `{meta['target']}`")
    md.append(f"- draft: `{meta.get('draft')}`")
    md.append(f"- commit: `{meta['git']['commit']}`")
    server_cmd = meta.get("server_cmd")
    if isinstance(server_cmd, list):
        md.append(f"- server_cmd: `{shlex.join(str(part) for part in server_cmd)}`")
    else:
        md.append("- server_cmd: —")
    md.append(f"- prompt_regime: `{meta.get('prompt_regime')}`")
    md.append(f"- wall_s: **{tot['wall_s']:.2f}**" if tot["wall_s"] else "- wall_s: —")
    md.append(f"- POSTs: **{summary['post_count']}**")
    md.append(f"- total prompt tokens (sum across POSTs): {tot['total_prompt_tokens']}")
    md.append(f"- total decode tokens: {tot['total_decode_tokens']}")
    md.append(f"- decode tps avg: {tot['decode_tps_avg']:.2f}" if tot["decode_tps_avg"] else "- decode tps avg: —")
    md.append(f"- weighted acceptance: {tot['weighted_acceptance']}")
    md.append(f"- cycles: {tot.get('total_cycles', 0)}")
    md.append(
        f"- avg tokens/cycle: {tot['avg_tokens_per_cycle']:.2f}"
        if tot.get("avg_tokens_per_cycle") is not None
        else "- avg tokens/cycle: —"
    )
    md.append(f"- tool calls: {tot.get('total_tool_calls', 0)}")
    md.append(f"- total cache hit tokens: {tot['total_cache_hit_tokens']}")
    if tot.get("first_tool_call_ms_avg") is not None:
        md.append(f"- first_tool_call_ms (avg over POSTs that emit a tool call): {tot['first_tool_call_ms_avg']:.0f}")
    md.append("")
    cs = summary.get("cache_summary")
    if cs:
        md.append(
            f"- prefix-cache: lookups={cs.get('n_lookups')} hits={cs.get('n_hits')} "
            f"hit_rate={cs.get('hit_rate'):.2%}".rstrip()
            if isinstance(cs.get("hit_rate"), float)
            else f"- prefix-cache: lookups={cs.get('n_lookups')} hits={cs.get('n_hits')}"
        )
        gate = cs.get("deeper_hit_gate")
        if isinstance(gate, dict):
            md.append(
                "- deeper-hit gate: "
                f"{'pass' if gate.get('pass') else 'fail'} "
                f"advancing_hits={gate.get('advancing_hits', 0)} "
                f"stalled={len(gate.get('stalled_hits') or [])} "
                f"regressions={len(gate.get('regressions') or [])} "
                f"resets={len(gate.get('resets') or [])}"
            )
            reset = (gate.get("resets") or [])[:3]
            if reset:
                rendered = ", ".join(
                    (
                        f"#{item.get('request_id')} {item.get('miss_reason')}"
                        f"@{item.get('first_divergence_pos')}"
                    )
                    for item in reset
                )
                md.append(f"- deeper-hit resets: {rendered}")
    pt = summary.get("prompt_transitions")
    if isinstance(pt, dict) and pt.get("n_transitions"):
        md.append(f"- prompt transitions: {pt.get('change_kinds')}")
        reset_annotations = (pt.get("cache_reset_annotations") or [])[:3]
        if reset_annotations:
            rendered = ", ".join(
                (
                    f"#{item.get('request_id')} {item.get('prompt_change_kind')} "
                    f"msg_prefix={item.get('common_message_prefix')}"
                )
                for item in reset_annotations
            )
            md.append(f"- cache reset prompt causes: {rendered}")
    md.append("")
    if peer is not None:
        md.append(_render_peer_comparison(summary, peer))
        md.append("")
    md.append("## Per-POST")
    md.append("")
    md.append("| # | prompt | decode | wall_s | tps | accept | tpc | cache_hit | ttft_srv | prefill_srv | decode_srv | cycles | tools | first_byte | first_content | first_tool | finish | src |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in summary["posts"]:
        v = _post_view(p)
        lm = p.get("landmarks") or {}
        sm = p.get("server_metric") or {}
        md.append(
            "| {idx} | {prompt} | {decode} | {wall} | {tps} | {accept} | {tpc} | {hit} | {ttft_s} | {pf_s} | {dc_s} | {cyc} | {tools} | {fb} | {fc} | {ftc} | {fin} | {src} |".format(
                idx=p["idx"],
                prompt=v["prompt_tokens"] if v["prompt_tokens"] is not None else "—",
                decode=v["decode_tokens"] if v["decode_tokens"] is not None else "—",
                wall=f"{v['wall_s']:.2f}" if v["wall_s"] is not None else "—",
                tps=f"{v['tps']:.1f}" if v["tps"] is not None else "—",
                accept=f"{v['accept']*100:.1f}%" if v["accept"] is not None else "—",
                tpc=f"{v['tokens_per_cycle']:.2f}" if v["tokens_per_cycle"] is not None else "—",
                hit=v["cache_hit_tokens"] if v["cache_hit_tokens"] is not None else "—",
                ttft_s=_ms(sm.get("ttft_ms_server")),
                pf_s=_ms(sm.get("prefill_ms_server")),
                dc_s=_ms(sm.get("decode_ms_server")),
                cyc=sm.get("cycles_completed", "—") if sm.get("cycles_completed") is not None else "—",
                tools=v["tool_call_count"] if v["tool_call_count"] is not None else "—",
                fb=_ms(lm.get("first_byte_ms")),
                fc=_ms(lm.get("first_content_token_ms")),
                ftc=_ms(lm.get("first_tool_call_sent_ms")),
                fin=v["finish_reason"] or "—",
                src=v["source"],
            )
        )
    md.append("")

    cycle_lines = []
    for p in summary["posts"]:
        sm = p.get("server_metric") or {}
        cyc = sm.get("cycles_summary")
        if cyc:
            cycle_lines.append(
                f"| {p['idx']} | {cyc['n_cycles']} | {cyc['total_commits']} | "
                f"{cyc['tokens_per_cycle']:.2f} | {cyc['mean_acceptance_len']:.2f} | {cyc['mean_block_len']:.2f} | "
                f"{cyc['verify_us_p50']/1000:.1f} | {cyc['verify_us_p99']/1000:.1f} |"
            )
    if cycle_lines:
        md.append("## Cycle stats (per-POST)")
        md.append("")
        md.append("| # | cycles | commits | tpc | avg_accept_len | avg_block_len | verify_p50_ms | verify_p99_ms |")
        md.append("|---|---|---|---|---|---|---|---|")
        md.extend(cycle_lines)
        md.append("")
    md.append("## Workspace files")
    md.append("")
    for f in summary["workspace_files"]:
        md.append(f"- `{f['path']}` ({f['bytes']} bytes)")
    return "\n".join(md) + "\n"

def _fmt_num(v: Any, digits: int = 2) -> str:
    return f"{float(v):.{digits}f}" if isinstance(v, (int, float)) else "—"

def _fmt_int_or_na(v: Any) -> str:
    return str(int(v)) if isinstance(v, int) else "n/a"

def _fmt_delta_int(a: Any, b: Any) -> str:
    return f"{int(a) - int(b):+d}" if isinstance(a, int) and isinstance(b, int) else "—"

def _fmt_pct_delta(a: Any, b: Any) -> str:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b != 0:
        return f"{((float(a) - float(b)) / float(b)) * 100:+.1f}%"
    return "—"

def _fmt_ms_delta(a: Any, b: Any) -> str:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return f"{float(a) - float(b):+.1f} ms"
    return "—"

def _fmt_delta_num(a: Any, b: Any, digits: int = 3) -> str:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return f"{float(a) - float(b):+.{digits}f}"
    return "—"

def _fmt_ratio(a: Any, b: Any) -> str:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b != 0:
        return f"{float(a) / float(b):.3f}"
    return "—"

def _observed_response_ms_per_output_token(totals: dict[str, Any]) -> float | None:
    tokens = totals.get("total_decode_tokens")
    wall_s = totals.get("total_decode_wall_s")
    if isinstance(tokens, int) and tokens > 0 and isinstance(wall_s, (int, float)):
        return (float(wall_s) / float(tokens)) * 1000.0
    return None

def _fmt_acceptance(summary: dict[str, Any]) -> str:
    value = (summary.get("totals") or {}).get("weighted_acceptance")
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    backend = (summary.get("metadata") or {}).get("backend")
    return "n/a (no draft)" if backend != "dflash" else "n/a"

def _decode_tokens_aligned(a: int, b: int) -> bool:
    if a == b == 0:
        return True
    if b == 0:
        return False
    return abs(a - b) / b <= 0.05

def _render_peer_comparison(this: dict[str, Any], peer: dict[str, Any]) -> str:
    md: list[str] = []
    this_label = this["metadata"]["label"]
    peer_label = peer["metadata"]["label"]
    this_tot = this.get("totals") or {}
    peer_tot = peer.get("totals") or {}
    posts_this = this.get("posts") or []
    posts_peer = peer.get("posts") or []
    n_this = len(posts_this)
    n_peer = len(posts_peer)
    decode_this = int(this_tot.get("total_decode_tokens") or 0)
    decode_peer = int(peer_tot.get("total_decode_tokens") or 0)
    tools_this = int(this_tot.get("total_tool_calls") or 0)
    tools_peer = int(peer_tot.get("total_tool_calls") or 0)
    trajectories_aligned = n_this == n_peer and _decode_tokens_aligned(decode_this, decode_peer)
    this_ms_per_token = _observed_response_ms_per_output_token(this_tot)
    peer_ms_per_token = _observed_response_ms_per_output_token(peer_tot)
    this_prefill_saved = this_tot.get("prefill_tokens_saved_cumulative")
    peer_prefill_saved = peer_tot.get("prefill_tokens_saved_cumulative")

    md.append(f"## Verdict — {this_label} vs {peer_label}")
    md.append("")
    md.append("### Trajectory parity check")
    md.append("")
    md.append(f"- {this_label}: {n_this} POSTs, {decode_this} decode tokens, {tools_this} tool calls")
    md.append(f"- {peer_label}: {n_peer} POSTs, {decode_peer} decode tokens, {tools_peer} tool calls")
    md.append("")
    if trajectories_aligned:
        md.append("✅ Trajectories aligned — per-POST comparison valid below.")
    else:
        md.append("⚠️  **TRAJECTORY DIVERGED** — per-POST timing comparison is invalid.")
        md.append("")
        md.append("Trajectories diverged; per-POST timing comparison is invalid.")
        md.append("Use trajectory-robust aggregate metrics for cross-runtime comparison.")
    md.append("")
    md.append("### Trajectory-robust aggregate metrics")
    md.append("")
    md.append(f"| Metric | {this_label} | {peer_label} | delta |")
    md.append("|---|---|---|---|")
    md.append(
        f"| decode_tps_avg | {_fmt_num(this_tot.get('decode_tps_avg'))} | "
        f"{_fmt_num(peer_tot.get('decode_tps_avg'))} | "
        f"{_fmt_pct_delta(this_tot.get('decode_tps_avg'), peer_tot.get('decode_tps_avg'))} |"
    )
    md.append(
        f"| observed_response_ms_per_output_token | {_fmt_num(this_ms_per_token, 1)} | "
        f"{_fmt_num(peer_ms_per_token, 1)} | {_fmt_ms_delta(this_ms_per_token, peer_ms_per_token)} |"
    )
    md.append(
        f"| prefill_tokens_saved (cumulative) | {_fmt_int_or_na(this_prefill_saved)} | "
        f"{_fmt_int_or_na(peer_prefill_saved)} | {_fmt_delta_int(this_prefill_saved, peer_prefill_saved)} |"
    )
    md.append(
        f"| total_cache_hit_tokens | {int(this_tot.get('total_cache_hit_tokens') or 0)} | "
        f"{int(peer_tot.get('total_cache_hit_tokens') or 0)} | "
        f"{int(this_tot.get('total_cache_hit_tokens') or 0) - int(peer_tot.get('total_cache_hit_tokens') or 0):+d} |"
    )
    md.append(
        f"| weighted_acceptance | {_fmt_acceptance(this)} | {_fmt_acceptance(peer)} | "
        f"{_fmt_delta_num(this_tot.get('weighted_acceptance'), peer_tot.get('weighted_acceptance'))} |"
    )
    md.append("")
    md.append("### Trajectory-dependent metrics (informational only)")
    md.append("")
    md.append("These depend on which tokens each runtime decoded — different runtimes reach")
    md.append("the goal via different paths, so direct comparison is misleading.")
    md.append("")
    md.append(f"| Metric | {this_label} | {peer_label} | ratio (this/peer) |")
    md.append("|---|---|---|---|")
    md.append(
        f"| wall_s | {_fmt_num(this_tot.get('wall_s'))} | {_fmt_num(peer_tot.get('wall_s'))} | "
        f"{_fmt_ratio(this_tot.get('wall_s'), peer_tot.get('wall_s'))} |"
    )
    md.append(f"| total_decode_tokens | {decode_this} | {decode_peer} | — |")
    md.append(
        f"| total_prompt_tokens | {int(this_tot.get('total_prompt_tokens') or 0)} | "
        f"{int(peer_tot.get('total_prompt_tokens') or 0)} | — |"
    )
    md.append(f"| total_tool_calls | {tools_this} | {tools_peer} | — |")
    md.append(f"| POST count | {n_this} | {n_peer} | — |")
    md.append("")
    md.append("⚠️  The wall_s ratio reflects total elapsed time including how many agentic")
    md.append("turns each runtime took. If POST counts differ, wall_s is **not** a runtime-")
    md.append("speed comparison.")
    md.append("")
    md.append("### Per-POST timing alignment")
    md.append("")
    if trajectories_aligned:
        md.append("Per-POST `first_tool_call_ms` gap (this − peer; negative = this is faster):")
        md.append("")
        md.append("| # | this | peer | gap_ms |")
        md.append("|---|---|---|---|")
        gaps: list[float] = []
        for i in range(n_this):
            a = (posts_this[i].get("landmarks") or {}).get("first_tool_call_sent_ms")
            b = (posts_peer[i].get("landmarks") or {}).get("first_tool_call_sent_ms")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                gap = float(a) - float(b)
                gaps.append(gap)
                md.append(f"| {i+1} | {a:.0f} | {b:.0f} | {gap:+.0f} |")
            else:
                md.append(f"| {i+1} | {_ms(a)} | {_ms(b)} | — |")
        md.append("")
        a_tot = this_tot.get("first_tool_call_ms_sum")
        b_tot = peer_tot.get("first_tool_call_ms_sum")
        if isinstance(a_tot, (int, float)) and isinstance(b_tot, (int, float)):
            gap_sum = float(a_tot) - float(b_tot)
        else:
            gap_sum = sum(gaps) if gaps else None
        if gap_sum is not None:
            md.append(f"- **tool_call_latency_gap (sum)**: {gap_sum:+.0f} ms ({this_label} − {peer_label})")
        if gaps:
            avg = sum(gaps) / len(gaps)
            md.append(f"- **tool_call_latency_gap (avg per POST)**: {avg:+.0f} ms")
    else:
        md.append("Skipped — trajectories diverged (see warning above).")
    return "\n".join(md)

if __name__ == "__main__":
    raise SystemExit(main())
