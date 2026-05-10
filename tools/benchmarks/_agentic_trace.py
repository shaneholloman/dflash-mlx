# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)


from __future__ import annotations

import argparse
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
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from datetime import datetime
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
    if has_l2_options and not bool(args.prefix_cache_l2):
        raise SystemExit("--prefix-cache-l2-dir/--prefix-cache-l2-max-bytes require --prefix-cache-l2")

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

def post_event_to_server_metric(pe: dict[str, Any], cycles_summary: dict[str, Any] | None) -> dict[str, Any]:
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

        "ttft_ms_server": pe.get("ttft_ms"),
        "prefill_ms_server": pe.get("prefill_ms"),
        "decode_ms_server": pe.get("decode_ms"),
        "cycles_completed": pe.get("cycles_completed"),
        "finish_reason_server": pe.get("finish_reason"),
        "cache_lookup_ms": pe.get("cache_lookup_ms"),
        "cache_insert_ms": pe.get("cache_insert_ms"),
        "mode_used": pe.get("mode_used"),
        "prompt_regime": pe.get("prompt_regime") or {},
        "request_id": pe.get("request_id"),
        "cycles_summary": cycles_summary,
        "_source": "events",
    }

def summarize_cache_events(cache: list[dict[str, Any]]) -> dict[str, Any]:
    lookups = [e for e in cache if e.get("op") == "lookup"]

    hits = [e for e in lookups if e.get("result") and e.get("result") != "miss"]
    inserts = [e for e in cache if e.get("op") == "insert"]
    fingerprint_reject = sum(1 for e in lookups if e.get("fingerprint_reject"))
    return {
        "n_lookups": len(lookups),
        "n_hits": len(hits),
        "hit_rate": (len(hits) / len(lookups)) if lookups else None,
        "n_inserts": len(inserts),
        "fingerprint_rejects": fingerprint_reject,
        "total_matched_tokens": sum(e.get("matched_len", 0) for e in hits),
    }

_DFLASH_TPS_RE = re.compile(
    r"\[dflash\]\s+([\d.]+)\s+tok/s\s+\|\s+([\d.]+)%\s+accepted\s+\|\s+(\d+)\s+tokens\s+\|\s+([\d.]+)s\s+\|\s+prompt:\s+(\d+)\s+tokens"
)
_DFLASH_HIT_RE = re.compile(
    r"\[dflash\]\s+prefix\s+cache\s+hit\s+(\d+)/(\d+)\s+tokens"
)
_DFLASH_STATS_RE = re.compile(
    r"\[dflash\]\s+prefix-cache-stats.*?prefill_tokens_saved=(\d+)"
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

def attach_dflash_metrics_to_posts(events: list[dict[str, Any]], n_posts: int) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    pending: dict[str, Any] = {"hit": None, "stats": None, "all": []}
    for ev in events:
        pending["all"].append(ev)
        if ev["kind"] == "hit":
            pending["hit"] = ev
        elif ev["kind"] == "stats":
            pending["stats"] = ev
        elif ev["kind"] == "tps":
            buckets.append({
                "tps": ev["tps"],
                "accept": ev["accept"],
                "tokens": ev["tokens"],
                "wall_s": ev["wall_s"],
                "prompt_tokens": ev["prompt_tokens"],
                "cache_hit_tokens": pending["hit"]["hit"] if pending["hit"] else None,
                "stable_prefix": pending["hit"]["stable"] if pending["hit"] else None,
                "prefill_tokens_saved_cumulative": pending["stats"]["prefill_tokens_saved"] if pending["stats"] else None,
            })
            pending = {"hit": None, "stats": None, "all": []}

    merged: list[dict[str, Any]] = []
    for b in buckets:
        if merged and merged[-1]["prompt_tokens"] == b["prompt_tokens"] and b["tokens"] >= merged[-1]["tokens"]:
            merged[-1] = b
        else:
            merged.append(b)
    return merged

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
    accumulated_tool_call_args: list[str] = []
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
        if args.profile:
            cmd.extend(["--profile", args.profile])
        if args.prefill_step_size is not None:
            cmd.extend(["--prefill-step-size", str(int(args.prefill_step_size))])
        if args.fastpath_max_tokens is not None:
            cmd.extend(["--fastpath-max-tokens", str(int(args.fastpath_max_tokens))])
        if args.max_snapshot_tokens is not None:
            cmd.extend(["--max-snapshot-tokens", str(int(args.max_snapshot_tokens))])
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
) -> tuple[int, float]:
    source_requests = sorted((source_trace / "requests").glob("*.json"))
    if request_limit > 0:
        source_requests = source_requests[:request_limit]
    if not source_requests:
        raise SystemExit(f"no captured requests found under {source_trace / 'requests'}")
    start = time.perf_counter()
    for idx, request_path in enumerate(source_requests, start=1):
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
    p.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prefix-cache-l2", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prefix-cache-l2-max-bytes", type=int, default=None)
    p.add_argument("--prefix-cache-l2-dir", default=None)
    p.add_argument("--diagnostics", choices=("off", "basic", "full"), default=None)
    p.add_argument("--draft-quant", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--prefill-step-size", type=int, default=None)
    p.add_argument("--fastpath-max-tokens", type=int, default=None)
    p.add_argument("--max-snapshot-tokens", type=int, default=None)
    p.add_argument("--target-fa-window", type=int, default=None)
    p.add_argument("--chat-template-args", default='{"enable_thinking":true}')
    p.add_argument("--compare-to", default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

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
            "--profile": args.profile,
            "--prefill-step-size": args.prefill_step_size,
            "--fastpath-max-tokens": args.fastpath_max_tokens,
            "--max-snapshot-tokens": args.max_snapshot_tokens,
            "--target-fa-window": args.target_fa_window,
        }
        used = [flag for flag, value in dflash_only.items() if value is not None]
        if used:
            raise SystemExit(f"{', '.join(used)} require --backend dflash")
    for flag_name in (
        "prefill_step_size",
        "fastpath_max_tokens",
        "max_snapshot_tokens",
        "request_limit",
    ):
        value = getattr(args, flag_name)
        if value is not None and int(value) < 0:
            raise SystemExit(f"--{flag_name.replace('_', '-')} must be >= 0")
    if args.target_fa_window is not None and args.target_fa_window < 0:
        raise SystemExit("--target-fa-window must be >= 0")
    if args.prefix_cache_l2_max_bytes is not None and int(args.prefix_cache_l2_max_bytes) < 0:
        raise SystemExit("--prefix-cache-l2-max-bytes must be >= 0")
    if args.backend == "dflash":
        _validate_l2_flags(args)
    if args.backend == "dflash":
        args.prefix_cache = True if args.prefix_cache is None else bool(args.prefix_cache)
        args.prefix_cache_l2 = False if args.prefix_cache_l2 is None else bool(args.prefix_cache_l2)
        args.prefix_cache_l2_max_bytes = (
            50 * 1024**3
            if args.prefix_cache_l2_max_bytes is None
            else int(args.prefix_cache_l2_max_bytes)
        )
        args.diagnostics = "basic" if args.diagnostics is None else args.diagnostics
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
        if args.prefix_cache:
            server_cmd.extend(
                [
                    "--prefix-cache",
                    "--prefix-cache-max-entries",
                    "8",
                    "--prefix-cache-max-bytes",
                    "10737418240",
                ]
            )
        else:
            server_cmd.append("--no-prefix-cache")
        if args.prefix_cache_l2:
            l2_dir = (
                Path(args.prefix_cache_l2_dir)
                if args.prefix_cache_l2_dir
                else run_dir / "l2_cache"
            )
            l2_dir.mkdir(parents=True, exist_ok=True)
            server_cmd.extend(
                [
                    "--prefix-cache-l2",
                    "--prefix-cache-l2-dir",
                    str(l2_dir),
                    "--prefix-cache-l2-max-bytes",
                    str(int(args.prefix_cache_l2_max_bytes)),
                ]
            )

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
    }
    if args.backend == "dflash":
        metadata["dflash_runtime_overrides"] = {
            "draft_quant": args.draft_quant,
            "profile": args.profile,
            "prefill_step_size": args.prefill_step_size,
            "fastpath_max_tokens": args.fastpath_max_tokens,
            "max_snapshot_tokens": args.max_snapshot_tokens,
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
        request_count, replay_wall_s = _run_replay_requests(
            source_trace=source_trace,
            run_dir=run_dir,
            upstream_url=upstream_url,
            target=args.target,
            request_limit=int(args.request_limit),
            request_timeout_s=float(args.request_timeout_s),
        )
    finally:
        if server_proc is not None:
            _terminate(server_proc, "server")

    cache_summary: dict[str, Any] = {}
    per_post_metrics: list[dict[str, Any]] = []
    if args.backend == "dflash":
        events_dir = run_dir / "events"
        post_evts, cycles_by_req, cache_evts = read_dflash_events(events_dir)
        for pe in sorted(post_evts, key=lambda e: e.get("request_id", 0)):
            rid = pe.get("request_id")
            cycles_summary = summarize_cycles(cycles_by_req.get(rid, []))
            per_post_metrics.append(post_event_to_server_metric(pe, cycles_summary))
        cache_summary = summarize_cache_events(cache_evts)
        (run_dir / "server" / "metrics.jsonl").write_text(
            "\n".join(json.dumps(m) for m in per_post_metrics)
            + ("\n" if per_post_metrics else "")
        )
    else:
        (run_dir / "server" / "metrics.jsonl").write_text("")

    request_files = sorted((run_dir / "requests").glob("*.json"))
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
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

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
    p.add_argument("--proxy-port", type=int, default=DEFAULT_PROXY_PORT)
    p.add_argument("--dflash-port", type=int, default=DEFAULT_DFLASH_PORT)
    p.add_argument("--mlxlm-port", type=int, default=DEFAULT_MLXLM_PORT)
    p.add_argument("--server-ready-timeout-s", type=float, default=300.0)
    p.add_argument("--proxy-ready-timeout-s", type=float, default=30.0)
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
        "--profile",
        default=None,
        help="dflash only: pass --profile to dflash serve.",
    )
    p.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="dflash only: pass --prefill-step-size to dflash serve.",
    )
    p.add_argument(
        "--fastpath-max-tokens",
        type=int,
        default=None,
        help="dflash only: pass --fastpath-max-tokens to dflash serve.",
    )
    p.add_argument(
        "--max-snapshot-tokens",
        type=int,
        default=None,
        help="dflash only: pass --max-snapshot-tokens to dflash serve.",
    )
    p.add_argument(
        "--target-fa-window",
        type=int,
        default=None,
        help="dflash only: pass --target-fa-window to dflash_mlx.serve",
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
            "--profile": args.profile,
            "--prefill-step-size": args.prefill_step_size,
            "--fastpath-max-tokens": args.fastpath_max_tokens,
            "--max-snapshot-tokens": args.max_snapshot_tokens,
            "--target-fa-window": args.target_fa_window,
        }
        used = [flag for flag, value in dflash_only.items() if value is not None]
        if used:
            raise SystemExit(f"{', '.join(used)} require --backend dflash")
    for flag_name in ("prefill_step_size", "fastpath_max_tokens", "max_snapshot_tokens"):
        value = getattr(args, flag_name)
        if value is not None and int(value) < 0:
            raise SystemExit(f"--{flag_name.replace('_', '-')} must be >= 0")
    if args.target_fa_window is not None and args.target_fa_window < 0:
        raise SystemExit("--target-fa-window must be >= 0")
    if args.prefix_cache_l2_max_bytes is not None and int(args.prefix_cache_l2_max_bytes) < 0:
        raise SystemExit("--prefix-cache-l2-max-bytes must be >= 0")
    if args.backend == "dflash":
        _validate_l2_flags(args)
    if args.backend == "dflash":
        args.prefix_cache = True if args.prefix_cache is None else bool(args.prefix_cache)
        args.prefix_cache_l2 = False if args.prefix_cache_l2 is None else bool(args.prefix_cache_l2)
        args.prefix_cache_l2_max_bytes = (
            50 * 1024**3
            if args.prefix_cache_l2_max_bytes is None
            else int(args.prefix_cache_l2_max_bytes)
        )
        args.diagnostics = "basic" if args.diagnostics is None else args.diagnostics
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
        if args.prefix_cache:
            server_cmd.extend([
                "--prefix-cache",
                "--prefix-cache-max-entries",
                "8",
                "--prefix-cache-max-bytes",
                "10737418240",
            ])
        else:
            server_cmd.append("--no-prefix-cache")
        if args.prefix_cache_l2:
            l2_dir = Path(args.prefix_cache_l2_dir) if args.prefix_cache_l2_dir else (run_dir / "l2_cache")
            l2_dir.mkdir(parents=True, exist_ok=True)
            server_cmd.extend([
                "--prefix-cache-l2",
                "--prefix-cache-l2-dir",
                str(l2_dir),
                "--prefix-cache-l2-max-bytes",
                str(int(args.prefix_cache_l2_max_bytes)),
            ])

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
    }
    if args.backend == "dflash":
        metadata["dflash_runtime_overrides"] = {
            "draft_quant": args.draft_quant,
            "profile": args.profile,
            "prefill_step_size": args.prefill_step_size,
            "fastpath_max_tokens": args.fastpath_max_tokens,
            "max_snapshot_tokens": args.max_snapshot_tokens,
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
    client_returncode = None
    client_wall_s = None

    try:

        sys.stderr.write(f"[orch] starting server: {' '.join(server_cmd)}\n")
        server_proc = _spawn(server_cmd, run_dir / "server" / "stdout.log", run_dir / "server" / "stderr.log")
        if not _wait_health(server_health_url, args.server_ready_timeout_s, "server"):
            raise SystemExit("server not ready")

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
        if server_proc is not None:
            _terminate(server_proc, "server")

    server_stderr_text = (run_dir / "server" / "stderr.log").read_text()
    cache_summary: dict[str, Any] = {}
    per_post_metrics: list[dict[str, Any]] = []
    if args.backend == "dflash":
        events_dir = run_dir / "events"
        post_evts, cycles_by_req, cache_evts = read_dflash_events(events_dir)
        if post_evts:

            for pe in sorted(post_evts, key=lambda e: e.get("request_id", 0)):
                rid = pe.get("request_id")
                cycles_summary = summarize_cycles(cycles_by_req.get(rid, []))
                per_post_metrics.append(post_event_to_server_metric(pe, cycles_summary))
            cache_summary = summarize_cache_events(cache_evts)
        else:
            per_post_metrics = []
        (run_dir / "server" / "metrics.jsonl").write_text(
            "\n".join(json.dumps(m) for m in per_post_metrics) + ("\n" if per_post_metrics else "")
        )
    else:
        (run_dir / "server" / "metrics.jsonl").write_text("")

    request_files = sorted((run_dir / "requests").glob("*.json"))
    sse_files = sorted((run_dir / "sse").glob("*.jsonl"))

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
        "workspace_files": workspace_files,
        "totals": totals,
        "cache_summary": cache_summary if args.backend == "dflash" else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

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
