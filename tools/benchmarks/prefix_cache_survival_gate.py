# Copyright 2026 bstnxbt
# MIT License - see LICENSE file

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TARGET = "mlx-community/Qwen3.6-27B-4bit"
DEFAULT_DRAFT = "z-lab/Qwen3.6-27B-DFlash"
DEFAULT_CTX_TOKENS = "16K,32K,64K"
DEFAULT_OUT_ROOT = ".artifacts/dflash/prefix_cache_survival_gate"


@dataclass(frozen=True)
class SurvivalCase:
    ctx_tokens: int
    seed: int
    case_index: int
    insert_frac: float
    key: str
    value: str
    salt: str
    messages: list[dict[str, str]]
    prompt_tokens: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prefix-cache survival gate: warm a long prompt, replay it with a "
            "divergent suffix, and verify the answer plus cache-hit size."
        ),
        epilog=(
            "This is not a statistical NIAH benchmark. It is a cache-correctness "
            "gate. Defaults exercise 16K/32K/64K; 64K can be slow or OOM on "
            "64GB machines, and 128K should be run manually only after the 64K "
            "cell is clean. The server is started by this tool, AR fast-path is "
            "disabled, and outputs go under .artifacts/dflash/..."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target model ref.")
    parser.add_argument("--draft", default=DEFAULT_DRAFT, help="DFlash draft model ref.")
    parser.add_argument(
        "--ctx-tokens",
        default=DEFAULT_CTX_TOKENS,
        help="Comma-separated context targets. Supports suffix K, e.g. 16K,32K,64K.",
    )
    parser.add_argument("--cases-per-ctx", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Passed to the chat template. Default: true.",
    )
    parser.add_argument("--prefill-step-size", type=int, default=4096)
    parser.add_argument("--prefix-cache-max-entries", type=int, default=4)
    parser.add_argument("--prefix-cache-max-bytes", type=int, default=12 * 1024**3)
    parser.add_argument("--max-snapshot-tokens", type=int, default=0)
    parser.add_argument(
        "--min-cache-hit-ratio",
        type=float,
        default=0.80,
        help="Warm hit must be at least this fraction of cold prompt tokens.",
    )
    parser.add_argument(
        "--token-tolerance-pct",
        type=float,
        default=0.5,
        help="Prompt builder tries to stay within this percent below target.",
    )
    parser.add_argument(
        "--wrong-haystack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also ask the same key with a different value to catch stale-cache answers.",
    )
    parser.add_argument(
        "--diagnostics",
        choices=("basic", "full"),
        default="basic",
        help="basic is enough for post_events.jsonl; full adds per-cycle overhead.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--server-timeout-s", type=float, default=300.0)
    parser.add_argument("--request-timeout-s", type=float, default=1200.0)
    parser.add_argument("--out", default=DEFAULT_OUT_ROOT)
    return parser


def build_survival_case(
    tokenizer: Any,
    *,
    ctx_tokens: int,
    seed: int,
    case_index: int,
    insert_frac: float,
    key: str,
    value: str,
    salt: str,
    suffix: str,
    enable_thinking: bool,
    token_tolerance_pct: float = 0.5,
) -> SurvivalCase:
    system = _haystack_for_budget(
        tokenizer,
        target_tokens=ctx_tokens,
        insert_frac=insert_frac,
        key=key,
        value=value,
        salt=salt,
        suffix=suffix,
        enable_thinking=enable_thinking,
        token_tolerance_pct=token_tolerance_pct,
    )
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Return only the value associated with key {key}. "
                f"Request nonce: {suffix}."
            ),
        },
    ]
    return SurvivalCase(
        ctx_tokens=int(ctx_tokens),
        seed=int(seed),
        case_index=int(case_index),
        insert_frac=float(insert_frac),
        key=key,
        value=value,
        salt=salt,
        messages=messages,
        prompt_tokens=chat_token_count(
            tokenizer,
            messages,
            enable_thinking=enable_thinking,
        ),
    )


def build_followup_case(
    tokenizer: Any,
    *,
    previous: SurvivalCase,
    assistant_content: str,
    suffix: str,
    enable_thinking: bool,
) -> SurvivalCase:
    messages = [
        *previous.messages,
        {"role": "assistant", "content": str(assistant_content)},
        {
            "role": "user",
            "content": (
                f"Return only the value associated with key {previous.key}. "
                f"Request nonce: {suffix}."
            ),
        },
    ]
    return SurvivalCase(
        ctx_tokens=previous.ctx_tokens,
        seed=previous.seed,
        case_index=previous.case_index,
        insert_frac=previous.insert_frac,
        key=previous.key,
        value=previous.value,
        salt=previous.salt,
        messages=messages,
        prompt_tokens=chat_token_count(
            tokenizer,
            messages,
            enable_thinking=enable_thinking,
        ),
    )


def chat_template_kwargs(*, enable_thinking: bool) -> dict[str, bool]:
    return {"enable_thinking": bool(enable_thinking)}


def chat_token_count(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
) -> int:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **chat_template_kwargs(enable_thinking=enable_thinking),
    )
    if hasattr(ids, "ids"):
        ids = ids.ids
    return len(ids)


def score_response(content: str, value: str) -> bool:
    return value.strip() in str(content)


def parse_token_list(value: str) -> list[int]:
    out = [_parse_token_count(item) for item in str(value).split(",") if item.strip()]
    if not out:
        raise ValueError("--ctx-tokens must contain at least one value")
    return out


def make_case_specs(*, ctx_values: list[int], seed: int, cases_per_ctx: int) -> list[dict[str, Any]]:
    if cases_per_ctx < 1:
        raise ValueError("--cases-per-ctx must be >= 1")
    rng = random.Random(int(seed))
    specs: list[dict[str, Any]] = []
    for ctx_tokens in ctx_values:
        for case_index in range(cases_per_ctx):
            specs.append(
                {
                    "ctx_tokens": int(ctx_tokens),
                    "seed": int(seed),
                    "case_index": int(case_index),
                    "insert_frac": rng.uniform(0.05, 0.95),
                    "key": f"key_{rng.getrandbits(48):012x}",
                    "value": f"value_{rng.getrandbits(64):016x}",
                    "wrong_value": f"value_{rng.getrandbits(64):016x}",
                    "salt": f"salt_{rng.getrandbits(48):012x}",
                }
            )
    return specs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ctx_values = parse_token_list(args.ctx_tokens)
    if not (0.0 <= float(args.min_cache_hit_ratio) <= 1.0):
        raise ValueError("--min-cache-hit-ratio must be in [0, 1]")
    if float(args.token_tolerance_pct) < 0.0:
        raise ValueError("--token-tolerance-pct must be >= 0")

    label = "ctx" + "-".join(_short_token_count(v) for v in ctx_values)
    run_dir = _make_run_dir(Path(args.out), label)
    diagnostics_dir = run_dir / "diagnostics"
    log_path = run_dir / "server.log"
    rows_path = run_dir / "rows.jsonl"
    specs = make_case_specs(
        ctx_values=ctx_values,
        seed=int(args.seed),
        cases_per_ctx=int(args.cases_per_ctx),
    )

    _write_json_atomic(
        run_dir / "manifest.json",
        {
            "kind": "prefix_cache_survival_gate",
            "target": args.target,
            "draft": args.draft,
            "ctx_tokens": ctx_values,
            "cases_per_ctx": args.cases_per_ctx,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "enable_thinking": bool(args.enable_thinking),
            "prefill_step_size": args.prefill_step_size,
            "prefix_cache_max_entries": args.prefix_cache_max_entries,
            "prefix_cache_max_bytes": args.prefix_cache_max_bytes,
            "max_snapshot_tokens": args.max_snapshot_tokens,
            "min_cache_hit_ratio": args.min_cache_hit_ratio,
            "token_tolerance_pct": args.token_tolerance_pct,
            "wrong_haystack": bool(args.wrong_haystack),
            "diagnostics": args.diagnostics,
            "argv": [
                sys.executable,
                "-m",
                "tools.benchmarks.prefix_cache_survival_gate",
                *(argv or sys.argv[1:]),
            ],
            "cwd": os.getcwd(),
            "git_sha": _git_sha(),
        },
    )

    print(f"out={run_dir}", flush=True)
    print(f"target={args.target}", flush=True)
    print(f"ctx_tokens={','.join(str(v) for v in ctx_values)} cases={len(specs)}", flush=True)

    tokenizer = _load_tokenizer(args.target)
    port = int(args.port) if int(args.port) > 0 else _find_free_port(args.host)
    server_url = f"http://{args.host}:{port}"
    server_proc = _start_server(args, port=port, diagnostics_dir=diagnostics_dir, log_path=log_path)
    try:
        _wait_for_server(server_url, timeout_s=float(args.server_timeout_s))
        rows = _run_cases(
            args=args,
            tokenizer=tokenizer,
            specs=specs,
            server_url=server_url,
            diagnostics_dir=diagnostics_dir,
            rows_path=rows_path,
        )
        summary = _summary(rows, error=None)
        _write_json_atomic(run_dir / "results.json", summary)
        print(f"results={run_dir / 'results.json'}", flush=True)
        return 0 if summary["passed"] else 1
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        _write_json_atomic(run_dir / "results.json", _summary([], error=error))
        print(f"results={run_dir / 'results.json'}", flush=True)
        raise
    finally:
        _terminate(server_proc)


def _run_cases(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    specs: list[dict[str, Any]],
    server_url: str,
    diagnostics_dir: Path,
    rows_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    post_path = diagnostics_dir / "post_events.jsonl"
    for spec in specs:
        cold = _case_from_spec(args, tokenizer, spec, suffix="cold", value=spec["value"])
        wrong = (
            _case_from_spec(args, tokenizer, spec, suffix="wrong-haystack", value=spec["wrong_value"])
            if bool(args.wrong_haystack)
            else None
        )

        cold_result = _request_with_post_event(args, server_url, cold, post_path)
        warm = build_followup_case(
            tokenizer,
            previous=cold,
            assistant_content=str(cold_result.get("content") or ""),
            suffix=f"warm-divergent-ctx{spec['ctx_tokens']}-case{spec['case_index']}",
            enable_thinking=bool(args.enable_thinking),
        )
        warm_result = _request_with_post_event(args, server_url, warm, post_path)
        wrong_result = (
            _request_with_post_event(args, server_url, wrong, post_path)
            if wrong is not None
            else None
        )
        row = _row(
            spec=spec,
            cold=cold,
            warm=warm,
            wrong=wrong,
            cold_result=cold_result,
            warm_result=warm_result,
            wrong_result=wrong_result,
            min_cache_hit_ratio=float(args.min_cache_hit_ratio),
        )
        rows.append(row)
        _write_jsonl_atomic(rows_path, rows)
        print(
            "ctx={} seed={} case={} frac={:.3f} cold={} warm={} hit={}/{} physical={}/{} restored={} wrong={} wall_ms={}".format(
                row["ctx_tokens"],
                row["seed"],
                row["case_index"],
                row["insert_frac"],
                "ok" if row["cold_ok"] else "fail",
                "ok" if row["warm_ok"] else "fail",
                row["warm_cache_hit_tokens"],
                row["warm_min_cache_hit_tokens"],
                row["warm_physical_prefill_tokens"],
                row["warm_logical_ctx_tokens"],
                row["warm_prefill_tokens_restored"],
                row["wrong_ok"] if wrong is not None else "skipped",
                _fmt(row["warm_wall_ms"]),
            ),
            flush=True,
        )
    return rows


def _case_from_spec(
    args: argparse.Namespace,
    tokenizer: Any,
    spec: dict[str, Any],
    *,
    suffix: str,
    value: str,
) -> SurvivalCase:
    return build_survival_case(
        tokenizer,
        ctx_tokens=int(spec["ctx_tokens"]),
        seed=int(spec["seed"]),
        case_index=int(spec["case_index"]),
        insert_frac=float(spec["insert_frac"]),
        key=str(spec["key"]),
        value=str(value),
        salt=str(spec["salt"]),
        suffix=f"{suffix}-ctx{spec['ctx_tokens']}-case{spec['case_index']}",
        enable_thinking=bool(args.enable_thinking),
        token_tolerance_pct=float(args.token_tolerance_pct),
    )


def _request_with_post_event(
    args: argparse.Namespace,
    server_url: str,
    case: SurvivalCase,
    post_path: Path,
) -> dict[str, Any]:
    before = _jsonl_len(post_path)
    t0 = time.perf_counter_ns()
    response = _post_chat(
        server_url=server_url,
        model=str(args.target),
        messages=case.messages,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        enable_thinking=bool(args.enable_thinking),
        timeout_s=float(args.request_timeout_s),
    )
    wall_ms = (time.perf_counter_ns() - t0) / 1e6
    post_event = _wait_for_post_event(post_path, min_count=before + 1)
    message = _response_message(response)
    content = str(message.get("content") or "")
    return {
        "content": content,
        "message": message,
        "wall_ms": wall_ms,
        "response": response,
        "post_event": post_event,
    }


def _post_chat(
    *,
    server_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        "chat_template_kwargs": chat_template_kwargs(enable_thinking=enable_thinking),
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read())


def _start_server(
    args: argparse.Namespace,
    *,
    port: int,
    diagnostics_dir: Path,
    log_path: Path,
) -> subprocess.Popen[str]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "dflash_mlx.cli",
        "serve",
        "--model",
        str(args.target),
        "--draft",
        str(args.draft),
        "--host",
        str(args.host),
        "--port",
        str(port),
        "--fastpath-max-tokens",
        "0",
        "--diagnostics",
        str(args.diagnostics),
        "--diagnostics-dir",
        str(diagnostics_dir),
        "--prefix-cache",
        "--no-prefix-cache-l2",
        "--prefix-cache-max-entries",
        str(args.prefix_cache_max_entries),
        "--prefix-cache-max-bytes",
        str(args.prefix_cache_max_bytes),
        "--max-snapshot-tokens",
        str(args.max_snapshot_tokens),
        "--prefill-step-size",
        str(args.prefill_step_size),
        "--chat-template-args",
        json.dumps(chat_template_kwargs(enable_thinking=bool(args.enable_thinking))),
    ]
    print("server_cmd=" + " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)


def _wait_for_server(server_url: str, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"{server_url.rstrip('/')}/v1/models"
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5.0) as resp:
                if 200 <= int(resp.status) < 500:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"server not ready after {timeout_s:.1f}s: {last_error}")


def _wait_for_post_event(path: Path, *, min_count: int) -> dict[str, Any]:
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        rows = _read_jsonl(path)
        if len(rows) >= min_count:
            return rows[-1]
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for {path} row {min_count}")


def _response_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    return dict(message) if isinstance(message, dict) else {}


def _row(
    *,
    spec: dict[str, Any],
    cold: SurvivalCase,
    warm: SurvivalCase,
    wrong: SurvivalCase | None,
    cold_result: dict[str, Any],
    warm_result: dict[str, Any],
    wrong_result: dict[str, Any] | None,
    min_cache_hit_ratio: float,
) -> dict[str, Any]:
    cold_post = dict(cold_result.get("post_event") or {})
    warm_post = dict(warm_result.get("post_event") or {})
    wrong_post = dict((wrong_result or {}).get("post_event") or {})
    warm_hit = int(warm_post.get("cache_hit_tokens") or 0)
    warm_logical = _optional_int(warm_post.get("logical_ctx_tokens"))
    warm_physical = _optional_int(warm_post.get("physical_prefill_tokens"))
    warm_restored = _optional_int(warm_post.get("prefill_tokens_restored"))
    warm_computed = _optional_int(warm_post.get("prefill_tokens_computed"))
    min_hit_base = warm_logical if warm_logical is not None else warm.prompt_tokens
    min_hit = int(min_hit_base * float(min_cache_hit_ratio))
    wrong_content = str((wrong_result or {}).get("content", ""))
    wrong_ok = True
    wrong_stale = False
    if wrong is not None:
        wrong_ok = score_response(wrong_content, wrong.value)
        wrong_stale = score_response(wrong_content, cold.value)
    cold_ok = score_response(str(cold_result.get("content", "")), cold.value)
    warm_ok = score_response(str(warm_result.get("content", "")), warm.value)
    warm_hit_ok = warm_hit >= min_hit
    warm_mode_ok = str(warm_post.get("mode_used") or "") == "dflash"
    warm_physical_reuse_ok = (
        warm_logical is not None
        and warm_physical is not None
        and warm_restored is not None
        and warm_computed is not None
        and warm_logical > 0
        and warm_restored + warm_computed == warm_logical
        and warm_physical < warm_logical
        and warm_restored >= min_hit
        and warm_computed == warm_physical
    )
    passed = bool(
        cold_ok
        and warm_ok
        and warm_hit_ok
        and warm_mode_ok
        and warm_physical_reuse_ok
        and wrong_ok
        and not wrong_stale
    )
    return {
        "passed": passed,
        "ctx_tokens": int(spec["ctx_tokens"]),
        "seed": int(spec["seed"]),
        "case_index": int(spec["case_index"]),
        "insert_frac": float(spec["insert_frac"]),
        "key": str(spec["key"]),
        "cold_value": cold.value,
        "wrong_value": wrong.value if wrong is not None else None,
        "cold_prompt_tokens": int(cold.prompt_tokens),
        "warm_prompt_tokens": int(warm.prompt_tokens),
        "wrong_prompt_tokens": int(wrong.prompt_tokens) if wrong is not None else None,
        "cold_ok": cold_ok,
        "warm_ok": warm_ok,
        "wrong_ok": wrong_ok if wrong is not None else None,
        "wrong_stale": wrong_stale if wrong is not None else None,
        "warm_hit_ok": warm_hit_ok,
        "warm_mode_ok": warm_mode_ok,
        "warm_physical_reuse_ok": warm_physical_reuse_ok,
        "warm_min_cache_hit_tokens": min_hit,
        "cold_content": str(cold_result.get("content", "")),
        "warm_content": str(warm_result.get("content", "")),
        "wrong_content": wrong_content if wrong is not None else None,
        "cold_wall_ms": float(cold_result.get("wall_ms", 0.0)),
        "warm_wall_ms": float(warm_result.get("wall_ms", 0.0)),
        "wrong_wall_ms": float((wrong_result or {}).get("wall_ms", 0.0)) if wrong is not None else None,
        "cold_cache_hit_tokens": int(cold_post.get("cache_hit_tokens") or 0),
        "warm_cache_hit_tokens": warm_hit,
        "wrong_cache_hit_tokens": int(wrong_post.get("cache_hit_tokens") or 0) if wrong is not None else None,
        "warm_mode_used": warm_post.get("mode_used"),
        "warm_logical_ctx_tokens": warm_logical,
        "warm_physical_prefill_tokens": warm_physical,
        "warm_prefill_tokens_restored": warm_restored,
        "warm_prefill_tokens_computed": warm_computed,
        "cold_prefill_ms": cold_post.get("prefill_ms"),
        "warm_prefill_ms": warm_post.get("prefill_ms"),
        "wrong_prefill_ms": wrong_post.get("prefill_ms") if wrong is not None else None,
        "cold_prefill_tok_s": cold_post.get("prefill_tok_s"),
        "warm_prefill_tok_s": warm_post.get("prefill_tok_s"),
        "wrong_prefill_tok_s": wrong_post.get("prefill_tok_s") if wrong is not None else None,
        "cold_decode_ms": cold_post.get("decode_ms"),
        "warm_decode_ms": warm_post.get("decode_ms"),
        "wrong_decode_ms": wrong_post.get("decode_ms") if wrong is not None else None,
        "cold_acceptance": cold_post.get("acceptance_ratio"),
        "warm_acceptance": warm_post.get("acceptance_ratio"),
        "wrong_acceptance": wrong_post.get("acceptance_ratio") if wrong is not None else None,
    }


def _summary(rows: list[dict[str, Any]], *, error: dict[str, str] | None) -> dict[str, Any]:
    total = len(rows)
    passed_rows = sum(1 for row in rows if row.get("passed"))
    ctx_summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("ctx_tokens"))
        bucket = ctx_summary.setdefault(key, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += 1 if row.get("passed") else 0
    passed = bool(total and passed_rows == total and error is None)
    return {
        "passed": passed,
        "error": error,
        "total": total,
        "passed_rows": passed_rows,
        "pass_rate": (passed_rows / total) if total else 0.0,
        "ctx_summary": ctx_summary,
        "rows": rows,
    }


def _haystack_for_budget(
    tokenizer: Any,
    *,
    target_tokens: int,
    insert_frac: float,
    key: str,
    value: str,
    salt: str,
    suffix: str,
    enable_thinking: bool,
    token_tolerance_pct: float,
) -> str:
    lo = 1
    hi = 1
    best = _haystack(total_lines=1, insert_frac=insert_frac, key=key, value=value, salt=salt)
    while True:
        candidate = _haystack(total_lines=hi, insert_frac=insert_frac, key=key, value=value, salt=salt)
        count = _case_token_count(
            tokenizer,
            candidate,
            key=key,
            suffix=suffix,
            enable_thinking=enable_thinking,
        )
        if count > target_tokens:
            break
        best = candidate
        lo = hi + 1
        hi *= 2
    search_hi = hi - 1
    while lo <= search_hi:
        mid = (lo + search_hi) // 2
        candidate = _haystack(total_lines=mid, insert_frac=insert_frac, key=key, value=value, salt=salt)
        count = _case_token_count(
            tokenizer,
            candidate,
            key=key,
            suffix=suffix,
            enable_thinking=enable_thinking,
        )
        if count <= target_tokens:
            best = candidate
            lo = mid + 1
        else:
            search_hi = mid - 1
    return _pad_to_tolerance(
        tokenizer,
        best,
        target_tokens=target_tokens,
        key=key,
        suffix=suffix,
        enable_thinking=enable_thinking,
        token_tolerance_pct=token_tolerance_pct,
    )


def _pad_to_tolerance(
    tokenizer: Any,
    system: str,
    *,
    target_tokens: int,
    key: str,
    suffix: str,
    enable_thinking: bool,
    token_tolerance_pct: float,
) -> str:
    min_tokens = int(target_tokens * (1.0 - float(token_tolerance_pct) / 100.0))
    current = system
    current_count = _case_token_count(
        tokenizer,
        current,
        key=key,
        suffix=suffix,
        enable_thinking=enable_thinking,
    )
    pad_idx = 0
    while current_count < min_tokens and pad_idx < 512:
        candidate = f"{current} budget_pad_{pad_idx:04d}"
        candidate_count = _case_token_count(
            tokenizer,
            candidate,
            key=key,
            suffix=suffix,
            enable_thinking=enable_thinking,
        )
        if candidate_count > target_tokens:
            break
        current = candidate
        current_count = candidate_count
        pad_idx += 1
    return current


def _case_token_count(
    tokenizer: Any,
    system: str,
    *,
    key: str,
    suffix: str,
    enable_thinking: bool,
) -> int:
    return chat_token_count(
        tokenizer,
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Return only the value associated with key {key}. "
                    f"Request nonce: {suffix}."
                ),
            },
        ],
        enable_thinking=enable_thinking,
    )


def _haystack(*, total_lines: int, insert_frac: float, key: str, value: str, salt: str) -> str:
    insert_frac = max(0.0, min(1.0, float(insert_frac)))
    before = int(total_lines * insert_frac)
    after = max(0, int(total_lines) - before)
    before_text = "\n".join(_filler_line(i, salt=salt) for i in range(before))
    after_text = "\n".join(_filler_line(before + i, salt=salt) for i in range(after))
    return (
        "You are reading a long deterministic repository note. "
        "Ignore filler, but preserve exact records.\n"
        f"CASE_SALT: {salt}\n"
        f"{before_text}\n"
        f"NEEDLE_RECORD key={key} value={value}\n"
        f"{after_text}"
    )


def _filler_line(index: int, *, salt: str) -> str:
    return (
        f"filler-{index:06d}-{salt}: stable cache-survival context line with "
        "irrelevant code review notes and no requested value."
    )


def _parse_token_count(value: str) -> int:
    text = str(value).strip().lower().replace("_", "")
    mult = 1
    if text.endswith("k"):
        mult = 1024
        text = text[:-1]
    number = int(text)
    if number <= 0:
        raise ValueError(f"token count must be > 0, got {value!r}")
    return number * mult


def _short_token_count(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def _load_tokenizer(model_ref: str) -> Any:
    from mlx_lm.utils import hf_repo_to_path, load_tokenizer

    return load_tokenizer(hf_repo_to_path(model_ref))


def _make_run_dir(root: Path, label: str) -> Path:
    base = root / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}"
    path = base
    suffix = 2
    while path.exists():
        path = Path(f"{base}-{suffix}")
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def _jsonl_len(path: Path) -> int:
    return len(_read_jsonl(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)

def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
