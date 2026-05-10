# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dflash_mlx.artifacts import create_run_dir, write_json, write_manifest
from dflash_mlx.benchmark import (
    _generate_dflash_stream_once,
    _generate_stock_baseline_once,
    _hardware_info,
    _load_pristine_target_bundle,
    _release_loaded_models,
)
from dflash_mlx.observability.memory import process_memory_snapshot
from dflash_mlx.runtime import get_stop_token_ids
from dflash_mlx.runtime.bundle import load_runtime_bundle
from dflash_mlx.runtime.context import build_runtime_context, runtime_config_from_profile

GB = 1_000_000_000.0
DEFAULT_CONTEXTS = "512,1k,2k,4k,8k,16k,32k"
DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent working inside a large Python repository. "
    "Preserve user changes, measure before patching, and report evidence."
)
DEFAULT_FINAL_REQUEST = (
    "\n\n# Final user request\n"
    "Write code only. Create a single Python file that behaves like a small "
    "production package for deterministic benchmark runs. No prose outside code. "
    "Use Python 3.11, dataclasses, pathlib, json, argparse, time, hashlib, "
    "statistics, and typing. Keep it compact but complete.\n\n"
    "Implement prompt schema dataclasses, validation helpers, an LRU cache, "
    "rolling metrics, benchmark record serialization, a deterministic sampler "
    "config, an atomic JSONL event log writer, a run registry, a CLI, and a "
    "small self-test.\n"
)


def parse_contexts(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).replace(";", ",").split(","):
        token = part.strip().lower()
        if not token:
            continue
        match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]?)", token)
        if match is None:
            raise ValueError(f"invalid context bucket: {part!r}")
        multiplier = {
            "": 1,
            "k": 1024,
            "m": 1024**2,
            "g": 1024**3,
            "t": 1024**4,
        }[match.group(2)]
        value = int(float(match.group(1)) * multiplier)
        if value < 1:
            raise ValueError(f"context bucket must be positive: {part!r}")
        values.append(value)
    if not values:
        raise ValueError("at least one context bucket is required")
    return values


def format_context(value: int) -> str:
    value = int(value)
    if value == 512:
        return "0.5k"
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def build_context_prompt_tokens(
    tokenizer: Any,
    target_tokens: int,
    *,
    prompt_format: str = "chat",
    enable_thinking: bool | None = False,
) -> list[int]:
    prompt_format = _normalize_prompt_format(prompt_format)
    tail = DEFAULT_FINAL_REQUEST
    tail_ids = _encode_prompt_content(
        tokenizer,
        tail,
        prompt_format=prompt_format,
        enable_thinking=enable_thinking,
    )
    if len(tail_ids) >= int(target_tokens):
        return tail_ids[-int(target_tokens) :]

    filler = _model_prompt_text()
    filler_ids = _tokenize(tokenizer, filler)
    filler_target = max(1, int(target_tokens) - len(tail_ids))
    while len(filler_ids) < filler_target:
        filler += "\n\n" + _model_prompt_text()
        filler_ids = _tokenize(tokenizer, filler)

    while True:
        content = _decode_tokens(tokenizer, filler_ids[:filler_target]) + tail
        token_ids = _encode_prompt_content(
            tokenizer,
            content,
            prompt_format=prompt_format,
            enable_thinking=enable_thinking,
        )
        if len(token_ids) >= int(target_tokens):
            return token_ids[-int(target_tokens) :]
        filler_target += max(1, int(target_tokens) - len(token_ids))
        while len(filler_ids) < filler_target:
            filler += "\n\n" + _model_prompt_text()
            filler_ids = _tokenize(tokenizer, filler)


def run_grid(args: argparse.Namespace, argv: Sequence[str]) -> dict[str, Any]:
    contexts = parse_contexts(args.contexts)
    run_dir = create_run_dir(
        "benchmark",
        args.label or f"context-grid-{Path(str(args.target)).name}",
        args.out,
    )
    write_manifest(
        run_dir,
        kind="benchmark",
        label=args.label or "context-grid",
        argv=list(argv),
        model=args.target,
        draft=args.draft,
        profile=args.profile,
        effective_config=_effective_config(args, contexts),
    )

    rows: list[dict[str, Any]] = []
    prompt_tokens_by_context: dict[int, list[int]] | None = None
    prompt_hashes: dict[int, str] = {}

    def record(row: dict[str, Any], samples: list[dict[str, Any]]) -> None:
        rows.append(row)
        _append_jsonl(run_dir / "rows.jsonl", [row])
        _append_jsonl(run_dir / "memory_samples.jsonl", samples)
        summary = _summary_payload(args, contexts, prompt_hashes, rows)
        write_json(run_dir / "summary.json", summary)
        (run_dir / "summary.md").write_text(render_markdown(summary) + "\n")

    if args.backend in ("both", "mlxlm"):
        model, tokenizer, meta = _load_pristine_target_bundle(args.target)
        prompt_tokens_by_context = {
            ctx: build_context_prompt_tokens(
                tokenizer,
                ctx,
                prompt_format=args.prompt_format,
                enable_thinking=args.enable_thinking,
            )
            for ctx in contexts
        }
        prompt_hashes.update(
            {
                ctx: _token_hash(prompt_tokens)
                for ctx, prompt_tokens in prompt_tokens_by_context.items()
            }
        )
        try:
            for idx, ctx in enumerate(contexts):
                with MemorySampler(
                    backend="mlxlm",
                    context_tokens=ctx,
                    interval_s=args.memory_sample_interval,
                ) as sampler:
                    raw = _generate_stock_baseline_once(
                        target_model=model,
                        tokenizer=tokenizer,
                        prompt="",
                        max_new_tokens=args.max_tokens,
                        no_eos=args.no_eos,
                        use_chat_template=False,
                        prompt_tokens_override=prompt_tokens_by_context[ctx],
                    )
                row = _with_memory(_baseline_row(raw, ctx, meta, args), sampler.summary())
                record(row, sampler.samples)
                _after_case(args, idx, len(contexts))
        finally:
            del model
            del tokenizer
            _release_loaded_models()
        if args.backend == "both" and args.cooldown > 0:
            time.sleep(float(args.cooldown))

    if args.backend in ("both", "dflash"):
        runtime_context = build_runtime_context(
            runtime_config_from_profile(
                profile=args.profile,
                prefix_cache=False,
                prefix_cache_l2=False,
                clear_cache_boundaries=args.clear_cache_boundaries,
                target_fa_window=(
                    0 if args.target_fa_window is None else args.target_fa_window
                ),
                prefill_step_size=args.prefill_step_size,
                draft_sink_size=args.draft_sink_size,
                draft_window_size=args.draft_window_size,
                verify_len_cap=args.verify_len_cap,
                verify_mode=args.verify_mode,
            )
        )
        bundle = load_runtime_bundle(
            model_ref=args.target,
            draft_ref=args.draft,
            draft_quant=args.draft_quant,
            verify_config=runtime_context.verify,
            split_full_attention_sdpa=args.split_sdpa,
        )
        if prompt_tokens_by_context is None:
            prompt_tokens_by_context = {
                ctx: build_context_prompt_tokens(
                    bundle.tokenizer,
                    ctx,
                    prompt_format=args.prompt_format,
                    enable_thinking=args.enable_thinking,
                )
                for ctx in contexts
            }
            prompt_hashes.update(
                {
                    ctx: _token_hash(prompt_tokens)
                    for ctx, prompt_tokens in prompt_tokens_by_context.items()
                }
            )
        stop_ids = get_stop_token_ids(bundle.tokenizer)
        try:
            for idx, ctx in enumerate(contexts):
                with MemorySampler(
                    backend="dflash",
                    context_tokens=ctx,
                    interval_s=args.memory_sample_interval,
                ) as sampler:
                    raw = _generate_dflash_stream_once(
                        target_model=bundle.target_model,
                        target_ops=bundle.target_ops,
                        tokenizer=bundle.tokenizer,
                        draft_model=bundle.draft_model,
                        draft_backend=bundle.draft_backend,
                        prompt="",
                        max_new_tokens=args.max_tokens,
                        use_chat_template=False,
                        block_tokens=args.block_tokens,
                        stop_token_ids=[] if args.no_eos else stop_ids,
                        suppress_token_ids=stop_ids if args.no_eos else None,
                        prompt_tokens_override=prompt_tokens_by_context[ctx],
                        runtime_context=runtime_context,
                    )
                row = _with_memory(_dflash_row(raw, ctx, bundle, args), sampler.summary())
                record(row, sampler.samples)
                _after_case(args, idx, len(contexts))
        finally:
            del bundle
            _release_loaded_models()

    summary = _summary_payload(args, contexts, prompt_hashes, rows)
    print(render_markdown(summary))
    print(f"\nResults written to {run_dir}")
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    meta = summary["metadata"]
    lines = [
        f"# Context Grid - {meta['label']}",
        "",
        f"- target: `{meta['target']}`",
        f"- draft: `{meta.get('draft')}`",
        f"- backend: `{meta['backend']}`",
        f"- max_tokens: `{meta['max_tokens']}`",
        f"- cooldown_s: `{meta['cooldown_s']}`",
        f"- prompt_format: `{meta['prompt_format']}`",
        "",
        "| context | mlx pp tok/s | dflash pp tok/s | mlx gen tok/s | dflash gen tok/s | mlx peak GB | mlx footprint GB | dflash peak GB | dflash footprint GB | dflash footprint delta GB | dflash cache peak GB | compare | wall dflash/mlx |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in summary.get("comparison", []):
        lines.append(
            "| {ctx} | {mlx_pp} | {df_pp} | {mlx_gen} | {df_gen} | {mlx_mem} | {mlx_phys} | {df_mem} | {df_phys} | {df_phys_delta} | {df_cache} | {status} | {wall_ratio} |".format(
                ctx=format_context(row["context_tokens"]),
                mlx_pp=_fmt(row.get("mlxlm_prompt_tps")),
                df_pp=_fmt(row.get("dflash_prompt_tps")),
                mlx_gen=_fmt(row.get("mlxlm_generation_tps")),
                df_gen=_fmt(row.get("dflash_generation_tps")),
                mlx_mem=_fmt(row.get("mlxlm_mlx_peak_gb")),
                mlx_phys=_fmt(row.get("mlxlm_phys_footprint_peak_gb")),
                df_mem=_fmt(row.get("dflash_mlx_peak_gb")),
                df_phys=_fmt(row.get("dflash_phys_footprint_peak_gb")),
                df_phys_delta=_fmt(row.get("dflash_phys_footprint_delta_gb")),
                df_cache=_fmt(row.get("dflash_sampled_mlx_cache_peak_gb")),
                status=row.get("compare_status", "pending"),
                wall_ratio=_fmt(row.get("dflash_wall_ratio"), digits=3),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Rows",
            "",
            "| backend | context | prompt tokens | generated | wall s | ttft ms | prefill tok/s | gen tok/s | mlx peak GB | foot start GB | foot peak GB | foot delta GB | mlx cache peak GB | accept | cycles |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("rows", []):
        lines.append(
            "| {backend} | {ctx} | {prompt} | {gen} | {wall} | {ttft} | {pp} | {gtps} | {mem} | {phys_start} | {phys} | {phys_delta} | {cache_peak} | {accept} | {cycles} |".format(
                backend=row["backend"],
                ctx=format_context(row["context_tokens"]),
                prompt=row["prompt_tokens"],
                gen=row["generated_tokens"],
                wall=_fmt(row.get("wall_s"), digits=3),
                ttft=_fmt(row.get("ttft_ms")),
                pp=_fmt(row.get("prompt_tps")),
                gtps=_fmt(row.get("generation_tps")),
                mem=_fmt(row.get("mlx_peak_gb")),
                phys_start=_fmt(row.get("phys_footprint_start_gb")),
                phys=_fmt(row.get("phys_footprint_peak_gb")),
                phys_delta=_fmt(row.get("phys_footprint_delta_gb")),
                cache_peak=_fmt(row.get("sampled_mlx_cache_peak_gb")),
                accept=_fmt(row.get("acceptance_ratio"), digits=3),
                cycles=row.get("cycles") if row.get("cycles") is not None else "n/a",
            )
        )
    return "\n".join(lines)


class MemorySampler:
    def __init__(self, *, backend: str, context_tokens: int, interval_s: float) -> None:
        self.backend = str(backend)
        self.context_tokens = int(context_tokens)
        self.interval_s = max(0.05, float(interval_s))
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemorySampler":
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()

    def summary(self) -> dict[str, float | None]:
        post_reset_samples = self.samples[1:] if len(self.samples) > 1 else self.samples
        return {
            "rss_peak_gb": _max_gb(self.samples, "rss_bytes"),
            "rss_start_gb": _first_gb(self.samples, "rss_bytes"),
            "rss_end_gb": _last_gb(self.samples, "rss_bytes"),
            "rss_delta_gb": _delta_gb(self.samples, "rss_bytes"),
            "phys_footprint_peak_gb": _max_gb(self.samples, "phys_footprint_bytes"),
            "phys_footprint_start_gb": _first_gb(self.samples, "phys_footprint_bytes"),
            "phys_footprint_end_gb": _last_gb(self.samples, "phys_footprint_bytes"),
            "phys_footprint_delta_gb": _delta_gb(self.samples, "phys_footprint_bytes"),
            "sampled_mlx_active_peak_gb": _max_gb(self.samples, "mlx_active_bytes"),
            "sampled_mlx_active_start_gb": _first_gb(self.samples, "mlx_active_bytes"),
            "sampled_mlx_active_end_gb": _last_gb(self.samples, "mlx_active_bytes"),
            "sampled_mlx_active_delta_gb": _delta_gb(self.samples, "mlx_active_bytes"),
            "sampled_mlx_cache_peak_gb": _max_gb(self.samples, "mlx_cache_bytes"),
            "sampled_mlx_cache_start_gb": _first_gb(self.samples, "mlx_cache_bytes"),
            "sampled_mlx_cache_end_gb": _last_gb(self.samples, "mlx_cache_bytes"),
            "sampled_mlx_cache_delta_gb": _delta_gb(self.samples, "mlx_cache_bytes"),
            "sampled_mlx_peak_gb": _max_gb(post_reset_samples, "mlx_peak_bytes"),
            "system_wired_peak_gb": _max_gb(self.samples, "system_wired_bytes"),
            "system_wired_start_gb": _first_gb(self.samples, "system_wired_bytes"),
            "system_wired_end_gb": _last_gb(self.samples, "system_wired_bytes"),
            "system_wired_delta_gb": _delta_gb(self.samples, "system_wired_bytes"),
            "darwin_device_peak_gb": _max_gb(self.samples, "darwin_device_bytes"),
            "darwin_internal_peak_gb": _max_gb(self.samples, "darwin_internal_bytes"),
            "darwin_compressed_peak_gb": _max_gb(self.samples, "darwin_compressed_bytes"),
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        snapshot = dict(process_memory_snapshot())
        snapshot.update(
            {
                "ts": time.time(),
                "backend": self.backend,
                "context_tokens": self.context_tokens,
            }
        )
        self.samples.append(snapshot)


def _baseline_row(
    raw: dict[str, Any],
    context: int,
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_tokens = int(raw.get("prompt_token_count") or context)
    prefill_s = float(raw.get("prefill_us") or 0.0) / 1e6
    return {
        "backend": "mlxlm",
        **_row_regime_fields(args),
        "context_tokens": int(context),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(raw.get("generation_tokens") or 0),
        "wall_s": float(raw.get("elapsed_us") or 0.0) / 1e6,
        "ttft_ms": float(raw.get("prefill_us") or 0.0) / 1000.0,
        "prefill_s": prefill_s if prefill_s > 0.0 else None,
        "prompt_tps": (prompt_tokens / prefill_s) if prefill_s > 0.0 else None,
        "generation_tps": raw.get("generation_tps"),
        "mlx_peak_gb": raw.get("peak_memory_gb"),
        "token_sha256": _token_hash(raw.get("generated_token_ids") or ()),
        "model": meta.get("resolved_model_ref"),
    }


def _dflash_row(
    raw: dict[str, Any],
    context: int,
    bundle: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt_tokens = int(raw.get("prompt_token_count") or context)
    phase_timings = dict(raw.get("phase_timings_us") or {})
    prefill_us = float(raw.get("prefill_us") or phase_timings.get("prefill") or 0.0)
    prefill_s = prefill_us / 1e6
    elapsed_us = float(raw.get("elapsed_us") or 0.0)
    generated_tokens = int(raw.get("generation_tokens") or 0)
    generation_us = max(0.0, elapsed_us - prefill_us)
    cycles = int(raw.get("cycles_completed") or 0)
    return {
        "backend": "dflash",
        **_row_regime_fields(args),
        "context_tokens": int(context),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "wall_s": elapsed_us / 1e6,
        "ttft_ms": float(raw.get("ttft_us") or prefill_us) / 1000.0,
        "prefill_s": prefill_s if prefill_s > 0.0 else None,
        "prompt_tps": (prompt_tokens / prefill_s) if prefill_s > 0.0 else None,
        "generation_tps": (generated_tokens / (generation_us / 1e6)) if generation_us > 0 else 0.0,
        "mlx_peak_gb": raw.get("peak_memory_gb"),
        "acceptance_ratio": raw.get("acceptance_ratio"),
        "cycles": cycles,
        "tokens_per_cycle": (generated_tokens / cycles) if cycles else None,
        "token_sha256": _token_hash(raw.get("generated_token_ids") or ()),
        "model": bundle.resolved_model_ref,
        "draft": bundle.resolved_draft_ref,
        "phase_timings_us": phase_timings,
    }


def _summary_payload(
    args: argparse.Namespace,
    contexts: list[int],
    prompt_hashes: dict[int, str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metadata": {
            "label": args.label or "context-grid",
            "target": str(args.target),
            "draft": str(args.draft) if args.draft else None,
            "backend": args.backend,
            "contexts": contexts,
            "max_tokens": int(args.max_tokens),
            "cooldown_s": float(args.cooldown),
            "no_eos": bool(args.no_eos),
            "clear_cache_between_cases": bool(args.clear_cache_between_cases),
            "process_regime": _process_regime(args),
            "cleanup_policy": _cleanup_policy(args),
            "clear_cache_boundaries": args.clear_cache_boundaries,
            "prompt_format": args.prompt_format,
            "hardware": _hardware_info(),
        },
        "prompt_hashes": {str(ctx): prompt_hashes[ctx] for ctx in prompt_hashes},
        "rows": list(rows),
        "comparison": _compare_rows(rows, contexts),
    }


def _process_regime(args: argparse.Namespace) -> str:
    return (
        "process_reused_cleanup"
        if bool(args.clear_cache_between_cases)
        else "process_reused_no_cleanup"
    )


def _cleanup_policy(args: argparse.Namespace) -> str:
    return "clear_between_cases" if bool(args.clear_cache_between_cases) else "none"


def _row_regime_fields(args: argparse.Namespace) -> dict[str, str]:
    return {
        "process_regime": _process_regime(args),
        "cleanup_policy": _cleanup_policy(args),
        "prompt_regime": str(args.prompt_format),
    }


def _compare_rows(rows: list[dict[str, Any]], contexts: list[int]) -> list[dict[str, Any]]:
    by_key = {(row["backend"], int(row["context_tokens"])): row for row in rows}
    out = []
    for ctx in contexts:
        mlx = by_key.get(("mlxlm", ctx))
        dflash = by_key.get(("dflash", ctx))
        item = {"context_tokens": ctx, "compare_status": "pending"}
        if mlx:
            item.update({f"mlxlm_{key}": mlx.get(key) for key in _COMPARE_FIELDS})
        if dflash:
            item.update({f"dflash_{key}": dflash.get(key) for key in _COMPARE_FIELDS})
        if mlx and dflash:
            same_count = mlx.get("generated_tokens") == dflash.get("generated_tokens")
            same_tokens = mlx.get("token_sha256") == dflash.get("token_sha256")
            if same_count and same_tokens and mlx.get("wall_s"):
                item["compare_status"] = "ok"
                item["dflash_wall_ratio"] = float(dflash["wall_s"]) / float(mlx["wall_s"])
            else:
                item["compare_status"] = "diverged"
                item["dflash_wall_ratio"] = None
        out.append(item)
    return out


_COMPARE_FIELDS = (
    "prompt_tps",
    "generation_tps",
    "mlx_peak_gb",
    "phys_footprint_peak_gb",
    "phys_footprint_start_gb",
    "phys_footprint_delta_gb",
    "sampled_mlx_cache_peak_gb",
    "sampled_mlx_cache_delta_gb",
    "wall_s",
    "generated_tokens",
    "token_sha256",
)


def _with_memory(row: dict[str, Any], memory: dict[str, float | None]) -> dict[str, Any]:
    out = dict(row)
    out.update(memory)
    if out.get("mlx_peak_gb") is None:
        out["mlx_peak_gb"] = out.get("sampled_mlx_peak_gb")
    return out


def _model_prompt_text() -> str:
    module = (
        "from __future__ import annotations\n\n"
        "import dataclasses\nimport json\nimport time\n"
        "from pathlib import Path\nfrom typing import Any\n\n"
        "@dataclasses.dataclass(frozen=True)\n"
        "class RequestState:\n"
        "    request_id: str\n"
        "    prompt_tokens: int\n"
        "    started_at: float\n"
        "    metadata: dict[str, Any]\n\n"
        "def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:\n"
        "    out = dict(payload)\n"
        "    out.setdefault('created_at', time.time())\n"
        "    out.setdefault('source', 'context-grid')\n"
        "    return out\n"
    )
    return "\n".join(f"# file_{idx}.py\n{module}" for idx in range(96))


def _encode_prompt_content(
    tokenizer: Any,
    content: str,
    *,
    prompt_format: str,
    enable_thinking: bool | None,
) -> list[int]:
    if prompt_format == "raw":
        return _tokenize(tokenizer, content)
    if not hasattr(tokenizer, "apply_chat_template"):
        return _tokenize(tokenizer, content)
    kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return [
        int(token)
        for token in tokenizer.apply_chat_template(
            [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            **kwargs,
        )
    ]


def _normalize_prompt_format(value: str) -> str:
    value = str(value or "").strip().lower()
    if value not in {"chat", "raw"}:
        raise ValueError("--prompt-format must be 'chat' or 'raw'")
    return value


def _context_line(index: int) -> str:
    return f"# context_record_{index:06d}: preserve behavior, measure latency, update artifacts.\n"


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return [int(token) for token in tokenizer.encode(text)]
    return [int(token) for token in tokenizer(text)]


def _decode_tokens(tokenizer: Any, tokens: Sequence[int]) -> str:
    if hasattr(tokenizer, "decode"):
        return str(tokenizer.decode(list(tokens)))
    return "".join(chr(int(token)) for token in tokens)


def _token_hash(tokens: Sequence[int]) -> str:
    h = hashlib.sha256()
    for token in tokens:
        h.update(int(token).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def _append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fp:
        for row in rows:
            fp.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def _max_gb(samples: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [int(sample[key]) for sample in samples if sample.get(key) is not None]
    if not values:
        return None
    return max(values) / GB


def _first_gb(samples: Sequence[dict[str, Any]], key: str) -> float | None:
    for sample in samples:
        if sample.get(key) is not None:
            return int(sample[key]) / GB
    return None


def _last_gb(samples: Sequence[dict[str, Any]], key: str) -> float | None:
    for sample in reversed(samples):
        if sample.get(key) is not None:
            return int(sample[key]) / GB
    return None


def _delta_gb(samples: Sequence[dict[str, Any]], key: str) -> float | None:
    first = _first_gb(samples, key)
    last = _last_gb(samples, key)
    if first is None or last is None:
        return None
    return last - first


def _after_case(args: argparse.Namespace, index: int, total: int) -> None:
    if index >= total - 1:
        return
    if args.clear_cache_between_cases:
        _release_loaded_models()
    if args.cooldown > 0:
        time.sleep(float(args.cooldown))


def _fmt(value: Any, *, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _effective_config(args: argparse.Namespace, contexts: list[int]) -> dict[str, Any]:
    return {
        "contexts": contexts,
        "backend": args.backend,
        "max_tokens": int(args.max_tokens),
        "cooldown": float(args.cooldown),
        "profile": args.profile,
        "draft_quant": args.draft_quant,
        "block_tokens": args.block_tokens,
        "prefill_step_size": args.prefill_step_size,
        "target_fa_window": args.target_fa_window,
        "draft_sink_size": args.draft_sink_size,
        "draft_window_size": args.draft_window_size,
        "verify_len_cap": args.verify_len_cap,
        "verify_mode": args.verify_mode,
        "split_sdpa": args.split_sdpa,
        "clear_cache_boundaries": args.clear_cache_boundaries,
        "clear_cache_between_cases": bool(args.clear_cache_between_cases),
        "no_eos": bool(args.no_eos),
        "prompt_format": args.prompt_format,
        "memory_sample_interval": float(args.memory_sample_interval),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Internal long-context speed grid for DFlash vs mlx_lm.",
    )
    p.add_argument("--target", required=True, help="HF target model ref or local path")
    p.add_argument("--draft", default=None, help="DFlash draft model ref or local path")
    p.add_argument("--backend", choices=("both", "mlxlm", "dflash"), default="both")
    p.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--cooldown", type=float, default=180.0)
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None, help="Output run directory")
    p.add_argument("--profile", default="balanced")
    p.add_argument("--draft-quant", default=None)
    p.add_argument("--block-tokens", type=int, default=None)
    p.add_argument("--prefill-step-size", type=int, default=None)
    p.add_argument("--target-fa-window", type=int, default=None)
    p.add_argument("--draft-sink-size", type=int, default=None)
    p.add_argument("--draft-window-size", type=int, default=None)
    p.add_argument("--verify-len-cap", type=int, default=None)
    p.add_argument("--verify-mode", choices=("auto", "adaptive", "off"), default=None)
    p.add_argument("--clear-cache-boundaries", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--clear-cache-between-cases", action="store_true")
    p.add_argument("--split-sdpa", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--memory-sample-interval", type=float, default=0.25)
    eos = p.add_mutually_exclusive_group()
    eos.add_argument(
        "--no-eos",
        dest="no_eos",
        action="store_true",
        default=True,
        help="Force generation to run to --max-tokens. Default: enabled.",
    )
    eos.add_argument(
        "--allow-eos",
        dest="no_eos",
        action="store_false",
        help="Allow EOS to stop generation before --max-tokens.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.backend == "dflash" and not args.draft:
        raise SystemExit("--draft is required when --backend=dflash")
    if args.backend == "both" and not args.draft:
        raise SystemExit("--draft is required when --backend=both")
    run_grid(args, sys.argv[1:] if argv is None else list(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
