# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import argparse
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dflash_mlx.benchmark_suites import BenchmarkPrompt, ctx_tokens

def aggregate_prompt_reports(prompt_reports: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [dict(report.get("summary", {})) for report in prompt_reports]
    configs = [dict(report.get("config", {})) for report in prompt_reports]
    prompt_tokens = [
        int(config.get("prompt_tokens", 0))
        for config in configs
        if config.get("prompt_tokens") is not None
    ]
    return {
        "prompt_tok_avg": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None,
        "baseline_tps_median": _median([summary.get("baseline_tps_median") for summary in summaries]),
        "dflash_tps_median": _median([summary.get("dflash_tps_median") for summary in summaries]),
        "speedup_median": _median([summary.get("speedup_median") for summary in summaries]),
        "baseline_ttft_ms_median": _median(
            run.get("baseline", {}).get("ttft_ms")
            for report in prompt_reports
            for run in report.get("runs", [])
        ),
        "dflash_ttft_ms_median": _median(
            run.get("dflash", {}).get("ttft_ms")
            for report in prompt_reports
            for run in report.get("runs", [])
        ),
        "baseline_peak_memory_gb_median": _median(
            summary.get("baseline_peak_memory_gb_median") for summary in summaries
        ),
        "dflash_peak_memory_gb_median": _median(
            summary.get("dflash_peak_memory_gb_median") for summary in summaries
        ),
        "acceptance_ratio_median": _median(
            summary.get("acceptance_ratio_median") for summary in summaries
        ),
        "prefix_saved_tokens": None,
        "prefix_cache_stats": None,
        "prefill_tps_baseline": None,
        "prefill_tps_dflash": None,
    }

def suite_report(
    *,
    prompts: list[BenchmarkPrompt],
    prompt_reports: list[dict[str, Any]],
    args: argparse.Namespace,
    include_memory: bool,
) -> dict[str, Any]:
    first = prompt_reports[0] if prompt_reports else {}
    config = dict(first.get("config", {}))
    config.update(
        {
            "suite": str(args.suite),
            "limit": int(args.limit),
            "ctx_tokens": ctx_tokens(args),
            "ctx": ctx_tokens(args),
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
            "prompt_source": prompts[0].source if prompts else None,
            "hf_dataset_name": prompts[0].hf_dataset_name if prompts else None,
            "hf_dataset_config": prompts[0].hf_dataset_config if prompts else None,
            "hf_dataset_split": prompts[0].hf_dataset_split if prompts else None,
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "hf_shuffle_seed": int(args.seed) if args.shuffle else None,
            "selected_row_indices": [prompt.row_index for prompt in prompts],
            "prompt_ids": [prompt.id for prompt in prompts],
            "prompt_count": len(prompts),
            "benchmark_mode": str(args.suite),
            "max_tokens": int(args.max_tokens),
            "block_tokens": int(args.block_tokens),
            "include_memory": bool(include_memory),
            "no_memory": not bool(include_memory),
            "repeat": int(args.repeat),
            "cooldown": int(args.cooldown),
            "use_chat_template": not bool(args.no_chat_template),
            "draft_quant": args.draft_quant,
            "no_eos": bool(args.no_eos),
            "split_sdpa": bool(args.split_sdpa),
            "target_fa_window": int(args.target_fa_window),
            "draft_sink_size": int(args.draft_sink_size),
            "draft_window_size": int(args.draft_window_size),
            "verify_len_cap": int(args.verify_len_cap),
            "prompt_tokenization_mode": "chat_template"
            if not bool(args.no_chat_template)
            else "raw",
        }
    )
    return {
        "hardware": first.get("hardware", {}),
        "config": config,
        "prompts": prompt_reports,
        "runs": flatten_prompt_runs(prompt_reports, suite_config=config),
        "summary": aggregate_prompt_reports(prompt_reports),
    }

def flatten_prompt_runs(
    prompt_reports: list[dict[str, Any]],
    *,
    suite_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in prompt_reports:
        config = dict(report.get("config", {}))
        for run in report.get("runs", []):
            row = dict(run)
            row["prompt_id"] = config.get("prompt_id")
            row["prompt_suite"] = config.get("prompt_suite")
            row["suite"] = suite_config.get("suite")
            row["model"] = suite_config.get("model")
            row["draft"] = suite_config.get("draft")
            row["git_hash"] = suite_config.get("git_hash")
            row["prompt_tokenization_mode"] = suite_config.get("prompt_tokenization_mode")
            row["max_tokens"] = suite_config.get("max_tokens")
            row["block_tokens"] = suite_config.get("block_tokens")
            row["use_chat_template"] = suite_config.get("use_chat_template")
            row["split_sdpa"] = suite_config.get("split_sdpa")
            row["target_fa_window"] = suite_config.get("target_fa_window")
            row["draft_sink_size"] = suite_config.get("draft_sink_size")
            row["draft_window_size"] = suite_config.get("draft_window_size")
            row["verify_len_cap"] = suite_config.get("verify_len_cap")
            rows.append(row)
    return rows

def print_summary(result: dict[str, Any], output_path: Path) -> None:
    summary = result.get("summary", {})
    config = result.get("config", {})
    print("Summary:")
    print(f"  suite                         : {config.get('suite')}")
    print(f"  prompts                       : {config.get('prompt_count')}")
    print(f"  baseline generation_tps median: {summary.get('baseline_tps_median')}")
    print(f"  dflash generation_tps median  : {summary.get('dflash_tps_median')}")
    print(f"  speedup median                : {summary.get('speedup_median')}")
    print(f"  acceptance median             : {summary.get('acceptance_ratio_median')}")
    if "baseline_peak_memory_gb_median" in summary or "dflash_peak_memory_gb_median" in summary:
        print(f"  baseline peak memory median   : {summary.get('baseline_peak_memory_gb_median')}")
        print(f"  dflash peak memory median     : {summary.get('dflash_peak_memory_gb_median')}")
    print(f"Artifacts written to: {output_path}")

def summary_markdown(result: dict[str, Any]) -> str:
    config = result.get("config", {})
    summary = result.get("summary", {})
    speedup = summary.get("speedup_median")
    speedup_text = "n/a" if speedup is None else f"{float(speedup):.2f}x"
    ttft = summary.get("dflash_ttft_ms_median")
    peak_memory = summary.get("dflash_peak_memory_gb_median")
    acceptance = summary.get("acceptance_ratio_median")
    prompt_rows = []
    for prompt_report in result.get("prompts", []):
        prompt_config = dict(prompt_report.get("config", {}))
        prompt_summary = dict(prompt_report.get("summary", {}))
        prompt_rows.append(
            "| {id} | {tokens} | {base} | {dflash} | {speedup} | {acceptance} |".format(
                id=prompt_config.get("prompt_id"),
                tokens=_md_value(prompt_config.get("prompt_tokens"), precision=0),
                base=_md_value(prompt_summary.get("baseline_tps_median")),
                dflash=_md_value(prompt_summary.get("dflash_tps_median")),
                speedup=(
                    "n/a"
                    if prompt_summary.get("speedup_median") is None
                    else f"{float(prompt_summary['speedup_median']):.2f}x"
                ),
                acceptance=_md_value(prompt_summary.get("acceptance_ratio_median")),
            )
        )
    lines = [
        "# DFlash Benchmark",
        "",
        "| suite | prompts | prompt tok avg | baseline tok/s | dflash tok/s | speedup | TTFT | peak memory | acceptance | prefix saved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {suite} | {prompts} | {prompt_tok_avg} | {baseline} | {dflash} | {speedup} | {ttft} | {peak} | {acceptance} | {prefix} |".format(
            suite=config.get("suite", config.get("benchmark_mode")),
            prompts=_md_value(config.get("prompt_count", 1), precision=0),
            prompt_tok_avg=_md_value(summary.get("prompt_tok_avg")),
            baseline=_md_value(summary.get("baseline_tps_median")),
            dflash=_md_value(summary.get("dflash_tps_median")),
            speedup=speedup_text,
            ttft=_md_value(ttft, suffix=" ms"),
            peak=_md_value(peak_memory, suffix=" GB"),
            acceptance=_md_value(acceptance),
            prefix=_md_value(summary.get("prefix_saved_tokens"), precision=0),
        ),
        "",
        f"- mode: {config.get('benchmark_mode')}",
        f"- suite: {config.get('suite')}",
        f"- model: {config.get('model')}",
        f"- draft: {config.get('draft')}",
        f"- git_hash: {config.get('git_hash')}",
        f"- max_tokens: {config.get('max_tokens')}",
        f"- block_tokens: {config.get('block_tokens')}",
        f"- repeat: {config.get('repeat')}",
        f"- cooldown: {config.get('cooldown')}",
        f"- prompt_count: {config.get('prompt_count', 1)}",
        f"- prompt_ids: {', '.join(str(item) for item in config.get('prompt_ids', []))}",
        f"- prompt_source: {config.get('prompt_source')}",
        f"- prompt_tokenization_mode: {config.get('prompt_tokenization_mode')}",
        f"- use_chat_template: {config.get('use_chat_template')}",
        f"- split_sdpa: {config.get('split_sdpa')}",
        f"- target_fa_window: {config.get('target_fa_window')}",
        f"- draft_window: {config.get('draft_sink_size')}+{config.get('draft_window_size')}",
        f"- verify_len_cap: {config.get('verify_len_cap')}",
        "",
        "## Per Prompt",
        "",
        "| prompt id | prompt tokens | baseline tok/s | dflash tok/s | speedup | acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(prompt_rows or ["| n/a | n/a | n/a | n/a | n/a | n/a |"])
    return "\n".join(lines) + "\n"

def _median(values: Sequence[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    return statistics.median(filtered) if filtered else None

def _md_value(value: Any, *, suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}{suffix}"
    return f"{value}{suffix}"
