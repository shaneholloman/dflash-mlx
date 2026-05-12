# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.benchmarks import context_grid
from tools.benchmarks.context_grid import (
    DEFAULT_FINAL_REQUEST,
    build_context_prompt_tokens,
    format_context,
    parse_contexts,
    run_grid,
)


class FakeTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]

    def decode(self, tokens):
        return "".join(chr(int(token)) for token in tokens)

    def apply_chat_template(self, messages, **kwargs):
        text = "\n".join(str(message["content"]) for message in messages)
        if kwargs.get("add_generation_prompt"):
            text += "\nAssistant:"
        if kwargs.get("tokenize"):
            return self.encode(text)
        return text


def test_parse_contexts_accepts_suffixes_and_fractional_k():
    assert parse_contexts("512,1k,2.5k") == [512, 1024, 2560]
    assert format_context(512) == "0.5k"
    assert format_context(2048) == "2k"


def test_parse_contexts_rejects_invalid_values():
    with pytest.raises(ValueError, match="invalid context bucket"):
        parse_contexts("2kb")
    with pytest.raises(ValueError, match="at least one"):
        parse_contexts("")


def test_build_context_prompt_tokens_hits_target_and_preserves_final_task():
    tokenizer = FakeTokenizer()
    tokens = build_context_prompt_tokens(tokenizer, 2048, prompt_format="raw")

    assert len(tokens) == 2048
    assert tokens[-len(tokenizer.encode(DEFAULT_FINAL_REQUEST)) :] == tokenizer.encode(
        DEFAULT_FINAL_REQUEST
    )


def test_run_grid_writes_incremental_rows_and_gates_diverged_compare(monkeypatch, tmp_path):
    order = []
    bundle_kwargs = {}
    tokenizer = FakeTokenizer()
    bundle = SimpleNamespace(
        target_model=object(),
        target_ops=object(),
        tokenizer=tokenizer,
        draft_model=object(),
        draft_backend=object(),
        resolved_model_ref="target",
        resolved_draft_ref="draft",
    )

    monkeypatch.setattr(
        context_grid,
        "_load_pristine_target_bundle",
        lambda _target: (object(), tokenizer, {"resolved_model_ref": "target"}),
    )
    def fake_load_runtime_bundle(**kwargs):
        bundle_kwargs.update(kwargs)
        return bundle

    monkeypatch.setattr(context_grid, "load_runtime_bundle", fake_load_runtime_bundle)
    monkeypatch.setattr(context_grid, "get_stop_token_ids", lambda _tokenizer: [0])
    release_calls = []
    monkeypatch.setattr(context_grid, "_release_loaded_models", lambda: release_calls.append("release"))
    monkeypatch.setattr(context_grid, "_hardware_info", lambda: {"chip": "test"})
    monkeypatch.setattr(context_grid.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        context_grid,
        "process_memory_snapshot",
        lambda: {
            "rss_bytes": 2_000_000_000,
            "phys_footprint_bytes": 5_000_000_000,
            "system_wired_bytes": 1_000_000_000,
            "mlx_active_bytes": 3_000_000_000,
            "mlx_cache_bytes": 1_000_000_000,
            "mlx_peak_bytes": 4_000_000_000,
            "untracked_bytes": 0,
        },
    )

    def baseline(**kwargs):
        order.append(("mlxlm", len(kwargs["prompt_tokens_override"])))
        return {
            "prompt_token_count": len(kwargs["prompt_tokens_override"]),
            "generation_tokens": 2,
            "generated_token_ids": [1, 2],
            "elapsed_us": 20_000.0,
            "prefill_us": 10_000.0,
            "generation_tps": 200.0,
            "peak_memory_gb": 4.0,
        }

    def dflash(**kwargs):
        order.append(("dflash", len(kwargs["prompt_tokens_override"])))
        return {
            "prompt_token_count": len(kwargs["prompt_tokens_override"]),
            "generation_tokens": 2,
            "generated_token_ids": [1, 3],
            "elapsed_us": 10_000.0,
            "prefill_us": 5_000.0,
            "phase_timings_us": {"prefill": 5_000.0},
            "acceptance_ratio": 0.5,
            "cycles_completed": 1,
            "peak_memory_gb": 4.5,
        }

    monkeypatch.setattr(context_grid, "_generate_stock_baseline_once", baseline)
    monkeypatch.setattr(context_grid, "_generate_dflash_stream_once", dflash)

    args = context_grid.build_parser().parse_args(
        [
            "--target",
            "target",
            "--draft",
            "draft",
            "--contexts",
            "512,1k",
            "--cooldown",
            "0",
            "--clear-cache-between-cases",
            "--verify-mode",
            "adaptive",
            "--split-sdpa",
            "--out",
            str(tmp_path / "grid"),
        ]
    )
    summary = run_grid(args, [])

    assert order == [("mlxlm", 512), ("mlxlm", 1024), ("dflash", 512), ("dflash", 1024)]
    assert release_calls == ["release", "release", "release", "release"]
    assert (tmp_path / "grid" / "summary.json").exists()
    assert (tmp_path / "grid" / "summary.md").exists()
    assert (tmp_path / "grid" / "memory_samples.jsonl").exists()
    rows = (tmp_path / "grid" / "rows.jsonl").read_text().splitlines()
    assert len(rows) == 4
    assert json.loads(rows[0])["phys_footprint_peak_gb"] == 5.0
    assert json.loads(rows[0])["phys_footprint_delta_gb"] == 0.0
    assert json.loads(rows[0])["sampled_mlx_cache_peak_gb"] == 1.0
    assert json.loads(rows[0])["process_regime"] == "process_reused_cleanup"
    assert json.loads(rows[0])["cleanup_policy"] == "clear_between_cases"
    assert json.loads(rows[0])["prompt_regime"] == "chat"
    assert summary["metadata"]["process_regime"] == "process_reused_cleanup"
    assert summary["metadata"]["cleanup_policy"] == "clear_between_cases"
    assert summary["comparison"][0]["compare_status"] == "diverged"
    assert summary["comparison"][0]["dflash_wall_ratio"] is None
    assert bundle_kwargs["verify_config"].mode == "adaptive"
    assert bundle_kwargs["split_full_attention_sdpa"] is True
    manifest = json.loads((tmp_path / "grid" / "manifest.json").read_text())
    assert manifest["effective_config"]["verify_mode"] == "adaptive"
    assert manifest["effective_config"]["split_sdpa"] is True


def test_memory_sampler_ignores_first_stale_mlx_peak(monkeypatch):
    snapshots = iter(
        [
            {
                "rss_bytes": 1_000_000_000,
                "phys_footprint_bytes": 2_000_000_000,
                "system_wired_bytes": 3_000_000_000,
                "mlx_active_bytes": 4_000_000_000,
                "mlx_cache_bytes": 5_000_000_000,
                "mlx_peak_bytes": 100_000_000_000,
                "untracked_bytes": 0,
            },
            {
                "rss_bytes": 1_500_000_000,
                "phys_footprint_bytes": 4_000_000_000,
                "system_wired_bytes": 3_500_000_000,
                "mlx_active_bytes": 4_500_000_000,
                "mlx_cache_bytes": 8_000_000_000,
                "mlx_peak_bytes": 6_000_000_000,
                "untracked_bytes": 0,
            },
        ]
    )
    monkeypatch.setattr(context_grid, "process_memory_snapshot", lambda: next(snapshots))

    sampler = context_grid.MemorySampler(
        backend="dflash",
        context_tokens=512,
        interval_s=1.0,
    )
    sampler._sample()
    sampler._sample()

    summary = sampler.summary()

    assert summary["sampled_mlx_peak_gb"] == 6.0
    assert summary["phys_footprint_start_gb"] == 2.0
    assert summary["phys_footprint_delta_gb"] == 2.0
    assert summary["sampled_mlx_cache_delta_gb"] == 3.0
