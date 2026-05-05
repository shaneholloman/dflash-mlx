# Copyright 2026 bstnxbt
# MIT License - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
from types import SimpleNamespace

from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
from dflash_mlx.server.metrics import log_bench_post
from dflash_mlx.serve import _build_prompt_regime


def test_diagnostics_post_event_records_prefill_details(tmp_path):
    diagnostics = DiagnosticsConfig(
        mode="full",
        run_dir=tmp_path,
        trace=TraceConfig(log_dir=tmp_path, cycle_events=True),
    )

    log_bench_post(
        request_id=7,
        summary_event={
            "generation_tokens": 1,
            "acceptance_ratio": 0.0,
            "cycles_completed": 0,
            "tokens_per_cycle": 0.0,
            "phase_timings_us": {"prefill": 2_000_000.0},
        },
        request_start_ns=1_000_000_000,
        request_done_ns=3_500_000_000,
        first_token_ns=3_000_000_000,
        prefill_done_ns=3_000_000_000,
        prompt_token_count=4096,
        live_token_count=1,
        cache_lookup_ms=0.0,
        cache_hit_tokens=0,
        cache_insert_ms=0.0,
        finish_reason="length",
        max_tokens=1,
        prefill_event={
            "event": "prefill",
            "prompt_token_count": 4096,
            "logical_ctx_tokens": 4096,
            "physical_prefill_tokens": 1024,
            "prefill_tokens_restored": 3072,
            "prefill_tokens_computed": 1024,
            "phase_cold_us": 1_900_000.0,
            "phase_seam_us": 100_000.0,
        },
        runtime_config=SimpleNamespace(
            profile="balanced",
            prefill_step_size=8192,
            draft_sink_size=64,
            draft_window_size=1024,
            verify_len_cap=0,
            prefix_cache=False,
            prefix_cache_l2=False,
            target_fa_window=0,
            dflash_max_ctx=0,
            memory_waterfall=True,
            max_snapshot_tokens=24000,
            clear_cache_boundaries=False,
            verify_mode="auto",
        ),
        diagnostics=diagnostics,
    )

    row = json.loads((tmp_path / "post_events.jsonl").read_text().splitlines()[-1])

    assert row["prefill_tok_s"] == 2048.0
    assert row["logical_ctx_tokens"] == 4096
    assert row["physical_prefill_tokens"] == 1024
    assert row["prefill_tokens_restored"] == 3072
    assert row["prefill_tokens_computed"] == 1024
    assert row["runtime_config"]["prefill_step_size"] == 8192
    assert row["prefill_phase_timings_us"] == {
        "phase_cold_us": 1_900_000.0,
        "phase_seam_us": 100_000.0,
    }
    assert "prefill_tok_s" in (tmp_path / "summary.md").read_text()


def test_prompt_regime_distinguishes_text_completion_from_chat():
    tokenizer = SimpleNamespace(has_chat_template=True)
    args = SimpleNamespace(
        chat_template_args={"enable_thinking": True},
        use_default_chat_template=True,
        chat_template="template",
    )

    completion = _build_prompt_regime(
        args,
        tokenizer,
        SimpleNamespace(request_type="text"),
    )
    chat = _build_prompt_regime(
        args,
        tokenizer,
        SimpleNamespace(request_type="chat"),
    )

    assert completion["request_type"] == "text"
    assert completion["chat_template"] is False
    assert completion["chat_template_args"] == {}
    assert chat["request_type"] == "chat"
    assert chat["chat_template"] is True
    assert chat["enable_thinking"] is True
