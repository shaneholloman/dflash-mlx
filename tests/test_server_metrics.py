# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
from dflash_mlx.server.metrics import (
    configure_live_metrics,
    record_request_metrics,
    reset_live_metrics_for_tests,
    start_live_request,
    update_live_request,
)
from dflash_mlx import serve
from dflash_mlx.serve import DFlashAPIHandler, _build_prompt_regime


def _runtime_config(**overrides):
    values = {
        "profile": "balanced",
        "prefill_step_size": 4096,
        "draft_sink_size": 64,
        "draft_window_size": 1024,
        "verify_len_cap": 0,
        "prefix_cache": True,
        "prefix_cache_l2": False,
        "target_fa_window": 0,
        "dflash_max_ctx": 0,
        "memory_waterfall": False,
        "max_snapshot_tokens": 24000,
        "clear_cache_boundaries": False,
        "verify_mode": "auto",
        "prefix_cache_max_entries": 4,
        "prefix_cache_max_bytes": 8 * 1024**3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _configure_metrics(runtime_config=None):
    runtime_config = runtime_config or _runtime_config()
    configure_live_metrics(
        version="test-version",
        model_provider=SimpleNamespace(
            model_key=("target-model", None, "draft-model"),
            cli_args=SimpleNamespace(
                model="target-model",
                draft_model=None,
                runtime_config=runtime_config,
                diagnostics="off",
                metal_limits=SimpleNamespace(wired_bytes=64 * 1024**3),
            ),
        ),
    )
    return runtime_config


def _call_metrics_endpoint():
    statuses = []
    headers = []
    handler = object.__new__(DFlashAPIHandler)
    handler.path = "/metrics"
    handler.wfile = BytesIO()
    handler._set_completion_headers = lambda status=200: statuses.append(status)
    handler.send_header = lambda *args: headers.append(args)
    handler.end_headers = lambda: None

    DFlashAPIHandler.do_GET(handler)

    assert statuses == [200]
    assert any(header[0] == "Content-Length" for header in headers)
    return json.loads(handler.wfile.getvalue().decode())


class FakePrefixCache:
    def stats(self):
        return {
            "current_entries": 2,
            "max_entries": 4,
            "current_bytes": 1024,
            "max_bytes": 4096,
            "exact_hits": 1,
            "prefix_hits": 2,
            "misses": 4,
            "insertions": 6,
            "evictions": 1,
            "prefill_tokens_saved": 750,
        }


def test_diagnostics_post_event_records_prefill_details(tmp_path):
    diagnostics = DiagnosticsConfig(
        mode="full",
        run_dir=tmp_path,
        trace=TraceConfig(log_dir=tmp_path, cycle_events=True),
    )

    record_request_metrics(
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


def test_metrics_endpoint_returns_json_before_request(monkeypatch):
    reset_live_metrics_for_tests()
    _configure_metrics()
    monkeypatch.setattr(serve, "_current_dflash_prefix_cache", lambda: None)

    payload = _call_metrics_endpoint()

    assert payload["server"]["version"] == "test-version"
    assert payload["server"]["model"] == "target-model"
    assert payload["server"]["draft"] == "draft-model"
    assert payload["runtime"]["prefill_step_size"] == 4096
    assert payload["memory"].keys() >= {
        "rss_gb",
        "rss_peak_gb",
        "mlx_active_gb",
        "mlx_cache_gb",
        "mlx_peak_gb",
        "wired_gb",
        "wired_limit_gb",
    }
    assert payload["current_request"] is None
    assert payload["last_request"] is None
    assert payload["recent_requests"] == []
    assert payload["prefix_cache"]["entries"] == 0
    assert payload["totals"]["requests"] == 0


def test_metrics_endpoint_reports_current_request(monkeypatch):
    reset_live_metrics_for_tests()
    _configure_metrics()
    monkeypatch.setattr(serve, "_current_dflash_prefix_cache", lambda: None)

    start_live_request(
        request_id=9,
        mode_used="dflash",
        prompt_tokens=4096,
        max_tokens=64,
        cache_hit_tokens=3072,
        cache_lookup_ms=1.5,
    )
    update_live_request(
        request_id=9,
        state="prefill",
        prefill_tokens_processed=1024,
        prefill_tokens_total=4096,
    )

    payload = _call_metrics_endpoint()

    current = payload["current_request"]
    assert current["request_id"] == 9
    assert current["state"] == "prefill"
    assert current["prompt_tokens"] == 4096
    assert current["max_tokens"] == 64
    assert current["cache_hit_tokens"] == 3072
    assert current["cache_lookup_ms"] == 1.5
    assert current["prefill_tokens_processed"] == 1024
    assert current["prefill_tokens_total"] == 4096
    assert current["elapsed_s"] >= 0.0
    assert payload["last_request"] is None

    update_live_request(
        request_id=9,
        state="decode",
        generated_tokens=3,
        decode_tok_s=24.0,
        acceptance_rate=0.75,
    )

    payload = _call_metrics_endpoint()

    assert payload["current_request"]["state"] == "decode"
    assert payload["current_request"]["generated_tokens"] == 3
    assert payload["current_request"]["decode_tok_s"] == 24.0
    assert payload["current_request"]["acceptance_rate"] == 0.75


def test_metrics_endpoint_reports_last_request_and_prefix_cache(monkeypatch):
    reset_live_metrics_for_tests()
    runtime_config = _configure_metrics()
    monkeypatch.setattr(serve, "_current_dflash_prefix_cache", lambda: FakePrefixCache())

    record_request_metrics(
        request_id=12,
        summary_event={
            "generation_tokens": 100,
            "acceptance_ratio": 0.81,
            "cycles_completed": 44,
            "tokens_per_cycle": 2.27,
            "phase_timings_us": {"prefill": 1_000_000.0},
        },
        request_start_ns=0,
        request_done_ns=4_000_000_000,
        first_token_ns=1_000_000_000,
        prefill_done_ns=1_000_000_000,
        prompt_token_count=1000,
        live_token_count=100,
        cache_lookup_ms=0.0,
        cache_hit_tokens=750,
        cache_insert_ms=0.0,
        finish_reason="stop",
        max_tokens=100,
        prefill_event={
            "event": "prefill",
            "prompt_token_count": 1000,
            "logical_ctx_tokens": 1000,
            "physical_prefill_tokens": 250,
            "prefill_tokens_restored": 750,
            "prefill_tokens_computed": 250,
        },
        runtime_config=runtime_config,
        diagnostics=None,
    )

    payload = _call_metrics_endpoint()

    assert payload["current_request"] is None
    last = payload["last_request"]
    assert last["request_id"] == 12
    assert last["prompt_tokens"] == 1000
    assert last["generated_tokens"] == 100
    assert last["prefill_tok_s_physical"] == 250.0
    assert last["prefill_tok_s_apparent"] == 1000.0
    assert round(last["decode_tok_s"], 2) == 33.33
    assert last["acceptance_rate"] == 0.81
    assert last["cycles"] == 44
    assert last["finish_reason"] == "stop"
    assert payload["prefix_cache"]["hits"] == 3
    assert payload["prefix_cache"]["misses"] == 4
    assert payload["prefix_cache"]["last_restored_tokens"] == 750
    assert payload["prefix_cache"]["last_computed_tokens"] == 250
    assert payload["totals"]["requests"] == 1
    assert payload["totals"]["generated_tokens"] == 100
    assert payload["totals"]["prefill_tokens_physical"] == 250
    assert payload["totals"]["prefill_tokens_restored"] == 750
    assert len(payload["recent_requests"]) == 1
    assert payload["recent_requests"][0]["request_id"] == 12


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
