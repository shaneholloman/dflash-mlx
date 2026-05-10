# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import hashlib
import http.client
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import dflash_mlx.artifacts as artifacts
from tools.benchmarks import _agentic_trace as agentic_trace
from tools.benchmarks import _agentic_proxy as agentic_proxy
from tools.benchmarks._agentic_trace import (
    _apply_dflash_stderr_totals,
    _aggregate,
    _build_server_cmd,
    _ensure_replay_outputs,
    _replay_body_bytes,
    _render_peer_comparison,
    derive_post_landmarks,
    summarize_cycles,
)

def test_summarize_cycles_reports_tokens_per_cycle():
    summary = summarize_cycles(
        [
            {
                "commit_count": 3,
                "verify_us": 1000,
                "block_len": 16,
                "acceptance_len": 2,
            },
            {
                "commit_count": 5,
                "verify_us": 2000,
                "block_len": 16,
                "acceptance_len": 4,
            },
        ]
    )

    assert summary is not None
    assert summary["n_cycles"] == 2
    assert summary["total_commits"] == 8
    assert summary["tokens_per_cycle"] == 4.0

def test_derive_post_landmarks_counts_unique_streamed_tool_calls(tmp_path):
    sse_path = tmp_path / "001.jsonl"
    events = [
        {"type": "first_byte", "t_ms": 10.0},
        {
            "type": "event",
            "t_ms": 20.0,
            "payload": {
                "data": {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": "{\"path\""},
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        },
        {
            "type": "event",
            "t_ms": 25.0,
            "payload": {
                "data": {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": ": \"a.py\"}"},
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        },
        {
            "type": "event",
            "t_ms": 30.0,
            "payload": {
                "data": {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 1,
                                        "function": {"arguments": "{\"cmd\": \"pytest\"}"},
                                    }
                                ]
                            },
                        }
                    ]
                }
            },
        },
    ]
    sse_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    landmarks = derive_post_landmarks(sse_path)

    assert landmarks["first_tool_call_sent_ms"] == 20.0
    assert landmarks["tool_call_complete_ms"] == 30.0
    assert landmarks["tool_call_delta_count"] == 3
    assert landmarks["tool_call_count"] == 2
    assert landmarks["finish_reason"] == "tool_calls"

def test_derive_post_landmarks_rejects_malformed_jsonl(tmp_path):
    sse_path = tmp_path / "001.jsonl"
    sse_path.write_text("{bad json\n")

    with pytest.raises(RuntimeError, match="malformed SSE trace JSONL"):
        derive_post_landmarks(sse_path)

def test_aggregate_reports_cycle_and_tool_totals():
    posts = [
        {
            "landmarks": {"tool_call_count": 2, "first_tool_call_sent_ms": 100.0},
            "server_metric": {
                "prompt_tokens": 10,
                "tokens": 6,
                "wall_s": 0.5,
                "accept": 0.75,
                "cache_hit_tokens": 8,
                "cycles_summary": {"n_cycles": 2, "total_commits": 6},
            },
        }
    ]

    totals = _aggregate(posts, wall_s=1.0)

    assert totals["total_cycles"] == 2
    assert totals["total_cycle_commits"] == 6
    assert totals["avg_tokens_per_cycle"] == 3.0
    assert totals["total_tool_calls"] == 2

def test_aggregate_falls_back_to_post_event_cycles_when_cycle_events_absent():
    posts = [
        {
            "landmarks": {"tool_call_count": 1},
            "server_metric": {
                "prompt_tokens": 10,
                "tokens": 7,
                "wall_s": 0.5,
                "cache_hit_tokens": 8,
                "cycles_completed": 3,
                "cycles_summary": None,
            },
        }
    ]

    totals = _aggregate(posts, wall_s=1.0)

    assert totals["total_cycles"] == 3
    assert totals["total_cycle_commits"] == 7
    assert totals["avg_tokens_per_cycle"] == pytest.approx(7 / 3)

def test_aggregate_uses_usage_cached_tokens_without_dflash_metrics():
    posts = [
        {
            "landmarks": {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
                "first_byte_ms": 10.0,
                "end_t_ms": 60.0,
            },
            "server_metric": None,
        }
    ]

    totals = _aggregate(posts, wall_s=0.1)

    assert totals["total_prompt_tokens"] == 100
    assert totals["total_decode_tokens"] == 5
    assert totals["total_cache_hit_tokens"] == 80

def test_agentic_rows_normalize_dflash_post_metrics():
    summary = {
        "metadata": {
            "label": "row-test",
            "backend": "dflash",
            "target": "target-model",
            "draft": "draft-model",
        },
        "client": "replay",
        "posts": [
            {
                "idx": 2,
                "request": {"tools_count": 1},
                "landmarks": {"tool_call_count": 1, "finish_reason": "tool_calls"},
                "server_metric": {
                    "prompt_tokens": 13370,
                    "tokens": 68,
                    "wall_s": 4.148,
                    "ttft_ms_server": 956.0,
                    "prefill_ms_server": 955.0,
                    "decode_ms_server": 3193.0,
                    "decode_tok_s": 21.3,
                    "cache_hit_tokens": 13136,
                    "logical_ctx_tokens": 13370,
                    "prefill_tokens_computed": 234,
                    "accept": 0.706,
                    "tokens_per_cycle": 3.4,
                    "cycles_completed": 20,
                    "cache_hit_source": "L2",
                    "memory_waterfall_peak": {
                        "phys_footprint_gb": 25.2,
                        "mlx_cache_gb": 6.1,
                    },
                    "memory_boundary_start": {"phys_footprint_bytes": 21_000_000_000},
                    "memory_boundary_end": {"phys_footprint_bytes": 24_500_000_000},
                },
                "effective_finish_reason": "tool_calls",
            }
        ],
    }

    rows = agentic_trace._normalized_post_rows(summary)
    rendered = agentic_trace._render_rows_markdown(summary, rows)

    assert rows == [
        {
            "run": "row-test",
            "backend": "dflash",
            "mode": "replay",
            "turn_post": 2,
            "cache": "warm-l2",
            "cache_hit_source": "L2",
            "prompt_tok": 13370,
            "ctx_tok": 13370,
            "cached_tok": 13136,
            "computed_tok": 234,
            "cache_hit": pytest.approx(13136 / 13370),
            "ttft_ms": 956.0,
            "prefill_ms": 955.0,
            "decode_ms": 3193.0,
            "decode_tok_s": 21.3,
            "out_tok": 68,
            "wall_s": 4.148,
            "acceptance": 0.706,
            "tokens_per_cycle": 3.4,
            "cycles": 20,
            "phys_footprint_peak_gb": 25.2,
            "phys_footprint_start_gb": 21.0,
            "phys_footprint_end_gb": 24.5,
            "phys_footprint_delta_gb": 3.5,
            "rss_peak_gb": None,
            "mlx_active_peak_gb": None,
            "mlx_cache_peak_gb": 6.1,
            "mlx_peak_gb": None,
            "l1_snapshot_gb": None,
            "l2_disk_gb": None,
            "finish_reason": "tool_calls",
            "tool_calls": 1,
            "source": "server",
        }
    ]
    assert "| # | cache | src | prompt | cached | computed |" in rendered
    assert "| 2 | warm-l2 | L2 | 13370 | 13136 | 234 | 98.2% |" in rendered

def test_cache_lookup_events_by_request_classifies_l1_and_l2_sources():
    post_events = [{"request_id": 2}, {"request_id": 1}]
    cache_events = [
        {"op": "lookup", "request_id": 1, "result": "prefix_hit", "matched_len": 10},
        {"op": "insert", "result": "admitted"},
        {"op": "lookup", "request_id": 2, "result": "l2_hit", "matched_len": 20},
    ]

    by_request = agentic_trace.cache_lookup_events_by_request(post_events, cache_events)
    first = agentic_trace.post_event_to_server_metric(
        {"request_id": 1, "wall_ms": 1.0, "generated_tokens": 1},
        None,
        by_request[1],
    )
    second = agentic_trace.post_event_to_server_metric(
        {"request_id": 2, "wall_ms": 1.0, "generated_tokens": 1},
        None,
        by_request[2],
    )

    assert first["cache_hit_source"] == "L1"
    assert first["cache_lookup_result"] == "prefix_hit"
    assert second["cache_hit_source"] == "L2"
    assert second["cache_lookup_result"] == "l2_hit"


def test_summarize_cache_events_reports_deeper_hit_gate():
    summary = agentic_trace.summarize_cache_events(
        [
            {
                "op": "lookup",
                "request_id": 1,
                "result": "miss",
                "matched_len": 0,
                "miss_reason": "empty_cache",
            },
            {
                "op": "lookup",
                "request_id": 2,
                "result": "prefix_hit",
                "matched_len": 10,
            },
            {
                "op": "lookup",
                "request_id": 3,
                "result": "prefix_hit",
                "matched_len": 20,
            },
            {
                "op": "lookup",
                "request_id": 4,
                "result": "miss",
                "matched_len": 0,
                "miss_reason": "token_divergence",
                "first_divergence_pos": 7,
            },
            {
                "op": "lookup",
                "request_id": 5,
                "result": "prefix_hit",
                "matched_len": 12,
            },
        ]
    )

    assert summary["miss_reasons"] == {"empty_cache": 1, "token_divergence": 1}
    assert summary["deeper_hit_gate"]["pass"] is True
    assert summary["deeper_hit_gate"]["advancing_hits"] == 3
    assert summary["deeper_hit_gate"]["resets"] == [
        {
            "request_id": 4,
            "miss_reason": "token_divergence",
            "first_divergence_pos": 7,
            "previous_matched_len": 20,
        }
    ]


def test_summarize_deeper_hit_gate_flags_stalled_and_regressed_hits():
    gate = agentic_trace.summarize_deeper_hit_gate(
        [
            {"op": "lookup", "request_id": 1, "result": "prefix_hit", "matched_len": 10},
            {"op": "lookup", "request_id": 2, "result": "prefix_hit", "matched_len": 10},
            {"op": "lookup", "request_id": 3, "result": "prefix_hit", "matched_len": 8},
        ]
    )

    assert gate["pass"] is False
    assert gate["stalled_hits"] == [
        {"request_id": 2, "matched_len": 10, "previous_matched_len": 10}
    ]
    assert gate["regressions"] == [
        {"request_id": 3, "matched_len": 8, "previous_matched_len": 10}
    ]


def test_summarize_prompt_transitions_annotates_cache_reset_cause(tmp_path):
    first = tmp_path / "001.json"
    second = tmp_path / "002.json"
    first.write_text(
        json.dumps(
            {
                "body": {
                    "messages": [
                        {"role": "system", "content": "repo instructions v1"},
                        {"role": "user", "content": "edit the file"},
                    ]
                }
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "body": {
                    "messages": [
                        {"role": "system", "content": "repo instructions v2"},
                        {"role": "user", "content": "edit the file"},
                    ]
                }
            }
        )
    )

    summary = agentic_trace.summarize_prompt_transitions(
        [first, second],
        {
            "deeper_hit_gate": {
                "resets": [
                    {
                        "request_id": 2,
                        "miss_reason": "token_divergence",
                        "first_divergence_pos": 11457,
                    }
                ]
            }
        },
    )

    assert summary["change_kinds"] == {"system_prompt_mutation": 1}
    assert summary["cache_reset_annotations"] == [
        {
            "request_id": 2,
            "miss_reason": "token_divergence",
            "first_divergence_pos": 11457,
            "prompt_change_kind": "system_prompt_mutation",
            "common_message_prefix": 0,
            "first_diff_index": 0,
            "first_diff_roles": "system->system",
        }
    ]


def test_summarize_prompt_transitions_classifies_tool_result_append(tmp_path):
    first = tmp_path / "001.json"
    second = tmp_path / "002.json"
    prefix = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"},
                }
            ],
        },
    ]
    first.write_text(json.dumps({"body": {"messages": prefix}}))
    second.write_text(
        json.dumps(
            {
                "body": {
                    "messages": [
                        *prefix,
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": "README.md\n",
                        },
                    ]
                }
            }
        )
    )

    summary = agentic_trace.summarize_prompt_transitions([first, second])

    transition = summary["transitions"][0]
    assert transition["kind"] == "tool_result_append"
    assert transition["common_message_prefix"] == 3
    assert transition["added_roles"] == ["tool"]


def test_summarize_prompt_transitions_classifies_middle_truncation(tmp_path):
    first = tmp_path / "001.json"
    second = tmp_path / "002.json"
    first.write_text(
        json.dumps(
            {
                "body": {
                    "messages": [
                        {"role": "system", "content": "rules"},
                        {"role": "user", "content": "old task"},
                        {"role": "assistant", "content": "old answer"},
                        {"role": "user", "content": "latest task"},
                    ]
                }
            }
        )
    )
    second.write_text(
        json.dumps(
            {
                "body": {
                    "messages": [
                        {"role": "system", "content": "rules"},
                        {"role": "user", "content": "latest task"},
                    ]
                }
            }
        )
    )

    summary = agentic_trace.summarize_prompt_transitions([first, second])

    transition = summary["transitions"][0]
    assert transition["kind"] == "prompt_truncation"
    assert transition["common_message_prefix"] == 1
    assert transition["common_message_suffix"] == 1


def test_summarize_prompt_transitions_classifies_generation_config_change(tmp_path):
    first = tmp_path / "001.json"
    second = tmp_path / "002.json"
    messages = [{"role": "user", "content": "same prompt"}]
    first.write_text(
        json.dumps({"body": {"messages": messages, "temperature": 0, "top_p": 1}})
    )
    second.write_text(
        json.dumps({"body": {"messages": messages, "temperature": 0.2, "top_p": 1}})
    )

    summary = agentic_trace.summarize_prompt_transitions([first, second])

    transition = summary["transitions"][0]
    assert transition["kind"] == "request_config_change"
    assert transition["option_changes"] == ["temperature"]


def test_cache_lookup_events_without_request_id_do_not_claim_l2_source():
    post_events = [{"request_id": 1}]
    cache_events = [{"op": "lookup", "result": "l2_hit", "matched_len": 20}]

    by_request = agentic_trace.cache_lookup_events_by_request(post_events, cache_events)
    row = agentic_trace.post_event_to_server_metric(
        {
            "request_id": 1,
            "wall_ms": 1.0,
            "generated_tokens": 1,
            "prompt_tokens": 100,
            "cache_hit_tokens": 80,
        },
        None,
        by_request[1],
    )
    rows = agentic_trace._normalized_post_rows(
        {
            "metadata": {"label": "ambiguous", "backend": "dflash"},
            "client": "replay",
            "posts": [{"idx": 1, "landmarks": {}, "server_metric": row}],
        }
    )

    assert row["cache_hit_source"] == "unknown"
    assert rows[0]["cache"] == "warm"
    assert rows[0]["cache_hit_source"] == "unknown"


def test_write_timeline_artifacts_joins_request_proxy_and_opencode(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "requests").mkdir(parents=True)
    (run_dir / "opencode").mkdir()
    (run_dir / "requests" / "001.json").write_text(
        json.dumps(
            {
                "body_bytes": 1234,
                "body_sha256": hashlib.sha256(b"body-one").hexdigest(),
                "body": {
                    "messages": [
                        {"role": "system", "content": "rules"},
                        {"role": "user", "content": "build"},
                    ],
                    "tools": [{"function": {"name": "bash"}}],
                },
            }
        )
    )
    (run_dir / "requests" / "002.json").write_text(
        json.dumps(
            {
                "body_bytes": 2345,
                "body_sha256": hashlib.sha256(b"body-two").hexdigest(),
                "source_body_sha256": hashlib.sha256(b"source-two").hexdigest(),
                "body": {
                    "messages": [
                        {"role": "system", "content": "rules"},
                        {"role": "user", "content": "build"},
                        {"role": "assistant", "content": "", "reasoning_content": "think"},
                        {"role": "tool", "content": "tool output"},
                    ],
                    "tools": [{"function": {"name": "bash"}}],
                },
            }
        )
    )
    (run_dir / "proxy.log").write_text(
        "\n".join(
            [
                "2026-05-09 10:00:00 req#1 /v1/chat/completions stream=True body_bytes=1234 max_tokens=1 model=m",
                "2026-05-09 10:00:00 req#1 done t_total_ms=1000.0",
                "2026-05-09 10:00:03 req#2 /v1/chat/completions stream=True body_bytes=2345 max_tokens=1 model=m",
                "2026-05-09 10:00:03 req#2 done t_total_ms=500.0",
            ]
        )
        + "\n"
    )
    (run_dir / "opencode" / "stdout.jsonl").write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"type": "step_start", "timestamp": 1000, "part": {"messageID": "m1"}},
                {
                    "type": "tool_use",
                    "timestamp": 1500,
                    "part": {
                        "messageID": "m1",
                        "tool": "bash",
                        "state": {"time": {"start": 1500, "end": 1700}},
                    },
                },
                {
                    "type": "step_finish",
                    "timestamp": 2000,
                    "part": {
                        "messageID": "m1",
                        "tokens": {"input": 10, "output": 2, "cache": {"read": 0, "write": 0}},
                    },
                },
                {"type": "step_start", "timestamp": 3500, "part": {"messageID": "m2"}},
                {"type": "step_finish", "timestamp": 4000, "part": {"messageID": "m2"}},
            ]
        )
        + "\n"
    )
    summary = {
        "metadata": {
            "label": "timeline",
            "backend": "dflash",
            "pause_before_request": {"2": 0.0},
        },
        "client": "opencode",
        "posts": [
            {
                "idx": 1,
                "landmarks": {"first_byte_ms": 1.0, "first_tool_call_sent_ms": 20.0},
                "server_metric": {
                    "prompt_tokens": 10,
                    "cache_hit_tokens": 0,
                    "prefill_tokens_computed": 10,
                    "decode_tok_s": 25.0,
                    "tokens": 2,
                    "accept": 0.5,
                    "cache_lookup_result": "miss",
                },
            },
            {
                "idx": 2,
                "landmarks": {"first_byte_ms": 2.0, "first_tool_call_sent_ms": 30.0},
                "server_metric": {
                    "prompt_tokens": 12,
                    "cache_hit_tokens": 10,
                    "prefill_tokens_computed": 2,
                    "decode_tok_s": 30.0,
                    "tokens": 3,
                    "accept": 0.75,
                    "cache_lookup_result": "prefix_hit",
                },
            },
        ],
    }

    agentic_trace._write_timeline_artifacts(run_dir, summary)

    rows = [
        json.loads(line)
        for line in (run_dir / "timeline.jsonl").read_text().splitlines()
    ]
    assert rows[0]["body_bytes"] == 1234
    assert rows[0]["body_sha256"] == hashlib.sha256(b"body-one").hexdigest()
    assert rows[0]["message_count"] == 2
    assert rows[0]["opencode_tool_exec_ms"] == 200
    assert rows[1]["proxy_http_gap_before_s"] == 2.0
    assert rows[1]["proxy_http_gap_excluding_replay_pause_s"] == 2.0
    assert rows[1]["replay_pause_before_s"] == 0.0
    assert rows[1]["opencode_step_gap_before_s"] == 1.5
    assert rows[1]["assistant_reasoning_chars"] == 5
    assert rows[1]["tool_output_chars"] == len("tool output")
    assert rows[1]["source_body_sha256"] == hashlib.sha256(b"source-two").hexdigest()
    assert (run_dir / "rows_timeline.md").read_text().startswith("# Agentic timeline")


def test_timeline_subtracts_replay_pause_from_http_gap(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "requests").mkdir(parents=True)
    for idx in (1, 2):
        (run_dir / "requests" / f"{idx:03d}.json").write_text(
            json.dumps({"body": {"messages": [{"role": "user", "content": str(idx)}]}})
        )
    (run_dir / "proxy.log").write_text(
        "\n".join(
            [
                "2026-05-09 10:00:00 req#1 /v1/chat/completions stream=True body_bytes=100 max_tokens=1 model=m",
                "2026-05-09 10:00:00 req#1 done t_total_ms=1000.0",
                "2026-05-09 10:01:04 req#2 /v1/chat/completions stream=True body_bytes=100 max_tokens=1 model=m",
                "2026-05-09 10:01:04 req#2 done t_total_ms=500.0",
            ]
        )
        + "\n"
    )
    summary = {
        "metadata": {
            "label": "timeline",
            "backend": "dflash",
            "pause_before_request": {"2": 60.0},
        },
        "client": "replay",
        "posts": [
            {"idx": 1, "landmarks": {}, "server_metric": {"prompt_tokens": 1, "tokens": 1}},
            {"idx": 2, "landmarks": {}, "server_metric": {"prompt_tokens": 1, "tokens": 1}},
        ],
    }

    agentic_trace._write_timeline_artifacts(run_dir, summary)

    rows = [
        json.loads(line)
        for line in (run_dir / "timeline.jsonl").read_text().splitlines()
    ]
    assert rows[1]["proxy_http_gap_before_s"] == 63.0
    assert rows[1]["replay_pause_before_s"] == 60.0
    assert rows[1]["proxy_http_gap_excluding_replay_pause_s"] == 3.0


def test_timeline_rejects_malformed_request_json(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "requests").mkdir(parents=True)
    (run_dir / "requests" / "001.json").write_text("{bad json\n")
    summary = {
        "metadata": {"label": "timeline", "backend": "dflash"},
        "client": "replay",
        "posts": [{"idx": 1, "landmarks": {}, "server_metric": {"prompt_tokens": 1}}],
    }

    with pytest.raises(RuntimeError, match="malformed request JSON"):
        agentic_trace._write_timeline_artifacts(run_dir, summary)


def test_timeline_rejects_malformed_opencode_stdout_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "requests").mkdir(parents=True)
    (run_dir / "opencode").mkdir()
    (run_dir / "requests" / "001.json").write_text(
        json.dumps({"body": {"messages": [{"role": "user", "content": "hi"}]}})
    )
    (run_dir / "opencode" / "stdout.jsonl").write_text("{bad json\n")
    summary = {
        "metadata": {"label": "timeline", "backend": "dflash"},
        "client": "opencode",
        "posts": [{"idx": 1, "landmarks": {}, "server_metric": {"prompt_tokens": 1}}],
    }

    with pytest.raises(RuntimeError, match="malformed OpenCode stdout JSONL"):
        agentic_trace._write_timeline_artifacts(run_dir, summary)

def test_apply_dflash_stderr_totals_records_prefill_saved():
    totals = {"prefill_tokens_saved_cumulative": None}

    _apply_dflash_stderr_totals(
        totals,
        "\n".join(
            [
                "[dflash] prefix-cache-stats lookups=1 hits=1 prefill_tokens_saved=128",
                "[dflash] prefix-cache-stats lookups=2 hits=2 prefill_tokens_saved=512",
            ]
        ),
    )

    assert totals["prefill_tokens_saved_cumulative"] == 512


def test_summarize_cycles_ignores_memory_boundary_events():
    summary = agentic_trace.summarize_cycles(
        [
            {"memory_phase": "request_start", "phys_footprint_bytes": 10},
            {"commit_count": 3, "acceptance_len": 2, "block_len": 4, "verify_us": 100.0},
            {"memory_phase": "request_end", "phys_footprint_bytes": 12},
        ]
    )

    assert summary is not None
    assert summary["n_cycles"] == 1
    assert summary["total_commits"] == 3


def test_summarize_cycles_returns_none_for_memory_only_events():
    assert agentic_trace.summarize_cycles(
        [{"memory_phase": "request_start", "phys_footprint_bytes": 10}]
    ) is None

def test_replay_outputs_rejects_empty_success_streams():
    posts = [
        {
            "landmarks": {
                "n_chunks": 0,
                "end_t_ms": 12.0,
            },
            "server_metric": {"tokens": 0},
        }
    ]

    with pytest.raises(SystemExit, match="without model output"):
        _ensure_replay_outputs(posts)

    _ensure_replay_outputs([{"landmarks": {}, "server_metric": {"tokens": 1}}])

    _ensure_replay_outputs(
        [
            {
                "landmarks": {"usage": {"completion_tokens": 1}},
                "server_metric": None,
            }
        ]
    )

def _comparison_summary(label: str, backend: str, post_count: int, decode_tokens: int) -> dict:
    posts = [
        {"landmarks": {"first_tool_call_sent_ms": 100.0 + idx}}
        for idx in range(post_count)
    ]
    return {
        "metadata": {"label": label, "backend": backend},
        "posts": posts,
        "totals": {
            "wall_s": 10.0,
            "total_decode_tokens": decode_tokens,
            "total_decode_wall_s": decode_tokens / 100.0,
            "decode_tps_avg": 100.0,
            "total_prompt_tokens": 1000,
            "total_tool_calls": post_count,
            "total_cache_hit_tokens": 128 if backend == "dflash" else 0,
            "prefill_tokens_saved_cumulative": 512 if backend == "dflash" else None,
            "weighted_acceptance": 0.75 if backend == "dflash" else None,
            "first_tool_call_ms_sum": sum(100.0 + idx for idx in range(post_count)),
        },
    }

def test_render_peer_comparison_keeps_per_post_table_when_trajectories_align():
    rendered = _render_peer_comparison(
        _comparison_summary("dflash", "dflash", 4, 100),
        _comparison_summary("mlxlm", "mlxlm", 4, 104),
    )

    assert "Trajectory-robust aggregate metrics" in rendered
    assert "Trajectories aligned" in rendered
    assert "observed_response_ms_per_output_token" in rendered
    assert "| # | this | peer | gap_ms |" in rendered

def test_render_peer_comparison_skips_per_post_table_when_trajectories_diverge():
    rendered = _render_peer_comparison(
        _comparison_summary("dflash", "dflash", 9, 220),
        _comparison_summary("mlxlm", "mlxlm", 4, 100),
    )

    assert "Trajectory-robust aggregate metrics" in rendered
    assert "TRAJECTORY DIVERGED" in rendered
    assert "Skipped — trajectories diverged" in rendered
    assert "| # | this | peer | gap_ms |" not in rendered
    assert "tool_call_latency_gap" not in rendered
    assert "split_sdpa" not in rendered
    assert "fused matmul" not in rendered

def test_dflash_l2_options_require_l2_enablement():
    with pytest.raises(SystemExit, match="require --prefix-cache-l2"):
        agentic_trace.main(
            [
                "--backend",
                "dflash",
                "--target",
                "m",
                "--draft",
                "d",
                "--prefix-cache-l2-dir",
                "cache",
            ]
        )

def test_replay_l2_options_require_l2_enablement(tmp_path):
    source = tmp_path / "source-trace"
    (source / "requests").mkdir(parents=True)

    with pytest.raises(SystemExit, match="require --prefix-cache-l2"):
        agentic_trace.replay_main(
            [
                "--source-trace",
                str(source),
                "--backend",
                "dflash",
                "--target",
                "m",
                "--draft",
                "d",
                "--prefix-cache-l2-max-bytes",
                "1",
            ]
        )

@pytest.mark.parametrize(
    "flag_args",
    [
        ["--verify-len-cap", "8"],
        ["--verify-mode", "off"],
        ["--wired-limit", "48GB"],
        ["--cache-limit", "4GB"],
        ["--draft-sink-size", "64"],
        ["--draft-window-size", "2048"],
        ["--clear-cache-boundaries"],
        ["--no-clear-cache-boundaries"],
    ],
)
def test_replay_dflash_runtime_overrides_reject_mlxlm_backend(tmp_path, flag_args):
    source = tmp_path / "source-trace"
    (source / "requests").mkdir(parents=True)

    with pytest.raises(SystemExit, match="require --backend dflash"):
        agentic_trace.replay_main(
            [
                "--source-trace",
                str(source),
                "--backend",
                "mlxlm",
                "--target",
                "m",
                *flag_args,
            ]
        )

def test_parse_pause_before_request():
    assert agentic_trace._parse_pause_before_request(["3:1.5", "7:0"]) == {
        3: 1.5,
        7: 0.0,
    }

    with pytest.raises(SystemExit, match="IDX:SECONDS"):
        agentic_trace._parse_pause_before_request(["bad"])
    with pytest.raises(SystemExit, match="IDX must be >= 1"):
        agentic_trace._parse_pause_before_request(["0:1"])
    with pytest.raises(SystemExit, match="SECONDS must be >= 0"):
        agentic_trace._parse_pause_before_request(["1:-1"])

def test_run_replay_requests_can_pause_before_specific_request(tmp_path, monkeypatch):
    source = tmp_path / "source-trace"
    (source / "requests").mkdir(parents=True)
    for idx in (1, 2):
        (source / "requests" / f"{idx:03d}.json").write_text(
            json.dumps({"body": {"messages": [], "stream": True}})
        )
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    calls = []
    monkeypatch.setattr(agentic_trace.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))

    def fake_post_replay_request(*, idx, request_path, **_kwargs):
        calls.append(("post", idx, request_path.name))

    monkeypatch.setattr(agentic_trace, "_post_replay_request", fake_post_replay_request)

    count, wall_s = agentic_trace._run_replay_requests(
        source_trace=source,
        run_dir=run_dir,
        upstream_url="http://127.0.0.1:1",
        target="target",
        request_limit=0,
        request_timeout_s=1.0,
        pause_before_request={2: 3.5},
    )

    assert count == 2
    assert wall_s >= 0
    assert calls == [("post", 1, "001.json"), ("sleep", 3.5), ("post", 2, "002.json")]

def _start_json_upstream(response_body: bytes = b'{"ok":true}'):
    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_, **__):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length else b""
            self.server.bodies.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server.bodies = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

def test_proxy_capture_hashes_raw_body_bytes(tmp_path):
    upstream = _start_json_upstream()
    proxy = None
    conn = None
    try:
        out_dir = tmp_path / "proxy"
        out_dir.mkdir()
        handler = agentic_proxy._make_handler(
            out_dir,
            f"http://127.0.0.1:{upstream.server_address[1]}",
        )
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()

        raw_body = (
            b'{ "stream" : false, "model":"captured",'
            b' "messages":[{"role":"user","content":"hi"}] }'
        )
        conn = http.client.HTTPConnection(
            "127.0.0.1",
            proxy.server_address[1],
            timeout=5,
        )
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=raw_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer secret",
            },
        )
        response = conn.getresponse()
        response.read()

        assert response.status == 200
        assert upstream.bodies == [raw_body]

        captured = json.loads((out_dir / "requests" / "001.json").read_text())
        assert captured["body_bytes"] == len(raw_body)
        assert captured["body_sha256"] == hashlib.sha256(raw_body).hexdigest()
        assert captured["body"]["model"] == "captured"
        assert "Authorization" not in captured["headers"]
    finally:
        if conn is not None:
            conn.close()
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        upstream.shutdown()
        upstream.server_close()

def test_post_replay_request_records_replay_and_source_body_hashes(tmp_path):
    upstream = _start_json_upstream()
    try:
        source = tmp_path / "source"
        (source / "requests").mkdir(parents=True)
        source_raw = (
            b'{"model":"captured","messages":[{"role":"user","content":"hi"}],'
            b'"stream":false}'
        )
        source_request = {
            "path": "/v1/chat/completions",
            "headers": {"Content-Type": "application/json"},
            "body": {
                "model": "captured",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
            "body_bytes": len(source_raw),
            "body_sha256": hashlib.sha256(source_raw).hexdigest(),
        }
        request_path = source / "requests" / "001.json"
        request_path.write_text(json.dumps(source_request))
        run_dir = tmp_path / "run"

        agentic_trace._post_replay_request(
            idx=1,
            request_path=request_path,
            out_dir=run_dir,
            upstream_url=f"http://127.0.0.1:{upstream.server_address[1]}",
            target="fresh-target",
            timeout_s=5.0,
        )

        expected_replay_body = agentic_trace._replay_body_bytes(
            source_request,
            "fresh-target",
        )
        replay = json.loads((run_dir / "requests" / "001.json").read_text())
        assert upstream.bodies == [expected_replay_body]
        assert replay["body"]["model"] == "fresh-target"
        assert replay["body_bytes"] == len(expected_replay_body)
        assert replay["body_sha256"] == hashlib.sha256(expected_replay_body).hexdigest()
        assert replay["source_body_bytes"] == len(source_raw)
        assert replay["source_body_sha256"] == hashlib.sha256(source_raw).hexdigest()
    finally:
        upstream.shutdown()
        upstream.server_close()

def test_read_dflash_events_rejects_malformed_jsonl(tmp_path):
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    (events_dir / "post_events.jsonl").write_text("{bad json\n")

    with pytest.raises(RuntimeError, match="malformed JSONL"):
        agentic_trace.read_dflash_events(events_dir)

def test_ensure_port_available_rejects_bound_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]

        with pytest.raises(SystemExit, match="already in use"):
            agentic_trace._ensure_port_available("127.0.0.1", port, "server")

def test_verify_ready_model_rejects_mismatch(monkeypatch):
    monkeypatch.setattr(agentic_trace, "_health_model_ids", lambda _url: {"other-model"})

    with pytest.raises(SystemExit, match="server identity mismatch"):
        agentic_trace._verify_ready_model("http://127.0.0.1:1/v1/models", "expected-model")

def test_replay_aborts_when_spawned_server_exits(tmp_path, monkeypatch):
    source = tmp_path / "source-trace"
    (source / "requests").mkdir(parents=True)
    (source / "requests" / "001.json").write_text(
        json.dumps(
            {
                "idx": 1,
                "path": "/v1/chat/completions",
                "body": {"model": "captured", "messages": [], "stream": True},
            }
        )
    )
    monkeypatch.setattr(agentic_trace, "_ensure_port_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_terminate", lambda *_args, **_kwargs: None)

    class ExitedProc:
        pid = 1

        def poll(self):
            return 7

        def wait(self, timeout=None):
            return 7

    def fake_spawn(_cmd, stdout_path, stderr_path, env=None):
        stdout_path.write_text("")
        stderr_path.write_text("bind failed\n")
        return ExitedProc()

    monkeypatch.setattr(agentic_trace, "_spawn", fake_spawn)

    with pytest.raises(SystemExit, match="server exited before replay"):
        agentic_trace.replay_main(
            [
                "--source-trace",
                str(source),
                "--backend",
                "mlxlm",
                "--target",
                "fresh-target",
                "--out-root",
                str(tmp_path),
                "--label",
                "server-exited",
            ]
        )

def _server_args(**overrides):
    defaults = {
        "backend": "dflash",
        "target": "target-model",
        "draft": "draft-model",
        "dflash_port": 8123,
        "mlxlm_port": 8124,
        "draft_quant": None,
        "wired_limit": None,
        "cache_limit": None,
        "profile": None,
        "prefill_step_size": None,
        "draft_sink_size": None,
        "draft_window_size": None,
        "fastpath_max_tokens": None,
        "verify_len_cap": None,
        "max_snapshot_tokens": None,
        "verify_mode": None,
        "clear_cache_boundaries": None,
        "target_fa_window": 0,
        "chat_template_args": '{"enable_thinking":true}',
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)

def test_build_server_cmd_disables_dflash_fastpath_by_default():
    cmd, port, url = _build_server_cmd(_server_args())

    assert port == 8123
    assert url == "http://127.0.0.1:8123"
    assert "--draft-quant" not in cmd
    assert "--profile" not in cmd
    assert "--prefill-step-size" not in cmd
    rendered = " ".join(cmd)
    assert "--fastpath-max-tokens 0" in rendered
    assert "--max-snapshot-tokens" not in cmd
    assert "--target-fa-window" not in cmd

def test_build_server_cmd_forwards_gemma_runtime_overrides():
    cmd, _, _ = _build_server_cmd(
        _server_args(
            draft_quant="w4",
            wired_limit="48GB",
            cache_limit="4GB",
            profile="long-session",
            prefill_step_size=1024,
            draft_sink_size=64,
            draft_window_size=2048,
            fastpath_max_tokens=0,
            verify_len_cap=8,
            max_snapshot_tokens=32000,
            verify_mode="off",
            clear_cache_boundaries=True,
            target_fa_window=2048,
            chat_template_args='{"enable_thinking":false}',
        )
    )

    rendered = " ".join(cmd)
    assert "--draft-quant w4" in rendered
    assert "--wired-limit 48GB" in rendered
    assert "--cache-limit 4GB" in rendered
    assert "--profile long-session" in rendered
    assert "--prefill-step-size 1024" in rendered
    assert "--draft-sink-size 64" in rendered
    assert "--draft-window-size 2048" in rendered
    assert "--fastpath-max-tokens 0" in rendered
    assert "--verify-len-cap 8" in rendered
    assert "--max-snapshot-tokens 32000" in rendered
    assert "--verify-mode off" in rendered
    assert "--clear-cache-boundaries" in cmd
    assert "--target-fa-window 2048" in rendered
    idx = cmd.index("--chat-template-args")
    assert cmd[idx + 1] == '{"enable_thinking":false}'

@pytest.mark.parametrize(
    "flag_args",
    [
        ["--prefix-cache"],
        ["--no-prefix-cache"],
        ["--prefix-cache-l2"],
        ["--prefix-cache-l2-max-bytes", "1"],
        ["--prefix-cache-l2-dir", "cache"],
        ["--diagnostics", "off"],
        ["--draft-quant", "w4"],
        ["--wired-limit", "48GB"],
        ["--cache-limit", "4GB"],
        ["--profile", "long-session"],
        ["--prefill-step-size", "1024"],
        ["--draft-sink-size", "64"],
        ["--draft-window-size", "2048"],
        ["--fastpath-max-tokens", "0"],
        ["--verify-len-cap", "8"],
        ["--max-snapshot-tokens", "32000"],
        ["--verify-mode", "off"],
        ["--clear-cache-boundaries"],
        ["--no-clear-cache-boundaries"],
        ["--target-fa-window", "0"],
    ],
)
def test_dflash_runtime_overrides_reject_mlxlm_backend(flag_args):
    with pytest.raises(SystemExit, match="requires? --backend dflash"):
        agentic_trace.main(["--backend", "mlxlm", "--target", "m", *flag_args])

def test_agentic_trace_metadata_records_dflash_runtime_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "opencode.jsonc"
    config_path.write_text('{"provider": {}}\n')
    monkeypatch.setattr(agentic_trace, "OPENCODE_CONFIG", config_path)
    monkeypatch.setattr(agentic_trace, "_wait_health", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(agentic_trace, "_ensure_port_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_verify_ready_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_terminate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_git", lambda _args: "test-git")
    monkeypatch.setattr(agentic_trace.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(artifacts, "_git", lambda _args: "test-git")
    monkeypatch.setattr(artifacts, "_git_dirty", lambda: False)

    class DummyProc:
        pid = 1

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_spawn(_cmd, stdout_path, stderr_path, env=None):
        stdout_path.write_text("")
        stderr_path.write_text("")
        return DummyProc()

    class DummyClientProc:
        def __init__(self, *_args, **kwargs):
            for key in ("stdout", "stderr"):
                stream = kwargs.get(key)
                if hasattr(stream, "close"):
                    stream.close()

        def wait(self, timeout=None):
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(agentic_trace, "_spawn", fake_spawn)
    monkeypatch.setattr(agentic_trace.subprocess, "Popen", DummyClientProc)
    l2_dir = tmp_path / "custom-l2"

    rc = agentic_trace.main(
        [
            "--backend",
            "dflash",
            "--target",
            "target-model",
            "--draft",
            "draft-model",
            "--draft-quant",
            "w4",
            "--profile",
            "long-session",
            "--prefill-step-size",
            "1024",
            "--draft-sink-size",
            "64",
            "--draft-window-size",
            "2048",
            "--fastpath-max-tokens",
            "0",
            "--verify-len-cap",
            "8",
            "--max-snapshot-tokens",
            "32000",
            "--verify-mode",
            "off",
            "--clear-cache-boundaries",
            "--prefix-cache-l2",
            "--prefix-cache-l2-dir",
            str(l2_dir),
            "--diagnostics",
            "full",
            "--out-root",
            str(tmp_path),
            "--label",
            "metadata",
            "--client-timeout-s",
            "1",
        ]
    )

    assert rc == 0
    run_dir = next(tmp_path.glob("*-opencode-metadata"))
    metadata = json.loads((run_dir / "metadata.json").read_text())
    overrides = metadata["dflash_runtime_overrides"]
    assert overrides["draft_quant"] == "w4"
    assert overrides["profile"] == "long-session"
    assert overrides["prefill_step_size"] == 1024
    assert overrides["draft_sink_size"] == 64
    assert overrides["draft_window_size"] == 2048
    assert overrides["fastpath_max_tokens"] == 0
    assert overrides["verify_len_cap"] == 8
    assert overrides["max_snapshot_tokens"] == 32000
    assert overrides["verify_mode"] == "off"
    assert overrides["clear_cache_boundaries"] is True
    assert overrides["prefix_cache"] is True
    assert overrides["prefix_cache_l2"] is True
    assert overrides["prefix_cache_l2_dir"] == str(l2_dir)
    assert overrides["diagnostics"] == "full"
    server_cmd = metadata["server_cmd"]
    rendered = " ".join(server_cmd)
    assert "--draft-quant w4" in rendered
    assert "--profile long-session" in rendered
    assert "--prefill-step-size 1024" in rendered
    assert "--draft-sink-size 64" in rendered
    assert "--draft-window-size 2048" in rendered
    assert "--fastpath-max-tokens 0" in rendered
    assert "--verify-len-cap 8" in rendered
    assert "--max-snapshot-tokens 32000" in rendered
    assert "--verify-mode off" in rendered
    assert "--clear-cache-boundaries" in server_cmd
    assert "--prefix-cache-l2" in server_cmd
    assert str(l2_dir) in server_cmd
    assert (run_dir / "server" / "cmd.txt").read_text().strip()

def test_replay_body_bytes_rewrites_model_for_target():
    payload = {
        "body": {
            "model": "captured-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
    }

    body = json.loads(_replay_body_bytes(payload, "fresh-target").decode())

    assert body["model"] == "fresh-target"
    assert body["messages"][0]["content"] == "hi"
    assert agentic_trace._body_sha256(b"abc") == hashlib.sha256(b"abc").hexdigest()

def test_system_sample_parsers_extract_memory_and_server_stats():
    vm = agentic_trace._parse_vm_stat(
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free: 100.\n"
        "Pages active: 20.\n"
        "Pages wired down: 30.\n"
        "Pages used by compressor: 4.\n"
    )

    assert vm["page_size"] == 16384
    assert vm["pages"]["pages_free"] == 100
    assert vm["free_gb"] == pytest.approx(100 * 16384 / 1e9)
    assert vm["wired_gb"] == pytest.approx(30 * 16384 / 1e9)
    assert vm["compressor_gb"] == pytest.approx(4 * 16384 / 1e9)
    assert agentic_trace._parse_ps_resource("1024 2048 12.5 3.0") == {
        "rss_gb": pytest.approx(1024 * 1024 / 1e9),
        "vsz_gb": pytest.approx(2048 * 1024 / 1e9),
        "cpu_pct": 12.5,
        "mem_pct": 3.0,
    }

def test_system_sampler_writes_jsonl_rows(tmp_path, monkeypatch):
    calls = []

    def fake_run_capture(cmd, *, timeout_s=2.0):
        calls.append(cmd)
        if cmd[0] == "ps":
            return {"cmd": cmd, "returncode": 0, "stdout": "1024 2048 12.5 3.0", "stderr": ""}
        if cmd[0] == "vm_stat":
            return {
                "cmd": cmd,
                "returncode": 0,
                "stdout": (
                    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                    "Pages free: 100.\n"
                    "Pages wired down: 30.\n"
                ),
                "stderr": "",
            }
        return {
            "cmd": cmd,
            "returncode": 0,
            "stdout": (
                "Note: No thermal warning level has been recorded\n"
                "Note: No performance warning level has been recorded\n"
                "Note: No CPU power status has been recorded\n"
            ),
            "stderr": "",
        }

    monkeypatch.setattr(agentic_trace, "_run_capture", fake_run_capture)
    sampler = agentic_trace._SystemSampler(run_dir=tmp_path, server_pid=123, interval_s=100.0)
    sampler._write_sample()

    rows = [json.loads(line) for line in (tmp_path / "system_samples.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["server_pid"] == 123
    assert rows[0]["server_ps"]["rss_gb"] == pytest.approx(1024 * 1024 / 1e9)
    assert rows[0]["vm_stat"]["wired_gb"] == pytest.approx(30 * 16384 / 1e9)
    assert rows[0]["thermal"]["thermal_warning_recorded"] is False
    assert [cmd[0] for cmd in calls] == ["ps", "vm_stat", "pmset"]

def test_agentic_replay_metadata_records_source_trace(tmp_path, monkeypatch):
    source = tmp_path / "source-trace"
    (source / "requests").mkdir(parents=True)
    (source / "requests" / "001.json").write_text(
        json.dumps(
            {
                "idx": 1,
                "path": "/v1/chat/completions",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "model": "captured",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            }
        )
    )
    monkeypatch.setattr(agentic_trace, "_wait_health", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(agentic_trace, "_ensure_port_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_verify_ready_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_terminate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_git", lambda _args: "test-git")
    monkeypatch.setattr(agentic_trace.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(artifacts, "_git", lambda _args: "test-git")
    monkeypatch.setattr(artifacts, "_git_dirty", lambda: False)
    sampler_calls = []

    class DummyProc:
        pid = 1

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_spawn(_cmd, stdout_path, stderr_path, env=None):
        stdout_path.write_text("")
        stderr_path.write_text("")
        return DummyProc()

    def fake_replay_requests(*, run_dir, target, **_kwargs):
        req = {
            "idx": 1,
            "method": "POST",
            "path": "/v1/chat/completions",
            "stream": True,
            "headers": {"Content-Type": "application/json"},
            "body": {
                "model": target,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            "body_bytes": 80,
        }
        (run_dir / "requests").mkdir(exist_ok=True)
        (run_dir / "sse").mkdir(exist_ok=True)
        (run_dir / "requests" / "001.json").write_text(json.dumps(req))
        (run_dir / "sse" / "001.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"type": "first_byte", "t_ms": 10.0}),
                    json.dumps(
                        {
                            "type": "event",
                            "t_ms": 20.0,
                            "payload": {
                                "data": {
                                    "choices": [{"delta": {"content": "ok"}}],
                                    "usage": {
                                        "prompt_tokens": 4,
                                        "completion_tokens": 1,
                                    },
                                }
                            },
                        }
                    ),
                    json.dumps({"type": "end", "t_ms": 30.0}),
                ]
            )
            + "\n"
        )
        return 1, 0.03

    monkeypatch.setattr(agentic_trace, "_spawn", fake_spawn)
    monkeypatch.setattr(agentic_trace, "_run_replay_requests", fake_replay_requests)

    class DummySampler:
        def stop(self):
            sampler_calls.append(("stop",))

    def fake_start_system_sampler(*, run_dir, server_pid, interval_s):
        sampler_calls.append(("start", run_dir, server_pid, interval_s))
        return DummySampler()

    monkeypatch.setattr(agentic_trace, "_start_system_sampler", fake_start_system_sampler)

    rc = agentic_trace.replay_main(
        [
            "--source-trace",
            str(source),
            "--backend",
            "mlxlm",
            "--target",
            "fresh-target",
            "--out-root",
            str(tmp_path),
            "--label",
            "replay-smoke",
            "--system-sample-interval-s",
            "7.5",
        ]
    )

    assert rc == 0
    run_dir = next(tmp_path.glob("*-replay-replay-smoke"))
    metadata = json.loads((run_dir / "metadata.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "rows.jsonl").read_text().splitlines()]
    assert metadata["source_trace"] == str(source)
    assert metadata["client"] == "replay"
    assert metadata["system_sample_interval_s"] == 7.5
    assert sampler_calls[0][0] == "start"
    assert sampler_calls[0][2:] == (1, 7.5)
    assert sampler_calls[-1] == ("stop",)
    assert summary["post_count"] == 1
    assert summary["posts"][0]["request"]["model"] == "fresh-target"
    assert rows[0]["backend"] == "mlxlm"
    assert rows[0]["prompt_tok"] == 4
    assert rows[0]["out_tok"] == 1
    assert (run_dir / "rows.md").read_text().startswith("# Agentic rows")


def test_replay_forwards_no_prefix_cache_l2_for_long_session(tmp_path, monkeypatch):
    source = tmp_path / "source-trace"
    (source / "requests").mkdir(parents=True)
    (source / "requests" / "001.json").write_text(
        json.dumps(
            {
                "idx": 1,
                "path": "/v1/chat/completions",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "model": "captured",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            }
        )
    )
    monkeypatch.setattr(agentic_trace, "_wait_health", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(agentic_trace, "_ensure_port_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_verify_ready_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_terminate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_git", lambda _args: "test-git")
    monkeypatch.setattr(agentic_trace.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(artifacts, "_git", lambda _args: "test-git")
    monkeypatch.setattr(artifacts, "_git_dirty", lambda: False)

    class DummyProc:
        pid = 1

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_spawn(_cmd, stdout_path, stderr_path, env=None):
        stdout_path.write_text("")
        stderr_path.write_text("")
        return DummyProc()

    def fake_replay_requests(*, run_dir, target, **_kwargs):
        req = {
            "idx": 1,
            "method": "POST",
            "path": "/v1/chat/completions",
            "stream": True,
            "headers": {"Content-Type": "application/json"},
            "body": {
                "model": target,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            "body_bytes": 80,
        }
        (run_dir / "requests").mkdir(exist_ok=True)
        (run_dir / "sse").mkdir(exist_ok=True)
        (run_dir / "requests" / "001.json").write_text(json.dumps(req))
        (run_dir / "sse" / "001.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"type": "first_byte", "t_ms": 10.0}),
                    json.dumps(
                        {
                            "type": "event",
                            "t_ms": 20.0,
                            "payload": {
                                "data": {
                                    "choices": [{"delta": {"content": "ok"}}],
                                    "usage": {
                                        "prompt_tokens": 4,
                                        "completion_tokens": 1,
                                    },
                                }
                            },
                        }
                    ),
                    json.dumps({"type": "end", "t_ms": 30.0}),
                ]
            )
            + "\n"
        )
        return 1, 0.03

    monkeypatch.setattr(agentic_trace, "_spawn", fake_spawn)
    monkeypatch.setattr(agentic_trace, "_run_replay_requests", fake_replay_requests)

    rc = agentic_trace.replay_main(
        [
            "--source-trace",
            str(source),
            "--backend",
            "dflash",
            "--target",
            "fresh-target",
            "--draft",
            "fresh-draft",
            "--profile",
            "long-session",
            "--no-prefix-cache-l2",
            "--out-root",
            str(tmp_path),
            "--label",
            "no-l2",
        ]
    )

    assert rc == 0
    run_dir = next(tmp_path.glob("*-replay-no-l2"))
    server_cmd = json.loads((run_dir / "metadata.json").read_text())["server_cmd"]
    rendered = " ".join(server_cmd)
    assert "--profile long-session" in rendered
    assert "--no-prefix-cache-l2" in server_cmd
    assert "--prefix-cache-l2" not in server_cmd

def test_replay_terminates_server_when_sampler_stop_fails(tmp_path, monkeypatch):
    source = tmp_path / "source-trace"
    (source / "requests").mkdir(parents=True)
    (source / "requests" / "001.json").write_text(
        json.dumps(
            {
                "idx": 1,
                "path": "/v1/chat/completions",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "model": "captured",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            }
        )
    )
    terminated = []
    monkeypatch.setattr(agentic_trace, "_wait_health", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(agentic_trace, "_ensure_port_available", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_verify_ready_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(agentic_trace, "_terminate", lambda proc, label: terminated.append((proc.pid, label)))
    monkeypatch.setattr(agentic_trace, "_git", lambda _args: "test-git")
    monkeypatch.setattr(agentic_trace.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(artifacts, "_git", lambda _args: "test-git")
    monkeypatch.setattr(artifacts, "_git_dirty", lambda: False)

    class DummyProc:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    class BrokenSampler:
        def stop(self):
            raise RuntimeError("sample failed")

    def fake_spawn(_cmd, stdout_path, stderr_path, env=None):
        stdout_path.write_text("")
        stderr_path.write_text("")
        return DummyProc()

    def fake_replay_requests(*, run_dir, target, **_kwargs):
        req = {
            "idx": 1,
            "method": "POST",
            "path": "/v1/chat/completions",
            "stream": True,
            "headers": {"Content-Type": "application/json"},
            "body": {
                "model": target,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            "body_bytes": 80,
        }
        (run_dir / "requests").mkdir(exist_ok=True)
        (run_dir / "sse").mkdir(exist_ok=True)
        (run_dir / "requests" / "001.json").write_text(json.dumps(req))
        (run_dir / "sse" / "001.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"type": "first_byte", "t_ms": 10.0}),
                    json.dumps(
                        {
                            "type": "event",
                            "t_ms": 20.0,
                            "payload": {
                                "data": {
                                    "choices": [{"delta": {"content": "ok"}}],
                                    "usage": {
                                        "prompt_tokens": 4,
                                        "completion_tokens": 1,
                                    },
                                }
                            },
                        }
                    ),
                    json.dumps({"type": "end", "t_ms": 30.0}),
                ]
            )
            + "\n"
        )
        return 1, 0.03

    monkeypatch.setattr(agentic_trace, "_spawn", fake_spawn)
    monkeypatch.setattr(agentic_trace, "_run_replay_requests", fake_replay_requests)
    monkeypatch.setattr(agentic_trace, "_start_system_sampler", lambda **_kwargs: BrokenSampler())

    rc = agentic_trace.replay_main(
        [
            "--source-trace",
            str(source),
            "--backend",
            "mlxlm",
            "--target",
            "fresh-target",
            "--out-root",
            str(tmp_path),
            "--label",
            "sampler-fails",
            "--system-sample-interval-s",
            "1",
        ]
    )

    assert rc == 0
    assert terminated == [(123, "server")]
    run_dir = next(tmp_path.glob("*-replay-sampler-fails"))
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "sample failed" in summary["system_sampler_error"]["error"]
