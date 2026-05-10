# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import json
import socket
from types import SimpleNamespace

import pytest

import dflash_mlx.artifacts as artifacts
from tools.benchmarks import _agentic_trace as agentic_trace
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
        "profile": None,
        "prefill_step_size": None,
        "fastpath_max_tokens": None,
        "max_snapshot_tokens": None,
        "target_fa_window": 0,
        "chat_template_args": '{"enable_thinking":true}',
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)

def test_build_server_cmd_omits_dflash_runtime_overrides_by_default():
    cmd, port, url = _build_server_cmd(_server_args())

    assert port == 8123
    assert url == "http://127.0.0.1:8123"
    assert "--draft-quant" not in cmd
    assert "--profile" not in cmd
    assert "--prefill-step-size" not in cmd
    assert "--fastpath-max-tokens" not in cmd
    assert "--max-snapshot-tokens" not in cmd
    assert "--target-fa-window" not in cmd

def test_build_server_cmd_forwards_gemma_runtime_overrides():
    cmd, _, _ = _build_server_cmd(
        _server_args(
            draft_quant="w4",
            profile="long-session",
            prefill_step_size=1024,
            fastpath_max_tokens=0,
            max_snapshot_tokens=32000,
            target_fa_window=2048,
            chat_template_args='{"enable_thinking":false}',
        )
    )

    rendered = " ".join(cmd)
    assert "--draft-quant w4" in rendered
    assert "--profile long-session" in rendered
    assert "--prefill-step-size 1024" in rendered
    assert "--fastpath-max-tokens 0" in rendered
    assert "--max-snapshot-tokens 32000" in rendered
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
        ["--profile", "long-session"],
        ["--prefill-step-size", "1024"],
        ["--fastpath-max-tokens", "0"],
        ["--max-snapshot-tokens", "32000"],
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
            "--fastpath-max-tokens",
            "0",
            "--max-snapshot-tokens",
            "32000",
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
    assert overrides["fastpath_max_tokens"] == 0
    assert overrides["max_snapshot_tokens"] == 32000
    assert overrides["prefix_cache"] is True
    assert overrides["prefix_cache_l2"] is True
    assert overrides["prefix_cache_l2_dir"] == str(l2_dir)
    assert overrides["diagnostics"] == "full"
    server_cmd = metadata["server_cmd"]
    rendered = " ".join(server_cmd)
    assert "--draft-quant w4" in rendered
    assert "--profile long-session" in rendered
    assert "--prefill-step-size 1024" in rendered
    assert "--fastpath-max-tokens 0" in rendered
    assert "--max-snapshot-tokens 32000" in rendered
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
            "mlxlm",
            "--target",
            "fresh-target",
            "--out-root",
            str(tmp_path),
            "--label",
            "replay-smoke",
        ]
    )

    assert rc == 0
    run_dir = next(tmp_path.glob("*-replay-replay-smoke"))
    metadata = json.loads((run_dir / "metadata.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert metadata["source_trace"] == str(source)
    assert metadata["client"] == "replay"
    assert summary["post_count"] == 1
    assert summary["posts"][0]["request"]["model"] == "fresh-target"
