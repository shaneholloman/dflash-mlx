# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from dataclasses import replace
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from dflash_mlx import benchmark
from dflash_mlx import benchmark_report
from dflash_mlx import benchmark_suites
from dflash_mlx.engine.events import (
    CycleCompleteEvent,
    PrefillCompleteEvent,
    SummaryEvent,
    TokenEvent,
)
from dflash_mlx.runtime.context import build_offline_runtime_context
from dflash_mlx.runtime.config import PROFILES

def test_prompt_slug_distinguishes_long_prompts_with_same_prefix():
    prefix = "same prefix " * 8
    first = benchmark_suites.slugify_prompt_id(prefix + "alpha")
    second = benchmark_suites.slugify_prompt_id(prefix + "beta")

    assert first != second
    assert first.startswith("same-prefix")
    assert second.startswith("same-prefix")

def test_benchmark_help_documents_public_flags(capsys):
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {
        "--suite",
        "--limit",
        "--ctx-tokens",
        "--prompt-file",
        "--shuffle",
        "--seed",
        "--prompt",
        "--max-tokens",
        "--block-tokens",
        "--ctx",
        "--no-memory",
        "--repeat",
        "--cooldown",
        "--model",
        "--draft",
        "--no-chat-template",
        "--draft-quant",
        "--no-eos",
        "--split-sdpa",
        "--no-split-sdpa",
        "--target-fa-window",
        "--draft-sink-size",
        "--draft-window-size",
        "--verify-len-cap",
        "--out",
    } <= option_strings
    assert "Target model. Required." in out
    assert ".artifacts/dflash/benchmarks/<timestamp>-<suite>-<model>" in out
    assert "--matrix" not in out
    assert "--memory" not in out
    assert "--agentic" not in out

@pytest.mark.parametrize("flag", ["--matrix", "--memory", "--agentic"])
def test_benchmark_rejects_removed_public_flags(flag):
    parser = benchmark.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([flag])

    assert exc.value.code == 2

def test_benchmark_finalize_uses_offline_runtime_validation():
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "m",
            "--draft-window-size",
            "0",
        ]
    )

    with pytest.raises(
        ValueError,
        match="--draft-window-size / draft_window_size must be > 0",
    ):
        benchmark._finalize_benchmark_args(args, ["--model", "m"])

def test_benchmark_cli_reports_offline_runtime_validation_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(
            [
                "--model",
                "m",
                "--draft-window-size",
                "0",
            ],
            prog="dflash benchmark",
        )

    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "--draft-window-size must be > 0" in err
    assert "draft_window_size" not in err
    assert "Traceback" not in err


def test_benchmark_cli_requires_model_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["--max-tokens", "1"], prog="dflash benchmark")

    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "--model" in err
    assert "Traceback" not in err


def test_benchmark_invocation_records_explicit_and_effective_values():
    parser = benchmark.build_parser()
    args = parser.parse_args(
        [
            "--suite",
            "longctx",
            "--limit",
            "1",
            "--prompt",
            "p",
            "--model",
            "target-alias",
            "--draft",
            "draft-alias",
            "--max-tokens",
            "8",
            "--ctx",
            "65536",
            "--no-memory",
            "--out",
            "/tmp/result.json",
            "--no-chat-template",
            "--no-split-sdpa",
            "--draft-sink-size",
            "32",
            "--draft-window-size",
            "512",
            "--verify-len-cap",
            "8",
        ]
    )
    args = benchmark._finalize_benchmark_args(args, ["--suite", "longctx", "--limit", "1", "--ctx", "65536"])
    config = {
        "model": "resolved-target",
        "draft": "resolved-draft",
        "draft_quant": "w4",
    }
    invocation = benchmark._build_invocation(
        args,
        Path("/tmp/result.json"),
        [
            "dflash benchmark",
            "--suite",
            "longctx",
            "--limit",
            "1",
            "--prompt",
            "p",
            "--model",
            "target-alias",
            "--draft",
            "draft-alias",
            "--max-tokens",
            "8",
            "--ctx",
            "65536",
            "--no-memory",
            "--out",
            "/tmp/result.json",
            "--no-chat-template",
            "--no-split-sdpa",
            "--draft-sink-size",
            "32",
            "--draft-window-size",
            "512",
            "--verify-len-cap",
            "8",
        ],
        config,
    )

    assert invocation["output_path"] == "/tmp/result.json"
    assert invocation["output_dir"] == "/tmp/result.json"
    assert invocation["command"].startswith("dflash benchmark --suite longctx")
    assert invocation["protocol_order"] == ["baseline", "dflash"]
    assert invocation["same_prompt_token_ids"] is True
    assert invocation["primary_metric"] == "post_prefill_generation_tps"
    assert invocation["explicit_flags"]["model"] == "resolved-target"
    assert invocation["explicit_flags"]["draft"] == "resolved-draft"
    assert invocation["explicit_flags"]["suite"] == "longctx"
    assert invocation["explicit_flags"]["limit"] == 1
    assert invocation["explicit_flags"]["max_tokens"] == 8
    assert invocation["explicit_flags"]["ctx"] == 65536
    assert invocation["explicit_flags"]["no_memory"] is True
    assert invocation["explicit_flags"]["out"] == "/tmp/result.json"
    assert invocation["explicit_flags"]["draft_sink_size"] == 32
    assert invocation["explicit_flags"]["draft_window_size"] == 512
    assert invocation["explicit_flags"]["verify_len_cap"] == 8
    assert invocation["effective"]["model"] == "resolved-target"
    assert invocation["effective"]["draft"] == "resolved-draft"
    assert invocation["effective"]["suite"] == "longctx"
    assert invocation["effective"]["ctx_tokens"] == 65536
    assert invocation["effective"]["shuffle"] is False
    assert invocation["effective"]["seed"] == 0
    assert invocation["effective"]["include_memory"] is False
    assert invocation["effective"]["use_chat_template"] is False
    assert invocation["effective"]["draft_quant"] == "w4"
    assert invocation["effective"]["split_sdpa"] is False

def test_benchmark_default_output_dir_is_artifact_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--model", "mlx-community/Qwen3.6-27B-4bit"]),
        ["--model", "mlx-community/Qwen3.6-27B-4bit"],
    )

    out = benchmark.create_run_dir("benchmark", benchmark._benchmark_label(args))

    assert out.parts[:3] == (".artifacts", "dflash", "benchmarks")
    assert "benchmark-results" not in str(out).replace("/", "-")
    assert not str(out).startswith("/tmp")

def test_benchmark_default_suite_is_smoke():
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(parser.parse_args([]), [])
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert args.suite == "smoke"
    assert args.limit == 1
    assert [prompt.id for prompt in prompts] == ["smoke-default"]

def _fake_hf_rows(suite: str, count: int = 5) -> list[dict[str, str]]:
    if suite == "humaneval":
        return [
            {"task_id": f"HumanEval/{idx}", "prompt": f"def f_{idx}():\n    pass\n"}
            for idx in range(count)
        ]
    if suite == "gsm8k":
        return [{"question": f"question {idx}"} for idx in range(count)]
    return [
        {"problem_id": f"problem-{idx}", "problem": f"problem text {idx}"}
        for idx in range(count)
    ]

@pytest.mark.parametrize(
    ("suite", "expected_call", "expected_ids"),
    [
        (
            "humaneval",
            ("openai_humaneval", None, "test"),
            ["humaneval:HumanEval/0", "humaneval:HumanEval/1", "humaneval:HumanEval/2"],
        ),
        (
            "gsm8k",
            ("gsm8k", "main", "test"),
            ["gsm8k:0", "gsm8k:1", "gsm8k:2"],
        ),
        (
            "math500",
            ("HuggingFaceH4/MATH-500", None, "test"),
            ["math500:problem-0", "math500:problem-1", "math500:problem-2"],
        ),
    ],
)
def test_benchmark_named_suite_loads_hf_dataset(monkeypatch, suite, expected_call, expected_ids):
    calls = []

    def fake_load(name, config, split):
        calls.append((name, config, split))
        return _fake_hf_rows(suite)

    monkeypatch.setattr(benchmark_suites, "load_hf_dataset", fake_load)
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", suite, "--limit", "3"]),
        ["--suite", suite, "--limit", "3"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert calls == [expected_call]
    assert len(prompts) == 3
    assert [prompt.suite for prompt in prompts] == [suite, suite, suite]
    assert [prompt.source for prompt in prompts] == ["hf", "hf", "hf"]
    assert [prompt.id for prompt in prompts] == expected_ids

def test_benchmark_hf_default_selection_is_not_shuffled(monkeypatch):
    monkeypatch.setattr(
        benchmark_suites,
        "load_hf_dataset",
        lambda name, config, split: _fake_hf_rows("gsm8k", count=5),
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--limit", "3"]),
        ["--suite", "gsm8k", "--limit", "3"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert [prompt.id for prompt in prompts] == ["gsm8k:0", "gsm8k:1", "gsm8k:2"]

def test_benchmark_hf_shuffle_is_explicit_and_seeded(monkeypatch):
    monkeypatch.setattr(
        benchmark_suites,
        "load_hf_dataset",
        lambda name, config, split: _fake_hf_rows("gsm8k", count=6),
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--limit", "3", "--shuffle", "--seed", "7"]),
        ["--suite", "gsm8k", "--limit", "3", "--shuffle", "--seed", "7"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)
    expected_indices = list(range(6))
    random.Random(7).shuffle(expected_indices)

    assert [prompt.id for prompt in prompts] == [
        f"gsm8k:{idx}" for idx in expected_indices[:3]
    ]

def test_benchmark_missing_datasets_has_clean_error(monkeypatch):
    def raise_import_error():
        raise ImportError("no datasets")

    monkeypatch.setattr(benchmark_suites, "datasets_load_dataset", raise_import_error)
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--limit", "1"]),
        ["--suite", "gsm8k", "--limit", "1"],
    )

    with pytest.raises(RuntimeError, match="Install datasets to use --suite humaneval/gsm8k/math500"):
        benchmark_suites.resolve_benchmark_prompts(args)

def test_benchmark_local_suites_do_not_call_hf(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("HF loader should not be called")

    monkeypatch.setattr(benchmark_suites, "load_hf_dataset", fail)
    parser = benchmark.build_parser()
    smoke = benchmark._finalize_benchmark_args(parser.parse_args(["--suite", "smoke"]), ["--suite", "smoke"])
    longctx = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "longctx", "--ctx-tokens", "1024"]),
        ["--suite", "longctx", "--ctx-tokens", "1024"],
    )

    assert benchmark_suites.resolve_benchmark_prompts(smoke)[0].source == "smoke"
    assert benchmark_suites.resolve_benchmark_prompts(longctx)[0].source == "synthetic"

def test_benchmark_longctx_resolves_configured_context():
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "longctx", "--ctx-tokens", "65536"]),
        ["--suite", "longctx", "--ctx-tokens", "65536"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert args.ctx_tokens == 65536
    assert args.limit == 1
    assert prompts[0].id == "longctx-65536"
    assert len(prompts[0].prompt) > len(benchmark_suites.DEFAULT_PROMPT)

def test_benchmark_prompt_file_loads_jsonl_deterministically(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"id":"b","suite":"custom","prompt":"second"}\n'
        '{"id":"a","suite":"custom","prompt":"first"}\n'
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--prompt-file", str(prompt_file), "--limit", "1"]),
        ["--suite", "gsm8k", "--prompt-file", str(prompt_file), "--limit", "1"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert [prompt.id for prompt in prompts] == ["b"]
    assert prompts[0].prompt == "second"
    assert prompts[0].source == "jsonl"

def test_benchmark_prompt_file_bypasses_hf(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise AssertionError("HF loader should not be called")

    monkeypatch.setattr(benchmark_suites, "load_hf_dataset", fail)
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"id":"local","suite":"gsm8k","prompt":"local prompt"}\n')
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "gsm8k", "--prompt-file", str(prompt_file)]),
        ["--suite", "gsm8k", "--prompt-file", str(prompt_file)],
    )

    prompts = benchmark_suites.resolve_benchmark_prompts(args)

    assert [prompt.id for prompt in prompts] == ["local"]

def test_benchmark_summary_markdown_handles_missing_metrics():
    result = {
        "config": {
            "suite": "smoke",
            "benchmark_mode": "smoke",
            "model": "target",
            "draft": "draft",
            "max_tokens": 8,
            "block_tokens": 16,
            "prompt_count": 1,
            "draft_quant": "w4",
            "prompt_tokenization_mode": "chat_template",
        },
        "summary": {
            "prompt_tok_avg": None,
            "baseline_tps_median": None,
            "dflash_tps_median": None,
            "speedup_median": None,
            "dflash_ttft_ms_median": None,
            "dflash_peak_memory_gb_median": None,
            "acceptance_ratio_median": None,
            "prefix_saved_tokens": None,
        },
        "prompts": [],
    }

    text = benchmark_report.summary_markdown(result)

    assert "| smoke | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |" in text
    assert "- draft_quant: w4" in text


def test_benchmark_uses_effective_runtime_default_draft_quant_label():
    assert (
        benchmark._effective_draft_quant_label(
            None,
            {"draft_quant_spec": "w4"},
        )
        == "w4"
    )
    assert (
        benchmark._effective_draft_quant_label(
            "none",
            {"draft_quant_spec": None},
        )
        == "none"
    )


def test_benchmark_manifest_config_fields_are_suite_aware(monkeypatch):
    monkeypatch.setattr(
        benchmark_suites,
        "load_hf_dataset",
        lambda name, config, split: _fake_hf_rows("humaneval", count=3),
    )
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "humaneval", "--limit", "2", "--model", "m", "--shuffle", "--seed", "9"]),
        ["--suite", "humaneval", "--limit", "2", "--model", "m", "--shuffle", "--seed", "9"],
    )
    prompts = benchmark_suites.resolve_benchmark_prompts(args)
    result = benchmark_report.suite_report(
        prompts=prompts,
        prompt_reports=[
            {
                "hardware": {},
                "config": {
                    "model": "m",
                    "draft": "d",
                    "prompt_id": prompts[0].id,
                    "prompt_suite": prompts[0].suite,
                    "prompt_tokens": 10,
                },
                "runs": [],
                "summary": {},
            },
            {
                "hardware": {},
                "config": {
                    "model": "m",
                    "draft": "d",
                    "prompt_id": prompts[1].id,
                    "prompt_suite": prompts[1].suite,
                    "prompt_tokens": 12,
                },
                "runs": [],
                "summary": {},
            },
        ],
        args=args,
        include_memory=True,
    )

    assert result["config"]["suite"] == "humaneval"
    assert result["config"]["prompt_source"] == "hf"
    assert result["config"]["hf_dataset_name"] == "openai_humaneval"
    assert result["config"]["hf_dataset_config"] is None
    assert result["config"]["hf_dataset_split"] == "test"
    assert result["config"]["shuffle"] is True
    assert result["config"]["seed"] == 9
    assert result["config"]["hf_shuffle_seed"] == 9
    assert result["config"]["prompt_ids"] == [prompt.id for prompt in prompts]
    assert result["config"]["selected_row_indices"] == [prompt.row_index for prompt in prompts]
    assert result["config"]["prompt_count"] == 2
    assert result["config"]["ctx_tokens"] == 0
    assert result["runs"] == []

def test_benchmark_runs_json_rows_keep_provenance():
    prompts = [
        benchmark_suites.BenchmarkPrompt("p0", "smoke", "prompt", "smoke"),
    ]
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(
            ["--suite", "smoke", "--model", "m", "--draft", "d", "--draft-quant", "w4"]
        ),
        ["--suite", "smoke", "--model", "m", "--draft", "d", "--draft-quant", "w4"],
    )
    result = benchmark_report.suite_report(
        prompts=prompts,
        prompt_reports=[
            {
                "hardware": {},
                "config": {
                    "model": "m",
                    "draft": "d",
                    "git_hash": "abc123",
                    "prompt_id": "p0",
                    "prompt_suite": "smoke",
                    "prompt_tokens": 8,
                    "split_sdpa": False,
                },
                "runs": [
                    {
                        "run": 1,
                        "baseline": {"generation_tps": 1.0},
                        "dflash": {"generation_tps": 2.0},
                        "speedup": 2.0,
                    }
                ],
                "summary": {},
            }
        ],
        args=args,
        include_memory=True,
    )

    row = result["runs"][0]
    assert row["suite"] == "smoke"
    assert row["prompt_id"] == "p0"
    assert row["model"] == "m"
    assert row["draft"] == "d"
    assert row["draft_quant"] == "w4"
    assert row["git_hash"] == "abc123"
    assert row["prompt_tokenization_mode"] == "chat_template"
    assert row["max_tokens"] == 64
    assert result["config"]["split_sdpa"] is False
    assert result["config"]["split_sdpa_applied"] is False
    assert row["split_sdpa"] is False
    assert row["split_sdpa_applied"] is False


def test_benchmark_manifest_effective_config_uses_applied_split_sdpa(tmp_path):
    parser = benchmark.build_parser()
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(["--suite", "smoke", "--model", "m"]),
        ["--suite", "smoke", "--model", "m"],
    )

    effective = benchmark._controlled_flag_values(
        args,
        tmp_path,
        {"split_sdpa": False},
    )

    assert args.split_sdpa is None
    assert effective["split_sdpa"] is False


def test_benchmark_invocation_separates_requested_and_applied_split_sdpa(tmp_path):
    parser = benchmark.build_parser()
    argv = ["dflash benchmark", "--model", "m", "--split-sdpa"]
    args = benchmark._finalize_benchmark_args(
        parser.parse_args(argv[1:]),
        argv[1:],
    )

    invocation = benchmark._build_invocation(
        args,
        tmp_path,
        argv,
        {"split_sdpa": False, "split_sdpa_requested": True},
    )

    assert invocation["explicit_flags"]["split_sdpa"] is True
    assert invocation["effective"]["split_sdpa"] is False


def test_benchmark_split_sdpa_cli_is_tri_state():
    parser = benchmark.build_parser()

    assert parser.parse_args(["--model", "m"]).split_sdpa is None
    assert parser.parse_args(["--model", "m", "--split-sdpa"]).split_sdpa is True
    assert parser.parse_args(["--model", "m", "--no-split-sdpa"]).split_sdpa is False


def test_benchmark_runtime_context_uses_product_verify_config():
    context = build_offline_runtime_context(
        target_fa_window=2048,
        draft_sink_size=32,
        draft_window_size=512,
        verify_len_cap=8,
    )

    assert context.runtime.target_fa_window == 2048
    assert context.runtime.draft_sink_size == 32
    assert context.runtime.draft_window_size == 512
    assert context.runtime.verify_len_cap == 8
    assert context.runtime.prefix_cache is False
    assert context.verify.mode == "auto"
    assert context.verify.enable_qmm is True

def test_benchmark_direct_defaults_follow_balanced_profile(monkeypatch):
    monkeypatch.setitem(
        PROFILES,
        "balanced",
        replace(PROFILES["balanced"], draft_window_size=1536),
    )

    values = benchmark._offline_runtime_values()

    assert values["draft_window_size"] == 1536

def test_benchmark_runtime_context_is_required():
    stream = benchmark.stream_dflash_generate()
    with pytest.raises(ValueError, match="runtime_context is required"):
        next(stream)

def test_dflash_stream_target_ops_is_required():
    stream = benchmark.stream_dflash_generate(runtime_context=object())
    with pytest.raises(ValueError, match="target_ops is required"):
        next(stream)


def test_dflash_stream_draft_backend_is_required():
    stream = benchmark.stream_dflash_generate(runtime_context=object(), target_ops=object())
    with pytest.raises(ValueError, match="draft_backend is required"):
        next(stream)


class _BenchmarkTokenizer:
    eos_token_ids = []
    eos_token_id = None

    def encode(self, prompt):
        return [1, 2, 3]


class _ClosableBenchmarkStream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self):
        self.closed = True


class _BindableBenchmarkDraft:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.bound = False
        self.bound_target = None
        self.bound_ops = None

    def bind_target_model(self, target_model, *, target_ops):
        if self.events is not None:
            self.events.append("bind")
        self.bound = True
        self.bound_target = target_model
        self.bound_ops = target_ops


def _fake_baseline_result() -> dict:
    return {
        "elapsed_us": 2_000_000.0,
        "prefill_us": 1_000_000.0,
        "prompt_token_count": 3,
        "generated_token_ids": [7],
        "generation_tokens": 1,
        "generation_tps": 1.0,
        "peak_memory_gb": 1.0,
    }


def _fake_dflash_result() -> dict:
    return {
        "elapsed_us": 1_000_000.0,
        "phase_timings_us": {"prefill": 100_000.0, "verify": 50_000.0},
        "ttft_us": 100_000.0,
        "generated_token_ids": [7],
        "generation_tokens": 1,
        "acceptance_ratio": 1.0,
        "cycles_completed": 1,
    }


def test_benchmark_omits_baseline_peak_memory_when_reset_fails(monkeypatch, capsys):
    def reset_fails():
        raise RuntimeError("reset failed")

    monkeypatch.setattr(benchmark.mx, "reset_peak_memory", reset_fails, raising=False)
    monkeypatch.setattr(
        benchmark,
        "mlx_stream_generate",
        lambda *_args, **_kwargs: iter(
            [
                SimpleNamespace(
                    token=7,
                    prompt_tokens=3,
                    prompt_tps=100.0,
                    generation_tokens=1,
                    generation_tps=10.0,
                    peak_memory=42.0,
                )
            ]
        ),
    )

    result = benchmark._generate_stock_baseline_once(
        target_model=object(),
        tokenizer=_BenchmarkTokenizer(),
        prompt="prompt",
        max_new_tokens=1,
        no_eos=False,
        use_chat_template=False,
    )

    assert result["peak_memory_gb"] is None
    assert "baseline peak memory reset failed" in capsys.readouterr().err


def test_benchmark_omits_dflash_peak_memory_when_reset_fails(monkeypatch, capsys):
    def reset_fails():
        raise RuntimeError("reset failed")

    stream = _ClosableBenchmarkStream(
        [
            PrefillCompleteEvent(
                prefill_us=100_000.0,
                prompt_token_count=3,
                logical_ctx_tokens=3,
                physical_prefill_tokens=3,
                prefill_tokens_restored=0,
                prefill_tokens_computed=3,
            ),
            TokenEvent(
                token_id=7,
                generated_tokens=1,
                acceptance_ratio=1.0,
                cycles_completed=1,
            ),
            SummaryEvent(
                elapsed_us=1_000_000.0,
                prompt_token_count=3,
                generated_token_ids=(7,),
                generation_tokens=1,
                accepted_from_draft=1,
                acceptance_ratio=1.0,
                cycles_completed=1,
                phase_timings_us={"prefill": 100_000.0},
                peak_memory_gb=42.0,
            ),
        ]
    )
    monkeypatch.setattr(benchmark.mx, "reset_peak_memory", reset_fails, raising=False)
    monkeypatch.setattr(benchmark, "stream_dflash_generate", lambda **_kwargs: stream)

    result = benchmark._generate_dflash_stream_once(
        target_model=object(),
        target_ops=object(),
        tokenizer=_BenchmarkTokenizer(),
        draft_model=object(),
        draft_backend=object(),
        prompt="prompt",
        max_new_tokens=1,
        use_chat_template=False,
        block_tokens=16,
        stop_token_ids=[],
        suppress_token_ids=None,
        runtime_context=object(),
    )

    assert stream.closed is True
    assert result["peak_memory_gb"] is None
    assert "dflash peak memory reset failed" in capsys.readouterr().err


def test_benchmark_rejects_stale_dict_engine_event(monkeypatch):
    stream = _ClosableBenchmarkStream(
        [
            PrefillCompleteEvent(
                prefill_us=100_000.0,
                prompt_token_count=3,
                logical_ctx_tokens=3,
                physical_prefill_tokens=3,
                prefill_tokens_restored=0,
                prefill_tokens_computed=3,
            ),
            {"event": "token", "token_id": 7},
        ]
    )
    monkeypatch.setattr(benchmark, "_reset_peak_memory_for_benchmark", lambda _mode: True)
    monkeypatch.setattr(benchmark, "stream_dflash_generate", lambda **_kwargs: stream)

    with pytest.raises(TypeError, match="Unsupported DFlash engine event: dict"):
        benchmark._generate_dflash_stream_once(
            target_model=object(),
            target_ops=object(),
            tokenizer=_BenchmarkTokenizer(),
            draft_model=object(),
            draft_backend=object(),
            prompt="prompt",
            max_new_tokens=1,
            use_chat_template=False,
            block_tokens=16,
            stop_token_ids=[],
            suppress_token_ids=None,
            runtime_context=object(),
        )

    assert stream.closed is True


def test_summary_event_keeps_cycle_profile_typed_until_serialization():
    cycle = CycleCompleteEvent(
        cycle=1,
        block_len=16,
        commit_count=8,
        acceptance_len=8,
        draft_us=1.0,
        verify_us=2.0,
        acceptance_us=3.0,
        hidden_extraction_us=4.0,
        rollback_us=5.0,
        other_us=6.0,
        cycle_total_us=21.0,
    )
    summary = SummaryEvent(
        elapsed_us=1_000_000.0,
        prompt_token_count=3,
        generated_token_ids=(7,),
        generation_tokens=1,
        accepted_from_draft=1,
        acceptance_ratio=1.0,
        cycles_completed=1,
        phase_timings_us={"prefill": 100_000.0},
        cycle_profile_us=(cycle,),
    )

    assert summary.cycle_profile_us == (cycle,)
    assert summary.to_payload()["cycle_profile_us"] == [cycle.to_payload()]


def test_benchmark_cleanup_failure_is_reported(monkeypatch, capsys):
    def clear_cache_fails():
        raise RuntimeError("clear failed")

    def metal_clear_cache_fails():
        raise RuntimeError("metal clear failed")

    monkeypatch.setattr(benchmark.mx, "clear_cache", clear_cache_fails, raising=False)
    monkeypatch.setattr(
        benchmark.mx,
        "metal",
        SimpleNamespace(clear_cache=metal_clear_cache_fails),
        raising=False,
    )

    benchmark._release_loaded_models()

    err = capsys.readouterr().err
    assert "MLX cache cleanup failed" in err
    assert "clear failed" in err
    assert "metal clear failed" in err


def test_benchmark_cleanup_primary_failure_with_fallback_success_is_reported(monkeypatch, capsys):
    fallback_calls = 0

    def clear_cache_fails():
        raise RuntimeError("clear failed")

    def metal_clear_cache_succeeds():
        nonlocal fallback_calls
        fallback_calls += 1

    monkeypatch.setattr(benchmark.mx, "clear_cache", clear_cache_fails, raising=False)
    monkeypatch.setattr(
        benchmark.mx,
        "metal",
        SimpleNamespace(clear_cache=metal_clear_cache_succeeds),
        raising=False,
    )

    benchmark._release_loaded_models()

    err = capsys.readouterr().err
    assert fallback_calls == 1
    assert "used metal.clear_cache after clear_cache failed" in err
    assert "clear failed" in err


def _patch_benchmark_target(monkeypatch, *, resolved_model_ref: str = "target"):
    target = object()
    tokenizer = _BenchmarkTokenizer()
    monkeypatch.setattr(
        benchmark,
        "_load_pristine_target_bundle",
        lambda model_ref: (object(), tokenizer, {"resolved_model_ref": model_ref}),
    )
    monkeypatch.setattr(
        benchmark,
        "_generate_stock_baseline_once",
        lambda **kwargs: _fake_baseline_result(),
    )
    return target, tokenizer, {
        "resolved_model_ref": resolved_model_ref,
        "split_full_attention_sdpa": False,
        "split_full_attention_sdpa_requested": None,
        "split_full_attention_sdpa_default": False,
        "split_full_attention_sdpa_resolved": False,
    }


def _run_one_fake_benchmark(
    *,
    draft_model_ref: str | None = None,
    split_sdpa: bool | None = None,
) -> dict:
    return benchmark._run_once_sequential(
        prompt="prompt",
        max_new_tokens=1,
        block_tokens=16,
        use_chat_template=False,
        target_model_ref="target",
        draft_model_ref=draft_model_ref,
        draft_quant=None,
        no_eos=True,
        split_sdpa=split_sdpa,
    )


def test_benchmark_dflash_path_binds_draft_before_generation(monkeypatch):
    events: list[str] = []
    target, tokenizer, target_meta = _patch_benchmark_target(monkeypatch)
    draft = _BindableBenchmarkDraft(events)
    draft_backend = object()
    ops = object()
    bundle_calls: list[dict] = []

    def fake_load_runtime_bundle(**kwargs):
        bundle_calls.append(kwargs)
        draft.bind_target_model(target, target_ops=ops)
        return SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=draft,
            draft_meta={"resolved_model_ref": "auto-draft"},
            draft_backend=draft_backend,
            target_ops=ops,
        )

    monkeypatch.setattr(benchmark, "load_runtime_bundle", fake_load_runtime_bundle)

    def fake_generate(**kwargs):
        events.append("generate")
        assert kwargs["draft_model"].bound is True
        assert kwargs["draft_model"].bound_target is target
        assert kwargs["draft_model"].bound_ops is ops
        assert kwargs["draft_backend"] is draft_backend
        assert kwargs["target_ops"] is ops
        return _fake_dflash_result()

    monkeypatch.setattr(benchmark, "_generate_dflash_stream_once", fake_generate)

    report = _run_one_fake_benchmark()

    assert events == ["bind", "generate"]
    assert len(bundle_calls) == 1
    assert bundle_calls[0]["model_ref"] == "target"
    assert bundle_calls[0]["draft_ref"] is None
    assert bundle_calls[0]["draft_quant"] is None
    assert bundle_calls[0]["verify_config"] is not None
    assert bundle_calls[0]["split_full_attention_sdpa"] is None
    assert report["draft_meta"]["resolved_model_ref"] == "auto-draft"
    assert report["token_match"] is True


def test_benchmark_records_applied_split_sdpa_state(monkeypatch):
    target, tokenizer, target_meta = _patch_benchmark_target(monkeypatch)
    target_meta["split_full_attention_sdpa"] = False
    target_meta["split_full_attention_sdpa_requested"] = True

    monkeypatch.setattr(
        benchmark,
        "load_runtime_bundle",
        lambda **kwargs: SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=_BindableBenchmarkDraft(),
            draft_meta={"resolved_model_ref": "auto-draft"},
            draft_backend=object(),
            target_ops=object(),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_generate_dflash_stream_once",
        lambda **kwargs: _fake_dflash_result(),
    )

    report = benchmark.benchmark_once(
        prompt="prompt",
        max_new_tokens=1,
        block_tokens=16,
        use_chat_template=False,
        target_model_ref="target",
        draft_model_ref=None,
        draft_quant=None,
        no_eos=True,
        split_sdpa=True,
        cooldown=0,
    )

    assert report["config"]["split_sdpa"] is False
    assert report["config"]["split_sdpa_applied"] is False
    assert report["config"]["split_sdpa_requested"] is True


def test_benchmark_preserves_dflash_phase_timings_in_artifact_entry(monkeypatch):
    target, tokenizer, target_meta = _patch_benchmark_target(monkeypatch)
    monkeypatch.setattr(
        benchmark,
        "load_runtime_bundle",
        lambda **kwargs: SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=_BindableBenchmarkDraft(),
            draft_meta={"resolved_model_ref": "auto-draft"},
            draft_backend=object(),
            target_ops=object(),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_generate_dflash_stream_once",
        lambda **kwargs: _fake_dflash_result(),
    )

    run = _run_one_fake_benchmark()
    assert "generated_token_ids" not in run["dflash"]
    assert run["dflash"]["prefill_us"] == 100_000.0

    entry = benchmark._format_run_entry({**run, "run_index": 1})

    assert entry["dflash"]["phase_timings_us"] == {
        "prefill": 100_000.0,
        "verify": 50_000.0,
    }


def test_benchmark_explicit_draft_wins_over_registry(monkeypatch):
    loaded_bundle_args: list[dict] = []
    target, tokenizer, target_meta = _patch_benchmark_target(
        monkeypatch,
        resolved_model_ref="mlx-community/gemma-4-26b-a4b-it-4bit",
    )

    def fake_load_runtime_bundle(**kwargs):
        loaded_bundle_args.append(kwargs)
        return SimpleNamespace(
            target_model=target,
            tokenizer=tokenizer,
            target_meta=target_meta,
            draft_model=_BindableBenchmarkDraft(),
            draft_meta={"resolved_model_ref": kwargs["draft_ref"]},
            draft_backend=object(),
            target_ops=object(),
        )

    monkeypatch.setattr(benchmark, "load_runtime_bundle", fake_load_runtime_bundle)
    monkeypatch.setattr(
        benchmark,
        "_generate_dflash_stream_once",
        lambda **kwargs: _fake_dflash_result(),
    )

    report = _run_one_fake_benchmark(draft_model_ref="manual/draft")

    assert [call["draft_ref"] for call in loaded_bundle_args] == ["manual/draft"]
    assert report["draft_meta"]["resolved_model_ref"] == "manual/draft"


def _patch_missing_auto_draft(monkeypatch) -> None:
    _patch_benchmark_target(monkeypatch, resolved_model_ref="unknown/model")
    monkeypatch.setattr(
        benchmark,
        "load_runtime_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("No DFlash draft model found for 'unknown/model'")
        ),
    )


def test_benchmark_missing_auto_draft_raises_clear_value_error(monkeypatch):
    _patch_missing_auto_draft(monkeypatch)

    with pytest.raises(
        ValueError,
        match="No DFlash draft model found for 'unknown/model'",
    ):
        _run_one_fake_benchmark()


def test_benchmark_cli_missing_auto_draft_exits_cleanly(monkeypatch, tmp_path, capsys):
    _patch_missing_auto_draft(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        benchmark.main(
            [
                "--model",
                "unknown/model",
                "--prompt",
                "prompt",
                "--max-tokens",
                "1",
                "--no-chat-template",
                "--no-memory",
                "--out",
                str(tmp_path / "missing-auto-draft"),
            ],
            prog="dflash benchmark",
        )

    err = capsys.readouterr().err
    assert exc.value.code == 2
    assert "No DFlash draft model found for 'unknown/model'" in err
    assert "Traceback" not in err
