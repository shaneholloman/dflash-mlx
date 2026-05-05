# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file

from __future__ import annotations

import pytest

from tools.benchmarks import prefix_cache_survival_gate as gate


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is True
        assert add_generation_prompt is True
        assert "enable_thinking" in kwargs
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text.split()


def test_build_survival_case_fits_budget_and_places_random_record():
    case = gate.build_survival_case(
        FakeTokenizer(),
        ctx_tokens=220,
        seed=7,
        case_index=1,
        insert_frac=0.50,
        key="key_abcd",
        value="value_1234",
        salt="salt_case",
        suffix="cold",
        enable_thinking=True,
        token_tolerance_pct=20.0,
    )

    assert 176 <= case.prompt_tokens <= 220
    assert "NEEDLE_RECORD key=key_abcd value=value_1234" in case.messages[0]["content"]
    assert "Return only the value associated with key key_abcd" in case.messages[1]["content"]


def test_build_followup_case_keeps_cold_turn_as_prefix():
    cold = gate.build_survival_case(
        FakeTokenizer(),
        ctx_tokens=220,
        seed=7,
        case_index=1,
        insert_frac=0.50,
        key="key_abcd",
        value="value_1234",
        salt="salt_case",
        suffix="cold",
        enable_thinking=False,
        token_tolerance_pct=20.0,
    )

    warm = gate.build_followup_case(
        FakeTokenizer(),
        previous=cold,
        assistant_content="value_1234",
        suffix="warm",
        enable_thinking=False,
    )

    assert warm.messages[: len(cold.messages)] == cold.messages
    assert warm.messages[-2] == {"role": "assistant", "content": "value_1234"}
    assert warm.messages[-1]["content"].endswith("Request nonce: warm.")
    assert warm.prompt_tokens > cold.prompt_tokens


def test_parse_token_list_accepts_k_suffix():
    assert gate.parse_token_list("16K,32768") == [16384, 32768]

    with pytest.raises(ValueError, match="token count"):
        gate.parse_token_list("0")


def test_score_response_requires_exact_value():
    assert gate.score_response("final: value_deadbeef", "value_deadbeef")
    assert not gate.score_response("final: value_dead", "value_deadbeef")


def test_response_message_missing_content_is_empty_not_crash():
    response = {"choices": [{"message": {"role": "assistant", "reasoning_content": "thinking"}}]}

    assert gate._response_message(response)["reasoning_content"] == "thinking"
    assert str(gate._response_message(response).get("content") or "") == ""


def test_summary_requires_strict_warm_hit_and_wrong_haystack_guard():
    rows = [{"passed": True, "ctx_tokens": 16384}]

    assert gate._summary(rows, error=None)["passed"] is True

    rows[0]["passed"] = False
    assert gate._summary(rows, error=None)["passed"] is False


def test_row_fails_on_tiny_warm_hit_or_stale_wrong_answer():
    spec = {
        "ctx_tokens": 1024,
        "seed": 0,
        "case_index": 0,
        "insert_frac": 0.5,
        "key": "key",
    }
    cold = gate.SurvivalCase(1024, 0, 0, 0.5, "key", "value_a", "salt", [], 1000)
    warm = gate.SurvivalCase(1024, 0, 0, 0.5, "key", "value_a", "salt", [], 1000)
    wrong = gate.SurvivalCase(1024, 0, 0, 0.5, "key", "value_b", "salt", [], 1000)
    cold_result = {"content": "value_a", "wall_ms": 1.0, "post_event": {"cache_hit_tokens": 0}}
    warm_result = {
        "content": "value_a",
        "wall_ms": 1.0,
        "post_event": {
            "mode_used": "dflash",
            "cache_hit_tokens": 100,
            "logical_ctx_tokens": 1000,
            "physical_prefill_tokens": 1000,
            "prefill_tokens_restored": 0,
            "prefill_tokens_computed": 1000,
        },
    }
    wrong_result = {"content": "value_a", "wall_ms": 1.0, "post_event": {"cache_hit_tokens": 0}}

    row = gate._row(
        spec=spec,
        cold=cold,
        warm=warm,
        wrong=wrong,
        cold_result=cold_result,
        warm_result=warm_result,
        wrong_result=wrong_result,
        min_cache_hit_ratio=0.8,
    )

    assert row["warm_hit_ok"] is False
    assert row["warm_physical_reuse_ok"] is False
    assert row["wrong_stale"] is True
    assert row["passed"] is False


def test_row_passes_with_substantial_runtime_reuse():
    spec = {
        "ctx_tokens": 1024,
        "seed": 0,
        "case_index": 0,
        "insert_frac": 0.5,
        "key": "key",
    }
    cold = gate.SurvivalCase(1024, 0, 0, 0.5, "key", "value_a", "salt", [], 1000)
    warm = gate.SurvivalCase(1024, 0, 0, 0.5, "key", "value_a", "salt", [], 1000)
    wrong = gate.SurvivalCase(1024, 0, 0, 0.5, "key", "value_b", "salt", [], 1000)
    cold_result = {"content": "value_a", "wall_ms": 1.0, "post_event": {"cache_hit_tokens": 0}}
    warm_result = {
        "content": "value_a",
        "wall_ms": 1.0,
        "post_event": {
            "mode_used": "dflash",
            "cache_hit_tokens": 850,
            "logical_ctx_tokens": 1000,
            "physical_prefill_tokens": 150,
            "prefill_tokens_restored": 850,
            "prefill_tokens_computed": 150,
        },
    }
    wrong_result = {"content": "value_b", "wall_ms": 1.0, "post_event": {"cache_hit_tokens": 100}}

    row = gate._row(
        spec=spec,
        cold=cold,
        warm=warm,
        wrong=wrong,
        cold_result=cold_result,
        warm_result=warm_result,
        wrong_result=wrong_result,
        min_cache_hit_ratio=0.8,
    )

    assert row["warm_hit_ok"] is True
    assert row["warm_mode_ok"] is True
    assert row["warm_physical_reuse_ok"] is True
    assert row["wrong_stale"] is False
    assert row["passed"] is True


def test_row_fails_on_fallback_even_with_accounting_fields():
    spec = {
        "ctx_tokens": 1024,
        "seed": 0,
        "case_index": 0,
        "insert_frac": 0.5,
        "key": "key",
    }
    cold = gate.SurvivalCase(1024, 0, 0, 0.5, "key", "value_a", "salt", [], 1000)
    warm = gate.SurvivalCase(1024, 0, 0, 0.5, "key", "value_a", "salt", [], 1000)
    cold_result = {"content": "value_a", "wall_ms": 1.0, "post_event": {"cache_hit_tokens": 0}}
    warm_result = {
        "content": "value_a",
        "wall_ms": 1.0,
        "post_event": {
            "mode_used": "dflash_fallback",
            "cache_hit_tokens": 850,
            "logical_ctx_tokens": 1000,
            "physical_prefill_tokens": 150,
            "prefill_tokens_restored": 850,
            "prefill_tokens_computed": 150,
        },
    }

    row = gate._row(
        spec=spec,
        cold=cold,
        warm=warm,
        wrong=None,
        cold_result=cold_result,
        warm_result=warm_result,
        wrong_result=None,
        min_cache_hit_ratio=0.8,
    )

    assert row["warm_mode_ok"] is False
    assert row["warm_physical_reuse_ok"] is True
    assert row["passed"] is False
