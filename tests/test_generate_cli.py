# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import pytest

from dflash_mlx import generate
from dflash_mlx.runtime_loading import parse_draft_quant_spec


def test_decode_token_prefers_list_decode():
    calls = []

    class _Tokenizer:
        def decode(self, token):
            calls.append(token)
            return "list" if isinstance(token, list) else "scalar"

    assert generate.decode_token(_Tokenizer(), 7) == "list"
    assert calls == [[7]]


def test_decode_token_falls_back_to_scalar_decode_on_type_error():
    calls = []

    class _Tokenizer:
        def decode(self, token):
            calls.append(token)
            if isinstance(token, list):
                raise TypeError("scalar required")
            return f"token-{token}"

    assert generate.decode_token(_Tokenizer(), 7) == "token-7"
    assert calls == [[7], 7]


def test_decode_token_propagates_unexpected_decode_errors():
    calls = []

    class _Tokenizer:
        def decode(self, token):
            calls.append(token)
            raise RuntimeError("decode failed")

    with pytest.raises(RuntimeError, match="decode failed"):
        generate.decode_token(_Tokenizer(), 7)

    assert calls == [[7]]


def test_generate_cli_passes_verify_mode(monkeypatch):
    calls = []
    metal_calls = []

    monkeypatch.setattr(generate, "apply_metal_limits", lambda: metal_calls.append(True))
    monkeypatch.setattr(generate, "run_generate", lambda **kwargs: calls.append(kwargs) or 0)

    with pytest.raises(SystemExit) as exc:
        generate.main(
            [
                "--model",
                "m",
                "--prompt",
                "p",
                "--verify-mode",
                "off",
                "--prefill-step-size",
                "8192",
                "--draft-sink-size",
                "32",
                "--draft-window-size",
                "512",
                "--verify-len-cap",
                "8",
                "--draft-quant",
                "w4",
            ]
        )

    assert exc.value.code == 0
    assert metal_calls == [True]
    assert calls == [
        {
            "model_ref": "m",
            "prompt": "p",
            "max_tokens": 2048,
            "use_chat_template": True,
            "draft_ref": None,
            "target_fa_window": 0,
            "prefill_step_size": 8192,
            "draft_sink_size": 32,
            "draft_window_size": 512,
            "verify_len_cap": 8,
            "verify_mode": "off",
            "draft_quant": "w4",
        }
    ]

def test_generate_cli_rejects_invalid_prefill_step_size(monkeypatch):
    monkeypatch.setattr(generate, "apply_metal_limits", lambda: None)

    with pytest.raises(SystemExit) as exc:
        generate.main(
            [
                "--model",
                "m",
                "--prompt",
                "p",
                "--prefill-step-size",
                "0",
            ]
        )

    assert exc.value.code == "--prefill-step-size must be > 0"

def test_draft_quant_parser_no_env_fallback(monkeypatch):
    monkeypatch.setenv("DFLASH_DRAFT_QUANT", "w4")
    from dflash_mlx.runtime_loading import _resolve_draft_quant

    assert _resolve_draft_quant(None) is None
    assert parse_draft_quant_spec("w4").weight_bits == 4
