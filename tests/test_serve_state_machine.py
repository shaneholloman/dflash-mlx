# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

from mlx_lm.generate import SequenceStateMachine

from dflash_mlx import serve
from dflash_mlx.serve import DFlashResponseGenerator, _match_stream_token
from dflash_mlx.server.protocol import build_generation_context, thinking_enabled_for_request

def test_match_stream_token_terminal_state_is_not_reused():
    sm = SequenceStateMachine(
        transitions={"normal": [((42,), None)]},
        initial="normal",
    )
    state = sm.make_state()

    state, match, current, terminal = _match_stream_token(sm, state, 42)

    assert match == (42,)
    assert current is None
    assert terminal is True

    state2, match2, current2, terminal2 = _match_stream_token(sm, state, 7)

    assert state2 is state
    assert match2 is None
    assert current2 is None
    assert terminal2 is True

def test_match_stream_token_absent_state_machine_is_normal():
    state, match, current, terminal = _match_stream_token(None, None, 42)

    assert state is None
    assert match is None
    assert current == "normal"
    assert terminal is False

def test_response_generator_uses_configured_ar_fastpath(monkeypatch):
    calls = []

    def fake_ar_serve(self, request_tuple):
        calls.append(request_tuple)
        assert self.model_provider.draft_model is None
        return "ar"

    monkeypatch.setattr(serve.mlx_server.ResponseGenerator, "_serve_single", fake_ar_serve)

    generator = object.__new__(DFlashResponseGenerator)
    generator.model_provider = SimpleNamespace(
        draft_model="draft",
        cli_args=SimpleNamespace(
            fastpath_max_tokens=32,
            runtime_context=SimpleNamespace(
                diagnostics=SimpleNamespace(trace=None),
            ),
        ),
    )
    request_tuple = (object(), object(), SimpleNamespace(max_tokens=32))

    assert generator._serve_single(request_tuple) == "ar"
    assert calls == [request_tuple]
    assert generator.model_provider.draft_model == "draft"

class _ThinkingTokenizer:
    eos_token_ids = {0}
    has_thinking = True
    think_start_tokens = (10,)
    think_end_tokens = (11,)
    think_start = "<think>"
    think_end = "</think>"
    has_tool_calling = False
    tool_parser = None

    def convert_ids_to_tokens(self, token_id):
        return f"<{token_id}>"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

def _state_after(sm, token):
    state = sm.make_state()
    state, _match, current = sm.match(state, token)
    return current

def test_response_generator_disables_reasoning_state_when_thinking_is_off():
    generator = object.__new__(DFlashResponseGenerator)
    generator._state_machine_cache = {}
    generator._current_thinking_enabled = False

    sm, sequences = generator._make_state_machine(
        ("model", None, "draft"),
        _ThinkingTokenizer(),
        [],
        initial_state="reasoning",
    )

    assert sm.make_state()[0] == "normal"
    assert _state_after(sm, 10) == "normal"
    assert (10,) not in sequences

def test_response_generator_keeps_reasoning_state_when_thinking_is_on():
    generator = object.__new__(DFlashResponseGenerator)
    generator._state_machine_cache = {}
    generator._current_thinking_enabled = True

    sm, sequences = generator._make_state_machine(
        ("model", None, "draft"),
        _ThinkingTokenizer(),
        [],
    )

    assert _state_after(sm, 10) == "reasoning"
    assert sequences[(10,)] == "<think>"

def test_generation_context_honors_effective_thinking_flag():
    tokenizer = _ThinkingTokenizer()

    ctx = build_generation_context(tokenizer, [1, 2, 3], has_thinking=False)

    assert ctx.has_thinking is False

def test_request_thinking_default_and_request_override():
    cli_args = SimpleNamespace(chat_template_args={"enable_thinking": False})

    assert thinking_enabled_for_request(cli_args) is False
    assert thinking_enabled_for_request(
        cli_args,
        SimpleNamespace(chat_template_kwargs={"enable_thinking": True}),
    ) is True
