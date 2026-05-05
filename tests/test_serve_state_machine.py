# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

from mlx_lm.generate import SequenceStateMachine

from dflash_mlx import serve
from dflash_mlx.serve import DFlashResponseGenerator, _match_stream_token

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
