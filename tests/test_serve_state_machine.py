# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mlx_lm.generate import SequenceStateMachine

from dflash_mlx import serve
from dflash_mlx.server import metrics as metrics_mod
from dflash_mlx.server import runtime as server_runtime_mod
from dflash_mlx.serve import DFlashResponseGenerator
from dflash_mlx.server.protocol import (
    build_generation_context,
    match_stream_token,
    thinking_enabled_for_request,
)
from dflash_mlx.server.metrics import (
    get_live_metrics_payload,
    reset_live_metrics_for_tests,
    start_live_request,
)
from dflash_mlx.server.runtime import ServerRuntime


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _FakeGroup:
    def __init__(self, rank=0):
        self._rank = rank

    def rank(self):
        return self._rank


class _FakePromptCache:
    def __init__(self, size):
        self.size = size


class _FakeResponseGenerator:
    def __init__(self, model_provider, prompt_cache, server_runtime):
        self.events = model_provider.events
        self.fail_join = getattr(model_provider, "fail_join", False)
        self.fail_stop = getattr(model_provider, "fail_stop", False)

    def stop_and_join(self):
        self.events.append("stop")
        if self.fail_stop:
            raise RuntimeError("stop failed")

    def join(self):
        self.events.append("join")
        if self.fail_join:
            raise RuntimeError("worker join failed")


def test_match_stream_token_terminal_state_is_not_reused():
    sm = SequenceStateMachine(
        transitions={"normal": [((42,), None)]},
        initial="normal",
    )
    state = sm.make_state()

    state, match, current, terminal = match_stream_token(sm, state, 42)

    assert match == (42,)
    assert current is None
    assert terminal is True

    state2, match2, current2, terminal2 = match_stream_token(sm, state, 7)

    assert state2 is state
    assert match2 is None
    assert current2 is None
    assert terminal2 is True

def test_match_stream_token_absent_state_machine_is_normal():
    state, match, current, terminal = match_stream_token(None, None, 42)

    assert state is None
    assert match is None
    assert current == "normal"
    assert terminal is False

@pytest.mark.parametrize("ar_raises", [False, True])
def test_response_generator_ar_fastpath_accounting(monkeypatch, ar_raises):
    calls = []
    runtime_calls = []

    def fake_ar_serve(self, request_tuple):
        calls.append(request_tuple)
        assert self.model_provider.draft_model is None
        if ar_raises:
            raise RuntimeError("ar failed")
        return "ar"

    server_runtime = SimpleNamespace(
        next_request_id=lambda: 21 if ar_raises else 11,
        start_target_only_request=lambda **kwargs: runtime_calls.append(("start", kwargs)),
        record_target_only_request=lambda **kwargs: runtime_calls.append(("record", kwargs)),
        clear_request=lambda **kwargs: runtime_calls.append(("clear", kwargs)),
    )

    monkeypatch.setattr(serve.mlx_server.ResponseGenerator, "_serve_single", fake_ar_serve)

    generator = object.__new__(DFlashResponseGenerator)
    generator.server_runtime = server_runtime
    generator.model_provider = SimpleNamespace(
        draft_model="draft",
        cli_args=SimpleNamespace(
            fastpath_max_tokens=32,
            runtime_context=SimpleNamespace(
                diagnostics=SimpleNamespace(trace=None),
            ),
        ),
    )
    request_tuple = (_Queue(), object(), SimpleNamespace(max_tokens=32))
    result = generator._serve_single(request_tuple)

    assert generator.model_provider.draft_model == "draft"
    assert calls == [request_tuple]
    assert runtime_calls[0] == (
        "start",
        {
            "request_id": 21 if ar_raises else 11,
            "mode_used": "ar_fastpath",
            "max_tokens": 32,
        },
    )
    if ar_raises:
        assert result is None
        assert [name for name, _ in runtime_calls] == ["start", "clear"]
        assert runtime_calls[1][1] == {"request_id": 21}
        assert isinstance(request_tuple[0].items[0], RuntimeError)
    else:
        assert result == "ar"
        assert runtime_calls[1][0] == "record"
        assert runtime_calls[1][1]["request_id"] == 11
        assert runtime_calls[1][1]["mode_used"] == "ar_fastpath"
        assert runtime_calls[1][1]["max_tokens"] == 32
        assert runtime_calls[1][1]["wall_ms"] >= 0.0


def test_response_generator_clears_failed_dflash_request_after_live_start(monkeypatch):
    reset_live_metrics_for_tests()
    monkeypatch.setattr(metrics_mod, "current_runtime_cache_manager", lambda: None)

    class RuntimeThatFails:
        def next_request_id(self):
            return 31

        def serve_dflash_request(self, **kwargs):
            start_live_request(
                request_id=kwargs["request_id"],
                mode_used="dflash",
                prompt_tokens=3,
                max_tokens=16,
            )
            raise RuntimeError("dflash failed after live start")

        def clear_request(self, *, request_id):
            metrics_mod.clear_live_request(request_id=request_id)

    model_provider = SimpleNamespace(
        tokenizer=_ThinkingTokenizer(),
        cli_args=SimpleNamespace(
            fastpath_max_tokens=0,
            chat_template_args={"enable_thinking": False},
        ),
    )
    generator = object.__new__(DFlashResponseGenerator)
    generator.server_runtime = RuntimeThatFails()
    generator.model_provider = model_provider
    generator._state_machine_cache = {}
    generator._prepare_dflash_request = lambda *_args, **_kwargs: object()
    rqueue = _Queue()

    assert generator._serve_single(
        (rqueue, object(), SimpleNamespace(max_tokens=16, stop_words=[], seed=None))
    ) is None

    payload = get_live_metrics_payload()
    assert payload["current_request"] is None
    assert payload["last_request"] is None
    assert payload["totals"]["requests"] == 0
    assert isinstance(rqueue.items[0], RuntimeError)
    assert "dflash failed after live start" in str(rqueue.items[0])


@pytest.mark.parametrize(
    ("failure_stage", "error"),
    [
        ("cache", "prompt cache failed"),
        ("factory", "response generator failed"),
        ("startup", "load failed"),
        ("http", "http server failed"),
    ],
)
def test_server_runtime_shutdowns_after_rank0_failure(monkeypatch, failure_stage, error):
    events = []
    model_provider = SimpleNamespace(
        cli_args=SimpleNamespace(prompt_cache_size=7),
        events=events,
    )
    runtime = ServerRuntime(
        host="127.0.0.1",
        port=8123,
        model_provider=model_provider,
        version="test-version",
    )

    def wait_until_ready(**_kwargs):
        if failure_stage == "startup":
            raise RuntimeError(error)
        events.append("ready")

    def run_http_server(*_args, **_kwargs):
        if failure_stage == "http":
            raise RuntimeError(error)
        events.append("http")

    def prompt_cache_factory(size):
        if failure_stage == "cache":
            raise RuntimeError(error)
        return _FakePromptCache(size)

    def response_generator_factory(model_provider, prompt_cache, server_runtime):
        if failure_stage == "factory":
            raise RuntimeError(error)
        return _FakeResponseGenerator(model_provider, prompt_cache, server_runtime)

    monkeypatch.setattr(server_runtime_mod.mx.distributed, "init", lambda: _FakeGroup())
    monkeypatch.setattr(
        "dflash_mlx.server.runtime.mlx_server.LRUPromptCache",
        prompt_cache_factory,
    )
    monkeypatch.setattr(runtime, "wait_until_ready", wait_until_ready)
    monkeypatch.setattr(runtime, "configure_metrics", lambda: events.append("metrics"))
    monkeypatch.setattr(runtime, "print_startup_banner", lambda: events.append("banner"))
    monkeypatch.setattr(runtime, "shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr(
        "dflash_mlx.server.runtime.mlx_server._run_http_server",
        run_http_server,
    )

    with pytest.raises(RuntimeError, match=error):
        runtime.serve_forever(
            response_generator_factory=response_generator_factory,
            handler_class=object,
        )

    if failure_stage in {"cache", "factory"}:
        assert events == ["shutdown"]
    else:
        assert events[-2:] == ["stop", "shutdown"]
    if failure_stage in {"cache", "factory", "startup"}:
        assert "http" not in events


@pytest.mark.parametrize("join_raises", [False, True])
def test_server_runtime_shutdowns_after_rank_worker_join(monkeypatch, join_raises):
    events = []
    model_provider = SimpleNamespace(
        cli_args=SimpleNamespace(prompt_cache_size=7),
        events=events,
        fail_join=join_raises,
    )
    runtime = ServerRuntime(
        host="127.0.0.1",
        port=8123,
        model_provider=model_provider,
        version="test-version",
    )

    monkeypatch.setattr(server_runtime_mod.mx.distributed, "init", lambda: _FakeGroup(1))
    monkeypatch.setattr(
        "dflash_mlx.server.runtime.mlx_server.LRUPromptCache",
        _FakePromptCache,
    )
    monkeypatch.setattr(runtime, "shutdown", lambda: events.append("shutdown"))

    if join_raises:
        with pytest.raises(RuntimeError, match="worker join failed"):
            runtime.serve_forever(
                response_generator_factory=_FakeResponseGenerator,
                handler_class=object,
            )
    else:
        runtime.serve_forever(
            response_generator_factory=_FakeResponseGenerator,
            handler_class=object,
        )

    assert events == ["join", "shutdown"]


def test_server_runtime_shutdowns_after_rank0_stop_failure(monkeypatch):
    events = []
    model_provider = SimpleNamespace(
        cli_args=SimpleNamespace(prompt_cache_size=7),
        events=events,
        fail_stop=True,
    )
    runtime = ServerRuntime(
        host="127.0.0.1",
        port=8123,
        model_provider=model_provider,
        version="test-version",
    )

    monkeypatch.setattr(server_runtime_mod.mx.distributed, "init", lambda: _FakeGroup())
    monkeypatch.setattr(
        "dflash_mlx.server.runtime.mlx_server.LRUPromptCache",
        _FakePromptCache,
    )
    monkeypatch.setattr(runtime, "wait_until_ready", lambda **_kwargs: events.append("ready"))
    monkeypatch.setattr(runtime, "configure_metrics", lambda: events.append("metrics"))
    monkeypatch.setattr(runtime, "print_startup_banner", lambda: events.append("banner"))
    monkeypatch.setattr(runtime, "shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr(
        "dflash_mlx.server.runtime.mlx_server._run_http_server",
        lambda *_args, **_kwargs: events.append("http"),
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        runtime.serve_forever(
            response_generator_factory=_FakeResponseGenerator,
            handler_class=object,
        )

    assert events == ["ready", "metrics", "banner", "http", "stop", "shutdown"]


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
