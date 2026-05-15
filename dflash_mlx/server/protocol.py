# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

import mlx_lm.server as mlx_server
from mlx_lm.generate import SequenceStateMachine

STATEFUL_SERVER_API = "state" in getattr(mlx_server.Response, "__annotations__", {})

def thinking_enabled_for_request(cli_args: Any, request_args: Any = None) -> bool:
    chat_args = getattr(cli_args, "chat_template_args", None)
    enabled = True
    if isinstance(chat_args, dict) and "enable_thinking" in chat_args:
        enabled = bool(chat_args.get("enable_thinking", False))

    request_chat_args = getattr(request_args, "chat_template_kwargs", None)
    if isinstance(request_chat_args, dict) and "enable_thinking" in request_chat_args:
        enabled = bool(request_chat_args["enable_thinking"])
    return enabled

def build_generation_context(
    tokenizer,
    prompt,
    stop_words=None,
    sequences=None,
    has_thinking: Optional[bool] = None,
):
    thinking = tokenizer.has_thinking if has_thinking is None else bool(has_thinking)
    if STATEFUL_SERVER_API:
        return mlx_server.GenerationContext(
            has_thinking=thinking,
            has_tool_calling=tokenizer.has_tool_calling,
            tool_parser=tokenizer.tool_parser,
            sequences=sequences or {},
            prompt=prompt,
        )
    return mlx_server.GenerationContext(
        has_tool_calling=tokenizer.has_tool_calling,
        tool_call_start=tokenizer.tool_call_start,
        tool_call_end=tokenizer.tool_call_end,
        tool_parser=tokenizer.tool_parser,
        has_thinking=thinking,
        think_start_id=tokenizer.think_start_id,
        think_end=tokenizer.think_end,
        think_end_id=tokenizer.think_end_id,
        eos_token_ids=tokenizer.eos_token_ids,
        stop_token_sequences=[
            tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in (stop_words or [])
        ],
        prompt=prompt,
    )

def make_state_machine(
    *,
    tokenizer: Any,
    stop_words: list[str],
    initial_state: str = "normal",
    include_thinking: bool = True,
):
    if not include_thinking and initial_state == "reasoning":
        initial_state = "normal"

    transitions: dict[str, list[tuple[tuple[int, ...], Optional[str]]]] = {}
    sequences: dict[tuple[int, ...], str] = {}

    common_stops: list[tuple[tuple[int, ...], Optional[str]]] = []
    for token_id in tokenizer.eos_token_ids:
        token_tuple = (int(token_id),)
        sequences[token_tuple] = tokenizer.convert_ids_to_tokens(token_id)
        common_stops.append((token_tuple, None))
    for stop_word in stop_words:
        tokens = tuple(tokenizer.encode(stop_word, add_special_tokens=False))
        sequences[tokens] = stop_word
        common_stops.append((tokens, None))

    transitions["normal"] = list(common_stops)

    if include_thinking and tokenizer.has_thinking:
        think_start = tokenizer.think_start_tokens
        think_end = tokenizer.think_end_tokens
        transitions["normal"].append((think_start, "reasoning"))
        transitions["reasoning"] = [(think_end, "normal")]
        transitions["reasoning"].extend(common_stops)
        sequences[think_start] = tokenizer.think_start
        sequences[think_end] = tokenizer.think_end

    if tokenizer.has_tool_calling:
        tool_start = tokenizer.tool_call_start_tokens
        tool_end = tokenizer.tool_call_end_tokens
        transitions["normal"].append((tool_start, "tool"))
        transitions["tool"] = [(tool_end, "normal")]
        transitions["tool"].extend(common_stops)
        sequences[tool_start] = tokenizer.tool_call_start
        sequences[tool_end] = tokenizer.tool_call_end

    return SequenceStateMachine(transitions, initial=initial_state), sequences

def make_response(
    *,
    text: str,
    token: int,
    state: Optional[str],
    match: Optional[tuple[int, ...]],
    finish_reason: Optional[str],
):
    if STATEFUL_SERVER_API:
        return mlx_server.Response(
            text,
            token,
            state or "normal",
            match,
            0.0,
            finish_reason,
            (),
        )
    return mlx_server.Response(
        text,
        token,
        0.0,
        finish_reason,
        (),
    )

def match_stream_token(
    sm: Optional[Any],
    sm_state: Optional[Any],
    token: int,
) -> tuple[Optional[Any], Optional[tuple[int, ...]], Optional[str], bool]:
    if sm is None:
        return sm_state, None, "normal", False
    if sm_state is None or sm_state[0] is None:
        return sm_state, None, None, True
    next_state, match_sequence, current_state = sm.match(sm_state, token)
    terminal = match_sequence is not None and current_state is None
    return next_state, match_sequence, current_state, terminal
