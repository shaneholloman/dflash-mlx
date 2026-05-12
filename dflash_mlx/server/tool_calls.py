# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from mlx_lm import tokenizer_utils


class ToolCallParseError(RuntimeError):
    pass


def install_tool_parser(tokenizer: Any) -> None:
    _refresh_parser_from_template(tokenizer)
    parser = getattr(tokenizer, "tool_parser", None)
    if parser is None or getattr(parser, "_dflash_tool_parser", False):
        return
    setattr(tokenizer, "_tool_parser", make_tool_parser(parser))


def make_tool_parser(
    upstream_parser: Callable[[str, Any], Any] | None,
) -> Callable[[str, Any], list[dict[str, Any]]]:
    def parse(text: str, tools: Any = None) -> list[dict[str, Any]]:
        return parse_tool_call_span(
            text,
            tools=tools,
            upstream_parser=upstream_parser,
        )

    parse._dflash_tool_parser = True  # type: ignore[attr-defined]
    return parse


def parse_tool_call_span(
    text: str,
    *,
    tools: Sequence[Mapping[str, Any]] | None,
    upstream_parser: Callable[[str, Any], Any] | None,
) -> list[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        raise ToolCallParseError("empty tool call")

    upstream_error: BaseException | None = None
    if upstream_parser is not None:
        try:
            parsed = upstream_parser(raw, tools)
            return _normalize_parsed_calls(parsed, tools=tools)
        except (ValueError, json.JSONDecodeError, SyntaxError) as exc:
            upstream_error = exc

    try:
        parsed = _parse_json_payload(raw)
        return _normalize_parsed_calls(parsed, tools=tools)
    except (ValueError, json.JSONDecodeError) as exc:
        if upstream_error is not None:
            raise ToolCallParseError(
                f"failed to parse tool call: {upstream_error}; JSON fallback: {exc}"
            ) from exc
        raise ToolCallParseError(f"failed to parse tool call: {exc}") from exc


def apply_tool_choice(request: Any, tool_choice: Any) -> None:
    if tool_choice is None:
        return
    if isinstance(tool_choice, str):
        choice = tool_choice.strip().lower()
        if choice == "auto":
            return
        if choice == "none":
            request.tools = None
            return
        raise ValueError(f"tool_choice={tool_choice!r} is not supported")
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type") or tool_choice.get("mode") or "")
        choice_type = choice_type.strip().lower()
        if choice_type == "none":
            request.tools = None
            return
        if choice_type == "auto":
            return
        if choice_type == "function":
            raise ValueError("function-specific tool_choice is not supported")
    raise ValueError("tool_choice must be 'auto', 'none', or null")


def normalize_parallel_tool_calls(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ValueError("parallel_tool_calls must be a boolean")
    return value


def _refresh_parser_from_template(tokenizer: Any) -> None:
    chat_template = getattr(tokenizer, "chat_template", None)
    parser_type = tokenizer_utils._infer_tool_parser(chat_template)
    if parser_type is None:
        return

    module = importlib.import_module(f"mlx_lm.tool_parsers.{parser_type}")
    tool_call_start = module.tool_call_start
    tool_call_end = module.tool_call_end
    encode = getattr(tokenizer, "encode")
    setattr(tokenizer, "_tool_parser", module.parse_tool_call)
    setattr(tokenizer, "_tool_call_start", tool_call_start)
    setattr(tokenizer, "_tool_call_end", tool_call_end)
    setattr(
        tokenizer,
        "_tool_call_start_tokens",
        tuple(encode(tool_call_start, add_special_tokens=False)),
    )
    setattr(
        tokenizer,
        "_tool_call_end_tokens",
        tuple(encode(tool_call_end, add_special_tokens=False)),
    )


def _parse_json_payload(text: str) -> list[Any]:
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    return [payload]


def _normalize_parsed_calls(
    parsed: Any,
    *,
    tools: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    values = parsed if isinstance(parsed, list) else [parsed]
    if not values:
        raise ToolCallParseError("tool call span did not contain any calls")
    known_names = _declared_tool_names(tools)
    calls: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        call = _normalize_one(value, index=index)
        _validate_declared_tool(call, known_names)
        calls.append(call)
    return calls


def _normalize_one(value: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ToolCallParseError(f"tool_call[{index}] must be an object")
    function = value.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = value.get("name")
        arguments = value.get("arguments", {})
    name_text = str(name or "").strip()
    if not name_text:
        raise ToolCallParseError(f"tool_call[{index}] missing function name")
    return {
        "id": value.get("id"),
        "name": name_text,
        "arguments": _json_object(arguments, context=f"tool_call[{index}]"),
    }


def _json_object(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        parsed: Any = {}
    elif isinstance(value, str):
        text = value.strip()
        parsed = json.loads(text) if text else {}
    else:
        parsed = value
    if not isinstance(parsed, dict):
        raise ToolCallParseError(f"{context} arguments must be a JSON object")
    return parsed


def _validate_declared_tool(call: dict[str, Any], known_names: set[str]) -> None:
    name = str(call["name"])
    if not known_names:
        raise ToolCallParseError(f"tool call '{name}' emitted with no declared tools")
    if name not in known_names:
        raise ToolCallParseError(f"unknown tool call '{name}'")


def _declared_tool_names(tools: Sequence[Mapping[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or ():
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names
