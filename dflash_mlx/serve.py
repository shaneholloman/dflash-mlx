# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import itertools
import json
import logging
import sys
import time
import warnings
from collections.abc import Sequence

warnings.filterwarnings("ignore", message="mlx_lm.server is not recommended")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
try:
    from huggingface_hub.utils import disable_progress_bars
except ImportError:
    disable_progress_bars = None

if disable_progress_bars is not None:
    try:
        disable_progress_bars()
    except Exception as exc:
        sys.stderr.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"[dflash] huggingface progress bar disable failed: {exc}\n"
        )
        sys.stderr.flush()

import mlx.core as mx
import mlx_lm.server as mlx_server

from dflash_mlx import __version__ as _DFLASH_VERSION
from dflash_mlx.server.config import (
    build_parser as _build_parser,
    configure_logging,
    configure_metal_limits,
    normalize_cli_args,
)
from dflash_mlx.server.protocol import (
    STATEFUL_SERVER_API as _STATEFUL_SERVER_API,
    build_generation_context as _build_generation_context,
    make_state_machine as _make_state_machine,
    match_stream_token as _match_stream_token,
    thinking_enabled_for_request as _thinking_enabled_for_request,
)
from dflash_mlx.bench_logger import (
    enabled as _bench_enabled,
    log_post as _bench_log_post,
)
from dflash_mlx.server.metrics import (
    clear_live_request as _clear_live_request,
    configure_live_metrics as _configure_live_metrics,
    get_live_metrics_payload as _get_live_metrics_payload,
    record_request_metrics as _record_request_metrics,
    record_target_only_request as _record_target_only_request,
    start_live_request as _start_live_request,
    write_post_request_memory_line as _write_post_request_memory_line,
    write_summary_line as _write_summary_line,
)
from dflash_mlx.server.model_provider import (
    DFlashModelProvider,
    wait_for_initial_model_load as _wait_for_initial_model_load,
)
from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
from dflash_mlx.runtime_context import with_metal_limits
from dflash_mlx.cache.manager import (
    current_runtime_cache_manager as _current_runtime_cache_manager,
    shutdown_runtime_cache_manager,
)
from dflash_mlx.server.prefix_cache_flow import PrefixCacheFlow
from dflash_mlx.server.responses_adapter import (
    ResponsesAdapterError,
    chat_response_to_responses,
    responses_to_chat_body,
)
from dflash_mlx.server.request_loop import consume_dflash_events

def _read_project_version() -> str:
    return _DFLASH_VERSION

def _bytes_to_gib(value: int) -> str:
    return f"{float(value) / (1024 ** 3):.1f} GiB"

def _format_limit_request(value) -> str:
    if isinstance(value, int):
        return _bytes_to_gib(value)
    return str(value)

def _format_metal_limit(label: str, request, bytes_value: int | None, applied: bool) -> str:
    action = _bytes_to_gib(bytes_value) if applied and bytes_value is not None else "not set"
    return f"{label}: {_format_limit_request(request)} -> {action}"

_DFLASH_REQUEST_COUNTER = itertools.count(1)

def _build_prompt_regime(args, tokenizer, request=None) -> dict[str, object]:
    chat_template_args = getattr(args, "chat_template_args", None)
    if not isinstance(chat_template_args, dict):
        chat_template_args = {}
    request_type = getattr(request, "request_type", None)
    is_chat = request_type == "chat"
    return {
        "request_type": request_type or "unknown",
        "request_tokenization": "mlx_lm.server",
        "runtime_prompt_input": "prompt_tokens_override",
        "chat_template": bool(is_chat and getattr(tokenizer, "has_chat_template", False)),
        "chat_template_args": dict(chat_template_args) if is_chat else {},
        "enable_thinking": bool(is_chat and chat_template_args.get("enable_thinking", False)),
        "use_default_chat_template": bool(
            is_chat and getattr(args, "use_default_chat_template", False)
        ),
        "custom_chat_template": bool(is_chat and getattr(args, "chat_template", None)),
        "tokenizer_class": type(tokenizer).__name__,
    }

class DFlashResponseGenerator(mlx_server.ResponseGenerator):
    def _restore_thinking_enabled(self, had_previous: bool, previous: object) -> None:
        if had_previous:
            self._current_thinking_enabled = previous
        elif hasattr(self, "_current_thinking_enabled"):
            delattr(self, "_current_thinking_enabled")

    def _tokenize(self, tokenizer, request, args):
        tokenized = super()._tokenize(tokenizer, request, args)
        if (
            not bool(getattr(self, "_current_thinking_enabled", True))
            and isinstance(tokenized, tuple)
            and len(tokenized) == 4
        ):
            prompt, segments, segment_types, _initial_state = tokenized
            return prompt, segments, segment_types, "normal"
        return tokenized

    def _make_state_machine(
        self,
        model_key,
        tokenizer,
        stop_words,
        initial_state="normal",
    ):
        include_thinking = bool(getattr(self, "_current_thinking_enabled", True))
        cache_key = (model_key, tuple(stop_words), initial_state, include_thinking)
        cached = self._state_machine_cache.get(cache_key)
        if cached is not None:
            return cached

        sm, sequences = _make_state_machine(
            tokenizer=tokenizer,
            stop_words=stop_words,
            initial_state=initial_state,
            include_thinking=include_thinking,
        )
        if len(self._state_machine_cache) > 100:
            self._state_machine_cache.clear()
        self._state_machine_cache[cache_key] = (sm, sequences)
        return sm, sequences

    def _serve_single(self, request):
        request_tuple = request
        rqueue, request, args = request_tuple

        request_id = next(_DFLASH_REQUEST_COUNTER)
        had_previous_thinking_enabled = hasattr(self, "_current_thinking_enabled")
        previous_thinking_enabled = getattr(self, "_current_thinking_enabled", None)
        self._current_thinking_enabled = _thinking_enabled_for_request(
            self.model_provider.cli_args,
            args,
        )
        try:
            runtime_context = self.model_provider.cli_args.runtime_context
            trace_config = runtime_context.diagnostics.trace
            bench_active = _bench_enabled(trace_config)
            fastpath_max_tokens = int(
                getattr(self.model_provider.cli_args, "fastpath_max_tokens", 256) or 0
            )

            if fastpath_max_tokens > 0 and args.max_tokens <= fastpath_max_tokens:
                sys.stderr.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] fast-path AR | "
                    f"max_tokens={args.max_tokens} threshold={fastpath_max_tokens}\n"
                )
                sys.stderr.flush()
                saved_draft_model = self.model_provider.draft_model
                wall_t0 = time.perf_counter_ns()
                _start_live_request(
                    request_id=request_id,
                    mode_used="ar_fastpath",
                    prompt_tokens=None,
                    max_tokens=int(args.max_tokens),
                )
                try:
                    self.model_provider.draft_model = None
                    return super()._serve_single((rqueue, request, args))
                finally:
                    self.model_provider.draft_model = saved_draft_model
                    wall_ms = (time.perf_counter_ns() - wall_t0) / 1e6
                    _record_target_only_request(
                        request_id=request_id,
                        mode_used="ar_fastpath",
                        wall_ms=wall_ms,
                        max_tokens=int(args.max_tokens),
                    )
                    if bench_active:
                        _bench_log_post(
                            trace_config,
                            request_id=request_id,
                            mode_used="ar_fastpath",
                            max_tokens=int(args.max_tokens),
                            wall_ms=wall_ms,
                        )

            model = self.model_provider.model
            tokenizer = self.model_provider.tokenizer
            draft_model = self.model_provider.draft_model
            draft_backend = getattr(self.model_provider, "draft_backend", None)
            target_ops = getattr(self.model_provider, "target_ops", None)
            if draft_backend is None:
                raise RuntimeError("DFlash draft backend is not loaded")
            if target_ops is None:
                raise RuntimeError("DFlash target ops are not loaded")
            tokenized = self._tokenize(tokenizer, request, args)
            if isinstance(tokenized, tuple):
                prompt, _, _, initial_state = tokenized
            else:
                prompt = tokenized
                initial_state = "normal"

            sm = None
            sm_state = None
            sequences = {}
            if _STATEFUL_SERVER_API and hasattr(self, "_make_state_machine"):
                sm, sequences = self._make_state_machine(
                    self.model_provider.model_key,
                    tokenizer,
                    args.stop_words,
                    initial_state=initial_state,
                )
                sm_state = sm.make_state()

            ctx = _build_generation_context(
                tokenizer,
                prompt,
                stop_words=args.stop_words,
                sequences=sequences,
                has_thinking=self._current_thinking_enabled,
            )
            rqueue.put(ctx)

            if args.seed is not None:
                mx.random.seed(args.seed)

            stop_token_ids = get_stop_token_ids(tokenizer)
            eos_token_ids = set(int(token_id) for token_id in tokenizer.eos_token_ids)
            request_start_ns = time.perf_counter_ns()
            prefix_flow = PrefixCacheFlow.for_request(
                model_provider=self.model_provider,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt=prompt,
                runtime_context=runtime_context,
            )
            ctx.prompt_cache_count = prefix_flow.hit_tokens
            _start_live_request(
                request_id=request_id,
                mode_used="dflash",
                prompt_tokens=len(prompt),
                max_tokens=int(args.max_tokens),
                cache_hit_tokens=prefix_flow.hit_tokens,
                cache_lookup_ms=prefix_flow.lookup_ms,
            )

            event_iter = stream_dflash_generate(
                target_model=model,
                target_ops=target_ops,
                tokenizer=tokenizer,
                draft_model=draft_model,
                draft_backend=draft_backend,
                prompt="",
                max_new_tokens=args.max_tokens,
                use_chat_template=False,
                stop_token_ids=stop_token_ids,
                prompt_tokens_override=prompt,
                prefix_snapshot=prefix_flow.snapshot,
                prefix_snapshot_builder=prefix_flow.snapshot_builder,
                stable_prefix_len=prefix_flow.stable_prefix_len,
                prefix_cache_active=prefix_flow.cache_active,
                runtime_context=runtime_context,
            )
            loop_result = consume_dflash_events(
                event_iter=event_iter,
                rqueue=rqueue,
                ctx=ctx,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=int(args.max_tokens),
                eos_token_ids=eos_token_ids,
                request_start_ns=request_start_ns,
                prefix_flow=prefix_flow,
                sm=sm,
                sm_state=sm_state,
                bench_active=bench_active,
                request_id=request_id,
                runtime_context=runtime_context,
            )
            summary_event = loop_result.summary_event

            if summary_event is not None:
                _write_summary_line(
                    summary_event=summary_event,
                    prompt_token_count=len(prompt),
                )

            _record_request_metrics(
                request_id=request_id,
                summary_event=summary_event,
                request_start_ns=loop_result.request_start_ns,
                request_done_ns=time.perf_counter_ns(),
                first_token_ns=loop_result.first_token_ns,
                prefill_done_ns=loop_result.prefill_done_ns,
                prompt_token_count=len(prompt),
                live_token_count=loop_result.live_token_count,
                cache_lookup_ms=loop_result.cache_lookup_ms,
                cache_hit_tokens=loop_result.cache_hit_tokens,
                cache_insert_ms=loop_result.cache_insert_ms,
                finish_reason=loop_result.finish_reason,
                max_tokens=args.max_tokens,
                prompt_regime=_build_prompt_regime(args, tokenizer, request),
                memory_waterfall_peak=loop_result.memory_waterfall_peak,
                diagnostics=runtime_context.diagnostics,
                prefill_event=loop_result.prefill_event,
                runtime_config=runtime_context.runtime,
            )
            _write_post_request_memory_line(request_id=request_id)
            rqueue.put(None)
        except Exception as e:
            _clear_live_request(request_id=request_id)
            rqueue.put(e)
        finally:
            self._restore_thinking_enabled(
                had_previous_thinking_enabled,
                previous_thinking_enabled,
            )

class DFlashAPIHandler(mlx_server.APIHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] == "/metrics":
            self.handle_metrics_request()
            return
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] == "/v1/responses":
            self.handle_responses_request()
            return
        return super().do_POST()

    def handle_metrics_request(self):
        payload = _get_live_metrics_payload(
            prefix_cache_manager=_current_runtime_cache_manager(),
        )
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        self._set_completion_headers(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def handle_responses_request(self):
        try:
            body = self._read_json_body()
            chat_body = responses_to_chat_body(body)
            self._serve_responses_as_chat(chat_body)
        except ResponsesAdapterError as e:
            self._write_json_response(400, {"error": str(e)})

    def _serve_responses_as_chat(self, chat_body: dict[str, object]) -> None:
        previous_body = getattr(self, "body", None)
        previous_path = self.path
        previous_responses_mode = getattr(self, "_responses_mode", None)
        self.body = chat_body
        self.path = "/v1/chat/completions"
        self._responses_mode = True
        try:
            stop_words = self._load_completion_fields(chat_body)
            request = self.handle_chat_completions()
            self.handle_completion(request, stop_words)
        except ValueError as e:
            self._write_json_response(400, {"error": str(e)})
        finally:
            self.body = previous_body
            self.path = previous_path
            if previous_responses_mode is None:
                if hasattr(self, "_responses_mode"):
                    delattr(self, "_responses_mode")
            else:
                self._responses_mode = previous_responses_mode

    def _read_json_body(self) -> dict[str, object]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ResponsesAdapterError("Content-Length header is required")
        try:
            length = int(content_length)
        except ValueError as exc:
            raise ResponsesAdapterError("Invalid Content-Length header") from exc
        try:
            body = json.loads(self.rfile.read(length).decode())
        except json.JSONDecodeError as exc:
            raise ResponsesAdapterError(f"Invalid JSON in request body: {exc}") from exc
        if not isinstance(body, dict):
            raise ResponsesAdapterError("Request should be a JSON dictionary")
        return body

    def _load_completion_fields(self, body: dict[str, object]) -> list[str]:
        self.stream = bool(body.get("stream", False))
        self.stream_options = body.get("stream_options", None)
        self.requested_model = body.get("model", "default_model")
        self.requested_draft_model = body.get("draft_model", "default_model")
        self.num_draft_tokens = body.get(
            "num_draft_tokens",
            self.response_generator.cli_args.num_draft_tokens,
        )
        self.adapter = body.get("adapters", None)
        self.max_tokens = body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = body.get(
                "max_tokens",
                self.response_generator.cli_args.max_tokens,
            )
        self.temperature = body.get(
            "temperature",
            self.response_generator.cli_args.temp,
        )
        self.top_p = body.get("top_p", self.response_generator.cli_args.top_p)
        self.top_k = body.get("top_k", self.response_generator.cli_args.top_k)
        self.min_p = body.get("min_p", self.response_generator.cli_args.min_p)
        self.repetition_penalty = body.get("repetition_penalty", 0.0)
        self.repetition_context_size = body.get("repetition_context_size", 20)
        self.presence_penalty = body.get("presence_penalty", 0.0)
        self.presence_context_size = body.get("presence_context_size", 20)
        self.frequency_penalty = body.get("frequency_penalty", 0.0)
        self.frequency_context_size = body.get("frequency_context_size", 20)
        self.xtc_probability = body.get("xtc_probability", 0.0)
        self.xtc_threshold = body.get("xtc_threshold", 0.0)
        self.logit_bias = body.get("logit_bias", None)
        self.logprobs = body.get("logprobs", False)
        self.top_logprobs = body.get("top_logprobs", -1)
        self.seed = body.get("seed", None)
        self.chat_template_kwargs = body.get("chat_template_kwargs")
        self.validate_model_parameters()

        stop_words = body.get("stop") or []
        if isinstance(stop_words, str):
            return [stop_words]
        if not isinstance(stop_words, list):
            raise ValueError("stop must be a string or list")
        return stop_words

    def _write_json_response(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self._set_completion_headers(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def handle_completion(self, request, stop_words):
        try:
            return super().handle_completion(request, stop_words)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            return
        except ValueError as e:
            logging.warning("Tool parser error (likely malformed tool call): %s", e)
            self.close_connection = True
            return

    def generate_response(self, *args, **kwargs):
        response = super().generate_response(*args, **kwargs)
        served_model = (
            self.response_generator.model_provider.model_key[0]
            if self.response_generator.model_provider.model_key is not None
            else None
        )
        if served_model:
            response["model"] = served_model
        if getattr(self, "_responses_mode", False):
            return chat_response_to_responses(response)
        return response

def _print_startup_banner(
    *,
    port: int,
    model_provider: DFlashModelProvider,
) -> None:
    dflash_version = _read_project_version()
    server_name = getattr(mlx_server, "__name__", "mlx_lm.server")
    target_ref = None
    draft_ref = None
    if model_provider.model_key is not None:
        target_ref = model_provider.model_key[0]
        draft_ref = model_provider.model_key[2]
    target_ref = target_ref or model_provider.cli_args.model or "unknown"
    if not draft_ref:
        raise RuntimeError("DFlash server requires a resolved draft model before startup.")

    if model_provider.cli_args.draft_model:
        draft_suffix = " (explicit)"
    else:
        draft_suffix = " (auto-detected)"
    chat_template_args = getattr(model_provider.cli_args, "chat_template_args", {})
    if not isinstance(chat_template_args, dict):
        chat_template_args = {}
    thinking_enabled = bool(chat_template_args.get("enable_thinking", False))
    fastpath_max_tokens = int(
        getattr(model_provider.cli_args, "fastpath_max_tokens", 256) or 0
    )
    runtime_config = getattr(model_provider.cli_args, "runtime_config", None)
    target_fa_window = (
        int(runtime_config.target_fa_window) if runtime_config is not None else 0
    )
    pc_enabled = bool(runtime_config.prefix_cache) if runtime_config is not None else False
    if target_fa_window > 0:
        pc_status = "disabled (--target-fa-window)"
    else:
        pc_status = "enabled" if pc_enabled else "disabled (--no-prefix-cache)"
    target_fa_status = (
        "full KV" if target_fa_window == 0 else f"rotating window {target_fa_window}"
    )
    metal_limits = getattr(model_provider.cli_args, "metal_limits", None)
    raw_lines = [
        f"DFlash v{dflash_version} - speculative decoding engine",
        f"Target:       {target_ref}",
        f"Draft:        {draft_ref}{draft_suffix}",
        "Mode:         DFlash (speculative decoding active)",
        f"Thinking:     {'enabled' if thinking_enabled else 'disabled'}",
        (
            f"Fast path:    AR <= {fastpath_max_tokens} tokens"
            if fastpath_max_tokens > 0
            else "Fast path:    off"
        ),
        f"Prefix cache: {pc_status}",
        f"Target FA KV: {target_fa_status}",
        f"Server:       {server_name} on port {port}",
    ]
    if runtime_config is not None:
        l2_status = (
            f"on ({runtime_config.prefix_cache_l2_dir}, "
            f"{_bytes_to_gib(runtime_config.prefix_cache_l2_max_bytes)})"
            if runtime_config.prefix_cache_l2
            else "off"
        )
        bench_log_dir = runtime_config.bench_log_dir or "off"
        diagnostics_mode = getattr(model_provider.cli_args, "diagnostics", "off")
        diagnostics_dir = getattr(model_provider.cli_args, "diagnostics_dir_resolved", "")
        raw_lines.extend(
            [
                f"Profile:      {runtime_config.profile}",
                f"Prefill step: {runtime_config.prefill_step_size}",
                (
                    "Draft cache:  "
                    f"sink={runtime_config.draft_sink_size} "
                    f"window={runtime_config.draft_window_size}"
                ),
                f"Verify cap:   {runtime_config.verify_len_cap or 'block'}",
                f"Clear cache:  {'boundary' if runtime_config.clear_cache_boundaries else 'off'}",
                (
                    "L1 cache:     "
                    f"{'on' if runtime_config.prefix_cache else 'off'} "
                    f"entries={runtime_config.prefix_cache_max_entries} "
                    f"bytes={_bytes_to_gib(runtime_config.prefix_cache_max_bytes)}"
                ),
                f"L2 cache:     {l2_status}",
                f"Max snapshot: {runtime_config.max_snapshot_tokens}",
                f"Waterfall:    {'on' if runtime_config.memory_waterfall else 'off'}",
                f"Bench logs:   {bench_log_dir}",
                f"Diagnostics:  {diagnostics_mode}",
                f"Diagnostics dir: {diagnostics_dir or 'off'}",
                f"Verify mode:  {runtime_config.verify_mode}",
            ]
        )
    if metal_limits is not None:
        if metal_limits.metal_available:
            raw_lines.extend(
                [
                    _format_metal_limit(
                        "Wired limit",
                        metal_limits.wired_request,
                        metal_limits.wired_bytes,
                        metal_limits.wired_applied,
                    ),
                    _format_metal_limit(
                        "Cache limit",
                        metal_limits.cache_request,
                        metal_limits.cache_bytes,
                        metal_limits.cache_applied,
                    ),
                ]
            )
        else:
            raw_lines.extend(
                [
                    "Wired limit:  Metal unavailable -> not set",
                    "Cache limit:  Metal unavailable -> not set",
                ]
            )

    width = max(len(line) for line in raw_lines)
    use_color = sys.stderr.isatty()
    reset = "\033[0m" if use_color else ""
    border_color = "\033[38;5;39m" if use_color else ""
    title_color = "\033[1;38;5;51m" if use_color else ""
    body_color = "\033[38;5;252m" if use_color else ""

    def style(text: str, color: str) -> str:
        return f"{color}{text}{reset}" if use_color else text

    border = style("+" + "-" * (width + 2) + "+", border_color)
    lines = [border]
    for index, raw_line in enumerate(raw_lines):
        padded = f"| {raw_line.ljust(width)} |"
        lines.append(style(padded, title_color if index == 0 else body_color))
    lines.append(border)

    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()

def _run_with_dflash_server(host: str, port: int, model_provider: DFlashModelProvider):
    group = mx.distributed.init()
    prompt_cache = mlx_server.LRUPromptCache(model_provider.cli_args.prompt_cache_size)

    response_generator = DFlashResponseGenerator(model_provider, prompt_cache)
    if group.rank() == 0:
        _wait_for_initial_model_load(model_provider, timeout_s=300.0)
        _configure_live_metrics(
            version=_read_project_version(),
            model_provider=model_provider,
        )
        _print_startup_banner(port=port, model_provider=model_provider)
        try:
            mlx_server._run_http_server(
                host,
                port,
                response_generator,
                handler_class=DFlashAPIHandler,
            )
        finally:
            shutdown_runtime_cache_manager()
    else:
        response_generator.join()

def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = _build_parser()
    if prog is not None:
        parser.prog = prog
    args = normalize_cli_args(parser.parse_args(list(argv) if argv is not None else None))
    metal_limits = configure_metal_limits(args)
    args.runtime_context = with_metal_limits(args.runtime_context, metal_limits)
    configure_logging(args.log_level)
    _run_with_dflash_server(args.host, args.port, DFlashModelProvider(args))

if __name__ == "__main__":
    main()
