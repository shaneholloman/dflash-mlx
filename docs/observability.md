# Observability

DFlash has three observability surfaces:

1. compact stderr lines that help interactive server use;
2. live JSON metrics for current server state;
3. structured diagnostics artifacts for debugging and benchmarking.

Structured diagnostics are opt-in. The compact server memory line is currently
always printed after DFlash requests.

## Live Metrics

`dflash serve` exposes an always-on JSON endpoint:

```bash
curl http://127.0.0.1:8000/metrics
```

The endpoint reads the in-memory server snapshot. It does not scan diagnostics
artifacts and does not force MLX evaluation. It works before the first request
with `last_request: null`, reports `current_request` while a request is in
prefill/decode, and keeps the 32 latest completed requests in
`recent_requests`.

`rss_gb` is the current process resident size. `wired_gb` remains `null` unless
the runtime has a true per-process wired-memory source; `wired_limit_gb` still
reports the configured Metal limit.

Prefill throughput is split deliberately:

- `prefill_tok_s_physical`: tokens actually computed after prefix-cache restore
  divided by user-visible prefill wall time.
- `prefill_tok_s_apparent`: logical prompt tokens divided by the same wall time.

Completed requests also expose `ttft_s`, `phase_timings_us`, and
`prefill_phase_timings_us` in `last_request` and `recent_requests`.
`phase_timings_us` carries the runtime phase totals reported by the engine,
including `draft`, `verify`, and `replay` when DFlash produced those phases.

Use live metrics for debugging, dashboards, and benchmark visibility. Use
diagnostics artifacts when you need reproducible traces.

## Diagnostics Modes

`dflash serve` accepts:

```bash
--diagnostics off
--diagnostics basic
--diagnostics full
--diagnostics-dir PATH
```

`off`
: No structured JSONL artifacts.

`basic`
: Request/post events and cache summaries.

`full`
: Basic diagnostics plus memory waterfall and per-cycle profiling.

Default output:

```text
.artifacts/dflash/diagnostics/<timestamp>-serve-<mode>/
```

Use the `--diagnostics` form for operation, bug reports, and benchmark
debugging.

## Files

When diagnostics are enabled, the runtime may write:

```text
manifest.json
invocation.json
effective_config.json
cache_events.jsonl
cycle_events.jsonl
post_events.jsonl
summary.md
```

`manifest.json`
: Run metadata: command, cwd, git state, platform, model, draft, and selected
runtime config.

`invocation.json`
: The resolved server invocation and diagnostics mode.

`effective_config.json`
: Runtime/cache/verify settings after CLI/env/default resolution.

`cache_events.jsonl`
: Prefix-cache lookup, hit/miss, prune, eviction, and snapshot events.

`cycle_events.jsonl`
: Per-cycle engine events. In full mode this includes cycle timing and memory
waterfall samples.

`post_events.jsonl`
: Per-request summary events. These include request ids, prompt length, token
counts, acceptance/cycle fields when applicable, cache stats, prefill accounting
(`logical_ctx_tokens`, `physical_prefill_tokens`, `prefill_tokens_restored`,
`prefill_tokens_computed`), and memory summary fields.

`summary.md`
: Human-readable append-only request summary.

## Memory Waterfall

Memory waterfall records named buckets at selected runtime boundaries. It is
used to separate active model memory, MLX cache memory, prefix snapshots, draft
cache, target KV/state, target hidden chunks, rollback tape, and untracked RSS.

Enable it with:

```bash
dflash serve --diagnostics full
```

Waterfall is for diagnosis. Do not enable full diagnostics for throughput
claims.

## Always-On Server Line

After a DFlash request, the server prints a compact memory line on stderr. This
is separate from JSONL diagnostics. It exists so a normal server run still gives
basic memory visibility without a diagnostics directory.

If you need reproducible analysis, use `--diagnostics basic` or
`--diagnostics full` instead of scraping stderr.

## Trace Discipline

For performance claims:

- diagnostics should be `off` unless the claim is specifically about
  diagnostics;
- full cycle tracing should not be used;
- record the exact command, model refs, runtime flags, prompt-token count, and output
  directory;
- prefer `dflash benchmark` for public baseline-vs-DFlash numbers.

For debugging:

- use `--diagnostics basic` for request/cache issues;
- use `--diagnostics full` for memory or cycle timing issues;
- keep the run directory local under `.artifacts/`;
- attach the whole diagnostics directory to bug reports rather than copying
  selected lines.
