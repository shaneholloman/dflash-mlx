# Internal Benchmark Tools

This directory contains lab harnesses. They are useful for diagnosis, but they
are not the public benchmark contract.

Use `dflash benchmark` first for public baseline-vs-DFlash claims.
Named suites such as `smoke`, `humaneval`, `gsm8k`, `math500`, and `longctx`
belong to that public command, not to these lab tools.

## Surfaces

```bash
PYTHONPATH=$PWD python -m tools.benchmarks.agentic_trace --help
PYTHONPATH=$PWD python -m tools.benchmarks.prefix_cache_survival_gate --help
PYTHONPATH=$PWD python -m tools.benchmarks.prefix_cache_probe --help
PYTHONPATH=$PWD python -m tools.benchmarks.analyze_trace --help
```

`agentic_trace.py`
: Agentic session/proxy trace tooling. Use it to study server behavior under a
real client shape, not as a public speed claim by default.

`prefix_cache_survival_gate.py`
: Long-context prefix-cache correctness gate. It is not a statistical NIAH benchmark.

`prefix_cache_probe.py`
: Prefix-cache and L2 mechanism probes.

`analyze_trace.py`
: Trace and prompt/memory analyzers.

## Capture vs replay

`agentic_trace run` is a capture/orchestration harness. It starts a server,
starts the recording proxy, drives a real OpenCode or pi session, and writes
the observed requests, SSE stream, server diagnostics, and workspace output.
By default it writes one run directory under `.artifacts/dflash/traces/...`.

It is not an exact replay engine. It does not force two runtimes to take the
same token trajectory, and it does not validate output quality. A future replay
mode must either replay deterministically or fail explicitly as not implemented.

## Cross-runtime opencode comparison (dflash vs mlxlm)

Run baseline first, then dflash with `--compare-to`:

```bash
python -m tools.benchmarks.agentic_trace run --backend mlxlm --target <model> \
    --task-file <task.txt> --label mlx_baseline
python -m tools.benchmarks.agentic_trace run --backend dflash --target <model> \
    --draft <draft> --task-file <task.txt> --label dflash_run \
    --prefix-cache --prefix-cache-l2 --diagnostics basic \
    --compare-to .artifacts/dflash/traces/<mlx_baseline_dir>
```

The dflash run emits `compare.md` with:

- **Trajectory-invariant metrics** (`decode_tps_avg`,
  `prefix_tokens_saved`, `weighted_acceptance`,
  `post_prefill_ms_per_token`) — the only metrics that are mathematically
  valid for cross-runtime comparison.
- **Trajectory-dependent metrics** (`wall_s`, POST count) shown for reference
  with caveats — if trajectories diverge, these metrics describe the captured
  sessions, not a direct runtime-speed comparison.
- **Per-POST gap table** emitted only when trajectories align: same POST
  count and decode tokens within ±5%. Otherwise it is omitted with an explicit
  `TRAJECTORY DIVERGED` warning.

For deterministic per-token A/B with no trajectory divergence by construction,
use `dflash benchmark` on a fixed prompt. That benchmark runs both runtimes
in-process on identical input.

Private `_*.py` files are implementation modules for these wrappers.

## Tool audit

| Tool | Verdict | One-sentence question |
| --- | --- | --- |
| `agentic_trace.py` | KEEP | On a real OpenCode/pi loop, how many POSTs, tokens, cache hits, tok/s, and acceptance did the runtime produce? |
| `prefix_cache_survival_gate.py` | KEEP | Does a warm prefix-cache request preserve the correct long-context record after a divergent suffix and reject a stale wrong-haystack answer? |
| `_agentic_trace.py` | KEEP | How does the orchestrated server+proxy+client capture build the trace and compare it to a peer run? |
| `_agentic_proxy.py` | KEEP | What exact OpenAI-compatible requests and SSE events did the client send and receive? |
| `_agentic_session.py` | FOLD later | What did a raw OpenCode/pi client session emit without the server/proxy orchestration? |
| `analyze_trace.py` | KEEP | In a captured trace, where did prompt tokens, memory buckets, and request costs concentrate? |
| `prefix_cache_probe.py` | KEEP | Does prefix-cache L1/L2 hit, evict, restore, and account bytes mechanically? |
| `_prefix_cache_multiturn.py` | KEEP | Does an in-process multi-turn prefix-cache path produce expected hit/reuse behavior? |
| `_prefix_l2_long_session.py` | KEEP | Does a real-model long-session L2 path restore snapshots across revisits? |
| `_prefix_l2_synthetic.py` | KEEP | What are L2 write/read/size mechanics without loading a model? |

## Output Policy

New lab outputs should go under `.artifacts/dflash/...` or an explicit local
`--out` path. Do not add new benchmark outputs to Git.

Some harnesses are intentionally narrower than `dflash benchmark` and may not
write a full public manifest. If a result will be quoted outside local
debugging, record the exact command, model refs, git hash, profile/flags, prompt
tokenization mode, and output directory.

## Rules

- Do not compare numbers across harnesses as if they were one protocol.
- Do not use full tracing for throughput claims.
- Do not overlap heavy model loads.
- Keep mechanism probes separate from product benchmark results.
- Prefer the smallest harness that answers the question.
