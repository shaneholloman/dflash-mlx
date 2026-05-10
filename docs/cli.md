# CLI

The public command is `dflash`.

```bash
dflash --help
```

Commands:

```text
dflash serve
dflash generate
dflash benchmark
dflash doctor
dflash profiles
dflash models
```

Legacy top-level commands are not part of the product surface.

## Serve

Start the OpenAI-compatible server:

```bash
dflash serve \
  --model Qwen/Qwen3.5-4B \
  --profile balanced
```

Useful profiles:

```bash
dflash profiles
dflash serve --profile fast
dflash serve --profile low-memory
dflash serve --profile long-session
```

Expert overrides stay explicit:

```bash
dflash serve \
  --profile balanced \
  --prefill-step-size 8192 \
  --prefix-cache-max-bytes 17179869184
```

Split-SDPA defaults are target-policy auto in both `serve` and `benchmark`; use
`--split-sdpa` or `--no-split-sdpa` only for an explicit A/B.

See [runtime-flags.md](runtime-flags.md) for the full flag surface.

## Generate

One prompt, no server:

```bash
PROMPT='The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.'

dflash generate \
  --model Qwen/Qwen3.5-4B \
  --prompt "$PROMPT"
```

This path is for local sanity checks. It does not enable cross-request prefix cache and
should not be used for public performance claims.

## Benchmark

Public local baseline-vs-DFlash runtime benchmark:

```bash
PROMPT='The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.'

dflash benchmark \
  --model Qwen/Qwen3.5-9B \
  --prompt "$PROMPT" \
  --max-tokens 1024 \
  --repeat 3 \
  --cooldown 60 \
  --no-eos
```

Outputs default to:

```text
.artifacts/dflash/benchmarks/<timestamp>-<suite>-<model>/
```

Named runtime suites:

```bash
dflash benchmark --suite humaneval --limit 10 --model Qwen/Qwen3.5-4B
dflash benchmark --suite gsm8k --limit 10 --model Qwen/Qwen3.5-4B
dflash benchmark --suite math500 --limit 10 --model Qwen/Qwen3.5-4B
dflash benchmark --suite longctx --ctx-tokens 65536 --model Qwen/Qwen3.5-4B
```

`humaneval`, `gsm8k`, and `math500` load real Hugging Face datasets through the
optional `datasets` package (`pip install 'dflash-mlx[bench]'`). They are runtime
prompt suites, not official accuracy scoring. First use may download/cache
dataset rows. `smoke` and `longctx` are local/offline; `smoke` is a CLI sanity
check only, not a performance claim.

Offline/custom prompts:

```bash
dflash benchmark --suite gsm8k --prompt-file data/gsm8k_sample.jsonl --limit 100
```

See [benchmarking.md](benchmarking.md) for protocol and artifact details.

## Doctor

Check local runtime state:

```bash
dflash doctor
dflash doctor --json
dflash doctor --strict
```

Validate an effective runtime profile:

```bash
dflash doctor --profile low-memory
dflash doctor --profile long-session --prefix-cache-l2 --json
```

Check model/draft resolution:

```bash
dflash doctor --model Qwen/Qwen3.5-4B
dflash doctor --model Qwen/Qwen3.5-4B --load-model
```

`doctor` accepts the same runtime config flags as the server for validation:
profile, prefill size, draft sink/window, verify cap, prefix cache, L2, target
FA window, and max context.

## Models

List the current built-in target-to-draft registry:

```bash
dflash models
```

Only listed families are supported by the automatic draft resolver. Passing a
different target without a compatible `--draft` is a load error, not a silent
fallback to a generic server.

## Diagnostics

Basic request/cache logs:

```bash
dflash serve --diagnostics basic
```

Full memory/cycle diagnostics:

```bash
dflash serve --diagnostics full
```

Custom directory:

```bash
dflash serve --diagnostics full --diagnostics-dir .artifacts/dflash/diagnostics/manual
```

See [observability.md](observability.md).

## Common Examples

Normal coding server:

```bash
dflash serve --model Qwen/Qwen3.5-27B --profile balanced
```

Throughput-oriented server:

```bash
dflash serve --model Qwen/Qwen3.5-27B --profile fast
```

Lower-memory server:

```bash
dflash serve --model Qwen/Qwen3.5-27B --profile low-memory
```

Long-session cache experiment:

```bash
dflash serve \
  --model Qwen/Qwen3.5-27B \
  --profile long-session \
  --prefix-cache-l2-dir .artifacts/dflash/prefix-l2
```

Synthetic 64k context stress:

```bash
dflash benchmark \
  --model Qwen/Qwen3.5-4B \
  --suite longctx \
  --ctx-tokens 65536 \
  --max-tokens 64
```
