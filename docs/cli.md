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
dflash models
```

Legacy top-level commands are not part of the product surface.

## Serve

Start the OpenAI-compatible server:

```bash
dflash serve \
  --model Qwen/Qwen3.5-4B
```

Expert overrides stay explicit:

```bash
dflash serve \
  --prefill-step-size 8192 \
  --prefix-cache-max-bytes 17179869184
```

Long-KV attention routing is target-owned. Qwen verify GQA and Gemma4
full-attention GQA routes are selected internally by shape and cache state; there
is no public SDPA override flag.

`POST /v1/responses` is supported as a minimal non-streaming adapter for text
input and function-call tools. Streaming, reasoning/text/truncation controls,
and persistent response storage are not implemented; use
`/v1/chat/completions` for streaming.

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
dflash benchmark \
  --suite aime25 \
  --limit 30 \
  --shuffle \
  --seed 42 \
  --model mlx-community/Qwen3.6-27B-4bit \
  --draft z-lab/Qwen3.6-27B-DFlash
dflash benchmark --suite longctx --ctx-tokens 65536 --model Qwen/Qwen3.5-4B
```

`humaneval`, `gsm8k`, `math500`, and `aime25` load real Hugging Face datasets through the
optional `datasets` package (`pip install 'dflash-mlx[bench]'`). They are runtime
prompt suites. `aime25` also records exact integer score fields for baseline and
DFlash and defaults to `65536` generated tokens. First use may download/cache dataset rows. `smoke` and `longctx` are
local/offline; `smoke` is a CLI sanity check only, not a performance claim.

Use `--only-dflash` to skip the baseline MLX leg on expensive dataset runs.

Offline/custom prompts:

```bash
dflash benchmark --suite gsm8k --prompt-file data/gsm8k_sample.jsonl --limit 100 --model Qwen/Qwen3.5-4B
```

See [benchmarking.md](benchmarking.md) for protocol and artifact details.

## Doctor

Check local runtime state:

```bash
dflash doctor
dflash doctor --json
dflash doctor --strict
```

Validate effective runtime flags:

```bash
dflash doctor --prefill-step-size 1024
dflash doctor --no-prefix-cache-l2 --json
```

Check model/draft resolution:

```bash
dflash doctor --model Qwen/Qwen3.5-4B
dflash doctor --model Qwen/Qwen3.5-4B --load-model
```

`doctor` accepts the same runtime config flags as the server for validation:
prefill size, draft sink/window, verify cap, prefix cache, L2, target FA window,
and max context.

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
dflash serve --model Qwen/Qwen3.5-27B
```

Explicit prefill override:

```bash
dflash serve --model Qwen/Qwen3.5-27B --prefill-step-size 8192
```

Lower-memory prefill override:

```bash
dflash serve --model Qwen/Qwen3.5-27B --prefill-step-size 1024
```

Custom L2 cache directory:

```bash
dflash serve \
  --model Qwen/Qwen3.5-27B \
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
