# Runtime Flags

This document describes the live product config surface. Historical experiment
flags belong in `old_status/`, not here.

## Config Model

For `dflash serve` and `dflash doctor`, runtime config is resolved once at
startup:

1. CLI flag;
2. startup env var;
3. profile value;
4. product default.

After that, engine code consumes typed `RuntimeContext`. Product behavior should
not depend on rereading `os.environ` during a request.

`dflash generate` and `dflash benchmark` expose a smaller explicit flag surface.
They do not support the full server profile system.

## Profiles

Use:

```bash
dflash profiles
```

Current profiles:

<!-- dflash-runtime-config:profiles:start -->
| Profile | Prefill | Draft window | Prefix cache | L1 entries / byte budget | L2 | Intent |
| --- | ---: | --- | --- | --- | --- | --- |
| `balanced` | 4096 | `64+1024` | on | `4 / 8GiB` | off | default normal coding |
| `fast` | 8192 | `64+1024` | on | `4 / 16GiB` | off | throughput first |
| `low-memory` | 1024 | `64+1024` | on | `2 / 2GiB` | off | lower pressure, slower prefill |
| `long-session` | 4096 | `64+1024` | on | `8 / 8GiB` | on / `50GiB` | revisit-oriented long sessions |
<!-- dflash-runtime-config:profiles:end -->

`clear_cache_boundaries` is currently off in all profiles.

Gemma4 long-context memory-pressure note: use the existing `low-memory` profile,
or keep another profile and set `--prefill-step-size 1024` explicitly. The
measured 31B serve probes also showed lower peak memory and TTFT with
`--prefill-step-size 512` under tighter memory pressure, but validate that on
your prompt; it is not a balanced-default change or a public benchmark
throughput claim.

## `dflash serve`

Core:

| Flag | Meaning |
| --- | --- |
| `--profile {balanced,fast,low-memory,long-session}` | preset defaults |
| `--list-profiles` | print profiles and exit |
| `--model REF_OR_PATH` | target model |
| `--draft-model REF_OR_PATH`, `--draft REF_OR_PATH` | draft override |
| `--host HOST` | server host |
| `--port PORT` | server port |
| `--log-level LEVEL` | Python log level |

Generation defaults:

| Flag | Meaning |
| --- | --- |
| `--temp FLOAT` | default request temperature |
| `--top-p FLOAT` | default nucleus sampling cutoff |
| `--top-k INT` | default top-k, `0` disables |
| `--min-p FLOAT` | default min-p filter |
| `--max-tokens INT` | default request max tokens |
| `--fastpath-max-tokens INT` | short-output target-only AR fast path threshold; default `256`, `0` disables; does not change DFlash max context |
| `--chat-template TEMPLATE` | inline chat template override |
| `--use-default-chat-template` | force tokenizer default template |
| `--enable-thinking` | set chat-template arg `enable_thinking=true`; default disabled |
| `--chat-template-args JSON` | JSON args for template rendering |

DFlash runtime:

<!-- dflash-runtime-config:serve-runtime:start -->
| Flag | Meaning |
| --- | --- |
| `--prefill-step-size INT` | target prefill chunk size |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward, `0` means block size |
| `--verify-mode {auto,off}` | verifier path mode; `off` is debug/parity only |
| `--dflash-max-ctx INT` | DFlash runtime context cap; `0` means no cap |
| `--target-fa-window INT` | experimental target FA rotating window; `0` means full KV |
| `--clear-cache-boundaries`, `--no-clear-cache-boundaries` | clear the MLX cache at safe request boundaries |
<!-- dflash-runtime-config:serve-runtime:end -->

Draft loading:

| Flag | Meaning |
| --- | --- |
| `--draft-quant SPEC` | optional in-memory draft quantization, e.g. `w4:gs64` |

Draft quantization is deliberately explicit. Current Gemma4 probes keep
`--draft-quant w4` as a 31B memory-headroom option; 26B-A4B should be treated as
a measured speed tradeoff because its peak memory increased in the local long
context probe.

Prefix cache:

<!-- dflash-runtime-config:prefix-cache:start -->
| Flag | Meaning |
| --- | --- |
| `--prefix-cache`, `--no-prefix-cache` | enable/disable DFlash prefix snapshots |
| `--prefix-cache-max-entries INT` | L1 snapshot entry budget |
| `--prefix-cache-max-bytes BYTES` | L1 snapshot byte budget |
| `--max-snapshot-tokens INT` | snapshot insert token cap; `0` disables the cap |
| `--prefix-cache-l2`, `--no-prefix-cache-l2` | enable/disable SSD L2 for evicted snapshots |
| `--prefix-cache-l2-dir PATH` | L2 root directory |
| `--prefix-cache-l2-max-bytes BYTES` | L2 disk budget |
<!-- dflash-runtime-config:prefix-cache:end -->

Notes:

- `target_fa_window > 0` disables prefix cache reuse because snapshot cache shape
  differs from full-KV verification.
- `max_snapshot_tokens` skips oversized inserts when L2 is absent. With L2
  enabled, oversized snapshots may still be accepted and later spilled by budget.
- L2 helps revisits after L1 eviction. It does not reduce the active request KV
  bucket.

Upstream prompt cache:

| Flag | Meaning |
| --- | --- |
| `--prompt-cache-size INT` | upstream `mlx_lm` prompt-cache entry count |

Diagnostics:

| Flag | Meaning |
| --- | --- |
| `--diagnostics {off,basic,full}` | structured diagnostics mode |
| `--diagnostics-dir PATH` | output directory |

Metal limits:

| Flag | Meaning |
| --- | --- |
| `--wired-limit auto\|none\|BYTES` | MLX wired memory limit policy |
| `--cache-limit auto\|none\|BYTES` | MLX cache limit policy |

## `dflash generate`

```bash
dflash generate --model Qwen/Qwen3.5-4B --prompt "Explain DFlash."
```

Flags:

| Flag | Meaning |
| --- | --- |
| `--model REF_OR_PATH` | target model, required |
| `--prompt TEXT` | prompt, required |
| `--max-tokens INT` | generated token cap |
| `--no-chat-template` | use raw prompt text |
| `--draft REF_OR_PATH` | draft override |
| `--draft-quant SPEC` | optional in-memory draft quantization, e.g. `w4:gs64` |

Runtime override flags:

<!-- dflash-runtime-config:generate-runtime:start -->
| Flag | Meaning |
| --- | --- |
| `--verify-mode {auto,off}` | verifier path mode; `off` is debug/parity only |
| `--prefill-step-size INT` | target prefill chunk size |
| `--target-fa-window INT` | experimental target FA rotating window; `0` means full KV |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward, `0` means block size |
<!-- dflash-runtime-config:generate-runtime:end -->

Generate disables cross-request prefix caching. Use it for local sanity checks, not for
benchmark claims.

## `dflash benchmark`

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

`dflash benchmark` measures runtime behavior only. `humaneval`, `gsm8k`, and
`math500` load real Hugging Face datasets via the optional `datasets` package,
but they are not official accuracy evaluations. `smoke` is a local CLI sanity
check only; `longctx` is local/offline synthetic context stress. Use
`--prompt-file PATH` for offline/custom prompt JSONL.

Flags:

| Flag | Meaning |
| --- | --- |
| `--suite {smoke,humaneval,gsm8k,math500,longctx}` | named runtime prompt suite |
| `--limit N` | deterministic prompt count limit |
| `--ctx-tokens N` | synthetic context target for `longctx` |
| `--prompt-file PATH` | JSONL prompt override with `id`, `suite`, `prompt` rows |
| `--shuffle` | shuffle HF dataset rows before applying `--limit` |
| `--seed INT` | shuffle seed used only with `--shuffle` |
| `--prompt TEXT` | prompt text |
| `--ctx INT` | existing shorthand for `--ctx-tokens` |
| `--max-tokens INT` | generated token count |
| `--block-tokens INT` | DFlash verify block size |
| `--repeat INT` | measured runs |
| `--cooldown SECONDS` | sleep between runs |
| `--model REF_OR_PATH` | target model |
| `--draft REF_OR_PATH` | draft override |
| `--no-chat-template` | raw prompt mode |
| `--draft-quant SPEC` | optional in-memory draft quantization, e.g. `w4:gs64` |
| `--no-eos` | suppress EOS so generation reaches token cap |
| `--split-sdpa`, `--no-split-sdpa` | target split-SDPA verifier path; enabled by default |

Runtime override flags:

<!-- dflash-runtime-config:benchmark-runtime:start -->
| Flag | Meaning |
| --- | --- |
| `--target-fa-window INT` | experimental target FA rotating window; `0` means full KV |
| `--draft-sink-size INT` | draft cache sink tokens |
| `--draft-window-size INT` | draft cache rolling window tokens |
| `--verify-len-cap INT` | max tokens per verify forward, `0` means block size |
<!-- dflash-runtime-config:benchmark-runtime:end -->

Benchmark output flags:

| Flag | Meaning |
| --- | --- |
| `--no-memory` | omit memory medians from summary |
| `--out PATH` | artifact directory |

New benchmark outputs default to `.artifacts/dflash/benchmarks/...`.

## `dflash doctor`

Doctor accepts the same runtime config flags as the server for validation:

```bash
dflash doctor --profile low-memory
dflash doctor --profile long-session --prefix-cache-l2 --json
dflash doctor --model Qwen/Qwen3.5-4B --load-model
```

Common flags:

| Flag | Meaning |
| --- | --- |
| `--json` | machine-readable report |
| `--strict` | return non-zero on warnings |
| `--load-model` | actually load model/draft |
| `--model REF_OR_PATH` | target model for registry/load checks |
| `--draft REF_OR_PATH` | draft override |
| runtime flags | profile, prefill, draft window, prefix cache, L2, FA window, max ctx |

## Startup Env Vars

These are accepted as startup inputs for server/doctor config. CLI flags override
them.

<!-- dflash-runtime-config:env:start -->
| Env var | Matching config |
| --- | --- |
| `DFLASH_RUNTIME_PROFILE` | `--profile {balanced,fast,low-memory,long-session}` |
| `DFLASH_PREFILL_STEP_SIZE` | `--prefill-step-size INT` |
| `DFLASH_DRAFT_SINK_SIZE` | `--draft-sink-size INT` |
| `DFLASH_DRAFT_WINDOW_SIZE` | `--draft-window-size INT` |
| `DFLASH_VERIFY_LEN_CAP` | `--verify-len-cap INT` |
| `DFLASH_CLEAR_CACHE_BOUNDARIES` | `--clear-cache-boundaries`, `--no-clear-cache-boundaries` |
| `DFLASH_VERIFY_MODE` | `--verify-mode {auto,off}` |
| `DFLASH_MAX_SNAPSHOT_TOKENS` | `--max-snapshot-tokens INT` |
| `DFLASH_PREFIX_CACHE_L2_ENABLED` | `--prefix-cache-l2`, `--no-prefix-cache-l2` |
| `DFLASH_PREFIX_CACHE_L2_DIR` | `--prefix-cache-l2-dir PATH` |
| `DFLASH_PREFIX_CACHE_L2_MAX_BYTES` | `--prefix-cache-l2-max-bytes BYTES` |
| `DFLASH_PREFIX_CACHE` | `--prefix-cache`, `--no-prefix-cache` |
| `DFLASH_PREFIX_CACHE_MAX_ENTRIES` | `--prefix-cache-max-entries INT` |
| `DFLASH_PREFIX_CACHE_MAX_BYTES` | `--prefix-cache-max-bytes BYTES` |
| `DFLASH_TARGET_FA_WINDOW` | `--target-fa-window INT` |
| `DFLASH_MAX_CTX` | `--dflash-max-ctx INT` |
<!-- dflash-runtime-config:env:end -->

Prefer CLI flags for reproducible local runs. Env vars are mainly for Docker,
CI, and wrappers.

## Internal Debug Env Vars

These are not product flags. They are kept for kernel and verifier experiments:

- `DFLASH_VERIFY_LINEAR`
- `DFLASH_VERIFY_QMM`
- `DFLASH_VERIFY_VARIANT`
- `DFLASH_VERIFY_MAX_N`
- `DFLASH_VERIFY_QMM_KPARTS`
- `DFLASH_VERIFY_INCLUDE`

Do not use them for public benchmark claims.

## Validation Rules

The runtime rejects invalid config before serving:

- `prefill_step_size > 0`
- `draft_sink_size >= 0`
- `draft_window_size > 0`
- `verify_len_cap >= 0`
- `prefix_cache_max_entries > 0`
- `prefix_cache_max_bytes >= 0`
- `max_snapshot_tokens >= 0`
- `prefix_cache_l2_dir` must be non-empty when L2 is enabled
- `prefix_cache_l2_max_bytes >= 0`
- `target_fa_window >= 0`
- `dflash_max_ctx >= 0`
- `verify_mode in {auto, off}`

Use `dflash doctor --json` to see the resolved effective config.
