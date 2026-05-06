# Benchmark Legacy Evidence

`benchmark/` is frozen historical evidence.

Do not add new benchmark runs, JSON outputs, trace directories, or generated
artifacts here.

Use the product benchmark instead:

```bash
PROMPT='The function $f$ satisfies the functional equation \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such integers, separated by commas. Please reason step by step, and put your final answer within \boxed{}.'

dflash benchmark \
  --model <target> \
  --prompt "$PROMPT" \
  --max-tokens 1024 \
  --repeat 3 \
  --cooldown 60 \
  --no-eos
```

New public benchmark outputs go under:

```text
.artifacts/dflash/benchmarks/
```

New trace and diagnostic outputs go under:

```text
.artifacts/dflash/traces/
.artifacts/dflash/diagnostics/
```

`benchmark/results/` remains in place for old comparisons only. Do not delete,
move, or append to it during normal development.
