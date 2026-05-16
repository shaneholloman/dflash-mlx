# Qwen3.6 27B 4-bit README Benchmark

Full benchmark artifacts for the README smoke prompt on Apple M5 Max.

Protocol:
- Target: `mlx-community/Qwen3.6-27B-4bit`
- Draft: `z-lab/Qwen3.6-27B-DFlash`
- Draft quant: `w4`
- Verify mode: `adaptive`
- Prompt: default README smoke prompt
- Tokenization: chat template, 102 prompt tokens
- Repeats: 3
- Cooldown: 120s
- EOS: disabled

Summary:

| Max tokens | Baseline tok/s | DFlash tok/s | Speedup | Acceptance |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 33.26 | 98.05 | 2.95x | 84.67% |
| 2048 | 32.34 | 90.67 | 2.81x | 84.62% |
| 4096 | 30.58 | 93.55 | 3.06x | 87.04% |
| 8192 | 26.03 | 79.12 | 3.04x | 83.45% |
| 16384 | 21.50 | 60.77 | 2.78x | 84.40% |

Directory layout:
- `tokens_1024/`
- `tokens_2048/`
- `tokens_4096/`
- `tokens_8192/`
- `tokens_16384/`

Each token directory contains the original benchmark artifact files:
`invocation.json`, `manifest.json`, `results.json`, `runs.jsonl`,
`summary.json`, and `summary.md`.
