# DFlash Benchmark

| suite | prompts | prompt tok avg | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | TTFT | peak memory | acceptance | prefix saved | baseline prefill tok/s | dflash prefill physical tok/s | dflash prefill apparent tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 102.00 | 32.34 | 90.67 | 2.81x | n/a | n/a | 212.36 ms | 16.76 GB | 0.85 | n/a | 137.89 | 483.59 | 483.59 |

- mode: smoke
- suite: smoke
- model: mlx-community/Qwen3.6-27B-4bit
- draft: z-lab/Qwen3.6-27B-DFlash
- draft_quant: w4
- git_hash: e962ea7
- max_tokens: 2048
- block_tokens: 16
- repeat: 3
- cooldown: 120
- prompt_count: 1
- prompt_ids: smoke-default
- prompt_source: smoke
- prompt_tokenization_mode: chat_template
- use_chat_template: True
- target_fa_window: 0
- draft_window: 64+1024
- verify_len_cap: 0
- verify_mode: adaptive
- only_dflash: False

## Per Prompt

| prompt id | prompt tokens | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke-default | 102 | 32.34 | 90.67 | 2.81x | n/a | n/a | 0.85 |
