# DFlash Benchmark

| suite | prompts | prompt tok avg | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | TTFT | peak memory | acceptance | prefix saved | baseline prefill tok/s | dflash prefill physical tok/s | dflash prefill apparent tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 102.00 | 21.50 | 60.77 | 2.78x | n/a | n/a | 289.88 ms | 17.82 GB | 0.84 | n/a | 113.65 | 353.60 | 353.60 |

- mode: smoke
- suite: smoke
- model: mlx-community/Qwen3.6-27B-4bit
- draft: z-lab/Qwen3.6-27B-DFlash
- draft_quant: w4
- git_hash: e962ea7
- max_tokens: 16384
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
| smoke-default | 102 | 21.50 | 60.77 | 2.78x | n/a | n/a | 0.84 |
