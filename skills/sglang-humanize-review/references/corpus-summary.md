# SGLang Human Review Corpus Summary

- Repo: `sgl-project/sglang`
- Source PR years: `2024` to `2026` inclusive
- Collected through (inclusive): `2026-07-27T15:49:55.233470+00:00`
- Generated at: `2026-07-27T18:06:12+00:00`
- Corpus file: `sglang-review-corpus.jsonl.gz`
- Threads: `40110`
- Comments in corpus: `93774`
- Human reviewer comments: `93774`
- Agent reviewer comments: `0`

## Collection Policy

- Pull requests are selected by PR `created_at` in the requested year range.
- Review/comment events are also capped at the requested end date.
- Pull requests authored by GitHub bots or obvious coding-agent accounts are excluded.
- Inline review comments are GitHub pull-review comments grouped by thread.
- Top-level PR conversation comments are included from GitHub PR comments and grouped by PR.
- Review submission bodies are included for COMMENT / REQUEST_CHANGES style review summaries; empty approvals are skipped.
- Comment bodies are kept in their original language; the corpus does not translate or drop non-English text.
- `diff_hunk` stores the code context that produced each review thread.

## Pull Request Stats

| Metric | Count |
| --- | ---: |
| `excluded_agent_prs` | 19 |
| `included_human_prs` | 25404 |
| `included_human_prs_2024` | 1772 |
| `included_human_prs_2025` | 9800 |
| `included_human_prs_2026` | 13832 |
| `seen_after_window` | 1 |
| `window_prs` | 25423 |

## Comment Stats

| Metric | Count |
| --- | ---: |
| `conversation_agent_pr_conversation_comments_on_target_prs` | 20757 |
| `conversation_all_pr_conversation_comments_seen` | 73251 |
| `conversation_human_pr_conversation_comments_on_target_prs` | 52482 |
| `conversation_pr_conversation_comments_after_window_skipped` | 12 |
| `conversation_pr_conversation_paginated_prs` | 8 |
| `conversation_threads` | 14287 |
| `inline_agent_reviewer_comments_on_target_prs` | 24121 |
| `inline_all_review_comments_seen` | 60039 |
| `inline_human_reviewer_comments_on_target_prs` | 35895 |
| `inline_threads` | 21941 |
| `submission_agent_review_submissions_on_target_prs` | 15279 |
| `submission_all_review_submissions_seen` | 55402 |
| `submission_empty_review_submissions_skipped` | 34715 |
| `submission_human_review_submissions_on_target_prs` | 5397 |
| `submission_review_submission_paginated_prs` | 4 |
| `submission_review_submissions_after_window_skipped` | 11 |
| `submission_threads` | 3882 |

## Episode Types

| Type | Threads |
| --- | ---: |
| `inline_review_thread` | 21941 |
| `pr_conversation` | 14287 |
| `review_submission` | 3882 |

## Event Kinds

| Kind | Events |
| --- | ---: |
| `pr_conversation` | 52482 |
| `inline_review_comment` | 35895 |
| `review_submission` | 5397 |

## Review States

| State | Events |
| --- | ---: |
| `APPROVED` | 2335 |
| `COMMENTED` | 1544 |
| `CHANGES_REQUESTED` | 1503 |
| `DISMISSED` | 15 |

## Top Categories

| Category | Threads |
| --- | ---: |
| `tests-ci` | 16921 |
| `models-quant` | 16737 |
| `correctness` | 16573 |
| `gpu-kernel` | 10938 |
| `api-compat` | 9034 |
| `performance` | 8623 |
| `memory-cache` | 8593 |
| `docs-examples` | 7270 |
| `style-maintainability` | 6562 |
| `distributed-concurrency` | 6335 |
| `build-deps` | 5436 |
| `observability` | 4683 |
| `general-review` | 2070 |

## Code Languages

| Language | Threads |
| --- | ---: |
| `python` | 17834 |
| `conversation` | 14287 |
| `review` | 3882 |
| `markdown` | 1049 |
| `rust` | 590 |
| `cuda` | 564 |
| `yaml` | 316 |
| `cpp` | 297 |
| `mdx` | 289 |
| `notebook` | 186 |
| `toml` | 186 |
| `shell` | 113 |
| `jsx` | 100 |
| `extensionless` | 93 |
| `dockerfile` | 70 |
| `text` | 60 |
| `unknown` | 32 |
| `mu` | 27 |
| `json` | 24 |
| `rst` | 17 |
| `proto` | 17 |
| `rocm` | 10 |
| `gb200` | 10 |
| `metal` | 9 |
| `jinja` | 5 |
| `npu` | 5 |
| `jpg` | 4 |
| `png` | 4 |
| `jpeg` | 3 |
| `hip` | 3 |
| `sycl` | 2 |
| `xeon` | 2 |
| `musa` | 2 |
| `sample` | 2 |
| `cmake` | 2 |
| `muh` | 2 |
| `gif` | 1 |
| `mp4` | 1 |
| `jsonconfig` | 1 |
| `po` | 1 |
| `dev` | 1 |
| `in` | 1 |
| `sagemaker` | 1 |
| `blackwell` | 1 |
| `router` | 1 |
| `py‎` | 1 |
| `jinja2` | 1 |
| `patch` | 1 |

## Comment Language Hints

| Hint | Comments |
| --- | ---: |
| `en_or_ascii` | 85794 |
| `non_ascii_other` | 7642 |
| `zh_or_cjk` | 337 |
| `ja` | 1 |

## Top Paths

| Path | Threads |
| --- | ---: |
| `<conversation>` | 18169 |
| `python/sglang/srt/server_args.py` | 681 |
| `python/sglang/srt/managers/scheduler.py` | 355 |
| `python/sglang/srt/models/deepseek_v2.py` | 333 |
| `python/sglang/srt/model_executor/model_runner.py` | 319 |
| `python/sglang/srt/managers/schedule_batch.py` | 231 |
| `python/sglang/srt/managers/tokenizer_manager.py` | 172 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | 136 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | 135 |
| `python/sglang/srt/mem_cache/memory_pool.py` | 131 |
| `python/sglang/srt/managers/io_struct.py` | 127 |
| `python/pyproject.toml` | 123 |
| `python/sglang/srt/layers/quantization/modelopt_quant.py` | 116 |
| `python/sglang/srt/entrypoints/http_server.py` | 115 |
| `python/sglang/srt/disaggregation/decode.py` | 114 |
| `python/sglang/srt/utils.py` | 109 |
| `docs/backend/server_arguments.md` | 106 |
| `python/sglang/srt/layers/moe/ep_moe/layer.py` | 106 |
| `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` | 99 |
| `python/sglang/srt/layers/moe/topk.py` | 96 |
| `python/sglang/srt/entrypoints/engine.py` | 95 |
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | 93 |
| `python/sglang/srt/mem_cache/hiradix_cache.py` | 93 |
| `python/sglang/srt/managers/cache_controller.py` | 93 |
| `python/sglang/srt/utils/common.py` | 92 |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | 90 |
| `python/sglang/srt/disaggregation/encode_server.py` | 88 |
| `python/sglang/multimodal_gen/runtime/managers/gpu_worker.py` | 88 |
| `python/sglang/srt/layers/quantization/fp8.py` | 87 |
| `test/srt/run_suite.py` | 84 |

## Top Human Reviewers

| Reviewer | Comments |
| --- | ---: |
| `Fridge003` | 3594 |
| `ShangmingCai` | 2926 |
| `BBuf` | 2895 |
| `mickqian` | 2861 |
| `merrymercy` | 2835 |
| `zhyncs` | 2695 |
| `hnyls2002` | 2585 |
| `JustinTong0323` | 2577 |
| `fzyzcjy` | 2493 |
| `zhaochenyang20` | 2276 |
| `yuan-luo` | 1722 |
| `ispobock` | 1446 |
| `yhyang201` | 1433 |
| `HaiShaw` | 1410 |
| `b8zhong` | 1406 |
| `ch-wan` | 1203 |
| `mingfeima` | 1085 |
| `ping1jing2` | 1027 |
| `alisonshao` | 959 |
| `CatherineSue` | 888 |
| `hzh0425` | 888 |
| `yizhang2077` | 764 |
| `DarkSharpness` | 725 |
| `Qiaolin-Yu` | 708 |
| `kpham-sgl` | 682 |
| `ishandhanani` | 673 |
| `alexnails` | 650 |
| `Kangyan-Zhou` | 552 |
| `yeahdongcn` | 549 |
| `nvpohanh` | 544 |

## Query Examples

```bash
python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py --query cuda --limit 5
python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py --path python/sglang/srt --category correctness --limit 8
python3 skills/sglang-humanize-review/scripts/query_sglang_review_corpus.py --query 'server_args' --format jsonl --limit 3
```
