# SGLang Upgrade Readiness Audit

> **SYNTHETIC FIXTURE:** Deployment profiles and canary results are invented for analyzer demonstration.

- Current: `v0.5.15`
- Target: `v0.5.16`
- Overall verdict: **NO_GO**
- Safety: read-only analysis; no command below was executed.

## Profile Verdicts

| Profile | Verdict | Findings | Proposed changes | Missing/failing canaries | Coverage gaps |
| --- | --- | ---: | --- | --- | --- |
| `deterministic-dp-graph` | **NO_GO** | 1 | no | temperature_zero_determinism | none |
| `legacy-fp4-pd` | **CONDITIONAL_GO** | 6 | yes | import_smoke, input_logprob_parity, long_context, pd_retry_abort, peak_memory, prefix_cache | pd_disaggregation |
| `plain-tp` | **GO** | 0 | no | none | none |

## Findings

### `deterministic-dp-graph`

- **BLOCKER** `dp-breakable-graph-temperature-zero-nondeterminism` — Temperature-zero nondeterminism in an affected DP graph path. The v0.5.16 known issue says identical temperature-zero requests can diverge with DP attention and breakable prefill CUDA Graph. ([source](https://github.com/sgl-project/sglang/pull/31125))

### `legacy-fp4-pd`

- **REQUIRED** `fp4-cutlass-backend-removed` — CUTLASS FP4 backend value removed. The in-tree NVFP4 JIT path and cutlass value were removed; v0.5.16 release guidance says to use auto. ([source](https://github.com/sgl-project/sglang/pull/30448))
- **REQUIRED** `deepep-waterfill-flag-renamed` — Waterfill flag renamed without alias. --enable-deepep-waterfill became --enable-waterfill and the old spelling is rejected. ([source](https://github.com/sgl-project/sglang/pull/27350))
- **REQUIRED** `optimistic-prefill-attempts-rename` — Optimistic prefill retries flag renamed. --optimistic-prefill-retries became --optimistic-prefill-attempts with no deprecated alias. ([source](https://github.com/sgl-project/sglang/pull/30951))
- **BEHAVIOR** `unified-radix-tree-default` — UnifiedRadixTree becomes the DSA/Mamba/SWA default. The cache implementation changes by default for DSA, Mamba, and SWA architectures. ([source](https://github.com/sgl-project/sglang/pull/30468))
- **BEHAVIOR** `chunked-input-logprob-default` — Chunked input-logprob processing enabled by default. Input-logprob requests use chunked processing by default to cap peak memory. ([source](https://github.com/sgl-project/sglang/pull/31498))
- **REQUIRED** `kernel-namespace-relocation` — Internal kernels move to sglang.kernels. The release relocates internal kernel imports into the sglang.kernels namespace. ([source](https://github.com/sgl-project/sglang/releases/tag/v0.5.16))

## Proposed Commands

### `deterministic-dp-graph`

Original argv:

```bash
python3 -m sglang.launch_server --model-path fixture/dsv4-fp4 --tp 8 --dp 8 --enable-dp-attention --enable-piecewise-cuda-graph
```

Proposed argv:

```bash
python3 -m sglang.launch_server --model-path fixture/dsv4-fp4 --tp 8 --dp 8 --enable-dp-attention --enable-piecewise-cuda-graph
```

### `legacy-fp4-pd`

Original argv:

```bash
python3 -m sglang.launch_server --model-path fixture/dsa-moe-model --tp 8 --fp4-gemm-backend cutlass --enable-deepep-waterfill --optimistic-prefill-retries 3
```

Proposed argv:

```bash
python3 -m sglang.launch_server --model-path fixture/dsa-moe-model --tp 8 --fp4-gemm-backend auto --enable-waterfill --optimistic-prefill-attempts 3
```

Proposed imports:

```text
sglang.kernels.fast_op
torch
```

### `plain-tp`

Original argv:

```bash
python3 -m sglang.launch_server --model-path fixture/dense-model --tp 8
```

Proposed argv:

```bash
python3 -m sglang.launch_server --model-path fixture/dense-model --tp 8
```

## Canaries and Rollback

### `deterministic-dp-graph`

- Required canaries: correctness, performance, server_health, temperature_zero_determinism
- Missing/failing: temperature_zero_determinism
- Rollback: Do not roll out this combination; disable the affected graph path or restore the previous release.

### `legacy-fp4-pd`

- Required canaries: correctness, import_smoke, input_logprob_parity, long_context, pd_retry_abort, peak_memory, performance, prefix_cache, server_health
- Missing/failing: import_smoke, input_logprob_parity, long_context, pd_retry_abort, peak_memory, prefix_cache
- Rollback: Restore the old extension and image if the exact relocated wrapper cannot be imported or validated.
- Rollback: Restore the prior image and old flag if waterfill startup or throughput validation fails.
- Rollback: Restore the prior release if logprob parity or long-context behavior changes unexpectedly.
- Rollback: Restore the v0.5.15 image if the FlashInfer-selected auto backend fails correctness or performance canaries.
- Rollback: Restore v0.5.15 if PD retry, abort, or parked-request canaries fail.
- Rollback: Return to the prior release if cache-hit, long-context, or output-parity canaries regress.

### `plain-tp`

- Required canaries: correctness, performance, server_health
- Missing/failing: none
- Rollback: restore the recorded current version and argv.

## Interpretation

- `GO`: no blocking or conditional finding remains and all required canaries are recorded as passing.
- `CONDITIONAL_GO`: apply/review proposed changes or complete required canaries before rollout.
- `NO_GO`: resolve the blocker or choose a different target/configuration.
- Review every proposed argv. This auditor never executes it.
