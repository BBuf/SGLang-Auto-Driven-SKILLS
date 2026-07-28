# Compatibility Proof and Bounded Search

## Contents

- [Evidence order](#evidence-order)
- [Compatibility record](#compatibility-record)
- [Algorithm checks](#algorithm-checks)
- [Search order](#search-order)
- [Stop conditions](#stop-conditions)
- [SGLang v0.5.16 example evidence](#sglang-v0516-example-evidence)

## Evidence order

Prove compatibility against the **selected SGLang revision** in this order:

1. `python -m sglang.launch_server --help` from the installed environment.
2. The exact release/tag and its upgrade or known-issue notes.
3. `server_args.py`, speculative algorithm enums, model runner, attention
   backend, and tests at the exact commit.
4. The model's immutable config and draft/MTP checkpoint contents.
5. Official cookbook commands scoped to the same release.

Current main-branch documentation can suggest what to inspect, but it does not
prove an older image supports a flag or combination.

Record the command or URL beside each conclusion. Mark a candidate `unknown`
and exclude it when evidence conflicts or remains incomplete.

## Compatibility record

Capture at least:

```text
candidate_id:
algorithm:
sglang_revision:
algorithm_symbol_or_cli_source:
target_model_revision:
draft_model_or_native_mtp_evidence:
attention_backend:
verify_mode:
quantization:
tp_pp_dp_ep_cp_pd:
cuda_graph_mode:
required_environment:
known_restrictions:
decision: include | exclude
reason:
```

Check both startup compatibility and semantic compatibility. A server that
accepts flags may still use an incompatible draft tokenizer, head, hidden size,
MTP layer layout, state cache, or verify backend.

## Algorithm checks

For every algorithm family, verify:

- the algorithm value exists in the selected CLI/source;
- a required draft checkpoint exists and matches the target model;
- native MTP weights and layer counts are present when claimed;
- target and draft tokenizers/vocabularies meet upstream requirements;
- attention prefill/decode/verify backends support the path;
- TP, PP, DP attention, EP, CP, PD disaggregation, and overlap scheduling are
  supported in the exact combination;
- quantization supports both target and draft paths;
- CUDA Graph capture supports the chosen steps, top-k, and draft-token shapes;
- architecture-specific recurrent or linear-attention state restrictions are
  satisfied;
- required environment variables are recorded with the candidate command.

Treat an upstream assertion scoped to one model or GPU as scoped evidence, not
a universal capability.

## Search order

Use a bounded, staged search:

| Stage | Change | Keep fixed | Advance when |
| --- | --- | --- | --- |
| Baseline | No speculative decoding | Everything else | Healthy and correct |
| Family | One proven algorithm/draft source | Workload and serving topology | At least one safe improvement |
| Shape | Steps/window/block size | Algorithm and backend | Stable gain exceeds noise |
| Width | Top-k/tree/draft tokens | Best safe shape | Correctness and memory pass |
| Backend | Proven compatible verify/attention mode | Best safe algorithm parameters | Gain survives clean restart |
| Combine | Previously winning choices only | Experiment contract | Revalidation passes |

Set explicit ceilings for:

- total candidates;
- candidates per family;
- startup failures;
- correctness failures;
- repeated regressions;
- wall-clock time;
- disk usage for artifacts.

When an interface offers an automatic parameter mode, benchmark that mode as
one candidate. Do not mix partially explicit flag groups when upstream requires
all related values to be either automatic or explicit.

## Stop conditions

Reject a candidate immediately for:

- worker crash, hang, or unhealthy endpoint;
- correctness mismatch;
- determinism failure when determinism is required;
- incompatible tokenizer/draft/MTP state;
- non-finite measurement;
- memory cap or hard SLA failure.

Stop exploring an algorithm family after the configured consecutive failure
budget or after multiple candidates are clearly dominated outside the noise
threshold. Preserve its logs and reason.

Return `no_safe_improvement` when:

- all speculative candidates are rejected;
- no candidate beats the baseline by the declared minimum improvement;
- the apparent gain falls inside measurement noise;
- compatibility remains unproven.

## SGLang v0.5.16 example evidence

Use these only as an example of source collection. Re-resolve them for the
requested version:

- [v0.5.16 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.16)
  documents DSpark, ReplaySSM Ring Spec-Verify, backend changes, and known
  issues.
- [Speculative decoding documentation](https://docs.sglang.io/docs/advanced_features/speculative_decoding)
  describes EAGLE/EAGLE3, DFlash, standalone, NGRAM, MTP-related parameters,
  and tuning groups.
- [DSpark implementation PR #30261](https://github.com/sgl-project/sglang/pull/30261)
  is direct evidence for the confidence-scheduled algorithm.
- [ReplaySSM PR #28695](https://github.com/sgl-project/sglang/pull/28695)
  documents architecture-specific state and top-k restrictions.
- [v0.5.16 tag source](https://github.com/sgl-project/sglang/tree/v0.5.16)
  is the immutable implementation snapshot.

For DSpark in v0.5.16, the release notes name
`--speculative-algorithm DSPARK`,
`--speculative-dspark-block-size`, and
`SGLANG_RAGGED_VERIFY_MODE=compact`. Verify those names again for any other
revision.
