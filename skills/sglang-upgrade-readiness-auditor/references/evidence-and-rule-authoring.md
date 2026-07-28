# Upgrade Evidence and Rule Authoring

## Contents

- [Source priority](#source-priority)
- [Build the change inventory](#build-the-change-inventory)
- [Applicability modes](#applicability-modes)
- [Classify severity](#classify-severity)
- [Write safe matches and transforms](#write-safe-matches-and-transforms)
- [Derive canaries and rollback](#derive-canaries-and-rollback)
- [Unknowns and confidence](#unknowns-and-confidence)
- [v0.5.16 example sources](#v0516-example-sources)

## Source priority

Use this order:

1. Official GitHub Release and explicit upgrade notes for the target.
2. Exact current-to-target GitHub compare and merged PR diffs.
3. CLI help and dependency metadata from both immutable revisions.
4. Source, tests, and official cookbooks at the target tag.
5. Open issues or reverted PRs, labeled as known-risk context rather than
   landed behavior.

Record direct URLs. Prefer an implementation PR or exact tag path over a
secondary summary. If a Release statement and source disagree, preserve the
conflict and withhold a `GO`.

Useful commands:

```bash
gh api repos/sgl-project/sglang/releases/tags/<target-tag>
gh api repos/sgl-project/sglang/compare/<current-tag>...<target-tag>
git diff <current-tag>...<target-tag> -- python/sglang/srt/server_args.py
```

Run CLI help from the exact installed images when available. Do not use a
main-branch CLI to prove an older release.

## Build the change inventory

For every relevant change, record:

```text
title:
category:
introduced_in:
fixed_in:
source:
affected flag/env/import/model/backend/topology/integration:
behavior before:
behavior after:
mechanical rewrite possible:
required canaries:
rollback:
confidence:
```

Search for:

- deleted/renamed arguments and environment variables;
- removed quantization, GEMM, attention, collective, or KV backends;
- new defaults and default-off/default-on changes;
- dependency pins and required extras;
- Python namespace/internal import moves;
- HTTP/gRPC/streaming/serialization schema changes;
- model-, quantization-, hardware-, and topology-specific known issues;
- reverted fixes still absent from the target;
- router, PD, KV-transfer, diffusion, RL, LoRA, and observability integrations.

Ignore unrelated models/features only after confirming no shared code or
default surface affects the profile.

## Applicability modes

Use `crossing` for migrations that matter when the interval crosses their
introduction:

```text
current < introduced_in <= target
```

Examples: a flag rename, removed backend, new default, or import relocation.

Use `target` for a condition that remains present in every affected target until
fixed:

```text
introduced_in <= target < fixed_in
```

Examples: a known issue or dependency incompatibility. Omit `fixed_in` while the
condition remains unresolved.

Do not use a `crossing` rule to represent a persistent known issue; an upgrade
from one already-affected version to another would incorrectly hide it.

## Classify severity

| Severity | Use when | Typical verdict |
| --- | --- | --- |
| `blocker` | Required guarantee cannot be met or no safe migration is known | `NO_GO` |
| `required` | Command/import/client must change before rollout | `CONDITIONAL_GO` |
| `behavior` | Default/semantics changed and regression canaries are required | `CONDITIONAL_GO` |
| `risk` | Target has a scoped known risk needing explicit acceptance/canary | `CONDITIONAL_GO` |
| `dependency` | Package/driver/library lockstep is required | `CONDITIONAL_GO` |
| `info` | Relevant but no action/canary remains | May remain `GO` |

Severity is deployment-specific. A known nondeterminism is a blocker only when
that profile requires determinism and matches the affected configuration.

## Write safe matches and transforms

Match the narrowest concrete profile surface. Prefer an `argv_value` predicate
over a broad model-family predicate when a removed value is the real trigger.
Combine predicates with `all` for configuration-specific issues.

Add transforms only for exact, reviewable changes:

- `rename_flag`;
- `remove_flag` with explicit arity 0 or 1;
- `replace_value` with an expected old value;
- `replace_import_prefix` when the source establishes a prefix-preserving move.

Do not encode:

- arbitrary shell;
- regex replacement over a command string;
- image pulls, restarts, or deployment actions;
- guessed flag arity;
- a backend substitution whose correctness/performance behavior is unknown;
- a Python import mapping that changes symbols or module layout unpredictably.

Keep the finding without a transform when mechanical migration is unsafe.

## Derive canaries and rollback

Map the changed surface to a falsifiable target test:

| Change | Minimum canary |
| --- | --- |
| Flag/backend | Startup, correctness, representative performance |
| Default cache behavior | Long context, prefix hit, eviction, output parity |
| Determinism issue | Repeated identical temperature-zero requests |
| PD/queue behavior | Retry, abort, parked/inflight request recovery |
| Logprob behavior | Token/logprob parity and peak memory |
| API transport | Client/server schema and tensor round trip |
| Internal import | Import smoke plus extension-level correctness |
| Dependency | Import/version check plus affected execution path |

Write a rollback that restores a known working image/configuration or disables
the affected feature. Do not use “monitor closely” as a rollback.

## Unknowns and confidence

Treat these as audit gaps:

- mutable current/target versions;
- a launch profile reconstructed from memory;
- undocumented private extension imports;
- an unmatched high-risk feature;
- a Release claim without source/CLI confirmation;
- a target canary not run on the stated hardware/topology.

Unknown does not mean safe. Keep the verdict conditional or blocked until the
gap is resolved.

## v0.5.16 example sources

The committed fixture uses direct SGLang sources:

- [v0.5.16 Release](https://github.com/sgl-project/sglang/releases/tag/v0.5.16)
- [FP4 backend removal PR #30448](https://github.com/sgl-project/sglang/pull/30448)
- [Waterfill rename PR #27350](https://github.com/sgl-project/sglang/pull/27350)
- [Optimistic prefill rename PR #30951](https://github.com/sgl-project/sglang/pull/30951)
- [UnifiedRadixTree default PR #30468](https://github.com/sgl-project/sglang/pull/30468)
- [Chunked input-logprob default PR #31498](https://github.com/sgl-project/sglang/pull/31498)
- [Determinism guard PR #31125](https://github.com/sgl-project/sglang/pull/31125)
- [Diffusion msgpack transport PR #31565](https://github.com/sgl-project/sglang/pull/31565)

These rules demonstrate authoring for one interval. Rebuild evidence for every
new target instead of treating the fixture as a permanent release database.
