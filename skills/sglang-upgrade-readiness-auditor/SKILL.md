---
name: sglang-upgrade-readiness-auditor
description: "Audit an SGLang deployment before changing releases, commits, or images by matching official release, CLI, dependency, default-behavior, known-issue, API, and internal-import changes to actual launch profiles. Use for GO/CONDITIONAL_GO/NO_GO decisions, proposed argv or import migrations, upgrade canaries, rollback plans, and version-specific compatibility reviews; especially when commands use quantization, speculative decoding, PD/DP/EP/CP, HiCache, custom kernels, diffusion rollout, or internal SGLang imports."
---

# SGLang Upgrade Readiness Auditor

## Objective

Turn a version diff into a deployment-specific decision. Inventory real launch
profiles, attach every finding to authoritative evidence, propose reviewable
argv/import changes, require relevant canaries, and issue `GO`,
`CONDITIONAL_GO`, or `NO_GO`.

Keep audit mode **read-only**. Never execute rendered commands, source shell
files, pull images, edit manifests, restart servers, or perform a rollout.
Canary execution and deployment changes require separate authorization after
the report.

## Required Inventory

Resolve before deciding:

| Field | Required detail |
| --- | --- |
| Current | Immutable tag/commit or image digest |
| Target | Immutable tag/commit or image digest |
| Launch surface | Exact argv arrays and relevant environment variables |
| Model | ID/revision, architecture family, quantization |
| Hardware | GPU model/count and topology |
| Parallelism | TP, PP, DP, EP, CP, PD/disaggregation |
| Features | Cache, graph, logprob, speculative, LoRA, router, rollout, custom extension paths |
| Guarantees | Correctness, determinism, API, availability, and performance requirements |
| Integrations | Routers, KV transfer, observability, RL/diffusion clients, private extensions |

If either version or the actual launch surface is unknown, return an incomplete
audit. Do not replace missing deployment facts with target-release defaults.

## Workflow

### 1. Freeze current behavior

Record:

- current command/environment and immutable revisions;
- a healthy current-version startup log;
- correctness and API smoke outputs;
- determinism evidence when required;
- representative performance and memory baseline;
- current rollback command/image;
- every in-scope deployment profile.

Represent commands as argv arrays. Do not evaluate shell strings or command
substitutions.

### 2. Resolve the version interval

Resolve mutable branch/image names to immutable commits or digests. Confirm the
target is newer than the current version. Record exact release and compare URLs.

Read [evidence-and-rule-authoring.md](references/evidence-and-rule-authoring.md).
Inspect official Releases, tag-to-tag compare/diffs, CLI help at both revisions,
dependency metadata, source, cookbooks, and known issues. Release-note absence
is not proof of compatibility.

### 3. Classify changes

Classify only deployment-relevant evidence:

- removed or renamed interface;
- required dependency/backend change;
- default behavior change;
- API/transport/schema change;
- internal import relocation;
- known issue or reverted fix;
- informational change.

Scope model-, hardware-, backend-, and topology-specific evidence precisely.
Do not generalize one configuration's issue to all deployments.

### 4. Author profiles and rules

Read [profile-and-rule-schema.md](references/profile-and-rule-schema.md). Write:

1. one profile document containing the version interval and all deployment
   profiles;
2. one evidence-backed rule document for the interval.

Attach a direct source URL, applicability mode/version, predicates, severity,
canaries, and rollback to every rule. Add a transformation only when the source
supports a safe exact rewrite.

### 5. Run the auditor

```bash
python3 skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py \
  --profiles /path/to/deployment-profiles.json \
  --rules /path/to/upgrade-rules.json \
  --output-markdown /path/to/upgrade-readiness.md \
  --output-json /path/to/upgrade-readiness.json
```

The analyzer validates the inputs, applies version and profile predicates,
detects conflicting transformations, proposes argv/import changes, derives
canaries, and aggregates verdicts. It never executes input or output commands.

### 6. Review every proposed change

For each matched profile:

- compare original and proposed argv token by token;
- inspect import-prefix changes at symbol level;
- reject a rewrite if flag arity or module mapping is ambiguous;
- keep findings without transformations as manual actions;
- resolve coverage gaps before calling the audit complete.

Treat proposals as patches for review, not as deployed configuration.

### 7. Run separately authorized canaries

After approval to test the target version, run only in an isolated canary
environment. Derive tests from matched rules plus the base contract:

- startup and worker health;
- model-output and logprob parity;
- temperature-zero determinism where required;
- API/streaming/tool/reasoning schemas;
- long-context and prefix-cache behavior;
- TP/DP/EP/CP/PD failure, abort, and retry paths;
- custom import/extension smoke tests;
- representative TTFT, TPOT, throughput, and peak memory.

Write `pass`, `fail`, or `not_run` results back to the profile and rerun the
auditor. Do not mark an unexecuted canary as passing.

### 8. Issue the decision

Use these meanings:

| Verdict | Meaning |
| --- | --- |
| `GO` | No blocking/conditional finding remains and all required canaries pass |
| `CONDITIONAL_GO` | Required changes, behavior/risk findings, or incomplete canaries remain |
| `NO_GO` | An unresolved blocker or unsafe/conflicting transformation remains |

Report per-profile and overall verdicts, findings with sources, proposed
commands/imports, coverage gaps, canaries, and rollback triggers.

## Handoffs

- Use `sglang-prod-incident-triage` when the current or target canary has already
  become an incident requiring replay-first diagnosis.
- Use `llm-serving-auto-benchmark` when the remaining question is a fair
  performance search rather than compatibility.
- Use `sglang-sota-humanize-loop` only when the authorized outcome requires an
  SGLang source change.

Preserve this audit's version evidence and profile artifacts during handoff.

## Safety Rules

- Keep audit mode read-only.
- Never execute a rendered command.
- Never source or interpolate profile input.
- Never modify production config or deployment state without new authority.
- Never declare compatibility from a release title alone.
- Never drop an unmatched high-risk feature silently; report it as coverage.
- Never propose a rewrite when flag arity, value semantics, or import mapping is
  ambiguous.
- Sanitize registry credentials, tokens, private image names, hosts, and paths
  from committed examples.

## Demonstration

Run the committed v0.5.15-to-v0.5.16 example:

```bash
python3 skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py \
  --profiles skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-profiles.json \
  --rules skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-rules.json \
  --output-markdown /tmp/v0516-upgrade-fixture.md \
  --output-json /tmp/v0516-upgrade-fixture.json
```

The report starts with **SYNTHETIC FIXTURE**. It demonstrates one `GO`, one
`CONDITIONAL_GO` with argv/import rewrites, and one `NO_GO` determinism blocker.
It is not evidence that a real deployment passed v0.5.16 canaries.

## Resources

- [evidence-and-rule-authoring.md](references/evidence-and-rule-authoring.md):
  read for every new version interval.
- [profile-and-rule-schema.md](references/profile-and-rule-schema.md): read
  before creating analyzer input or integrating deployment inventory.
- `scripts/audit_upgrade.py`: deterministic matching, rewrite, verdict, and
  reporting engine.
- `examples/fixture-report.md`: expected demonstration output.
