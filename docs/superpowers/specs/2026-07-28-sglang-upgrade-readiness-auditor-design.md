# SGLang Upgrade Readiness Auditor Design

## Purpose

Add a version-aware skill that determines whether an existing SGLang deployment
can safely move from one revision or image to another. The skill turns upstream
release notes, compare diffs, CLI evidence, dependency changes, known issues,
and the deployment's actual launch surface into a concrete migration report.

The motivating example is SGLang v0.5.15 to v0.5.16, which includes removed
quantization paths, renamed flags without deprecated aliases, default behavior
changes, dependency updates, and configuration-specific known issues. The skill
must generalize to future version pairs and must not hard-code a claim that the
latest release is safe for every deployment.

## Success Criteria

The pull request is complete when:

1. `sglang-upgrade-readiness-auditor` has a clear trigger distinct from
   production incident replay and performance optimization.
2. The workflow records immutable current and target revisions, deployment
   commands, model/quantization/parallelism features, hardware, and integrations
   before making a decision.
3. Evidence is classified as blocker, required change, behavior change, known
   risk, dependency change, or informational.
4. A deterministic analyzer matches structured upgrade rules against one or
   more deployment profiles and emits `GO`, `CONDITIONAL_GO`, or `NO_GO`.
5. Suggested command rewrites are explicit, reviewable, shell-safe
   argument-level transformations; the skill never silently edits production
   configuration.
6. The final report includes matched evidence, migrated-command suggestions,
   canary tests, rollback triggers, and unresolved unknowns.
7. Unit tests and a v0.5.15-to-v0.5.16 fixture demo run without installing
   either SGLang version.
8. The focused branch is pushed and opened as its own draft pull request
   against `main`.

## Approaches Considered

### 1. Release-note summarizer

Summarize the releases between two tags and let the user decide what applies.
This is broadly useful but does not inspect the actual deployment, rewrite
commands, or produce a testable readiness verdict.

### 2. Evidence workflow plus deterministic deployment matcher

Use `SKILL.md` to collect authoritative version evidence and encode
deployment-relevant findings in a small structured rule file. Use a
standard-library analyzer to match those rules against deployment profiles,
apply safe token-level rewrites, and generate a report and canary plan.

This is the chosen approach. It separates human/agent interpretation of upstream
changes from deterministic application to concrete deployments.

### 3. Automated in-place upgrader

Change images, launch scripts, manifests, or Helm values and run a rollout.
That crosses an operational safety boundary, assumes one deployment system, and
would make the skill destructive. The auditor will instead produce proposed
changes and validation gates that another authorized workflow can apply.

## User Contract

Required inputs:

- current SGLang version, commit, or image;
- target SGLang version, commit, or image;
- at least one exact launch command or structured deployment profile;
- model identifier and quantization;
- GPU type/count and TP/PP/DP/EP/CP/PD topology.

Optional inputs:

- environment variables;
- Docker, Kubernetes, Slurm, or systemd configuration paths;
- internal Python imports or extension packages;
- frontend, router, KV-transfer, observability, RL, or diffusion integrations;
- required correctness, determinism, availability, and performance guarantees.

When the current version, target version, or deployment command is unknown, the
skill reports an incomplete audit rather than assuming defaults.

## Architecture and Component Boundaries

### `SKILL.md`

Owns the evidence and decision workflow:

1. inventory every in-scope deployment profile and freeze current behavior;
2. resolve current and target immutable revisions;
3. collect official releases, compare links/diffs, CLI help, dependency
   manifests, migration notes, known issues, and relevant source changes;
4. convert applicable findings into structured rules with direct evidence
   links and confidence;
5. run the deterministic analyzer against each deployment profile;
6. review suggested command transformations;
7. construct correctness, determinism, API, performance, long-context, and
   failure-recovery canaries from matched risks;
8. issue a readiness verdict with unresolved unknowns and rollback triggers.

The skill is read-only by default. It does not pull images, modify manifests,
restart servers, or deploy the target release unless the user separately
authorizes those actions.

### Evidence reference

A concise reference defines source priority:

1. target release and official upgrade notes;
2. exact tag-to-tag compare and merged PR diffs;
3. CLI help and dependency metadata from both immutable revisions;
4. official docs/cookbooks scoped to those revisions;
5. issues only for explicitly labeled known-risk context.

It explains how to distinguish removed interfaces, renamed interfaces, default
changes, dependency constraints, known issues, fixes, and unrelated changes.
Release-note absence is not proof of compatibility.

### Deployment profile

The analyzer consumes a JSON profile containing:

- stable profile ID and full argv token list;
- relevant environment variables;
- model, quantization, hardware, topology, and enabled features;
- optional internal imports and integration tags;
- required guarantees such as temperature-zero determinism.

Commands are represented as argv arrays, not evaluated shell strings. Rendered
commands use shell quoting only for display.

### Upgrade rule schema

Each manually evidenced rule contains:

- stable rule ID, category, severity, title, and target version range;
- direct upstream source URL and evidence summary;
- match conditions over flags, environment variables, imports, model traits,
  topology, hardware, or integration tags;
- optional exact transformations such as rename flag, remove flag, replace
  value, or replace import prefix;
- required canary checks;
- notes and unresolved limitations.

Rules without a safe mechanical transformation still produce findings. A
rewrite is never invented merely to eliminate a warning.

### Deterministic analyzer

The Python CLI:

1. validates profile and rule schemas;
2. matches target-applicable rules to each profile;
3. detects conflicting transformations;
4. produces proposed argv after safe token-level rewrites;
5. derives a per-profile and overall verdict;
6. emits Markdown and JSON reports;
7. lists unmatched high-risk features as audit coverage gaps.

Verdict rules are explicit:

- `NO_GO`: an unresolved blocker or conflicting/unsafe required migration;
- `CONDITIONAL_GO`: required changes, known risks, or evidence gaps remain but
  no unresolved blocker is present;
- `GO`: no blockers or required changes remain and all required canaries pass.

Because the offline analyzer cannot run target-version canaries by itself, its
pre-canary result normally cannot exceed `CONDITIONAL_GO`. `GO` requires
recorded canary evidence.

## Data Flow

```text
current deployment profiles + current/target revisions
                              |
                              v
       releases + diffs + CLI + dependencies + known issues
                              |
                              v
             evidence-backed structured upgrade rules
                              |
                              v
       rule matching + safe argv rewrite + conflict detection
                              |
                              v
       findings + proposed commands + canary/rollback matrix
                              |
                              v
                 GO / CONDITIONAL_GO / NO_GO
```

## Safety and Integrity Rules

- Never execute a rendered launch command while auditing it.
- Never source user shell files or evaluate command substitutions.
- Never edit deployment files unless separately authorized after the report.
- Resolve both endpoints to immutable versions or commits.
- Attach every rule to a direct source and retain uncertainty when evidence is
  incomplete.
- Do not extrapolate a known issue beyond its documented configuration without
  labeling the inference.
- Default changes receive behavior canaries even when no flag is removed.
- Internal import scanning is opt-in and limited to user-scoped paths.
- Logs, tokens, registry credentials, and private image names are sanitized from
  committed examples and reports.

## Demonstration

The pull request includes a fixture audit for SGLang v0.5.15 to v0.5.16. The
fixture contains multiple deployment profiles so the report can demonstrate:

- removal of an unsupported quantization or GEMM path;
- exact flag renames without deprecated aliases;
- a default behavior change that requires a regression canary;
- a configuration-specific determinism risk;
- a profile unaffected by a model-specific change.

The demo report shows:

1. matched and non-matched rules per deployment;
2. original and proposed argv;
3. blockers versus conditional risks;
4. required canaries and rollback triggers;
5. an overall pre-canary verdict.

All deployment inputs are synthetic and clearly labeled. Release facts cite
their upstream sources; no fixture is represented as a production deployment.

## Error Handling

- Reject malformed profiles, duplicate IDs, missing versions, invalid rule
  ranges, unknown transformations, ambiguous flag arity, and conflicting
  rewrites.
- Preserve a finding and withhold a rewrite when the command structure is
  ambiguous.
- Return non-zero for invalid evidence or analyzer failure.
- Return a valid `NO_GO` or `CONDITIONAL_GO` report with zero status when the
  audit completed successfully and the verdict itself is unfavorable.
- Surface unmatched deployment features as coverage gaps rather than silently
  declaring compatibility.

## Testing and Validation

Validation includes:

1. unit tests for matching, version applicability, rewrites, conflicts,
   verdicts, quoting, unknowns, and multi-profile aggregation;
2. a stable fixture-demo report for v0.5.15 to v0.5.16;
3. Python compilation and CLI help checks;
4. direct-link verification for every committed example rule;
5. the repository's skill validator and metadata tests;
6. Markdown and link checks used by the repository;
7. a staged-diff review for unsafe execution guidance, unsupported upgrade
   claims, secrets, and overlap with incident-triage behavior.

The fixture proves the auditor's matching and reporting behavior, not that an
actual deployment passed its canaries.

## Planned Repository Layout

```text
skills/sglang-upgrade-readiness-auditor/
├── SKILL.md
├── references/
│   ├── evidence-and-rule-authoring.md
│   └── profile-and-rule-schema.md
├── scripts/
│   └── audit_upgrade.py
├── examples/
│   ├── v0.5.15-to-v0.5.16-profiles.json
│   ├── v0.5.15-to-v0.5.16-rules.json
│   └── fixture-report.md
└── tests/
    └── test_audit_upgrade.py
```

The root README, plugin manifest, marketplace metadata, and repository tests
will be updated only as required to register the new skill.

## Non-Goals

- Automatically deploying, restarting, or rolling back SGLang.
- Maintaining a permanent complete database of every historical release.
- Guaranteeing compatibility from release notes alone.
- Replacing production canaries or incident replay.
- Modifying SGLang source to restore a removed backend.
- Reading unrelated private repositories or deployment credentials.

## Commit and Pull Request Strategy

Use focused commits for:

1. this approved design and the implementation plan;
2. the skill workflow, references, schemas, and analyzer;
3. tests, v0.5.16 fixture demo, registration, and validation-driven
   corrections.

Push `codex/add-sglang-upgrade-readiness-auditor` and open a draft pull request
against `main`. The pull request will include the exact demo command, sample
output, validation commands, cited upstream evidence, and safety limitations.
