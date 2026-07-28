# SGLang Speculative Decoding Autotuner Design

## Purpose

Add an evidence-driven skill that finds a safe, workload-specific speculative
decoding configuration for SGLang. The skill will compare a non-speculative
baseline with only the speculative algorithms that the selected model, SGLang
revision, hardware, and attention backend can actually support. It will reject
incorrect or SLA-violating candidates before ranking performance.

The first upstream snapshot is SGLang v0.5.16, whose speculative paths include
MTP/EAGLE-family modes, DFlash, DSpark, and architecture-specific ReplaySSM
behavior. The skill must remain version-aware rather than treating the v0.5.16
flag set as timeless.

## Success Criteria

The pull request is complete when:

1. `sglang-speculative-decoding-autotuner` has an unambiguous trigger and does
   not overlap with the general `llm-serving-auto-benchmark` or model-patching
   `sglang-sota-humanize-loop`.
2. The workflow builds a bounded candidate matrix from proven compatibility
   evidence instead of blindly trying every speculative flag.
3. Every candidate is compared with the same model revision, precision,
   hardware allocation, workload, warmup policy, and correctness prompts.
4. Correctness, determinism requirements, server health, and SLA constraints
   are hard gates; throughput alone cannot select a winner.
5. A deterministic analysis script validates input measurements, reports
   rejected candidates, computes the Pareto frontier, and recommends a winner
   using an explicit objective.
6. The output includes runnable candidate commands, raw measurement references,
   a result table, a recommendation, and rollback conditions.
7. Unit tests and a clearly labeled fixture-based demo run without a GPU.
8. The focused branch is pushed and opened as its own draft pull request
   against `main`.

## Approaches Considered

### 1. Prompt-only tuning checklist

Document the relevant flags and tell the agent to benchmark them. This is easy
to maintain, but selection logic and evidence formatting would vary between
runs, making the claimed recommendation difficult to audit.

### 2. Guided workflow plus deterministic analyzer

Use `SKILL.md` to control discovery, launch, measurement, and stop conditions;
use a small standard-library Python analyzer for schema validation, hard-gate
filtering, Pareto analysis, and recommendation; include fixtures and tests.

This is the chosen approach. It makes the changing SGLang compatibility layer a
documented agent responsibility while keeping the final decision reproducible.

### 3. Fully autonomous server controller

Ship a controller that starts and stops every SGLang configuration and drives
all benchmarks. This would be convenient on one machine, but would duplicate
the existing benchmark skill, hard-code cluster assumptions, and make process
cleanup risky across local, Docker, Slurm, and multi-node environments.

## User Contract

Required inputs:

- model identifier and exact model revision when available;
- SGLang revision or image;
- GPU type and count;
- workload shape, concurrency or request-rate policy, and benchmark duration;
- optimization objective, such as minimum TPOT under an output-throughput
  floor;
- correctness policy and any latency or memory SLA.

Optional inputs:

- allowed or forbidden algorithms;
- an existing launch command to preserve;
- draft-model path or model-native MTP availability;
- attention, quantization, parallelism, or CUDA Graph constraints;
- a pre-existing measurement JSON file for analysis-only mode.

If required inputs cannot be discovered from the current environment, the
skill stops before launching candidates and reports the missing fields.

## Architecture and Component Boundaries

### `SKILL.md`

Owns the operational workflow:

1. freeze the experiment contract and record immutable revisions;
2. inspect SGLang help, model configuration, official docs, release notes, and
   source to build a compatibility table;
3. establish a healthy, correct, non-speculative baseline;
4. generate a bounded candidate set, changing one tuning dimension at a time
   before combining proven choices;
5. run identical warmup, correctness, and performance measurements;
6. write normalized measurements;
7. invoke the analyzer and interpret the report;
8. re-run the selected candidate against the baseline;
9. emit the final recommendation and rollback criteria.

The skill does not patch SGLang source. If evidence points to a kernel or
scheduler bottleneck rather than a configuration choice, it hands off to the
existing profiler/SOTA skills with the collected artifact paths.

### Compatibility reference

A concise reference explains how to prove, rather than assume:

- which algorithms exist in the selected SGLang revision;
- whether the model has a compatible draft model or native MTP layers;
- restrictions involving attention backends, DP/TP/EP/CP, quantization, CUDA
  Graphs, top-k, or architecture-specific state;
- flags and environment variables that changed between releases;
- which failures require rejecting a candidate rather than lowering its score.

The reference links to upstream evidence but does not copy a permanently frozen
support matrix that will become stale.

### Candidate and measurement schema

The committed JSON schema is represented by documented examples and enforced by
the analyzer. Each candidate records:

- stable candidate ID, algorithm, full launch command, and changed parameters;
- immutable model, framework, hardware, and workload identity;
- health and correctness outcomes;
- TTFT, TPOT/ITL, input/output throughput, request throughput, and peak memory
  when available;
- speculative metrics such as acceptance length or acceptance rate when
  exposed;
- repeat count, aggregate values, raw artifact paths, and failure reason.

Unknown optional metrics remain unknown. They are never coerced to zero.

### Deterministic analyzer

The standard-library Python CLI supports a measurement file and writes
Markdown plus JSON results. It performs:

1. schema and finite-number validation;
2. experiment-identity consistency checks;
3. hard-gate rejection for health, correctness, determinism, memory, and SLA;
4. Pareto-frontier calculation over explicitly selected minimize/maximize
   metrics;
5. objective-based recommendation with deterministic tie-breaking;
6. baseline-relative deltas;
7. an explicit no-winner result if no speculative candidate safely improves
   the baseline.

It never invents missing measurements and never interprets fixture data as a
real benchmark.

## Data Flow

```text
model + SGLang revision + hardware + workload/SLA
                         |
                         v
              compatibility evidence
                         |
                         v
        baseline and bounded candidate commands
                         |
                         v
       health -> correctness -> performance runs
                         |
                         v
            normalized measurement JSON
                         |
                         v
      hard gates -> Pareto set -> objective ranking
                         |
                         v
       recommendation + evidence + rollback rules
```

## Selection and Safety Rules

- The non-speculative baseline is mandatory and cannot be silently replaced by
  a speculative default.
- A candidate with wrong output, unhealthy workers, non-finite metrics,
  excessive memory, or a violated hard SLA is rejected.
- Algorithm availability is discovered from the selected revision. Current
  documentation is insufficient evidence for an older installed image.
- Search budgets cap candidate count and wall time. The skill does not perform
  an unbounded Cartesian product.
- Performance claims include exact workload and hardware scope.
- A small gain below the configured noise threshold is reported as a tie.
- The winning command is revalidated from a clean server start.
- Process cleanup targets only PIDs or containers started by the current run.

## Demonstration

The pull request includes a fixture with:

- a valid non-speculative baseline;
- one speculative candidate that wins on throughput but fails correctness;
- one candidate that improves throughput but violates the TPOT SLA;
- two valid candidates with different latency/throughput tradeoffs.

Running the analyzer produces a Markdown report that visibly:

1. rejects the unsafe candidates with reasons;
2. shows baseline-relative deltas;
3. identifies the Pareto frontier;
4. recommends the candidate matching the declared objective;
5. labels every value as fixture data.

This demonstrates the decision logic on any development machine. It is not
presented as measured SGLang or GPU performance.

## Error Handling

- Reject malformed JSON, duplicate candidate IDs, mixed experiment identities,
  missing baselines, non-finite values, invalid metric direction, and negative
  latency or throughput.
- Return a non-zero status for invalid evidence, but allow a valid
  `no_safe_improvement` result when all speculative candidates are rejected or
  fail to beat the baseline.
- Preserve raw logs and commands for failed candidates.
- Do not recommend flags that were not verified against the selected SGLang
  revision.

## Testing and Validation

Validation includes:

1. unit tests for schema errors, gates, Pareto dominance, objectives,
   deterministic tie-breaking, and no-winner behavior;
2. a fixture-demo snapshot with stable Markdown sections;
3. Python compilation and CLI help checks;
4. the repository's skill validator and metadata tests;
5. link and Markdown checks used by the repository;
6. a staged-diff review for unsupported performance claims and overlap with
   existing skills.

GPU execution is not required for the pull request because the analyzer and
workflow can be validated deterministically. Any later real-GPU demonstration
must report its exact hardware and raw artifacts separately.

## Planned Repository Layout

```text
skills/sglang-speculative-decoding-autotuner/
├── SKILL.md
├── references/
│   ├── compatibility-and-search.md
│   └── measurement-schema.md
├── scripts/
│   └── analyze_candidates.py
├── examples/
│   ├── fixture-measurements.json
│   └── fixture-report.md
└── tests/
    └── test_analyze_candidates.py
```

The root README, plugin manifest, marketplace metadata, and repository tests
will be updated only as required to register the new skill.

## Non-Goals

- Implementing a new speculative decoding algorithm.
- Modifying SGLang kernels, scheduler code, or model weights.
- Replacing `llm-serving-auto-benchmark` for general framework comparison.
- Claiming one algorithm is universally best across models or workloads.
- Managing shared clusters or killing processes not started by the workflow.
- Treating synthetic fixture values as hardware evidence.

## Commit and Pull Request Strategy

Use focused commits for:

1. this approved design and the implementation plan;
2. the skill workflow, references, and analyzer;
3. tests, fixture demo, registration, and validation-driven corrections.

Push `codex/add-sglang-speculative-decoding-autotuner` and open a draft pull
request against `main`. The pull request will include the exact demo command,
sample output, test commands, upstream snapshot, and limitations.
