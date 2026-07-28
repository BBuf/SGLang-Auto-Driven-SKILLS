---
name: sglang-speculative-decoding-autotuner
description: "Find a safe, workload-specific SGLang speculative decoding configuration by compatibility-gating and benchmarking a non-speculative baseline against revision-supported MTP/EAGLE, EAGLE3, DFlash, DSpark, NGRAM, or standalone-draft candidates. Use for acceptance-length, TTFT/TPOT, throughput, memory, correctness, and determinism tradeoffs; for deciding whether speculative decoding should be enabled; or for retuning after a model, SGLang, backend, parallelism, quantization, or workload change."
---

# SGLang Speculative Decoding Autotuner

## Objective

Select a measured configuration, not a fashionable algorithm. Reject unhealthy,
incorrect, non-deterministic, over-memory, or SLA-violating candidates before
comparing speed. Return `no_safe_improvement` when speculative decoding does not
produce a defensible win.

This skill owns a narrow SGLang speculative-decoding search. Use
`llm-serving-auto-benchmark` for general cross-framework or non-speculative
serving searches. Hand source or kernel changes to `sglang-sota-humanize-loop`
after preserving this run's evidence.

## Required Contract

Resolve these fields before launching a server:

| Field | Required evidence |
| --- | --- |
| Model | Model ID/path and immutable revision |
| SGLang | Commit, release tag, or immutable image digest |
| Hardware | GPU model/count and available topology |
| Workload | Dataset, input/output lengths, concurrency or request rate, duration |
| Objective | One primary metric and its direction |
| Gates | Correctness policy, determinism requirement, memory cap, latency/throughput SLA |
| Budget | Candidate count, repeats, and wall-clock ceiling |

If any identity field is unknown and cannot be discovered, stop before
benchmarking. Never compare measurements from different model revisions,
hardware allocations, workloads, or correctness policies.

## Workflow

### 1. Freeze the experiment

Record:

- full SGLang and model revisions;
- complete baseline launch command and relevant environment;
- GPU state before each run;
- fixed prompts, seeds, sampling parameters, warmup, repeats, and load shape;
- output artifact directory owned by this run.

Use temperature zero only when the deployment requires it. If determinism is a
hard requirement, repeat identical requests and record byte/token differences.

### 2. Prove compatibility

Read [compatibility-and-search.md](references/compatibility-and-search.md).
Inspect the selected SGLang revision's CLI, release notes, source, model config,
and official cookbook. Do not infer support from newer documentation.

Build a table with:

| Candidate | Proven algorithm/flags | Draft or native MTP evidence | Backend/topology restrictions | Decision |
| --- | --- | --- | --- | --- |

Exclude unproven candidates. Record the source URL or local source path for
every included algorithm and fragile flag.

### 3. Establish the non-speculative baseline

Launch without speculative decoding. Require:

1. healthy workers and endpoint readiness;
2. expected outputs on the fixed correctness set;
3. required determinism;
4. warmup completion;
5. repeated benchmark measurements;
6. complete command, logs, and raw result paths.

Stop if the baseline is unhealthy or violates correctness. A speculative result
cannot repair an invalid baseline.

### 4. Search within a hard budget

Start from defaults proven for the selected revision. Change one dimension at a
time before combining winners:

1. algorithm or draft source;
2. draft steps/window/block size;
3. top-k/tree width;
4. draft-token count;
5. compatible attention/verify mode;
6. only then safe combinations.

Keep the model, precision, topology, workload, and SLA fixed. Cap candidate
count and wall time. Stop an algorithm family after repeated health,
correctness, determinism, memory, or clear performance failures.

### 5. Measure every candidate identically

For each clean server start:

1. capture the exact argv and environment;
2. verify health;
3. run correctness and determinism gates;
4. run identical warmup and repeats;
5. collect TTFT, TPOT/ITL, request and token throughput, peak memory, and
   acceptance metrics when exposed;
6. retain raw logs and benchmark results;
7. stop only the PID/container created for this candidate.

Reuse the workload runner from `llm-serving-auto-benchmark` when helpful, but
do not allow it to change the experiment contract.

### 6. Normalize and analyze

Read [measurement-schema.md](references/measurement-schema.md), then write one
measurement document. Run:

```bash
python3 skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py \
  --input /path/to/speculative-measurements.json \
  --output-markdown /path/to/speculative-report.md \
  --output-json /path/to/speculative-result.json
```

The analyzer validates identities and metrics, applies health/correctness/
determinism/SLA gates, constructs the Pareto frontier, and ranks the primary
objective. Do not manually override a rejection to obtain a winner.

### 7. Revalidate and report

Restart the recommended command from a clean, run-owned process and repeat the
baseline contract. Report:

- exact winner and baseline commands;
- accepted/rejected candidates with reasons;
- metric table and baseline deltas;
- Pareto frontier and primary objective;
- correctness, determinism, SLA, and memory evidence;
- raw artifact paths;
- compatibility sources;
- rollback conditions.

If the analyzer returns `no_safe_improvement`, keep the baseline and state why.

## Safety Rules

- Never run an unbounded Cartesian search.
- Never treat acceptance length alone as a performance or correctness result.
- Never compare different workloads or GPU allocations in one decision.
- Never present fixture values as hardware evidence.
- Never kill shared processes or clear shared model caches.
- Never patch SGLang within this skill.
- Re-run compatibility discovery after any SGLang, model, quantization,
  attention-backend, parallelism, or draft-model change.

## Demonstration

Run the committed decision-logic example:

```bash
python3 skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py \
  --input skills/sglang-speculative-decoding-autotuner/examples/fixture-measurements.json \
  --output-markdown /tmp/speculative-fixture-report.md \
  --output-json /tmp/speculative-fixture-result.json
```

The generated report starts with **SYNTHETIC FIXTURE**. It demonstrates that a
fast wrong-output candidate and an SLA miss are rejected before the safe Pareto
set is ranked. It is not a GPU benchmark.

## Resources

- [compatibility-and-search.md](references/compatibility-and-search.md): read
  before authoring candidates or after any revision/configuration change.
- [measurement-schema.md](references/measurement-schema.md): read before
  writing analyzer input or integrating a benchmark harness.
- `scripts/analyze_candidates.py`: deterministic validation and selection.
- `examples/fixture-report.md`: expected demonstration output.
