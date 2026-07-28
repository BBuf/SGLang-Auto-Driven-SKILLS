# Measurement and Result Schema

## Contents

- [Input root](#input-root)
- [Experiment](#experiment)
- [Candidate](#candidate)
- [Hard gates](#hard-gates)
- [Selection](#selection)
- [Analyzer output](#analyzer-output)
- [Unknown values and fixtures](#unknown-values-and-fixtures)

## Input root

Provide one JSON object:

```json
{
  "schema_version": 1,
  "fixture": false,
  "experiment": {},
  "candidates": []
}
```

`fixture` is required only for examples. Set it to `true` for synthetic data;
the Markdown report will display a prominent warning.

## Experiment

Required fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | Stable experiment identity copied to every candidate |
| `model` | string | Model ID or path |
| `model_revision` | string | Immutable revision or digest |
| `sglang_revision` | string | Commit, tag, or immutable image |
| `hardware` | string | Exact GPU allocation/topology label |
| `workload` | object | Fixed load shape and dataset identity |
| `objective` | object | Primary metric, direction, and noise threshold |
| `hard_limits` | object | Required latency, throughput, and memory gates |
| `pareto_metrics` | array | Metrics and directions used for dominance |

Example objective:

```json
{
  "primary": "output_throughput",
  "direction": "maximize",
  "minimum_improvement_percent": 3.0
}
```

Directions are `maximize` or `minimize`.

Supported hard-limit keys:

| Limit | Metric | Behavior |
| --- | --- | --- |
| `max_ttft_ms` | `ttft_ms` | Reject above |
| `max_tpot_ms` | `tpot_ms` | Reject above |
| `max_peak_memory_gb` | `peak_memory_gb` | Reject above |
| `min_output_throughput` | `output_throughput` | Reject below |
| `min_request_throughput` | `request_throughput` | Reject below |

A metric referenced by an active hard limit, the objective, or the speculative
Pareto set is required for that candidate.

## Candidate

Each candidate has:

```json
{
  "id": "dspark-block-4",
  "baseline": false,
  "algorithm": "DSPARK",
  "command": ["python3", "-m", "sglang.launch_server"],
  "experiment_id": "experiment-1",
  "status": {
    "healthy": true,
    "correct": true,
    "deterministic": true
  },
  "metrics": {
    "ttft_ms": 125.0,
    "tpot_ms": 3.8,
    "output_throughput": 138.0,
    "request_throughput": 1.65,
    "peak_memory_gb": 151.0,
    "acceptance_length": 4.6,
    "acceptance_rate": null
  },
  "repeat_count": 3,
  "artifacts": ["runs/dspark-block-4/result.json"]
}
```

Rules:

- Supply exactly one candidate with `baseline: true`.
- Represent commands as argv arrays, not shell strings.
- Copy the root experiment ID exactly.
- Set every status field from recorded evidence.
- Store aggregate metrics from the declared repeat policy.
- Keep raw artifacts addressable from the final report.
- Do not encode missing values as zero.

## Hard gates

The analyzer rejects in this order:

1. health;
2. correctness;
3. determinism;
4. active hard limits;
5. missing objective or Pareto metrics.

Rejection reasons remain in JSON and Markdown. Rejected candidates still appear
in the metric table for diagnosis but cannot join the Pareto frontier or win.

## Selection

The analyzer:

1. validates one shared experiment identity;
2. gates every candidate;
3. excludes the baseline from speculative candidates;
4. calculates Pareto dominance using the declared directions;
5. ranks frontier candidates by the primary objective;
6. breaks exact ties by candidate ID for deterministic output;
7. compares the winner with the baseline;
8. returns `no_safe_improvement` below the declared threshold.

Baseline deltas are candidate minus baseline. Therefore a negative TTFT/TPOT
delta is normally favorable, while a positive throughput delta is favorable.

## Analyzer output

The JSON result contains:

| Field | Meaning |
| --- | --- |
| `schema_version` | Result schema version |
| `fixture` | Whether input was synthetic |
| `experiment` | Frozen experiment contract |
| `baseline_id` | Baseline candidate |
| `accepted` | IDs that passed hard gates |
| `rejected` | ID to ordered rejection reasons |
| `pareto_frontier` | Safe non-baseline frontier IDs |
| `recommendation` | Status, candidate, improvement percentage |
| `evaluations` | Metrics, deltas, commands, repeats, and artifacts |

The Markdown report presents the same information plus revalidation and
rollback guidance.

## Unknown values and fixtures

Unknown optional metrics remain `null` or are omitted. They are never converted
to zero. Metrics used by a gate or selection rule are not optional.

Set `fixture: true` for invented values used to test or demonstrate the
analyzer. The report must retain the **SYNTHETIC FIXTURE** warning. Never cite a
fixture report as SGLang, model, algorithm, or GPU performance evidence.
