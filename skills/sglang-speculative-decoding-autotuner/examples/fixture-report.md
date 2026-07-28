# SGLang Speculative Decoding Autotuner Report

> **SYNTHETIC FIXTURE:** These values demonstrate decision logic; they are not GPU measurements.

## Experiment

- ID: `synthetic-v0516-spec-search`
- Model: `fixture/long-context-moe` at `fixture-only`
- SGLang: `v0.5.16`
- Hardware: `synthetic-8x-blackwell`
- Objective: `maximize output_throughput`
- Minimum improvement: `3.00%`

## Gate Results

| Candidate | Algorithm | Result | Reasons |
| --- | --- | --- | --- |
| `baseline` | `NONE` | ACCEPTED | none |
| `dspark-balanced` | `DSPARK` | ACCEPTED | none |
| `dspark-wrong-output` | `DSPARK` | REJECTED | correctness_failed |
| `eagle-sla-miss` | `EAGLE3` | REJECTED | max_tpot_ms_exceeded |
| `mtp-low-latency` | `MTP` | ACCEPTED | none |

## Metrics and Baseline Deltas

| Candidate | TTFT ms | Δ TTFT | TPOT ms | Δ TPOT | Output tok/s | Δ Output tok/s | Peak GiB | Accept length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 160.00 | +0.00 | 4.40 | +0.00 | 100.00 | +0.00 | 145.00 | unknown |
| `dspark-balanced` | 125.00 | -35.00 | 3.80 | -0.60 | 138.00 | +38.00 | 151.00 | 4.60 |
| `dspark-wrong-output` | 105.00 | -55.00 | 2.80 | -1.60 | 155.00 | +55.00 | 153.00 | 5.40 |
| `eagle-sla-miss` | 115.00 | -45.00 | 5.10 | +0.70 | 142.00 | +42.00 | 158.00 | 4.80 |
| `mtp-low-latency` | 118.00 | -42.00 | 3.40 | -1.00 | 122.00 | +22.00 | 149.00 | 3.10 |

## Pareto Frontier

- `dspark-balanced`
- `mtp-low-latency`

## Recommendation

- Status: `recommended`
- Candidate: `dspark-balanced`
- Primary-metric improvement: `38.00%`

## Candidate Commands

### `baseline`

```bash
python3 -m sglang.launch_server --model-path fixture/long-context-moe --tp 8
```

Artifacts:
- `synthetic/raw/baseline.json`

### `dspark-balanced`

```bash
python3 -m sglang.launch_server --model-path fixture/long-context-moe --tp 8 --speculative-algorithm DSPARK --speculative-dspark-block-size 4
```

Artifacts:
- `synthetic/raw/dspark-balanced.json`

### `dspark-wrong-output`

```bash
python3 -m sglang.launch_server --model-path fixture/long-context-moe --tp 8 --speculative-algorithm DSPARK --speculative-dspark-block-size 8
```

Artifacts:
- `synthetic/raw/dspark-wrong-output.json`

### `eagle-sla-miss`

```bash
python3 -m sglang.launch_server --model-path fixture/long-context-moe --tp 8 --speculative-algorithm EAGLE3 --speculative-draft-model-path fixture/eagle3-draft --speculative-num-steps 5 --speculative-eagle-topk 8 --speculative-num-draft-tokens 64
```

Artifacts:
- `synthetic/raw/eagle-sla-miss.json`

### `mtp-low-latency`

```bash
python3 -m sglang.launch_server --model-path fixture/long-context-moe --tp 8 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

Artifacts:
- `synthetic/raw/mtp-low-latency.json`

## Revalidation and Rollback

- Restart the selected candidate from a clean, run-owned server process and repeat the baseline contract.
- Roll back to the recorded baseline command if health, correctness, determinism, memory, or latency gates fail.
- Treat `no_safe_improvement` as a successful audit outcome; do not force-enable speculative decoding.
