# SGLang Speculative Decoding Autotuner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a version-aware SGLang skill that rejects unsafe speculative decoding candidates and deterministically recommends the best workload-specific configuration from measured evidence.

**Architecture:** `SKILL.md` owns compatibility discovery and the bounded benchmark workflow. A standard-library Python analyzer owns measurement validation, hard gates, Pareto selection, objective ranking, and Markdown/JSON reports. Committed fixture inputs and reports demonstrate the decision logic without making GPU performance claims.

**Tech Stack:** Markdown, Python 3.10+ standard library, `unittest`, JSON, repository pre-commit and link checks.

---

## File Structure

- Create `skills/sglang-speculative-decoding-autotuner/SKILL.md`: operational workflow, trigger, safety gates, handoffs, and report contract.
- Create `skills/sglang-speculative-decoding-autotuner/references/compatibility-and-search.md`: version-aware compatibility proof and bounded search guidance.
- Create `skills/sglang-speculative-decoding-autotuner/references/measurement-schema.md`: exact JSON input/output schema.
- Create `skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py`: deterministic validator, gate evaluator, Pareto selector, and CLI reporter.
- Create `skills/sglang-speculative-decoding-autotuner/examples/fixture-measurements.json`: synthetic candidate measurements.
- Create `skills/sglang-speculative-decoding-autotuner/examples/fixture-report.md`: generated, visibly labeled demo report.
- Create `tests/test_speculative_decoding_autotuner.py`: analyzer, CLI, fixture, and documentation tests.
- Modify `README.md`: register the new core skill, installation commands, examples, and count.
- Modify `.claude-plugin/plugin.json`: mention speculative decoding tuning in the plugin description.
- Modify `.claude-plugin/marketplace.json`: mention the new capability in discovery metadata.
- Modify `tests/test_repository_metadata.py`: update the expected core-skill count and assert registration.

### Task 1: Validate Measurement Documents

**Files:**
- Create: `tests/test_speculative_decoding_autotuner.py`
- Create: `skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py`

- [ ] **Step 1: Write the failing loader and schema tests**

Create a module loader and a valid-document helper, then add tests for a valid
document, duplicate candidate IDs, a missing baseline, and mixed experiment
identity:

```python
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "skills"
    / "sglang-speculative-decoding-autotuner"
    / "scripts"
    / "analyze_candidates.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("spec_autotuner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_document() -> dict:
    return {
        "schema_version": 1,
        "fixture": True,
        "experiment": {
            "id": "fixture-search",
            "model": "fixture/model",
            "model_revision": "fixture-revision",
            "sglang_revision": "v0.5.16",
            "hardware": "fixture-hardware",
            "workload": {"input_tokens": 2048, "output_tokens": 256, "concurrency": 1},
            "objective": {
                "primary": "output_throughput",
                "direction": "maximize",
                "minimum_improvement_percent": 3.0,
            },
            "hard_limits": {"max_ttft_ms": 500.0, "max_tpot_ms": 5.0},
            "pareto_metrics": [
                {"name": "output_throughput", "direction": "maximize"},
                {"name": "tpot_ms", "direction": "minimize"},
            ],
        },
        "candidates": [
            {
                "id": "baseline",
                "baseline": True,
                "algorithm": "NONE",
                "command": ["python3", "-m", "sglang.launch_server", "--model-path", "fixture/model"],
                "experiment_id": "fixture-search",
                "status": {"healthy": True, "correct": True, "deterministic": True},
                "metrics": {
                    "ttft_ms": 120.0,
                    "tpot_ms": 4.5,
                    "output_throughput": 100.0,
                    "peak_memory_gb": 70.0,
                },
                "repeat_count": 3,
                "artifacts": ["examples/raw/baseline.json"],
            }
        ],
    }


class MeasurementValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_valid_document_is_accepted(self) -> None:
        document = valid_document()
        self.mod.validate_document(document)

    def test_duplicate_candidate_id_is_rejected(self) -> None:
        document = valid_document()
        document["candidates"].append(dict(document["candidates"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate candidate id"):
            self.mod.validate_document(document)

    def test_missing_baseline_is_rejected(self) -> None:
        document = valid_document()
        document["candidates"][0]["baseline"] = False
        with self.assertRaisesRegex(ValueError, "exactly one baseline"):
            self.mod.validate_document(document)

    def test_mixed_experiment_identity_is_rejected(self) -> None:
        document = valid_document()
        document["candidates"][0]["experiment_id"] = "other"
        with self.assertRaisesRegex(ValueError, "experiment_id"):
            self.mod.validate_document(document)
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner.MeasurementValidationTest -v
```

Expected: `FileNotFoundError` for `analyze_candidates.py`.

- [ ] **Step 3: Implement strict loading and validation**

Create the script with these public functions and validations:

```python
#!/usr/bin/env python3
"""Validate and rank measured SGLang speculative decoding candidates."""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any

VALID_DIRECTIONS = {"minimize", "maximize"}
NON_NEGATIVE_METRICS = {
    "ttft_ms",
    "tpot_ms",
    "output_throughput",
    "request_throughput",
    "peak_memory_gb",
    "acceptance_length",
    "acceptance_rate",
}


def load_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("measurement root must be an object")
    validate_document(value)
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_document(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    experiment = document.get("experiment")
    candidates = document.get("candidates")
    if not isinstance(experiment, dict):
        raise ValueError("experiment must be an object")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty array")
    experiment_id = experiment.get("id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment.id must be a non-empty string")
    ids: set[str] = set()
    baseline_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("each candidate must be an object")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate id must be a non-empty string")
        if candidate_id in ids:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        ids.add(candidate_id)
        baseline_count += candidate.get("baseline") is True
        if candidate.get("experiment_id") != experiment_id:
            raise ValueError(f"{candidate_id}: experiment_id does not match experiment.id")
        command = candidate.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(token, str) and token for token in command
        ):
            raise ValueError(f"{candidate_id}: command must be a non-empty argv array")
        status = candidate.get("status")
        if not isinstance(status, dict):
            raise ValueError(f"{candidate_id}: status must be an object")
        metrics = candidate.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{candidate_id}: metrics must be an object")
        for name, value in metrics.items():
            if value is None:
                continue
            number = _finite_number(value, f"{candidate_id}.metrics.{name}")
            if name in NON_NEGATIVE_METRICS and number < 0:
                raise ValueError(f"{candidate_id}.metrics.{name} must be non-negative")
    if baseline_count != 1:
        raise ValueError("candidates must contain exactly one baseline")
    objective = experiment.get("objective")
    if not isinstance(objective, dict) or objective.get("direction") not in VALID_DIRECTIONS:
        raise ValueError("experiment.objective.direction must be minimize or maximize")
    pareto_metrics = experiment.get("pareto_metrics")
    if not isinstance(pareto_metrics, list) or not pareto_metrics:
        raise ValueError("experiment.pareto_metrics must be a non-empty array")
    for metric in pareto_metrics:
        if (
            not isinstance(metric, dict)
            or not isinstance(metric.get("name"), str)
            or metric.get("direction") not in VALID_DIRECTIONS
        ):
            raise ValueError("each pareto metric needs a name and valid direction")
```

- [ ] **Step 4: Run the validation tests**

Run:

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner.MeasurementValidationTest -v
```

Expected: four tests pass.

- [ ] **Step 5: Commit the validated schema boundary**

```bash
git add tests/test_speculative_decoding_autotuner.py \
  skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py
git commit -m "feat: validate speculative decoding measurements"
```

### Task 2: Add Hard Gates, Pareto Selection, and Recommendation

**Files:**
- Modify: `tests/test_speculative_decoding_autotuner.py`
- Modify: `skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py`

- [ ] **Step 1: Add failing decision tests**

Add candidates with wrong output, an SLA violation, and two valid tradeoffs.
Assert exact rejection reasons, frontier IDs, and recommendation:

```python
class CandidateDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_hard_gates_precede_performance_ranking(self) -> None:
        document = valid_document()
        wrong = dict(document["candidates"][0])
        wrong.update(
            id="wrong-fast",
            baseline=False,
            algorithm="DSPARK",
            status={"healthy": True, "correct": False, "deterministic": True},
            metrics={"ttft_ms": 80.0, "tpot_ms": 2.0, "output_throughput": 180.0, "peak_memory_gb": 75.0},
        )
        slow = dict(document["candidates"][0])
        slow.update(
            id="slow-tpot",
            baseline=False,
            algorithm="EAGLE",
            status={"healthy": True, "correct": True, "deterministic": True},
            metrics={"ttft_ms": 90.0, "tpot_ms": 6.0, "output_throughput": 150.0, "peak_memory_gb": 76.0},
        )
        document["candidates"].extend([wrong, slow])
        result = self.mod.analyze(document)
        self.assertEqual(result["rejected"]["wrong-fast"], ["correctness_failed"])
        self.assertEqual(result["rejected"]["slow-tpot"], ["max_tpot_ms_exceeded"])

    def test_pareto_frontier_and_objective_select_safe_winner(self) -> None:
        document = valid_document()
        for candidate_id, tpot, throughput in [
            ("mtp-balanced", 3.9, 118.0),
            ("dspark-fast", 4.2, 135.0),
            ("dominated", 4.8, 110.0),
        ]:
            candidate = dict(document["candidates"][0])
            candidate.update(
                id=candidate_id,
                baseline=False,
                algorithm="MTP" if candidate_id == "mtp-balanced" else "DSPARK",
                status={"healthy": True, "correct": True, "deterministic": True},
                metrics={"ttft_ms": 100.0, "tpot_ms": tpot, "output_throughput": throughput, "peak_memory_gb": 75.0},
            )
            document["candidates"].append(candidate)
        result = self.mod.analyze(document)
        self.assertEqual(result["pareto_frontier"], ["dspark-fast", "mtp-balanced"])
        self.assertEqual(result["recommendation"]["candidate_id"], "dspark-fast")
        self.assertEqual(result["recommendation"]["status"], "recommended")

    def test_below_noise_threshold_returns_no_safe_improvement(self) -> None:
        document = valid_document()
        candidate = dict(document["candidates"][0])
        candidate.update(
            id="tiny-gain",
            baseline=False,
            algorithm="MTP",
            status={"healthy": True, "correct": True, "deterministic": True},
            metrics={"ttft_ms": 110.0, "tpot_ms": 4.4, "output_throughput": 102.0, "peak_memory_gb": 72.0},
        )
        document["candidates"].append(candidate)
        result = self.mod.analyze(document)
        self.assertEqual(result["recommendation"]["status"], "no_safe_improvement")
```

- [ ] **Step 2: Run the decision tests and verify missing behavior**

Run:

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner.CandidateDecisionTest -v
```

Expected: failures because `analyze` is not defined.

- [ ] **Step 3: Implement gates and Pareto dominance**

Add:

```python
def _gate_reasons(candidate: dict[str, Any], limits: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    status = candidate["status"]
    for key, reason in [
        ("healthy", "health_failed"),
        ("correct", "correctness_failed"),
        ("deterministic", "determinism_failed"),
    ]:
        if status.get(key) is not True:
            reasons.append(reason)
    metrics = candidate["metrics"]
    for limit_name, metric_name, relation in [
        ("max_ttft_ms", "ttft_ms", "max"),
        ("max_tpot_ms", "tpot_ms", "max"),
        ("max_peak_memory_gb", "peak_memory_gb", "max"),
        ("min_output_throughput", "output_throughput", "min"),
        ("min_request_throughput", "request_throughput", "min"),
    ]:
        if limit_name not in limits:
            continue
        if metric_name not in metrics:
            reasons.append(f"{metric_name}_missing")
            continue
        failed = (
            metrics[metric_name] > limits[limit_name]
            if relation == "max"
            else metrics[metric_name] < limits[limit_name]
        )
        if failed:
            reasons.append(f"{limit_name}_exceeded" if relation == "max" else f"{limit_name}_not_met")
    return reasons


def _dominates(
    left: dict[str, Any],
    right: dict[str, Any],
    metric_specs: list[dict[str, str]],
) -> bool:
    at_least_as_good = True
    strictly_better = False
    for spec in metric_specs:
        name = spec["name"]
        if name not in left["metrics"] or name not in right["metrics"]:
            return False
        left_value = left["metrics"][name]
        right_value = right["metrics"][name]
        if spec["direction"] == "maximize":
            at_least_as_good &= left_value >= right_value
            strictly_better |= left_value > right_value
        else:
            at_least_as_good &= left_value <= right_value
            strictly_better |= left_value < right_value
    return at_least_as_good and strictly_better


def pareto_frontier(
    candidates: list[dict[str, Any]], metric_specs: list[dict[str, str]]
) -> list[dict[str, Any]]:
    return sorted(
        [
            candidate
            for candidate in candidates
            if not any(
                other["id"] != candidate["id"]
                and _dominates(other, candidate, metric_specs)
                for other in candidates
            )
        ],
        key=lambda candidate: candidate["id"],
    )
```

- [ ] **Step 4: Implement objective ranking and threshold**

Add `analyze(document)` that validates, gates all candidates, excludes the
baseline from speculative recommendations, requires the primary metric, ranks
maximize objectives descending and minimize objectives ascending, uses
candidate ID as the final tie-breaker, and compares the winner with the
baseline:

```python
def _improvement_percent(baseline: float, candidate: float, direction: str) -> float:
    if baseline == 0:
        return 0.0 if candidate == 0 else math.inf
    delta = candidate - baseline if direction == "maximize" else baseline - candidate
    return delta / abs(baseline) * 100.0


def analyze(document: dict[str, Any]) -> dict[str, Any]:
    validate_document(document)
    experiment = document["experiment"]
    candidates = document["candidates"]
    baseline = next(candidate for candidate in candidates if candidate["baseline"])
    rejected: dict[str, list[str]] = {}
    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        reasons = _gate_reasons(candidate, experiment.get("hard_limits", {}))
        if reasons:
            rejected[candidate["id"]] = reasons
        else:
            accepted.append(candidate)
    speculative = [candidate for candidate in accepted if not candidate["baseline"]]
    frontier = pareto_frontier(speculative, experiment["pareto_metrics"])
    objective = experiment["objective"]
    primary = objective["primary"]
    rankable = [candidate for candidate in frontier if primary in candidate["metrics"]]
    if not rankable or baseline["id"] in rejected or primary not in baseline["metrics"]:
        recommendation = {"status": "no_safe_improvement", "candidate_id": None}
    else:
        reverse = objective["direction"] == "maximize"
        winner = sorted(
            rankable,
            key=lambda candidate: (
                candidate["metrics"][primary] * (-1 if reverse else 1),
                candidate["id"],
            ),
        )[0]
        improvement = _improvement_percent(
            baseline["metrics"][primary],
            winner["metrics"][primary],
            objective["direction"],
        )
        threshold = float(objective.get("minimum_improvement_percent", 0.0))
        recommendation = {
            "status": "recommended" if improvement >= threshold else "no_safe_improvement",
            "candidate_id": winner["id"] if improvement >= threshold else None,
            "improvement_percent": improvement,
        }
    return {
        "schema_version": 1,
        "fixture": document.get("fixture") is True,
        "experiment": experiment,
        "baseline_id": baseline["id"],
        "accepted": sorted(candidate["id"] for candidate in accepted),
        "rejected": rejected,
        "pareto_frontier": [candidate["id"] for candidate in frontier],
        "recommendation": recommendation,
        "candidates": candidates,
    }
```

- [ ] **Step 5: Run all decision tests**

Run:

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner -v
```

Expected: all current tests pass.

- [ ] **Step 6: Commit deterministic selection**

```bash
git add tests/test_speculative_decoding_autotuner.py \
  skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py
git commit -m "feat: rank safe speculative decoding candidates"
```

### Task 3: Add Reports and CLI

**Files:**
- Modify: `tests/test_speculative_decoding_autotuner.py`
- Modify: `skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py`

- [ ] **Step 1: Write failing report and CLI tests**

Test fixture labeling, rejection reasons, quoted commands, JSON output, and
non-zero invalid-input behavior:

```python
class ReportAndCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_markdown_labels_fixture_and_explains_rejections(self) -> None:
        document = valid_document()
        result = self.mod.analyze(document)
        report = self.mod.render_markdown(result)
        self.assertIn("SYNTHETIC FIXTURE", report)
        self.assertIn("## Gate Results", report)
        self.assertIn("## Pareto Frontier", report)
        self.assertIn("## Recommendation", report)
        self.assertIn("python3 -m sglang.launch_server", report)

    def test_cli_writes_markdown_and_json(self) -> None:
        document = valid_document()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.json"
            markdown_path = root / "report.md"
            json_path = root / "report.json"
            input_path.write_text(json.dumps(document), encoding="utf-8")
            exit_code = self.mod.main(
                [
                    "--input",
                    str(input_path),
                    "--output-markdown",
                    str(markdown_path),
                    "--output-json",
                    str(json_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertIn("SYNTHETIC FIXTURE", markdown_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["schema_version"], 1)
```

- [ ] **Step 2: Run report tests and verify missing functions**

Run:

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner.ReportAndCliTest -v
```

Expected: failures for missing `render_markdown` and `main`.

- [ ] **Step 3: Implement Markdown rendering and CLI**

Add a Markdown cell escaper, `shlex.join` command rendering, stable tables, and
the CLI:

```python
def _cell(value: Any) -> str:
    return str(value).replace("\n", "<br>").replace("|", "\\|")


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# SGLang Speculative Decoding Autotuner Report", ""]
    if result["fixture"]:
        lines.extend(
            [
                "> **SYNTHETIC FIXTURE:** These values demonstrate decision logic; they are not GPU measurements.",
                "",
            ]
        )
    experiment = result["experiment"]
    lines.extend(
        [
            "## Experiment",
            "",
            f"- ID: `{_cell(experiment['id'])}`",
            f"- Model: `{_cell(experiment['model'])}`",
            f"- SGLang: `{_cell(experiment['sglang_revision'])}`",
            f"- Hardware: `{_cell(experiment['hardware'])}`",
            "",
            "## Gate Results",
            "",
            "| Candidate | Result | Reasons |",
            "| --- | --- | --- |",
        ]
    )
    for candidate in sorted(result["candidates"], key=lambda item: item["id"]):
        reasons = result["rejected"].get(candidate["id"], [])
        lines.append(
            f"| `{_cell(candidate['id'])}` | {'REJECTED' if reasons else 'ACCEPTED'} | {_cell(', '.join(reasons))} |"
        )
    lines.extend(["", "## Pareto Frontier", ""])
    lines.extend(f"- `{_cell(candidate_id)}`" for candidate_id in result["pareto_frontier"])
    recommendation = result["recommendation"]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Status: `{recommendation['status']}`",
            f"- Candidate: `{recommendation.get('candidate_id') or 'none'}`",
            "",
            "## Candidate Commands",
            "",
        ]
    )
    for candidate in sorted(result["candidates"], key=lambda item: item["id"]):
        lines.extend(
            [
                f"### `{_cell(candidate['id'])}`",
                "",
                "```bash",
                shlex.join(candidate["command"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and rank measured SGLang speculative decoding candidates."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        document = load_document(args.input)
        result = analyze(document)
        args.output_markdown.write_text(render_markdown(result), encoding="utf-8")
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run report and full analyzer tests**

Run:

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner -v
python3 skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py --help
```

Expected: all tests pass and CLI help exits zero.

- [ ] **Step 5: Commit reports and CLI**

```bash
git add tests/test_speculative_decoding_autotuner.py \
  skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py
git commit -m "feat: report speculative decoding recommendations"
```

### Task 4: Add the Synthetic Demonstration

**Files:**
- Create: `skills/sglang-speculative-decoding-autotuner/examples/fixture-measurements.json`
- Create: `skills/sglang-speculative-decoding-autotuner/examples/fixture-report.md`
- Modify: `tests/test_speculative_decoding_autotuner.py`

- [ ] **Step 1: Add a failing fixture integration test**

```python
class FixtureDemoTest(unittest.TestCase):
    def test_committed_fixture_report_is_reproducible(self) -> None:
        module = load_module()
        skill = ROOT / "skills" / "sglang-speculative-decoding-autotuner"
        document = module.load_document(skill / "examples" / "fixture-measurements.json")
        result = module.analyze(document)
        generated = module.render_markdown(result)
        committed = (skill / "examples" / "fixture-report.md").read_text(encoding="utf-8")
        self.assertEqual(generated, committed)
        self.assertEqual(result["recommendation"]["candidate_id"], "dspark-balanced")
        self.assertEqual(result["rejected"]["dspark-wrong-output"], ["correctness_failed"])
        self.assertEqual(result["rejected"]["eagle-sla-miss"], ["max_tpot_ms_exceeded"])
```

- [ ] **Step 2: Run the fixture test and verify files are missing**

Run:

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner.FixtureDemoTest -v
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Create a five-candidate synthetic fixture**

Use the schema from Task 1 with:

```json
{
  "schema_version": 1,
  "fixture": true,
  "experiment": {
    "id": "synthetic-v0516-spec-search",
    "model": "fixture/long-context-moe",
    "model_revision": "fixture-only",
    "sglang_revision": "v0.5.16",
    "hardware": "synthetic-8x-blackwell",
    "workload": {"input_tokens": 4096, "output_tokens": 512, "concurrency": 8},
    "objective": {
      "primary": "output_throughput",
      "direction": "maximize",
      "minimum_improvement_percent": 3.0
    },
    "hard_limits": {
      "max_ttft_ms": 400.0,
      "max_tpot_ms": 4.5,
      "max_peak_memory_gb": 180.0
    },
    "pareto_metrics": [
      {"name": "output_throughput", "direction": "maximize"},
      {"name": "tpot_ms", "direction": "minimize"}
    ]
  }
}
```

Add `baseline`, `dspark-wrong-output`, `eagle-sla-miss`,
`mtp-low-latency`, and `dspark-balanced`. Give the unsafe candidates the best
headline throughput so the demo proves gates precede ranking. Set the two safe
candidates so neither dominates the other and `dspark-balanced` wins the
maximize-throughput objective.

- [ ] **Step 4: Generate the committed report**

Run:

```bash
python3 skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py \
  --input skills/sglang-speculative-decoding-autotuner/examples/fixture-measurements.json \
  --output-markdown skills/sglang-speculative-decoding-autotuner/examples/fixture-report.md \
  --output-json /tmp/sglang-speculative-decoding-fixture-result.json
```

Expected: exit zero; report contains the synthetic disclaimer, two rejection
reasons, two Pareto candidates, and `dspark-balanced`.

- [ ] **Step 5: Run the fixture and analyzer tests**

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the demo**

```bash
git add tests/test_speculative_decoding_autotuner.py \
  skills/sglang-speculative-decoding-autotuner/examples
git commit -m "test: demonstrate speculative decoding selection"
```

### Task 5: Write the Skill and References

**Files:**
- Create: `skills/sglang-speculative-decoding-autotuner/SKILL.md`
- Create: `skills/sglang-speculative-decoding-autotuner/references/compatibility-and-search.md`
- Create: `skills/sglang-speculative-decoding-autotuner/references/measurement-schema.md`
- Modify: `tests/test_speculative_decoding_autotuner.py`

- [ ] **Step 1: Add documentation contract tests**

Assert that the skill frontmatter and safety requirements stay present:

```python
class SkillDocumentationTest(unittest.TestCase):
    def test_skill_has_required_contract_and_handoffs(self) -> None:
        skill = (
            ROOT / "skills" / "sglang-speculative-decoding-autotuner" / "SKILL.md"
        ).read_text(encoding="utf-8")
        for required in [
            "name: sglang-speculative-decoding-autotuner",
            "non-speculative baseline",
            "compatibility",
            "correctness",
            "Pareto",
            "no_safe_improvement",
            "llm-serving-auto-benchmark",
            "sglang-sota-humanize-loop",
            "SYNTHETIC FIXTURE",
        ]:
            self.assertIn(required, skill)

    def test_references_define_evidence_and_schema(self) -> None:
        root = ROOT / "skills" / "sglang-speculative-decoding-autotuner"
        compatibility = (root / "references" / "compatibility-and-search.md").read_text(encoding="utf-8")
        schema = (root / "references" / "measurement-schema.md").read_text(encoding="utf-8")
        self.assertIn("selected SGLang revision", compatibility)
        self.assertIn("bounded", compatibility)
        self.assertIn("schema_version", schema)
        self.assertIn("Unknown optional metrics", schema)
```

- [ ] **Step 2: Run tests and verify missing docs**

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner.SkillDocumentationTest -v
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Write `SKILL.md`**

Use this frontmatter:

```yaml
---
name: sglang-speculative-decoding-autotuner
description: "Find a safe, workload-specific SGLang speculative decoding configuration by compatibility-gating and benchmarking baseline, MTP/EAGLE, DFlash, DSpark, or other revision-supported candidates. Use for acceptance-length, TTFT/TPOT, throughput, and memory tradeoffs."
---
```

The body must define:

- required inputs and a stop-before-launch rule for missing identity;
- immutable revision and GPU-state capture;
- authoritative compatibility evidence order;
- mandatory non-speculative baseline;
- bounded candidate generation and one-dimension-first search;
- health, correctness, determinism, memory, and SLA gates;
- fixed workload/warmup/repeat controls;
- analyzer invocation and artifact paths;
- clean-start revalidation of the winner;
- `no_safe_improvement` as a valid outcome;
- scoped cleanup;
- handoff to `llm-serving-auto-benchmark` for general framework comparison and
  `sglang-sota-humanize-loop` for source changes;
- exact fixture demo command and synthetic-data warning.

- [ ] **Step 4: Write compatibility and measurement references**

`compatibility-and-search.md` must explain how to inspect the selected SGLang
revision's CLI, release, source, model config, native MTP/draft availability,
attention backend, quantization, TP/DP/EP/CP/PD, CUDA Graph, and
architecture-specific restrictions. Include a bounded-search table and
candidate stop conditions.

`measurement-schema.md` must document every field accepted in Task 1, required
metric behavior, metric directions, hard limits, raw artifacts, fixture
labeling, analyzer outputs, and the fact that unknown optional metrics stay
unknown.

- [ ] **Step 5: Run documentation tests**

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the operational skill**

```bash
git add skills/sglang-speculative-decoding-autotuner/SKILL.md \
  skills/sglang-speculative-decoding-autotuner/references \
  tests/test_speculative_decoding_autotuner.py
git commit -m "docs: add speculative decoding autotuner workflow"
```

### Task 6: Register the Skill

**Files:**
- Modify: `README.md`
- Modify: `.claude-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`
- Modify: `tests/test_repository_metadata.py`

- [ ] **Step 1: Add failing metadata assertions**

Update the hard-coded badge expectation from `core_skills-11` to
`core_skills-12`, and add:

```python
def test_speculative_decoding_autotuner_is_registered() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    plugin = json.loads(
        (ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    marketplace = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert "sglang-speculative-decoding-autotuner" in readme
    assert "speculative decoding" in plugin["description"].lower()
    assert "speculative" in marketplace["plugins"][0]["description"].lower()
```

- [ ] **Step 2: Run metadata tests and verify failure**

```bash
python3 -m unittest tests.test_repository_metadata -v
```

Expected: failure because README and plugin metadata are not updated.

- [ ] **Step 3: Update README and plugin metadata**

Add the new skill to:

- the headline capability sentence;
- the core skill table;
- per-skill Claude and generic installation commands;
- the list of invocation examples;
- the repository map if it enumerates skill directories.

Change the badge to `core_skills-12`. Change the Claude plugin total from 12
to 13 because the model-history knowledge skill is installed in addition to
the 12 core skills. Mention speculative decoding tuning in both plugin
descriptions without changing the published plugin version.

- [ ] **Step 4: Run metadata and analyzer tests**

```bash
python3 -m unittest tests.test_repository_metadata -v
python3 -m unittest tests.test_speculative_decoding_autotuner -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit registration**

```bash
git add README.md .claude-plugin/plugin.json .claude-plugin/marketplace.json \
  tests/test_repository_metadata.py
git commit -m "docs: register speculative decoding autotuner"
```

### Task 7: Validate the Complete Pull Request

**Files:**
- Modify only when validation exposes an in-scope defect.

- [ ] **Step 1: Regenerate and diff the fixture report**

```bash
python3 skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py \
  --input skills/sglang-speculative-decoding-autotuner/examples/fixture-measurements.json \
  --output-markdown /tmp/speculative-decoding-fixture-report.md \
  --output-json /tmp/speculative-decoding-fixture-result.json
diff -u skills/sglang-speculative-decoding-autotuner/examples/fixture-report.md \
  /tmp/speculative-decoding-fixture-report.md
```

Expected: no diff.

- [ ] **Step 2: Run focused and repository tests**

```bash
python3 -m unittest tests.test_speculative_decoding_autotuner -v
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Run syntax and repository checks**

```bash
python3 -m py_compile \
  skills/sglang-speculative-decoding-autotuner/scripts/analyze_candidates.py \
  tests/test_speculative_decoding_autotuner.py
SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure
git diff --check origin/main...HEAD
```

Expected: all commands exit zero.

- [ ] **Step 4: Review scope and claims**

Run:

```bash
git status --short
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- \
  skills/sglang-speculative-decoding-autotuner README.md \
  .claude-plugin tests/test_speculative_decoding_autotuner.py \
  tests/test_repository_metadata.py
```

Expected: only the planned skill, registration, design, plan, tests, and
generated fixture report appear. Every fixture value is labeled synthetic and
there are no universal GPU-performance claims.

- [ ] **Step 5: Commit validation-driven fixes, if any**

```bash
git status --short
git add skills/sglang-speculative-decoding-autotuner \
  tests/test_speculative_decoding_autotuner.py README.md \
  .claude-plugin/plugin.json .claude-plugin/marketplace.json \
  tests/test_repository_metadata.py
git commit -m "test: validate speculative decoding autotuner"
```

Stage only paths that `git status --short` shows were corrected by validation;
omit unchanged paths. Skip this commit when validation required no file
changes.

- [ ] **Step 6: Push and open the draft PR**

```bash
git push -u origin codex/add-sglang-speculative-decoding-autotuner
gh pr create --draft --base main \
  --head codex/add-sglang-speculative-decoding-autotuner \
  --title "Add SGLang speculative decoding autotuner skill" \
  --body-file /tmp/sglang-speculative-decoding-autotuner-pr.md
```

The PR body must contain the user problem, workflow, exact fixture demo command
and result, test commands, SGLang v0.5.16 evidence scope, synthetic-data
disclaimer, and non-goals.
