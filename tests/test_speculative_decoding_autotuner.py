from __future__ import annotations

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "sglang-speculative-decoding-autotuner"
SCRIPT = SKILL_ROOT / "scripts" / "analyze_candidates.py"


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
            "workload": {
                "input_tokens": 2048,
                "output_tokens": 256,
                "concurrency": 1,
            },
            "objective": {
                "primary": "output_throughput",
                "direction": "maximize",
                "minimum_improvement_percent": 3.0,
            },
            "hard_limits": {
                "max_ttft_ms": 500.0,
                "max_tpot_ms": 5.0,
                "max_peak_memory_gb": 80.0,
            },
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
                "command": [
                    "python3",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    "fixture/model",
                ],
                "experiment_id": "fixture-search",
                "status": {
                    "healthy": True,
                    "correct": True,
                    "deterministic": True,
                },
                "metrics": {
                    "ttft_ms": 120.0,
                    "tpot_ms": 4.5,
                    "output_throughput": 100.0,
                    "peak_memory_gb": 70.0,
                    "acceptance_length": None,
                },
                "repeat_count": 3,
                "artifacts": ["examples/raw/baseline.json"],
            }
        ],
    }


def add_candidate(
    document: dict,
    *,
    candidate_id: str,
    algorithm: str,
    ttft_ms: float,
    tpot_ms: float,
    output_throughput: float,
    peak_memory_gb: float = 75.0,
    healthy: bool = True,
    correct: bool = True,
    deterministic: bool = True,
) -> dict:
    candidate = copy.deepcopy(document["candidates"][0])
    candidate.update(
        {
            "id": candidate_id,
            "baseline": False,
            "algorithm": algorithm,
            "status": {
                "healthy": healthy,
                "correct": correct,
                "deterministic": deterministic,
            },
            "metrics": {
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "output_throughput": output_throughput,
                "peak_memory_gb": peak_memory_gb,
                "acceptance_length": 4.0,
            },
        }
    )
    document["candidates"].append(candidate)
    return candidate


class MeasurementValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_valid_document_is_accepted(self) -> None:
        self.mod.validate_document(valid_document())

    def test_duplicate_candidate_id_is_rejected(self) -> None:
        document = valid_document()
        document["candidates"].append(copy.deepcopy(document["candidates"][0]))

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

    def test_non_finite_metric_is_rejected_but_unknown_optional_metric_is_allowed(
        self,
    ) -> None:
        document = valid_document()
        self.mod.validate_document(document)
        document["candidates"][0]["metrics"]["ttft_ms"] = math.inf

        with self.assertRaisesRegex(ValueError, "must be finite"):
            self.mod.validate_document(document)


class CandidateDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_hard_gates_precede_performance_ranking(self) -> None:
        document = valid_document()
        add_candidate(
            document,
            candidate_id="wrong-fast",
            algorithm="DSPARK",
            ttft_ms=80.0,
            tpot_ms=2.0,
            output_throughput=180.0,
            correct=False,
        )
        add_candidate(
            document,
            candidate_id="slow-tpot",
            algorithm="EAGLE",
            ttft_ms=90.0,
            tpot_ms=6.0,
            output_throughput=150.0,
        )

        result = self.mod.analyze(document)

        self.assertEqual(result["rejected"]["wrong-fast"], ["correctness_failed"])
        self.assertEqual(
            result["rejected"]["slow-tpot"], ["max_tpot_ms_exceeded"]
        )
        self.assertEqual(result["recommendation"]["status"], "no_safe_improvement")

    def test_pareto_frontier_and_objective_select_safe_winner(self) -> None:
        document = valid_document()
        add_candidate(
            document,
            candidate_id="mtp-low-latency",
            algorithm="MTP",
            ttft_ms=95.0,
            tpot_ms=3.7,
            output_throughput=118.0,
        )
        add_candidate(
            document,
            candidate_id="dspark-balanced",
            algorithm="DSPARK",
            ttft_ms=100.0,
            tpot_ms=4.1,
            output_throughput=135.0,
        )
        add_candidate(
            document,
            candidate_id="dominated",
            algorithm="EAGLE",
            ttft_ms=110.0,
            tpot_ms=4.8,
            output_throughput=110.0,
        )

        result = self.mod.analyze(document)

        self.assertEqual(
            result["pareto_frontier"], ["dspark-balanced", "mtp-low-latency"]
        )
        self.assertEqual(
            result["recommendation"]["candidate_id"], "dspark-balanced"
        )
        self.assertEqual(result["recommendation"]["status"], "recommended")
        self.assertAlmostEqual(
            result["recommendation"]["improvement_percent"], 35.0
        )

    def test_below_noise_threshold_returns_no_safe_improvement(self) -> None:
        document = valid_document()
        add_candidate(
            document,
            candidate_id="tiny-gain",
            algorithm="MTP",
            ttft_ms=110.0,
            tpot_ms=4.4,
            output_throughput=102.0,
        )

        result = self.mod.analyze(document)

        self.assertEqual(
            result["recommendation"]["status"], "no_safe_improvement"
        )
        self.assertIsNone(result["recommendation"]["candidate_id"])
        self.assertAlmostEqual(result["recommendation"]["improvement_percent"], 2.0)

    def test_missing_selection_metric_is_rejected(self) -> None:
        document = valid_document()
        candidate = add_candidate(
            document,
            candidate_id="missing-tpot",
            algorithm="MTP",
            ttft_ms=90.0,
            tpot_ms=4.0,
            output_throughput=120.0,
        )
        del candidate["metrics"]["tpot_ms"]

        result = self.mod.analyze(document)

        self.assertEqual(
            result["rejected"]["missing-tpot"],
            [
                "hard_limit_metric_missing:tpot_ms",
                "pareto_metric_missing:tpot_ms",
            ],
        )


class ReportAndCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_markdown_labels_fixture_and_explains_decision(self) -> None:
        document = valid_document()
        add_candidate(
            document,
            candidate_id="wrong-fast",
            algorithm="DSPARK",
            ttft_ms=80.0,
            tpot_ms=2.0,
            output_throughput=180.0,
            correct=False,
        )
        add_candidate(
            document,
            candidate_id="mtp-safe",
            algorithm="MTP",
            ttft_ms=100.0,
            tpot_ms=3.8,
            output_throughput=125.0,
        )

        report = self.mod.render_markdown(self.mod.analyze(document))

        self.assertIn("SYNTHETIC FIXTURE", report)
        self.assertIn("## Gate Results", report)
        self.assertIn("correctness_failed", report)
        self.assertIn("## Metrics and Baseline Deltas", report)
        self.assertIn("+25.00", report)
        self.assertIn("## Pareto Frontier", report)
        self.assertIn("## Recommendation", report)
        self.assertIn("`mtp-safe`", report)
        self.assertIn("python3 -m sglang.launch_server", report)
        self.assertIn("examples/raw/baseline.json", report)

    def test_cli_writes_markdown_and_json(self) -> None:
        document = valid_document()
        add_candidate(
            document,
            candidate_id="mtp-safe",
            algorithm="MTP",
            ttft_ms=100.0,
            tpot_ms=3.8,
            output_throughput=125.0,
        )
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
            self.assertIn(
                "SYNTHETIC FIXTURE", markdown_path.read_text(encoding="utf-8")
            )
            result = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], 1)
            self.assertEqual(
                result["recommendation"]["candidate_id"], "mtp-safe"
            )

    def test_cli_returns_two_for_invalid_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.json"
            input_path.write_text("[]", encoding="utf-8")

            exit_code = self.mod.main(
                [
                    "--input",
                    str(input_path),
                    "--output-markdown",
                    str(root / "report.md"),
                    "--output-json",
                    str(root / "report.json"),
                ]
            )

            self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
