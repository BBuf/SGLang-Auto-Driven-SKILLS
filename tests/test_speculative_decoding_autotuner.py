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


if __name__ == "__main__":
    unittest.main()
