#!/usr/bin/env python3
"""Validate and rank measured SGLang speculative decoding candidates."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
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
STATUS_FIELDS = ("healthy", "correct", "deterministic")
LIMITS = {
    "max_ttft_ms": ("ttft_ms", "max"),
    "max_tpot_ms": ("tpot_ms", "max"),
    "max_peak_memory_gb": ("peak_memory_gb", "max"),
    "min_output_throughput": ("output_throughput", "min"),
    "min_request_throughput": ("request_throughput", "min"),
}


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_document(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("measurement root must be an object")
    validate_document(value)
    return value


def validate_document(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")

    experiment = _require_mapping(document.get("experiment"), "experiment")
    experiment_id = _require_string(experiment.get("id"), "experiment.id")
    for field in ("model", "model_revision", "sglang_revision", "hardware"):
        _require_string(experiment.get(field), f"experiment.{field}")
    _require_mapping(experiment.get("workload"), "experiment.workload")

    objective = _require_mapping(
        experiment.get("objective"), "experiment.objective"
    )
    _require_string(objective.get("primary"), "experiment.objective.primary")
    if objective.get("direction") not in VALID_DIRECTIONS:
        raise ValueError(
            "experiment.objective.direction must be minimize or maximize"
        )
    if "minimum_improvement_percent" in objective:
        threshold = _finite_number(
            objective["minimum_improvement_percent"],
            "experiment.objective.minimum_improvement_percent",
        )
        if threshold < 0:
            raise ValueError(
                "experiment.objective.minimum_improvement_percent must be non-negative"
            )

    limits = experiment.get("hard_limits", {})
    _require_mapping(limits, "experiment.hard_limits")
    for name, value in limits.items():
        if name not in LIMITS:
            raise ValueError(f"unknown hard limit: {name}")
        if _finite_number(value, f"experiment.hard_limits.{name}") < 0:
            raise ValueError(f"experiment.hard_limits.{name} must be non-negative")

    pareto_metrics = experiment.get("pareto_metrics")
    if not isinstance(pareto_metrics, list) or not pareto_metrics:
        raise ValueError("experiment.pareto_metrics must be a non-empty array")
    pareto_names: set[str] = set()
    for index, metric in enumerate(pareto_metrics):
        spec = _require_mapping(metric, f"experiment.pareto_metrics[{index}]")
        name = _require_string(spec.get("name"), f"pareto metric {index}.name")
        if name in pareto_names:
            raise ValueError(f"duplicate pareto metric: {name}")
        pareto_names.add(name)
        if spec.get("direction") not in VALID_DIRECTIONS:
            raise ValueError(
                f"pareto metric {name} direction must be minimize or maximize"
            )

    candidates = document.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty array")

    candidate_ids: set[str] = set()
    baseline_count = 0
    for index, candidate_value in enumerate(candidates):
        candidate = _require_mapping(candidate_value, f"candidate {index}")
        candidate_id = _require_string(candidate.get("id"), f"candidate {index}.id")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        candidate_ids.add(candidate_id)
        if candidate.get("baseline") is True:
            baseline_count += 1
        elif candidate.get("baseline") is not False:
            raise ValueError(f"{candidate_id}: baseline must be true or false")
        _require_string(candidate.get("algorithm"), f"{candidate_id}.algorithm")
        if candidate.get("experiment_id") != experiment_id:
            raise ValueError(
                f"{candidate_id}: experiment_id does not match experiment.id"
            )
        command = candidate.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(token, str) and token for token in command)
        ):
            raise ValueError(f"{candidate_id}: command must be a non-empty argv array")
        status = _require_mapping(candidate.get("status"), f"{candidate_id}.status")
        for field in STATUS_FIELDS:
            if not isinstance(status.get(field), bool):
                raise ValueError(f"{candidate_id}.status.{field} must be boolean")
        metrics = _require_mapping(candidate.get("metrics"), f"{candidate_id}.metrics")
        for name, value in metrics.items():
            if value is None:
                continue
            number = _finite_number(value, f"{candidate_id}.metrics.{name}")
            if name in NON_NEGATIVE_METRICS and number < 0:
                raise ValueError(
                    f"{candidate_id}.metrics.{name} must be non-negative"
                )
        repeat_count = candidate.get("repeat_count")
        if not isinstance(repeat_count, int) or isinstance(repeat_count, bool):
            raise ValueError(f"{candidate_id}.repeat_count must be an integer")
        if repeat_count < 1:
            raise ValueError(f"{candidate_id}.repeat_count must be positive")
        artifacts = candidate.get("artifacts")
        if not isinstance(artifacts, list) or not all(
            isinstance(item, str) and item for item in artifacts
        ):
            raise ValueError(f"{candidate_id}.artifacts must be a string array")

    if baseline_count != 1:
        raise ValueError("candidates must contain exactly one baseline")


def _gate_reasons(
    candidate: dict[str, Any],
    experiment: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    status = candidate["status"]
    for field, reason in (
        ("healthy", "health_failed"),
        ("correct", "correctness_failed"),
        ("deterministic", "determinism_failed"),
    ):
        if status[field] is not True:
            reasons.append(reason)

    metrics = candidate["metrics"]
    for limit_name, (metric_name, direction) in LIMITS.items():
        if limit_name not in experiment["hard_limits"]:
            continue
        if metrics.get(metric_name) is None:
            reasons.append(f"hard_limit_metric_missing:{metric_name}")
            continue
        limit = float(experiment["hard_limits"][limit_name])
        value = float(metrics[metric_name])
        failed = value > limit if direction == "max" else value < limit
        if failed:
            reasons.append(
                f"{limit_name}_exceeded"
                if direction == "max"
                else f"{limit_name}_not_met"
            )

    primary = experiment["objective"]["primary"]
    if metrics.get(primary) is None:
        reasons.append(f"objective_metric_missing:{primary}")

    if candidate["baseline"] is not True:
        for spec in experiment["pareto_metrics"]:
            metric_name = spec["name"]
            if metrics.get(metric_name) is None:
                reasons.append(f"pareto_metric_missing:{metric_name}")

    return reasons


def _dominates(
    left: dict[str, Any],
    right: dict[str, Any],
    metric_specs: list[dict[str, str]],
) -> bool:
    at_least_as_good = True
    strictly_better = False
    for spec in metric_specs:
        metric_name = spec["name"]
        left_value = float(left["metrics"][metric_name])
        right_value = float(right["metrics"][metric_name])
        if spec["direction"] == "maximize":
            if left_value < right_value:
                at_least_as_good = False
            if left_value > right_value:
                strictly_better = True
        else:
            if left_value > right_value:
                at_least_as_good = False
            if left_value < right_value:
                strictly_better = True
    return at_least_as_good and strictly_better


def pareto_frontier(
    candidates: list[dict[str, Any]],
    metric_specs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    frontier = [
        candidate
        for candidate in candidates
        if not any(
            other["id"] != candidate["id"]
            and _dominates(other, candidate, metric_specs)
            for other in candidates
        )
    ]
    return sorted(frontier, key=lambda candidate: candidate["id"])


def _improvement_percent(
    baseline: float,
    candidate: float,
    direction: str,
) -> float:
    if baseline == 0:
        if candidate == 0:
            return 0.0
        favorable = candidate > 0 if direction == "maximize" else candidate < 0
        return math.inf if favorable else -math.inf
    favorable_delta = (
        candidate - baseline if direction == "maximize" else baseline - candidate
    )
    return favorable_delta / abs(baseline) * 100.0


def _metric_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name, baseline_value in baseline["metrics"].items():
        candidate_value = candidate["metrics"].get(name)
        if baseline_value is None or candidate_value is None:
            continue
        if isinstance(baseline_value, bool) or isinstance(candidate_value, bool):
            continue
        if not isinstance(baseline_value, (int, float)) or not isinstance(
            candidate_value, (int, float)
        ):
            continue
        deltas[name] = float(candidate_value) - float(baseline_value)
    return deltas


def analyze(document: dict[str, Any]) -> dict[str, Any]:
    validate_document(document)
    experiment = document["experiment"]
    candidates = document["candidates"]
    baseline = next(candidate for candidate in candidates if candidate["baseline"])

    rejected: dict[str, list[str]] = {}
    accepted: list[dict[str, Any]] = []
    for candidate in candidates:
        reasons = _gate_reasons(candidate, experiment)
        if reasons:
            rejected[candidate["id"]] = reasons
        else:
            accepted.append(candidate)

    speculative = [
        candidate for candidate in accepted if candidate["baseline"] is not True
    ]
    frontier = pareto_frontier(speculative, experiment["pareto_metrics"])
    objective = experiment["objective"]
    primary = objective["primary"]

    if baseline["id"] in rejected or not frontier:
        recommendation: dict[str, Any] = {
            "status": "no_safe_improvement",
            "candidate_id": None,
            "improvement_percent": None,
        }
    else:
        direction = objective["direction"]
        winner = sorted(
            frontier,
            key=lambda candidate: (
                -float(candidate["metrics"][primary])
                if direction == "maximize"
                else float(candidate["metrics"][primary]),
                candidate["id"],
            ),
        )[0]
        improvement = _improvement_percent(
            float(baseline["metrics"][primary]),
            float(winner["metrics"][primary]),
            direction,
        )
        threshold = float(objective.get("minimum_improvement_percent", 0.0))
        is_recommended = improvement >= threshold
        recommendation = {
            "status": "recommended" if is_recommended else "no_safe_improvement",
            "candidate_id": winner["id"] if is_recommended else None,
            "improvement_percent": improvement,
        }

    evaluations = []
    for candidate in sorted(candidates, key=lambda item: item["id"]):
        evaluations.append(
            {
                "id": candidate["id"],
                "baseline": candidate["baseline"],
                "algorithm": candidate["algorithm"],
                "accepted": candidate["id"] not in rejected,
                "reasons": rejected.get(candidate["id"], []),
                "metrics": candidate["metrics"],
                "baseline_deltas": _metric_deltas(baseline, candidate),
                "command": candidate["command"],
                "repeat_count": candidate["repeat_count"],
                "artifacts": candidate["artifacts"],
            }
        )

    return {
        "schema_version": 1,
        "fixture": document.get("fixture") is True,
        "experiment": experiment,
        "baseline_id": baseline["id"],
        "accepted": sorted(candidate["id"] for candidate in accepted),
        "rejected": rejected,
        "pareto_frontier": [candidate["id"] for candidate in frontier],
        "recommendation": recommendation,
        "evaluations": evaluations,
    }
