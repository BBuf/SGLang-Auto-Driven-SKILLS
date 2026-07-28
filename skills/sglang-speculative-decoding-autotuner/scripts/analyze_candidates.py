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
