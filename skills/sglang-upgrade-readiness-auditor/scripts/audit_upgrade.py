#!/usr/bin/env python3
"""Audit SGLang version changes against concrete deployment profiles."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


VERSION_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?$")
VALID_SEVERITIES = {
    "blocker",
    "required",
    "behavior",
    "risk",
    "dependency",
    "info",
}
VALID_TRANSFORMS = {
    "rename_flag",
    "remove_flag",
    "replace_value",
    "replace_import_prefix",
}
VALID_PREDICATES = {
    "argv_flag",
    "argv_value",
    "env",
    "feature",
    "guarantee",
    "integration",
    "import_prefix",
    "model_family",
    "quantization",
    "hardware",
    "topology",
}
VALID_APPLICABILITY_MODES = {"crossing", "target"}
VALID_CANARY_RESULTS = {"pass", "fail", "not_run"}


def parse_version(value: str) -> tuple[int, int, int, int]:
    if not isinstance(value, str):
        raise ValueError("SGLang version must be a string")
    match = VERSION_RE.fullmatch(value)
    if not match:
        raise ValueError(f"unsupported SGLang version: {value}")
    major, minor, patch, post = match.groups()
    return int(major), int(minor), int(patch), int(post or 0)


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a string array")
    return value


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def validate_profiles(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("profile schema_version must be 1")
    if "fixture" in document and not isinstance(document["fixture"], bool):
        raise ValueError("fixture must be boolean")

    audit = _require_mapping(document.get("audit"), "audit")
    current_text = _require_string(
        audit.get("current_version"), "audit.current_version"
    )
    target_text = _require_string(
        audit.get("target_version"), "audit.target_version"
    )
    current = parse_version(current_text)
    target = parse_version(target_text)
    if current >= target:
        raise ValueError("target_version must be newer than current_version")
    _require_string_list(
        audit.get("required_canaries", []), "audit.required_canaries"
    )

    profiles = document.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles must be a non-empty array")
    profile_ids: set[str] = set()
    for index, profile_value in enumerate(profiles):
        profile = _require_mapping(profile_value, f"profile {index}")
        profile_id = _require_string(profile.get("id"), f"profile {index}.id")
        if profile_id in profile_ids:
            raise ValueError(f"duplicate profile id: {profile_id}")
        profile_ids.add(profile_id)

        argv = profile.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(token, str) and token for token in argv)
        ):
            raise ValueError(f"{profile_id}: argv must be a non-empty string array")
        env = _require_mapping(profile.get("env", {}), f"{profile_id}.env")
        for name, value in env.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{profile_id}: env names must be non-empty strings")
            if not isinstance(value, str):
                raise ValueError(f"{profile_id}: env values must be strings")
        for field in ("model_family", "quantization", "hardware"):
            _require_string(profile.get(field), f"{profile_id}.{field}")
        topology = _require_mapping(
            profile.get("topology"), f"{profile_id}.topology"
        )
        for name, value in topology.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(
                    f"{profile_id}.topology values must be positive integers"
                )
        for field in ("features", "guarantees", "integrations", "imports"):
            _require_string_list(profile.get(field, []), f"{profile_id}.{field}")
        canary_results = _require_mapping(
            profile.get("canary_results", {}), f"{profile_id}.canary_results"
        )
        for name, value in canary_results.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"{profile_id}: canary names must be non-empty strings"
                )
            if value not in VALID_CANARY_RESULTS:
                raise ValueError(
                    f"{profile_id}.{name}: canary result must be pass, fail, or not_run"
                )


def validate_rules(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("rule schema_version must be 1")
    rules = document.get("rules")
    if not isinstance(rules, list):
        raise ValueError("rules must be an array")

    rule_ids: set[str] = set()
    for index, rule_value in enumerate(rules):
        rule = _require_mapping(rule_value, f"rule {index}")
        rule_id = _require_string(rule.get("id"), f"rule {index}.id")
        if rule_id in rule_ids:
            raise ValueError(f"duplicate rule id: {rule_id}")
        rule_ids.add(rule_id)
        _require_string(rule.get("category"), f"{rule_id}.category")
        if rule.get("severity") not in VALID_SEVERITIES:
            raise ValueError(f"{rule_id}: invalid severity")
        _require_string(rule.get("title"), f"{rule_id}.title")
        source_url = _require_string(rule.get("source_url"), f"{rule_id}.source_url")
        if not source_url.startswith(("https://", "http://")):
            raise ValueError(f"{rule_id}.source_url must be an HTTP URL")
        _require_string(rule.get("summary"), f"{rule_id}.summary")
        _require_string(rule.get("rollback"), f"{rule_id}.rollback")

        applies = _require_mapping(rule.get("applies"), f"{rule_id}.applies")
        if applies.get("mode", "crossing") not in VALID_APPLICABILITY_MODES:
            raise ValueError(f"{rule_id}: invalid applicability mode")
        introduced = parse_version(
            _require_string(
                applies.get("introduced_in"), f"{rule_id}.introduced_in"
            )
        )
        fixed_text = applies.get("fixed_in")
        if fixed_text is not None:
            fixed = parse_version(
                _require_string(fixed_text, f"{rule_id}.fixed_in")
            )
            if fixed <= introduced:
                raise ValueError(f"{rule_id}.fixed_in must follow introduced_in")

        match = _require_mapping(rule.get("match"), f"{rule_id}.match")
        predicates = match.get("all")
        if not isinstance(predicates, list) or not predicates:
            raise ValueError(f"{rule_id}: match.all must be a non-empty array")
        for predicate_value in predicates:
            predicate = _require_mapping(
                predicate_value, f"{rule_id}.match predicate"
            )
            if predicate.get("kind") not in VALID_PREDICATES:
                raise ValueError(f"{rule_id}: unknown predicate")

        transforms = rule.get("transforms", [])
        if not isinstance(transforms, list):
            raise ValueError(f"{rule_id}.transforms must be an array")
        for transform_value in transforms:
            transform = _require_mapping(
                transform_value, f"{rule_id}.transform"
            )
            if transform.get("kind") not in VALID_TRANSFORMS:
                raise ValueError(f"{rule_id}: unknown transform")
        _require_string_list(rule.get("canaries", []), f"{rule_id}.canaries")


def rule_applies(
    rule: dict[str, Any],
    current: str,
    target: str,
) -> bool:
    current_version = parse_version(current)
    target_version = parse_version(target)
    introduced = parse_version(rule["applies"]["introduced_in"])
    fixed_text = rule["applies"].get("fixed_in")
    if fixed_text is not None and target_version >= parse_version(fixed_text):
        return False
    mode = rule["applies"].get("mode", "crossing")
    if mode == "target":
        return target_version >= introduced
    return current_version < introduced <= target_version


def _flag_index(argv: list[str], name: str) -> int | None:
    try:
        return argv.index(name)
    except ValueError:
        return None


def _predicate_matches(
    predicate: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    kind = predicate["kind"]
    if kind == "argv_flag":
        return _flag_index(profile["argv"], predicate["name"]) is not None
    if kind == "argv_value":
        index = _flag_index(profile["argv"], predicate["name"])
        return (
            index is not None
            and index + 1 < len(profile["argv"])
            and profile["argv"][index + 1] == predicate["equals"]
        )
    if kind == "env":
        value = profile.get("env", {}).get(predicate["name"])
        if "equals" in predicate:
            return value == predicate["equals"]
        return value is not None
    if kind in {"feature", "guarantee", "integration"}:
        collection_name = {
            "feature": "features",
            "guarantee": "guarantees",
            "integration": "integrations",
        }[kind]
        return predicate["value"] in profile.get(collection_name, [])
    if kind == "import_prefix":
        prefix = predicate["value"]
        return any(
            value == prefix or value.startswith(prefix + ".")
            for value in profile.get("imports", [])
        )
    if kind in {"model_family", "quantization", "hardware"}:
        return profile.get(kind) == predicate["equals"]
    if kind == "topology":
        value = profile.get("topology", {}).get(predicate["name"])
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if "equals" in predicate and value != predicate["equals"]:
            return False
        if "min" in predicate and value < predicate["min"]:
            return False
        if "max" in predicate and value > predicate["max"]:
            return False
        return True
    raise ValueError(f"unsupported predicate: {kind}")


def rule_matches(
    rule: dict[str, Any],
    profile: dict[str, Any],
) -> bool:
    return all(
        _predicate_matches(predicate, profile)
        for predicate in rule["match"]["all"]
    )


def _transform_key(transform: dict[str, Any]) -> tuple[str, str]:
    kind = transform["kind"]
    if kind == "rename_flag":
        return "argv_flag", transform["from"]
    if kind == "remove_flag":
        return "argv_flag", transform["name"]
    if kind == "replace_value":
        return "argv_flag", transform["flag"]
    return "import_prefix", transform["from"]


def detect_transform_conflicts(transforms: list[dict[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for transform in transforms:
        key = _transform_key(transform)
        rendered = json.dumps(transform, sort_keys=True)
        if key in seen and seen[key] != rendered:
            raise ValueError(f"conflicting transformations for {key[1]}")
        seen[key] = rendered


def _single_flag_index(argv: list[str], name: str) -> int:
    indices = [index for index, token in enumerate(argv) if token == name]
    if not indices:
        raise ValueError(f"flag not found for transformation: {name}")
    if len(indices) != 1:
        raise ValueError(f"flag {name} appears {len(indices)} times")
    return indices[0]


def apply_transforms(
    argv: list[str],
    imports: list[str],
    transforms: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    detect_transform_conflicts(transforms)
    rewritten_argv = list(argv)
    rewritten_imports = list(imports)
    applied_signatures: set[str] = set()
    for transform in transforms:
        signature = json.dumps(transform, sort_keys=True)
        if signature in applied_signatures:
            continue
        applied_signatures.add(signature)
        kind = transform["kind"]
        if kind == "rename_flag":
            index = _single_flag_index(rewritten_argv, transform["from"])
            rewritten_argv[index] = transform["to"]
        elif kind == "remove_flag":
            index = _single_flag_index(rewritten_argv, transform["name"])
            arity = transform.get("arity", 0)
            if arity not in {0, 1}:
                raise ValueError(
                    f"invalid removal arity for {transform['name']}: {arity}"
                )
            if index + arity >= len(rewritten_argv):
                raise ValueError(
                    f"missing value for removal of {transform['name']}"
                )
            del rewritten_argv[index : index + arity + 1]
        elif kind == "replace_value":
            index = _single_flag_index(rewritten_argv, transform["flag"])
            if index + 1 >= len(rewritten_argv):
                raise ValueError(
                    f"missing value for replacement of {transform['flag']}"
                )
            actual = rewritten_argv[index + 1]
            if actual != transform["from"]:
                raise ValueError(
                    f"unexpected value for {transform['flag']}: {actual}"
                )
            rewritten_argv[index + 1] = transform["to"]
        elif kind == "replace_import_prefix":
            source = transform["from"]
            target = transform["to"]
            rewritten_imports = [
                target + value[len(source) :]
                if value == source or value.startswith(source + ".")
                else value
                for value in rewritten_imports
            ]
    return rewritten_argv, rewritten_imports
