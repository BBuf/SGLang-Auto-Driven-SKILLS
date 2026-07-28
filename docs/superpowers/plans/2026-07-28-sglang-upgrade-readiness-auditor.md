# SGLang Upgrade Readiness Auditor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only SGLang upgrade auditor that matches evidence-backed version changes against concrete deployment profiles, proposes safe argv rewrites, and emits per-profile and overall readiness verdicts with canary requirements.

**Architecture:** `SKILL.md` owns source collection and rule authoring from immutable releases and diffs. A standard-library Python analyzer owns version applicability, deployment matching, conflict-safe argv transformations, verdict aggregation, and Markdown/JSON output. A synthetic v0.5.15-to-v0.5.16 fixture demonstrates removed paths, flag renames, behavior changes, and known risks.

**Tech Stack:** Markdown, Python 3.10+ standard library, `unittest`, JSON, repository pre-commit and link checks.

---

## File Structure

- Create `skills/sglang-upgrade-readiness-auditor/SKILL.md`: read-only inventory, evidence, audit, canary, verdict, and rollback workflow.
- Create `skills/sglang-upgrade-readiness-auditor/references/evidence-and-rule-authoring.md`: source priority and rule-authoring guidance.
- Create `skills/sglang-upgrade-readiness-auditor/references/profile-and-rule-schema.md`: exact JSON schemas and transformations.
- Create `skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py`: validator, matcher, transformer, verdict engine, and reporters.
- Create `skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-profiles.json`: synthetic deployment inventory.
- Create `skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-rules.json`: source-linked release rules.
- Create `skills/sglang-upgrade-readiness-auditor/examples/fixture-report.md`: generated demonstration.
- Create `tests/test_upgrade_readiness_auditor.py`: version, matching, rewrite, verdict, report, fixture, and documentation tests.
- Modify `README.md`: register the new core skill, installation commands, examples, and count.
- Modify `.claude-plugin/plugin.json`: mention upgrade readiness auditing.
- Modify `.claude-plugin/marketplace.json`: mention upgrade auditing in discovery metadata.
- Modify `tests/test_repository_metadata.py`: update core-skill count and assert registration.

### Task 1: Validate Versions, Profiles, and Rules

**Files:**
- Create: `tests/test_upgrade_readiness_auditor.py`
- Create: `skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py`

- [ ] **Step 1: Write failing version and schema tests**

```python
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "skills"
    / "sglang-upgrade-readiness-auditor"
    / "scripts"
    / "audit_upgrade.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("upgrade_auditor", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_profiles() -> dict:
    return {
        "schema_version": 1,
        "fixture": True,
        "audit": {
            "current_version": "v0.5.15",
            "target_version": "v0.5.16",
            "required_canaries": ["server_health", "correctness", "performance"],
        },
        "profiles": [
            {
                "id": "plain-tp",
                "argv": ["python3", "-m", "sglang.launch_server", "--model-path", "fixture/model", "--tp", "8"],
                "env": {},
                "model_family": "dense",
                "quantization": "fp8",
                "hardware": "b200",
                "topology": {"tp": 8, "dp": 1, "ep": 1, "cp": 1},
                "features": [],
                "guarantees": [],
                "integrations": [],
                "imports": [],
                "canary_results": {
                    "server_health": "pass",
                    "correctness": "pass",
                    "performance": "pass"
                },
            }
        ],
    }


def valid_rules() -> dict:
    return {
        "schema_version": 1,
        "rules": [
            {
                "id": "rename-waterfill",
                "category": "required_change",
                "severity": "required",
                "title": "Waterfill flag renamed",
                "applies": {"introduced_in": "v0.5.16"},
                "source_url": "https://github.com/sgl-project/sglang/pull/27350",
                "summary": "The old flag has no deprecated alias.",
                "match": {
                    "all": [
                        {"kind": "argv_flag", "name": "--enable-deepep-waterfill"}
                    ]
                },
                "transforms": [
                    {
                        "kind": "rename_flag",
                        "from": "--enable-deepep-waterfill",
                        "to": "--enable-waterfill"
                    }
                ],
                "canaries": ["server_health", "performance"],
            }
        ],
    }


class SchemaAndVersionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_versions_support_post_releases(self) -> None:
        self.assertLess(
            self.mod.parse_version("v0.5.15"),
            self.mod.parse_version("v0.5.15.post1"),
        )

    def test_valid_documents_are_accepted(self) -> None:
        self.mod.validate_profiles(valid_profiles())
        self.mod.validate_rules(valid_rules())

    def test_duplicate_profile_id_is_rejected(self) -> None:
        document = valid_profiles()
        document["profiles"].append(dict(document["profiles"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate profile id"):
            self.mod.validate_profiles(document)

    def test_unknown_transform_is_rejected(self) -> None:
        rules = valid_rules()
        rules["rules"][0]["transforms"][0]["kind"] = "execute_shell"
        with self.assertRaisesRegex(ValueError, "unknown transform"):
            self.mod.validate_rules(rules)
```

- [ ] **Step 2: Run tests and verify the module is missing**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor.SchemaAndVersionTest -v
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Implement parsing and schema validation**

Create:

```python
#!/usr/bin/env python3
"""Audit SGLang version changes against deployment profiles."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any

VERSION_RE = re.compile(r"^v?(\\d+)\\.(\\d+)\\.(\\d+)(?:\\.post(\\d+))?$")
VALID_SEVERITIES = {"blocker", "required", "behavior", "risk", "dependency", "info"}
VALID_TRANSFORMS = {"rename_flag", "remove_flag", "replace_value", "replace_import_prefix"}
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


def parse_version(value: str) -> tuple[int, int, int, int]:
    match = VERSION_RE.fullmatch(value)
    if not match:
        raise ValueError(f"unsupported SGLang version: {value}")
    major, minor, patch, post = match.groups()
    return int(major), int(minor), int(patch), int(post or 0)


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_profiles(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("profile schema_version must be 1")
    audit = document.get("audit")
    profiles = document.get("profiles")
    if not isinstance(audit, dict):
        raise ValueError("audit must be an object")
    current = parse_version(_require_string(audit.get("current_version"), "current_version"))
    target = parse_version(_require_string(audit.get("target_version"), "target_version"))
    if current >= target:
        raise ValueError("target_version must be newer than current_version")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("profiles must be a non-empty array")
    ids: set[str] = set()
    for profile in profiles:
        profile_id = _require_string(profile.get("id"), "profile id")
        if profile_id in ids:
            raise ValueError(f"duplicate profile id: {profile_id}")
        ids.add(profile_id)
        argv = profile.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(token, str) and token for token in argv
        ):
            raise ValueError(f"{profile_id}: argv must be a non-empty string array")
        if not isinstance(profile.get("env", {}), dict):
            raise ValueError(f"{profile_id}: env must be an object")


def validate_rules(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1:
        raise ValueError("rule schema_version must be 1")
    rules = document.get("rules")
    if not isinstance(rules, list):
        raise ValueError("rules must be an array")
    ids: set[str] = set()
    for rule in rules:
        rule_id = _require_string(rule.get("id"), "rule id")
        if rule_id in ids:
            raise ValueError(f"duplicate rule id: {rule_id}")
        ids.add(rule_id)
        if rule.get("severity") not in VALID_SEVERITIES:
            raise ValueError(f"{rule_id}: invalid severity")
        parse_version(_require_string(rule.get("applies", {}).get("introduced_in"), f"{rule_id}.introduced_in"))
        fixed_in = rule.get("applies", {}).get("fixed_in")
        if fixed_in is not None:
            parse_version(_require_string(fixed_in, f"{rule_id}.fixed_in"))
        _require_string(rule.get("source_url"), f"{rule_id}.source_url")
        predicates = rule.get("match", {}).get("all")
        if not isinstance(predicates, list) or not predicates:
            raise ValueError(f"{rule_id}: match.all must be non-empty")
        for predicate in predicates:
            if predicate.get("kind") not in VALID_PREDICATES:
                raise ValueError(f"{rule_id}: unknown predicate")
        for transform in rule.get("transforms", []):
            if transform.get("kind") not in VALID_TRANSFORMS:
                raise ValueError(f"{rule_id}: unknown transform")
```

- [ ] **Step 4: Run schema tests**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor.SchemaAndVersionTest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit schema boundaries**

```bash
git add tests/test_upgrade_readiness_auditor.py \
  skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py
git commit -m "feat: validate SGLang upgrade audit inputs"
```

### Task 2: Match Versioned Rules to Deployment Profiles

**Files:**
- Modify: `tests/test_upgrade_readiness_auditor.py`
- Modify: `skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py`

- [ ] **Step 1: Add failing applicability and predicate tests**

```python
class RuleMatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_rule_applies_when_upgrade_crosses_introduced_version(self) -> None:
        rule = valid_rules()["rules"][0]
        self.assertTrue(self.mod.rule_applies(rule, "v0.5.15", "v0.5.16"))
        self.assertFalse(self.mod.rule_applies(rule, "v0.5.16", "v0.5.17"))

    def test_fixed_known_issue_does_not_apply_to_fixed_target(self) -> None:
        rule = valid_rules()["rules"][0]
        rule["applies"]["fixed_in"] = "v0.5.17"
        self.assertFalse(self.mod.rule_applies(rule, "v0.5.15", "v0.5.17"))

    def test_all_predicates_must_match(self) -> None:
        profile = valid_profiles()["profiles"][0]
        rule = valid_rules()["rules"][0]
        rule["match"]["all"] = [
            {"kind": "hardware", "equals": "b200"},
            {"kind": "topology", "name": "tp", "min": 8},
            {"kind": "feature", "value": "breakable_prefill_cuda_graph"},
        ]
        self.assertFalse(self.mod.rule_matches(rule, profile))
        profile["features"].append("breakable_prefill_cuda_graph")
        self.assertTrue(self.mod.rule_matches(rule, profile))
```

- [ ] **Step 2: Run matching tests and verify missing functions**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor.RuleMatchingTest -v
```

Expected: failures for missing `rule_applies` and `rule_matches`.

- [ ] **Step 3: Implement version applicability**

```python
def rule_applies(rule: dict[str, Any], current: str, target: str) -> bool:
    current_version = parse_version(current)
    target_version = parse_version(target)
    introduced = parse_version(rule["applies"]["introduced_in"])
    if not (current_version < introduced <= target_version):
        return False
    fixed_in = rule["applies"].get("fixed_in")
    return fixed_in is None or target_version < parse_version(fixed_in)
```

- [ ] **Step 4: Implement all supported predicates**

Add helpers for locating flag values and:

```python
def _flag_index(argv: list[str], name: str) -> int | None:
    try:
        return argv.index(name)
    except ValueError:
        return None


def _predicate_matches(predicate: dict[str, Any], profile: dict[str, Any]) -> bool:
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
        return value == predicate.get("equals") if "equals" in predicate else value is not None
    if kind in {"feature", "guarantee", "integration"}:
        collection_name = {
            "feature": "features",
            "guarantee": "guarantees",
            "integration": "integrations",
        }[kind]
        return predicate["value"] in profile.get(collection_name, [])
    if kind == "import_prefix":
        return any(
            value == predicate["value"] or value.startswith(predicate["value"] + ".")
            for value in profile.get("imports", [])
        )
    if kind in {"model_family", "quantization", "hardware"}:
        return profile.get(kind) == predicate["equals"]
    if kind == "topology":
        value = profile.get("topology", {}).get(predicate["name"])
        if not isinstance(value, int):
            return False
        if "equals" in predicate and value != predicate["equals"]:
            return False
        if "min" in predicate and value < predicate["min"]:
            return False
        if "max" in predicate and value > predicate["max"]:
            return False
        return True
    raise ValueError(f"unsupported predicate: {kind}")


def rule_matches(rule: dict[str, Any], profile: dict[str, Any]) -> bool:
    return all(
        _predicate_matches(predicate, profile)
        for predicate in rule["match"]["all"]
    )
```

- [ ] **Step 5: Run matching tests**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor.RuleMatchingTest -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit rule matching**

```bash
git add tests/test_upgrade_readiness_auditor.py \
  skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py
git commit -m "feat: match upgrade rules to deployments"
```

### Task 3: Add Safe Transformations and Conflict Detection

**Files:**
- Modify: `tests/test_upgrade_readiness_auditor.py`
- Modify: `skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py`

- [ ] **Step 1: Add failing argv and import transformation tests**

```python
class TransformationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_rename_and_replace_value_are_token_level(self) -> None:
        argv = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--enable-deepep-waterfill",
            "--fp4-gemm-backend",
            "cutlass",
        ]
        transforms = [
            {"kind": "rename_flag", "from": "--enable-deepep-waterfill", "to": "--enable-waterfill"},
            {"kind": "replace_value", "flag": "--fp4-gemm-backend", "from": "cutlass", "to": "auto"},
        ]
        rewritten, imports = self.mod.apply_transforms(argv, [], transforms)
        self.assertEqual(
            rewritten,
            [
                "python3",
                "-m",
                "sglang.launch_server",
                "--enable-waterfill",
                "--fp4-gemm-backend",
                "auto",
            ],
        )
        self.assertEqual(imports, [])

    def test_remove_flag_respects_arity(self) -> None:
        rewritten, _ = self.mod.apply_transforms(
            ["server", "--removed", "value", "--keep"],
            [],
            [{"kind": "remove_flag", "name": "--removed", "arity": 1}],
        )
        self.assertEqual(rewritten, ["server", "--keep"])

    def test_conflicting_rewrites_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting transformations"):
            self.mod.detect_transform_conflicts(
                [
                    {"kind": "rename_flag", "from": "--old", "to": "--new-a"},
                    {"kind": "rename_flag", "from": "--old", "to": "--new-b"},
                ]
            )
```

- [ ] **Step 2: Run transformation tests and verify failure**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor.TransformationTest -v
```

Expected: failures for missing transformation functions.

- [ ] **Step 3: Implement conflict keys and transformations**

Implement deterministic conflict keys and exact transformations:

```python
def _transform_key(transform: dict[str, Any]) -> tuple[str, str]:
    kind = transform["kind"]
    if kind == "rename_flag":
        return "flag", transform["from"]
    if kind == "remove_flag":
        return "flag", transform["name"]
    if kind == "replace_value":
        return "flag_value", transform["flag"]
    return "import", transform["from"]


def detect_transform_conflicts(transforms: list[dict[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for transform in transforms:
        key = _transform_key(transform)
        rendered = json.dumps(transform, sort_keys=True)
        if key in seen and seen[key] != rendered:
            raise ValueError(f"conflicting transformations for {key[1]}")
        seen[key] = rendered


def apply_transforms(
    argv: list[str],
    imports: list[str],
    transforms: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    detect_transform_conflicts(transforms)
    rewritten = list(argv)
    rewritten_imports = list(imports)
    for transform in transforms:
        kind = transform["kind"]
        if kind == "rename_flag":
            index = _flag_index(rewritten, transform["from"])
            if index is None:
                raise ValueError(f"flag not found for rename: {transform['from']}")
            rewritten[index] = transform["to"]
        elif kind == "remove_flag":
            index = _flag_index(rewritten, transform["name"])
            if index is None:
                raise ValueError(f"flag not found for removal: {transform['name']}")
            arity = transform.get("arity", 0)
            if arity not in {0, 1} or index + arity >= len(rewritten):
                raise ValueError(f"invalid removal arity for {transform['name']}")
            del rewritten[index : index + arity + 1]
        elif kind == "replace_value":
            index = _flag_index(rewritten, transform["flag"])
            if (
                index is None
                or index + 1 >= len(rewritten)
                or rewritten[index + 1] != transform["from"]
            ):
                raise ValueError(f"flag value not found for replacement: {transform['flag']}")
            rewritten[index + 1] = transform["to"]
        elif kind == "replace_import_prefix":
            source = transform["from"]
            target = transform["to"]
            rewritten_imports = [
                target + value[len(source) :]
                if value == source or value.startswith(source + ".")
                else value
                for value in rewritten_imports
            ]
    return rewritten, rewritten_imports
```

- [ ] **Step 4: Run transformation tests**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor.TransformationTest -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit safe rewrites**

```bash
git add tests/test_upgrade_readiness_auditor.py \
  skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py
git commit -m "feat: propose safe SGLang command rewrites"
```

### Task 4: Compute Verdicts and Render Reports

**Files:**
- Modify: `tests/test_upgrade_readiness_auditor.py`
- Modify: `skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py`

- [ ] **Step 1: Add failing verdict and report tests**

```python
class AuditVerdictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_unmatched_profile_with_passing_base_canaries_is_go(self) -> None:
        result = self.mod.audit(valid_profiles(), valid_rules())
        self.assertEqual(result["profiles"][0]["verdict"], "GO")

    def test_required_rewrite_is_conditional_go(self) -> None:
        profiles = valid_profiles()
        profiles["profiles"][0]["argv"].append("--enable-deepep-waterfill")
        result = self.mod.audit(profiles, valid_rules())
        profile = result["profiles"][0]
        self.assertEqual(profile["verdict"], "CONDITIONAL_GO")
        self.assertIn("--enable-waterfill", profile["proposed_argv"])

    def test_unresolved_blocker_is_no_go(self) -> None:
        profiles = valid_profiles()
        profiles["profiles"][0]["features"].extend(
            ["dp_attention", "breakable_prefill_cuda_graph"]
        )
        profiles["profiles"][0]["guarantees"].append("temperature_zero_determinism")
        rules = valid_rules()
        rules["rules"].append(
            {
                "id": "determinism-known-issue",
                "category": "known_issue",
                "severity": "blocker",
                "title": "Temperature-zero nondeterminism",
                "applies": {"introduced_in": "v0.5.16", "fixed_in": None},
                "source_url": "https://github.com/sgl-project/sglang/pull/31125",
                "summary": "Avoid the affected graph path.",
                "match": {
                    "all": [
                        {"kind": "feature", "value": "dp_attention"},
                        {"kind": "feature", "value": "breakable_prefill_cuda_graph"},
                        {"kind": "guarantee", "value": "temperature_zero_determinism"}
                    ]
                },
                "transforms": [],
                "canaries": ["temperature_zero_determinism"]
            }
        )
        result = self.mod.audit(profiles, rules)
        self.assertEqual(result["profiles"][0]["verdict"], "NO_GO")
        self.assertEqual(result["overall_verdict"], "NO_GO")

    def test_markdown_never_executes_or_hides_commands(self) -> None:
        result = self.mod.audit(valid_profiles(), valid_rules())
        report = self.mod.render_markdown(result)
        self.assertIn("SYNTHETIC FIXTURE", report)
        self.assertIn("## Profile Verdicts", report)
        self.assertIn("## Proposed Commands", report)
        self.assertIn("python3 -m sglang.launch_server", report)
```

- [ ] **Step 2: Run verdict tests and verify missing behavior**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor.AuditVerdictTest -v
```

Expected: failures for missing `audit` and `render_markdown`.

- [ ] **Step 3: Implement audit aggregation**

For every profile:

1. filter rules with `rule_applies` and `rule_matches`;
2. collect transformations and conflicts;
3. preserve source-linked findings even when no transform exists;
4. apply safe transforms;
5. union base and matched-rule canaries;
6. calculate missing/failing canaries;
7. assign `NO_GO` for blockers or transform conflicts,
   `CONDITIONAL_GO` for required/behavior/risk/dependency findings or missing
   canaries, otherwise `GO`;
8. aggregate overall verdict with `NO_GO > CONDITIONAL_GO > GO`.

Implement the result shape:

```python
{
    "schema_version": 1,
    "fixture": True,
    "current_version": "v0.5.15",
    "target_version": "v0.5.16",
    "overall_verdict": "CONDITIONAL_GO",
    "profiles": [
        {
            "id": "legacy-pd",
            "verdict": "CONDITIONAL_GO",
            "original_argv": [],
            "proposed_argv": [],
            "original_imports": [],
            "proposed_imports": [],
            "findings": [],
            "required_canaries": [],
            "missing_or_failing_canaries": []
        }
    ]
}
```

- [ ] **Step 4: Implement stable Markdown and CLI**

Render:

- fixture disclaimer;
- version pair and overall verdict;
- profile verdict table;
- per-profile findings with severity, source URL, and summary;
- original and proposed commands using `shlex.join`;
- import rewrites;
- required/missing canaries;
- rollback conditions copied from rule guidance.

Add CLI arguments:

```python
parser.add_argument("--profiles", required=True, type=Path)
parser.add_argument("--rules", required=True, type=Path)
parser.add_argument("--output-markdown", required=True, type=Path)
parser.add_argument("--output-json", required=True, type=Path)
```

Load both roots as JSON objects, validate them, write stable sorted JSON and
Markdown, return `2` on invalid input, and never execute any argv.

- [ ] **Step 5: Run verdict, CLI, and all current tests**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor -v
python3 skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py --help
```

Expected: all tests pass and help exits zero.

- [ ] **Step 6: Commit verdicts and reports**

```bash
git add tests/test_upgrade_readiness_auditor.py \
  skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py
git commit -m "feat: report SGLang upgrade readiness"
```

### Task 5: Add the v0.5.15-to-v0.5.16 Demonstration

**Files:**
- Create: `skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-profiles.json`
- Create: `skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-rules.json`
- Create: `skills/sglang-upgrade-readiness-auditor/examples/fixture-report.md`
- Modify: `tests/test_upgrade_readiness_auditor.py`

- [ ] **Step 1: Add a failing fixture reproducibility test**

```python
class FixtureDemoTest(unittest.TestCase):
    def test_v0516_fixture_report_is_reproducible(self) -> None:
        module = load_module()
        root = ROOT / "skills" / "sglang-upgrade-readiness-auditor"
        profiles = json.loads(
            (root / "examples" / "v0.5.15-to-v0.5.16-profiles.json").read_text(encoding="utf-8")
        )
        rules = json.loads(
            (root / "examples" / "v0.5.15-to-v0.5.16-rules.json").read_text(encoding="utf-8")
        )
        result = module.audit(profiles, rules)
        report = module.render_markdown(result)
        committed = (root / "examples" / "fixture-report.md").read_text(encoding="utf-8")
        self.assertEqual(report, committed)
        by_id = {profile["id"]: profile for profile in result["profiles"]}
        self.assertEqual(by_id["plain-tp"]["verdict"], "GO")
        self.assertEqual(by_id["legacy-fp4-pd"]["verdict"], "CONDITIONAL_GO")
        self.assertEqual(by_id["deterministic-dp-graph"]["verdict"], "NO_GO")
        self.assertEqual(result["overall_verdict"], "NO_GO")
```

- [ ] **Step 2: Run the fixture test and verify files are missing**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor.FixtureDemoTest -v
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Create three synthetic profiles**

Create:

- `plain-tp`: no affected features and passing base canaries;
- `legacy-fp4-pd`: argv contains `--fp4-gemm-backend cutlass`,
  `--enable-deepep-waterfill`, and `--optimistic-prefill-retries 3`;
- `deterministic-dp-graph`: features include `dp_attention` and
  `breakable_prefill_cuda_graph`, with
  `temperature_zero_determinism` as a required guarantee.

Every model path, image, hardware name, and measurement must be explicitly
synthetic.

- [ ] **Step 4: Create source-linked v0.5.16 rules**

Encode at least:

- `--fp4-gemm-backend cutlass` to `auto`, source SGLang PR #30448;
- `--enable-deepep-waterfill` to `--enable-waterfill`, source PR #27350;
- `--optimistic-prefill-retries` to
  `--optimistic-prefill-attempts`, source PR #30951;
- UnifiedRadixTree default behavior for SWA/Mamba/DSA, source PR #30468;
- chunked input-logprob default behavior, source PR #31498;
- `sglang.kernels` import relocation, sources PRs #30044 and #31582;
- temperature-zero DP-attention/breakable-graph known issue, source PR #31125;
- diffusion rollout msgpack transport change, source PR #31565.

Use only direct GitHub release or PR URLs. The exact fixture profiles should
match only the intended subset.

- [ ] **Step 5: Generate the committed fixture report**

```bash
python3 skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py \
  --profiles skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-profiles.json \
  --rules skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-rules.json \
  --output-markdown skills/sglang-upgrade-readiness-auditor/examples/fixture-report.md \
  --output-json /tmp/sglang-v0516-upgrade-fixture-result.json
```

Expected: `plain-tp=GO`, `legacy-fp4-pd=CONDITIONAL_GO`,
`deterministic-dp-graph=NO_GO`, overall `NO_GO`.

- [ ] **Step 6: Run fixture and analyzer tests**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the demonstration**

```bash
git add tests/test_upgrade_readiness_auditor.py \
  skills/sglang-upgrade-readiness-auditor/examples
git commit -m "test: demonstrate SGLang v0.5.16 upgrade audit"
```

### Task 6: Write the Skill and References

**Files:**
- Create: `skills/sglang-upgrade-readiness-auditor/SKILL.md`
- Create: `skills/sglang-upgrade-readiness-auditor/references/evidence-and-rule-authoring.md`
- Create: `skills/sglang-upgrade-readiness-auditor/references/profile-and-rule-schema.md`
- Modify: `tests/test_upgrade_readiness_auditor.py`

- [ ] **Step 1: Add failing documentation contract tests**

```python
class SkillDocumentationTest(unittest.TestCase):
    def test_skill_has_read_only_and_verdict_contract(self) -> None:
        skill = (
            ROOT / "skills" / "sglang-upgrade-readiness-auditor" / "SKILL.md"
        ).read_text(encoding="utf-8")
        for required in [
            "name: sglang-upgrade-readiness-auditor",
            "read-only",
            "immutable",
            "GO",
            "CONDITIONAL_GO",
            "NO_GO",
            "canary",
            "rollback",
            "never execute",
            "SYNTHETIC FIXTURE",
        ]:
            self.assertIn(required, skill)

    def test_references_define_source_priority_and_safe_argv(self) -> None:
        root = ROOT / "skills" / "sglang-upgrade-readiness-auditor"
        evidence = (root / "references" / "evidence-and-rule-authoring.md").read_text(encoding="utf-8")
        schema = (root / "references" / "profile-and-rule-schema.md").read_text(encoding="utf-8")
        self.assertIn("Release", evidence)
        self.assertIn("compare", evidence)
        self.assertIn("argv arrays", schema)
        self.assertIn("rename_flag", schema)
```

- [ ] **Step 2: Run tests and verify missing docs**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor.SkillDocumentationTest -v
```

Expected: `FileNotFoundError`.

- [ ] **Step 3: Write `SKILL.md`**

Use:

```yaml
---
name: sglang-upgrade-readiness-auditor
description: "Audit an SGLang deployment before changing versions by matching release, CLI, dependency, default-behavior, known-issue, and internal-import changes to actual launch profiles. Use for GO/CONDITIONAL_GO/NO_GO decisions, proposed command migrations, canaries, and rollback plans."
---
```

The workflow must require immutable version endpoints and actual argv/profile
inventory, prioritize official sources, author evidence-linked rules, run the
analyzer, review proposed commands without executing them, run separately
authorized canaries, update canary results, and issue a scoped verdict. State
that audit mode is read-only and never executes rendered commands, edits
production manifests, pulls images, or restarts services.

- [ ] **Step 4: Write the two references**

`evidence-and-rule-authoring.md` must cover release/compare/CLI/dependency/source
priority; rule classification; known-issue scope; confidence and unknowns;
version-crossing semantics; direct sources; canary derivation; and
future-release refresh.

`profile-and-rule-schema.md` must document every profile, audit, rule,
predicate, transformation, result, and verdict field; argv arrays; flag arity;
conflict behavior; fixture labels; and examples that never use shell
evaluation.

- [ ] **Step 5: Run documentation and analyzer tests**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit the operational skill**

```bash
git add skills/sglang-upgrade-readiness-auditor/SKILL.md \
  skills/sglang-upgrade-readiness-auditor/references \
  tests/test_upgrade_readiness_auditor.py
git commit -m "docs: add SGLang upgrade readiness workflow"
```

### Task 7: Register and Validate the Skill

**Files:**
- Modify: `README.md`
- Modify: `.claude-plugin/plugin.json`
- Modify: `.claude-plugin/marketplace.json`
- Modify: `tests/test_repository_metadata.py`
- Modify only validation-exposed defects elsewhere.

- [ ] **Step 1: Add failing metadata assertions**

Change `core_skills-11` to `core_skills-12` and add:

```python
def test_upgrade_readiness_auditor_is_registered() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    plugin = json.loads(
        (ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    marketplace = json.loads(
        (ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert "sglang-upgrade-readiness-auditor" in readme
    assert "upgrade" in plugin["description"].lower()
    assert "upgrade" in marketplace["plugins"][0]["description"].lower()
```

- [ ] **Step 2: Run metadata tests and verify failure**

```bash
python3 -m unittest tests.test_repository_metadata -v
```

Expected: failure until registration is added.

- [ ] **Step 3: Register the skill**

Add the skill to the README headline, core table, install commands, invocation
examples, and repository map. Change the badge to `core_skills-12` and the
Claude plugin installed total from 12 to 13. Mention upgrade readiness in both
plugin descriptions without changing the published plugin version.

- [ ] **Step 4: Regenerate and compare the fixture report**

```bash
python3 skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py \
  --profiles skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-profiles.json \
  --rules skills/sglang-upgrade-readiness-auditor/examples/v0.5.15-to-v0.5.16-rules.json \
  --output-markdown /tmp/upgrade-readiness-fixture-report.md \
  --output-json /tmp/upgrade-readiness-fixture-result.json
diff -u skills/sglang-upgrade-readiness-auditor/examples/fixture-report.md \
  /tmp/upgrade-readiness-fixture-report.md
```

Expected: no diff.

- [ ] **Step 5: Run focused and repository tests**

```bash
python3 -m unittest tests.test_upgrade_readiness_auditor -v
python3 -m unittest tests.test_repository_metadata -v
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Run syntax and repository checks**

```bash
python3 -m py_compile \
  skills/sglang-upgrade-readiness-auditor/scripts/audit_upgrade.py \
  tests/test_upgrade_readiness_auditor.py
SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure
git diff --check origin/main...HEAD
```

Expected: all commands exit zero.

- [ ] **Step 7: Review scope and safety**

```bash
git status --short
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- \
  skills/sglang-upgrade-readiness-auditor README.md \
  .claude-plugin tests/test_upgrade_readiness_auditor.py \
  tests/test_repository_metadata.py
```

Expected: only planned files appear; fixture deployments are synthetic; every
rule has a direct source; the analyzer never executes commands; no credentials
or private deployment identifiers appear.

- [ ] **Step 8: Commit registration and validation fixes**

```bash
git add README.md .claude-plugin/plugin.json .claude-plugin/marketplace.json \
  tests/test_repository_metadata.py
git commit -m "docs: register SGLang upgrade readiness auditor"
```

If validation changed another in-scope file, stage it explicitly in a separate
`test: validate SGLang upgrade readiness auditor` commit.

- [ ] **Step 9: Push and open the draft PR**

```bash
git push -u origin codex/add-sglang-upgrade-readiness-auditor
gh pr create --draft --base main \
  --head codex/add-sglang-upgrade-readiness-auditor \
  --title "Add SGLang upgrade readiness auditor skill" \
  --body-file /tmp/sglang-upgrade-readiness-auditor-pr.md
```

The PR body must include the read-only safety boundary, v0.5.15-to-v0.5.16
fixture result, exact demo and test commands, source-link policy, synthetic
deployment disclaimer, and non-goals.
