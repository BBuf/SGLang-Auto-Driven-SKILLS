from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "sglang-upgrade-readiness-auditor"
SCRIPT = SKILL_ROOT / "scripts" / "audit_upgrade.py"


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
            "required_canaries": [
                "server_health",
                "correctness",
                "performance",
            ],
        },
        "profiles": [
            {
                "id": "plain-tp",
                "argv": [
                    "python3",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    "fixture/model",
                    "--tp",
                    "8",
                ],
                "env": {},
                "model_family": "dense",
                "quantization": "fp8",
                "hardware": "b200",
                "topology": {"tp": 8, "pp": 1, "dp": 1, "ep": 1, "cp": 1},
                "features": [],
                "guarantees": [],
                "integrations": [],
                "imports": [],
                "canary_results": {
                    "server_health": "pass",
                    "correctness": "pass",
                    "performance": "pass",
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
                "applies": {
                    "mode": "crossing",
                    "introduced_in": "v0.5.16",
                    "fixed_in": None,
                },
                "source_url": "https://github.com/sgl-project/sglang/pull/27350",
                "summary": "The old flag has no deprecated alias.",
                "match": {
                    "all": [
                        {
                            "kind": "argv_flag",
                            "name": "--enable-deepep-waterfill",
                        }
                    ]
                },
                "transforms": [
                    {
                        "kind": "rename_flag",
                        "from": "--enable-deepep-waterfill",
                        "to": "--enable-waterfill",
                    }
                ],
                "canaries": ["server_health", "performance"],
                "rollback": "Restore the prior SGLang image and argv.",
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
        document["profiles"].append(copy.deepcopy(document["profiles"][0]))

        with self.assertRaisesRegex(ValueError, "duplicate profile id"):
            self.mod.validate_profiles(document)

    def test_target_must_be_newer(self) -> None:
        document = valid_profiles()
        document["audit"]["target_version"] = "v0.5.15"

        with self.assertRaisesRegex(ValueError, "newer"):
            self.mod.validate_profiles(document)

    def test_unknown_transform_is_rejected(self) -> None:
        rules = valid_rules()
        rules["rules"][0]["transforms"][0]["kind"] = "execute_shell"

        with self.assertRaisesRegex(ValueError, "unknown transform"):
            self.mod.validate_rules(rules)

    def test_commands_must_be_argv_arrays(self) -> None:
        document = valid_profiles()
        document["profiles"][0]["argv"] = "python -m sglang.launch_server"

        with self.assertRaisesRegex(ValueError, "argv"):
            self.mod.validate_profiles(document)


class RuleMatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_crossing_rule_applies_only_when_upgrade_crosses_introduction(self) -> None:
        rule = valid_rules()["rules"][0]

        self.assertTrue(self.mod.rule_applies(rule, "v0.5.15", "v0.5.16"))
        self.assertFalse(self.mod.rule_applies(rule, "v0.5.16", "v0.5.17"))

    def test_target_rule_persists_until_fixed_version(self) -> None:
        rule = valid_rules()["rules"][0]
        rule["applies"] = {
            "mode": "target",
            "introduced_in": "v0.5.16",
            "fixed_in": "v0.5.18",
        }

        self.assertTrue(self.mod.rule_applies(rule, "v0.5.16", "v0.5.17"))
        self.assertFalse(self.mod.rule_applies(rule, "v0.5.17", "v0.5.18"))

    def test_all_predicates_must_match(self) -> None:
        profile = valid_profiles()["profiles"][0]
        profile["env"]["SGLANG_TEST_MODE"] = "compact"
        profile["imports"] = ["sglang.jit_kernel.fast_op"]
        profile["features"] = ["breakable_prefill_cuda_graph"]
        rule = valid_rules()["rules"][0]
        rule["match"]["all"] = [
            {"kind": "hardware", "equals": "b200"},
            {"kind": "topology", "name": "tp", "min": 8},
            {"kind": "feature", "value": "breakable_prefill_cuda_graph"},
            {"kind": "env", "name": "SGLANG_TEST_MODE", "equals": "compact"},
            {"kind": "import_prefix", "value": "sglang.jit_kernel"},
            {"kind": "argv_value", "name": "--tp", "equals": "8"},
        ]

        self.assertTrue(self.mod.rule_matches(rule, profile))
        profile["features"] = []
        self.assertFalse(self.mod.rule_matches(rule, profile))

    def test_unset_optional_predicate_does_not_match(self) -> None:
        profile = valid_profiles()["profiles"][0]
        rule = valid_rules()["rules"][0]
        rule["match"]["all"] = [
            {"kind": "integration", "value": "diffusion_rollout"}
        ]

        self.assertFalse(self.mod.rule_matches(rule, profile))


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
            {
                "kind": "rename_flag",
                "from": "--enable-deepep-waterfill",
                "to": "--enable-waterfill",
            },
            {
                "kind": "replace_value",
                "flag": "--fp4-gemm-backend",
                "from": "cutlass",
                "to": "auto",
            },
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

    def test_import_prefix_is_rewritten_without_touching_other_imports(self) -> None:
        _, imports = self.mod.apply_transforms(
            ["server"],
            ["sglang.jit_kernel.fast_op", "torch"],
            [
                {
                    "kind": "replace_import_prefix",
                    "from": "sglang.jit_kernel",
                    "to": "sglang.kernels",
                }
            ],
        )

        self.assertEqual(imports, ["sglang.kernels.fast_op", "torch"])

    def test_conflicting_rewrites_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting transformations"):
            self.mod.detect_transform_conflicts(
                [
                    {"kind": "rename_flag", "from": "--old", "to": "--new-a"},
                    {"kind": "rename_flag", "from": "--old", "to": "--new-b"},
                ]
            )

    def test_ambiguous_duplicate_flag_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "appears 2 times"):
            self.mod.apply_transforms(
                ["server", "--old", "--old"],
                [],
                [{"kind": "rename_flag", "from": "--old", "to": "--new"}],
            )


class AuditVerdictTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = load_module()

    def test_unmatched_profile_with_passing_base_canaries_is_go(self) -> None:
        result = self.mod.audit(valid_profiles(), valid_rules())

        self.assertEqual(result["profiles"][0]["verdict"], "GO")
        self.assertEqual(result["overall_verdict"], "GO")

    def test_required_rewrite_is_conditional_go(self) -> None:
        profiles = valid_profiles()
        profiles["profiles"][0]["argv"].append("--enable-deepep-waterfill")

        result = self.mod.audit(profiles, valid_rules())
        profile = result["profiles"][0]

        self.assertEqual(profile["verdict"], "CONDITIONAL_GO")
        self.assertIn("--enable-waterfill", profile["proposed_argv"])
        self.assertNotIn("--enable-deepep-waterfill", profile["proposed_argv"])
        self.assertEqual(profile["findings"][0]["id"], "rename-waterfill")

    def test_unresolved_blocker_is_no_go(self) -> None:
        profiles = valid_profiles()
        profile = profiles["profiles"][0]
        profile["features"].extend(
            ["dp_attention", "breakable_prefill_cuda_graph"]
        )
        profile["guarantees"].append("temperature_zero_determinism")
        rules = valid_rules()
        rules["rules"].append(
            {
                "id": "determinism-known-issue",
                "category": "known_issue",
                "severity": "blocker",
                "title": "Temperature-zero nondeterminism",
                "applies": {
                    "mode": "target",
                    "introduced_in": "v0.5.16",
                    "fixed_in": None,
                },
                "source_url": "https://github.com/sgl-project/sglang/pull/31125",
                "summary": "Avoid the affected graph path.",
                "match": {
                    "all": [
                        {"kind": "feature", "value": "dp_attention"},
                        {
                            "kind": "feature",
                            "value": "breakable_prefill_cuda_graph",
                        },
                        {
                            "kind": "guarantee",
                            "value": "temperature_zero_determinism",
                        },
                    ]
                },
                "transforms": [],
                "canaries": ["temperature_zero_determinism"],
                "rollback": "Disable the affected graph path or restore v0.5.15.",
            }
        )

        result = self.mod.audit(profiles, rules)

        self.assertEqual(result["profiles"][0]["verdict"], "NO_GO")
        self.assertEqual(result["overall_verdict"], "NO_GO")
        self.assertIn(
            "temperature_zero_determinism",
            result["profiles"][0]["missing_or_failing_canaries"],
        )

    def test_markdown_labels_fixture_and_shows_commands(self) -> None:
        profiles = valid_profiles()
        profiles["profiles"][0]["argv"].append("--enable-deepep-waterfill")

        result = self.mod.audit(profiles, valid_rules())
        report = self.mod.render_markdown(result)

        self.assertIn("SYNTHETIC FIXTURE", report)
        self.assertIn("## Profile Verdicts", report)
        self.assertIn("## Findings", report)
        self.assertIn("## Proposed Commands", report)
        self.assertIn("--enable-waterfill", report)
        self.assertIn("github.com/sgl-project/sglang/pull/27350", report)
        self.assertIn("## Canaries and Rollback", report)

    def test_cli_writes_reports_without_executing_argv(self) -> None:
        profiles = valid_profiles()
        rules = valid_rules()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile_path = root / "profiles.json"
            rule_path = root / "rules.json"
            markdown_path = root / "report.md"
            json_path = root / "report.json"
            profile_path.write_text(json.dumps(profiles), encoding="utf-8")
            rule_path.write_text(json.dumps(rules), encoding="utf-8")

            exit_code = self.mod.main(
                [
                    "--profiles",
                    str(profile_path),
                    "--rules",
                    str(rule_path),
                    "--output-markdown",
                    str(markdown_path),
                    "--output-json",
                    str(json_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("SYNTHETIC FIXTURE", markdown_path.read_text("utf-8"))
            self.assertEqual(
                json.loads(json_path.read_text("utf-8"))["overall_verdict"], "GO"
            )


class FixtureDemoTest(unittest.TestCase):
    def test_v0516_fixture_report_is_reproducible(self) -> None:
        module = load_module()
        profiles = json.loads(
            (
                SKILL_ROOT
                / "examples"
                / "v0.5.15-to-v0.5.16-profiles.json"
            ).read_text(encoding="utf-8")
        )
        rules = json.loads(
            (
                SKILL_ROOT / "examples" / "v0.5.15-to-v0.5.16-rules.json"
            ).read_text(encoding="utf-8")
        )

        result = module.audit(profiles, rules)
        generated = module.render_markdown(result)
        committed = (SKILL_ROOT / "examples" / "fixture-report.md").read_text(
            encoding="utf-8"
        )
        by_id = {profile["id"]: profile for profile in result["profiles"]}

        self.assertEqual(generated, committed)
        self.assertEqual(by_id["plain-tp"]["verdict"], "GO")
        self.assertEqual(
            by_id["legacy-fp4-pd"]["verdict"], "CONDITIONAL_GO"
        )
        self.assertEqual(
            by_id["deterministic-dp-graph"]["verdict"], "NO_GO"
        )
        self.assertEqual(result["overall_verdict"], "NO_GO")
        self.assertIn("--enable-waterfill", by_id["legacy-fp4-pd"]["proposed_argv"])
        self.assertIn(
            "sglang.kernels.fast_op",
            by_id["legacy-fp4-pd"]["proposed_imports"],
        )


class SkillDocumentationTest(unittest.TestCase):
    def test_skill_has_read_only_and_verdict_contract(self) -> None:
        skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

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
        self.assertNotIn("TODO", skill)

    def test_references_define_source_priority_and_safe_argv(self) -> None:
        evidence = (
            SKILL_ROOT / "references" / "evidence-and-rule-authoring.md"
        ).read_text(encoding="utf-8")
        schema = (
            SKILL_ROOT / "references" / "profile-and-rule-schema.md"
        ).read_text(encoding="utf-8")

        self.assertIn("Release", evidence)
        self.assertIn("compare", evidence)
        self.assertIn("argv arrays", schema)
        self.assertIn("rename_flag", schema)

    def test_openai_metadata_mentions_the_skill(self) -> None:
        metadata = (SKILL_ROOT / "agents" / "openai.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("SGLang Upgrade Readiness Auditor", metadata)
        self.assertIn("$sglang-upgrade-readiness-auditor", metadata)


if __name__ == "__main__":
    unittest.main()
