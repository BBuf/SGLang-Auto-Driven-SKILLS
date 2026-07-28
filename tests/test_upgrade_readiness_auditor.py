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


if __name__ == "__main__":
    unittest.main()
