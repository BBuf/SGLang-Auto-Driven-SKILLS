# Profile, Rule, and Result Schema

## Contents

- [Profile document](#profile-document)
- [Deployment profile](#deployment-profile)
- [Rule document](#rule-document)
- [Predicates](#predicates)
- [Transformations](#transformations)
- [Verdicts and result](#verdicts-and-result)
- [Safety and fixtures](#safety-and-fixtures)

## Profile document

Use:

```json
{
  "schema_version": 1,
  "fixture": false,
  "audit": {
    "current_version": "v0.5.15",
    "target_version": "v0.5.16",
    "required_canaries": ["server_health", "correctness", "performance"]
  },
  "profiles": []
}
```

Versions accept `vMAJOR.MINOR.PATCH` and `.postN`. The target must be newer.
Use immutable tags/commits/images in the audit evidence even when the analyzer
version label is a release.

## Deployment profile

```json
{
  "id": "production-pd",
  "argv": ["python3", "-m", "sglang.launch_server", "--tp", "8"],
  "env": {"SGLANG_EXAMPLE": "1"},
  "model_family": "deepseek-v4",
  "quantization": "nvfp4",
  "hardware": "b200",
  "topology": {"tp": 8, "pp": 1, "dp": 8, "ep": 8, "cp": 1},
  "features": ["pd_disaggregation", "input_logprobs"],
  "guarantees": ["temperature_zero_determinism"],
  "integrations": ["custom_router"],
  "imports": ["sglang.jit_kernel.fast_op"],
  "canary_results": {
    "server_health": "pass",
    "correctness": "pass",
    "performance": "not_run"
  }
}
```

Use argv arrays, never shell command strings. Values are parsed as inert data.
The analyzer uses `shlex.join` only to display them.

Canary values are `pass`, `fail`, or `not_run`. A missing value behaves like
`not_run`. Record results only after executing the target canary separately.

## Rule document

```json
{
  "schema_version": 1,
  "rules": [
    {
      "id": "rename-example",
      "category": "renamed_interface",
      "severity": "required",
      "title": "Example flag renamed",
      "applies": {
        "mode": "crossing",
        "introduced_in": "v0.5.16",
        "fixed_in": null
      },
      "source_url": "https://github.com/sgl-project/sglang/pull/1",
      "summary": "The old spelling has no alias.",
      "match": {
        "all": [{"kind": "argv_flag", "name": "--old"}]
      },
      "transforms": [
        {"kind": "rename_flag", "from": "--old", "to": "--new"}
      ],
      "canaries": ["server_health"],
      "rollback": "Restore the prior image and argv."
    }
  ]
}
```

`mode` is `crossing` or `target`. `fixed_in` is optional/null. Every rule needs
a direct source, concrete summary, at least one predicate, explicit severity,
canaries, and a rollback.

## Predicates

All predicates in `match.all` must pass:

| Kind | Required fields | Match |
| --- | --- | --- |
| `argv_flag` | `name` | Exact token exists |
| `argv_value` | `name`, `equals` | Exact flag token followed by exact value |
| `env` | `name`, optional `equals` | Variable exists or equals value |
| `feature` | `value` | Value in `features` |
| `guarantee` | `value` | Value in `guarantees` |
| `integration` | `value` | Value in `integrations` |
| `import_prefix` | `value` | Exact import/prefix exists |
| `model_family` | `equals` | Exact model family |
| `quantization` | `equals` | Exact quantization |
| `hardware` | `equals` | Exact hardware label |
| `topology` | `name`, `equals`/`min`/`max` | Integer topology constraint |

Keep profile vocabulary stable within one audit. The analyzer does not infer
that two model-family aliases are equivalent.

## Transformations

### `rename_flag`

```json
{"kind": "rename_flag", "from": "--old", "to": "--new"}
```

Replace one exact flag token. Duplicate occurrences are ambiguous and rejected.

### `remove_flag`

```json
{"kind": "remove_flag", "name": "--removed", "arity": 1}
```

Remove the flag and zero or one following value. Only arity 0 or 1 is accepted.

### `replace_value`

```json
{
  "kind": "replace_value",
  "flag": "--backend",
  "from": "old",
  "to": "auto"
}
```

Require the exact old value before proposing the replacement.

### `replace_import_prefix`

```json
{
  "kind": "replace_import_prefix",
  "from": "sglang.jit_kernel",
  "to": "sglang.kernels"
}
```

Rewrite only exact imports or dotted children. Use only for source-proven,
prefix-preserving moves.

Conflicting transforms on the same flag/import prefix produce `NO_GO`. The
analyzer keeps original argv/imports when it cannot apply the proposal safely.

## Verdicts and result

The result contains:

```json
{
  "schema_version": 1,
  "fixture": false,
  "current_version": "v0.5.15",
  "target_version": "v0.5.16",
  "overall_verdict": "CONDITIONAL_GO",
  "profiles": [
    {
      "id": "production-pd",
      "verdict": "CONDITIONAL_GO",
      "original_argv": [],
      "proposed_argv": [],
      "original_imports": [],
      "proposed_imports": [],
      "findings": [],
      "required_canaries": [],
      "canary_results": {},
      "missing_or_failing_canaries": [],
      "transform_error": null,
      "coverage_gaps": []
    }
  ]
}
```

Overall verdict is the most restrictive profile verdict:
`NO_GO > CONDITIONAL_GO > GO`.

- `NO_GO`: blocker or transform conflict.
- `CONDITIONAL_GO`: required/behavior/risk/dependency finding or incomplete
  canary.
- `GO`: no conditional finding remains and all base/matched canaries pass.

`coverage_gaps` lists profile features that no target-applicable feature rule
covers. Review gaps manually; the analyzer cannot know whether every arbitrary
feature is high-risk.

## Safety and fixtures

The analyzer reads and writes files only. It never executes argv arrays,
imports user modules, reads environment variables named in a profile, pulls an
image, or changes deployment state.

Set `fixture: true` for synthetic examples. Preserve the **SYNTHETIC FIXTURE**
warning in generated Markdown and never use fixture verdicts as production
upgrade evidence.
