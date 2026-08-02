# Sol Engine SGLang Diffusion Skill Design

## Purpose

Replace the retired custom diffusion optimization skill with a thin integration
skill that runs the complete upstream Sol Engine campaign for SGLang Diffusion.
The skill adds domain knowledge and a model adapter, but it does not fork,
reproduce, or override Sol's orchestration strategy, technique registry,
Executor/Master contracts, verification, quality judgments, or termination
behavior.

The only final source deliverable is an apply-checkable patch against the
SGLang `main` commit fetched and frozen at campaign launch.

## User Contract

The caller supplies a machine, model/checkpoint, exact native SGLang Diffusion
baseline command, and any target requested by the current upstream Sol flow.
The agent resolves the matching host instructions and owns setup, campaign
launch, monitoring, and patch export without asking the user to operate Sol.

At launch the agent freezes two independent source locks:

- the latest fetched `NVlabs/Sana` `sol-engine` branch commit;
- the latest fetched SGLang `main` commit.

"Latest main" means latest at campaign launch. Neither lock moves during a
running campaign, so baseline and candidate measurements remain comparable.

## Non-Negotiable Boundary

The campaign uses upstream
`orchestration/run_orchestrated_experiment.py` as its outer state machine. The
skill may add campaign-local files required to make SGLang Diffusion a Sol
model and may enrich Executor seed goals. It must not edit or replace:

- `orchestration/**` or the upstream Master prompt;
- `orchestration/techniques.toml`;
- technique scope and loop-contract prompts;
- verifier, vision, correctness, quality, or plan-evaluation code;
- Sol scheduling, integration, recovery, or termination decisions.

The existing repository-local `sgl-engine-sglang-diffusion/` controller is a
legacy standalone implementation. The new skill neither invokes nor modifies
its implementation. Only its README receives a compatibility note so it no
longer points at the deleted skill.

## Architecture

```text
latest Sol Engine branch ----> frozen Sol campaign checkout
                                  + SGLang Diffusion model overlay
latest SGLang main ----------> external_copy source at sglang/
KDA-Pilot + SGLang history --> immutable knowledge manifest
                                  -> lane-specific Executor seed goals
                                  -> unchanged upstream Sol campaign
                                  -> integrated SGLang candidate tree
                                  -> binary patch against frozen main
                                  -> git apply --check
```

Sol already composes an Executor prompt from its seed `goal.md`, registered
technique scope, loop contract, and frozen baseline. The adapter pre-creates
the deterministic experiment directories through Sol's own
`create_model_experiment.py`, then appends a bounded, lane-specific knowledge
index to the seed goal. When Sol later spawns that Executor, it reuses the
experiment and keeps all authoritative Sol prompt sections unchanged.

## Campaign-Local Model Overlay

The skill documents how to add only the model-facing files that upstream Sol
requires:

- `models/sglang_diffusion/model.toml` for the external SGLang source copy;
- `models/sglang_diffusion.toml` for the runtime model profile;
- `candidates/sglang_diffusion_baseline.toml` for candidate launch;
- a small baseline/runtime adapter under `models/sglang_diffusion/baseline/`.

The adapter must execute the caller's exact SGLang Diffusion workload and emit
the canonical artifacts expected by the pinned Sol revision. It translates
outputs into Sol's artifact schema; it does not decide whether a result is
correct or good enough. Those judgments remain in upstream Sol.

The frozen SGLang checkout is copied under `sglang/` in each Sol experiment so
Executors edit SGLang itself. Knowledge inputs remain read-only references and
must never appear in the exported SGLang patch.

## Knowledge Pack

The skill builds a deterministic manifest instead of copying a second
controller or pasting an unbounded corpus into prompts. Each record contains
the source repository, frozen commit, relative path, SHA-256 digest, topic,
and eligible upstream Sol technique lanes.

The initial routing is deliberately conservative:

| Knowledge family | Upstream Sol lane | Examples |
| --- | --- | --- |
| lossless kernels and operator fusion | `kernel` | KDA diffusion kernels, KernelWiki, SGLang custom-op and compiler paths |
| exact or quality-gated reuse | `cache` | SGLang cache implementations and relevant PR history |
| sparse attention/token selection | `pisa` | attention sparsity, token selection, sequence reduction |
| distributed communication/layout | `topology` when registered | Ulysses, sequence/context parallel, collective layout |
| quantization below the frozen correctness contract | none | indexed as knowledge-only until upstream Sol registers a compatible lane |

Knowledge suggests hypotheses; it never supplies acceptance thresholds or
waives Sol evidence. Historical shapes, speedups, GPU limits, and quality
observations are marked non-authoritative.

## Patch Contract

After Sol produces its integrated delivery, the skill locates the integrated
SGLang candidate tree and exports a binary/full-index Git diff against the
frozen SGLang main commit. Export fails unless:

1. the base repository resolves the exact frozen commit;
2. every patch path belongs to the SGLang tree;
3. the patch applies cleanly to a fresh detached worktree at that commit;
4. no Sol overlay, prompt, campaign artifact, knowledge file, credential, or
   machine-local path is included.

The handoff includes `sglang.patch`, the SGLang base commit, the Sol Engine
commit, the knowledge-manifest digest, and the upstream Sol delivery/result
artifacts. The patch is the requested code output; Sol remains the authority
for performance and quality conclusions.

## Failure Behavior

The skill fails closed when source locks cannot be resolved, repositories are
dirty in a way that would contaminate frozen inputs, the model adapter does
not emit the pinned Sol artifact contract, a requested optimization has no
registered compatible Sol lane, or patch validation fails. It reports the
upstream Sol state instead of inventing replacement strategy or quality logic.

## Validation

Repository tests verify that the old skill is removed, the new skill is
discoverable, the knowledge manifest is deterministic and correctly routes
representative files, forbidden Sol paths are named in the boundary, and a
patch exported from a synthetic SGLang repository applies to its frozen base.
The standard skill validator and repository test suite must pass.
