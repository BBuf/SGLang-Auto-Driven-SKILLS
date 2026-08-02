# Sol Engine SGLang Diffusion Skill Implementation Plan

> Implement the approved thin-adapter design without changing the existing
> standalone controller or any upstream Sol decision logic.

**Goal:** Replace `sglang-diffusion-auto-optimize` with an installable
`sol-engine-sglang-diffusion` skill that feeds pinned KDA-Pilot/SGLang
knowledge into upstream Sol Engine and exports a validated patch against the
SGLang main commit frozen at launch.

**Architecture:** The skill resolves and pins upstream Sol Engine and SGLang,
adds a campaign-local SGLang model adapter, injects lane-specific knowledge
through Sol's existing seed-goal mechanism, runs the unchanged full Sol
orchestrator, and validates the final SGLang-only patch in a clean worktree.

**Implementation:** Markdown skill/reference contracts plus small Python
standard-library tools, tested with synthetic local Git repositories.

---

## Task 1: Scaffold the replacement skill

**Files:**

- Delete: `skills/sglang-diffusion-auto-optimize/`
- Create: `skills/sol-engine-sglang-diffusion/SKILL.md`
- Create: `skills/sol-engine-sglang-diffusion/agents/openai.yaml`
- Create: `skills/sol-engine-sglang-diffusion/references/sol-boundary.md`
- Create: `skills/sol-engine-sglang-diffusion/references/sglang-adapter.md`
- Create: `skills/sol-engine-sglang-diffusion/references/knowledge-routing.md`
- Create: `skills/sol-engine-sglang-diffusion/references/patch-contract.md`

1. Run the official `skill-creator` scaffold generator.
2. Remove the old skill directory with an explicit patch.
3. Write a concise `SKILL.md` whose mandatory flow invokes the complete
   upstream Sol runner and names the immutable boundaries.
4. Put detailed adapter, knowledge, and patch procedures in references.
5. Generate and then review `agents/openai.yaml`.

## Task 2: Build deterministic knowledge manifests

**Files:**

- Create: `skills/sol-engine-sglang-diffusion/scripts/build_knowledge_pack.py`
- Test: `tests/test_sol_engine_sglang_diffusion_skill.py`

1. Add a failing test with minimal synthetic KDA-Pilot and SGLang Git trees.
2. Implement full-commit resolution, allowlisted file discovery, SHA-256
   digests, conservative Sol-lane routing, and deterministic JSON/Markdown
   output.
3. Mark quantization-only material as knowledge-only when no compatible
   upstream Sol technique is registered.
4. Assert identical inputs produce byte-identical manifests.

## Task 3: Export a clean SGLang patch

**Files:**

- Create: `skills/sol-engine-sglang-diffusion/scripts/extract_sglang_patch.py`
- Test: `tests/test_sol_engine_sglang_diffusion_skill.py`

1. Add a failing test that edits, adds, deletes, and binary-modifies files in a
   candidate tree.
2. Materialize the candidate in a temporary detached worktree at the frozen
   base commit and produce a full-index binary diff.
3. Apply-check the patch in a second clean worktree and reject empty or
   out-of-tree delivery paths.
4. Cover dirty source rejection and cleanup behavior.

## Task 4: Make the new skill discoverable

**Files:**

- Modify: `README.md`
- Modify: `sgl-engine-sglang-diffusion/README.md` (compatibility note only)
- Modify: `tests/test_repository_metadata.py`

1. Replace current install, invocation, and repository-tree references to the
   deleted skill.
2. Rewrite the active diffusion section around the upstream Sol delegation
   boundary; do not rewrite historical design/plan records.
3. Update metadata tests to require the new skill and assert the legacy skill
   is absent.

## Task 5: Validate the implementation

**Files:** all changed files

1. Run the new focused tests.
2. Run `skill-creator`'s `quick_validate.py` on the new skill.
3. Run repository metadata and full test suites.
4. Run formatting/lint checks available in the repository.
5. Inspect `git diff --check`, the changed-file list, and the final diff to
   confirm the legacy controller implementation and historical documents are
   untouched.
