# Offline archive restore budget planner

The archived restorer lists selected file sizes, but fetching an archive member requires its entire compressed payload. This separate stdlib tool plans that cost before anyone approves a transfer. It reads eight pinned metadata/source files from the existing archive. It never downloads, restores, imports models or deserializes checkpoints.

From the repository root:

```sh
python3 -B auto-research/results/archive-restore-plan-20261023-v1/restore_plan.py --prefix 'icloud-documents/trin/.w560_bench_pkg_attempt1/README.md'
```

That saved selection has one 1,898-byte output file, but needs the 41,631,983-byte `research-source-evidence-006.tar.gz` payload. The saved summary marks its release as a draft. The command only prints a JSON plan.

`--prefix` uses the unchanged restorer's literal `startswith` rule. Include a trailing slash to select a directory and its descendants. An explicitly empty string selects the whole index. `--snapshot historical` uses the preserved original index instead of `FILES.current.json.gz`. Exit 0 means all selected objects have saved locations; exit 2 prints a blocked plan containing pending objects; exit 1 reports invalid metadata or a selection with no matches. None of these statuses verifies live availability or authorizes a restore.

The plan separates:

- Output paths and logical output bytes: aliases become separate copies.
- Unique selected content and its bytes: identical content counts once.
- Download/cache payloads keyed by SHA-256, as in the restorer: an entire compressed archive counts once even when many selected members use it.
- Pending objects and their known content bytes. Complete cache/combined totals are null for blocked selections; `known_download_payload_bytes` then covers only available recorded locations.
- Authenticated `gh` locators and draft tags found in saved release metadata. Missing release state stays unknown.

Assumptions are empty output/cache and the unchanged sequential restorer. Totals exclude retries, HTTP overhead, filesystem allocation/metadata, existing files and unrelated cache contents. Output and cache need their respective budgets; their sum matters if they share a filesystem. All recorded locators for a cache digest are listed; the restorer uses the first encountered locator. The planner does not choose an alternate location or examine credentials, disk space, cache validity, remote assets or scientific results. Actual transfers require exact authorization and must respect the sealed official-test restriction.

## Verify this contribution

```sh
python3 -B -m unittest discover -s auto-research/results/archive-restore-plan-20261023-v1 -p test_restore_plan.py -v
python3 -O -B -m unittest discover -s auto-research/results/archive-restore-plan-20261023-v1 -p test_restore_plan.py -v
python3 -B auto-research/results/archive-restore-plan-20261023-v1/verify_plans.py
python3 -B scripts/verify_evidence.py
```

The 14 tests include 13 malformed-metadata controls, tiny synthetic restores through the unchanged archived restorer, cold/warm cache checks, pending and CLI status checks, and standalone-file root coverage. All test network calls are replaced by in-memory bytes. No actual archive, model or dataset body is fetched.

`PLAN_RESULTS.json` retains both full-selection summaries, every one of 161 disjoint current first-two-component selections (including standalone files), a digest of their complete plans, implementation hashes and measured CPU/wall time. Re-running the verifier prints a fresh timestamp/timing; its deterministic plan hashes and byte totals should agree. `METADATA_ATTEMPT_01.json` and `development/verify_plans_attempt_01.py` preserve the initial harness failure and source. Seventeen standalone receipts/logs required correcting its directory-only grouping; the planner byte accounting did not change.

Protocol: `PROTOCOL.json`, preregistered at `d880a914c5589f12c86034f5ddc72082ed94bdc6`. Parent inventory/provenance: `INPUTS.json`. Findings and limits: [report](../../reports/archive-restore-plan-20261023-v1.md).

Credit: Rohan Arun and the original archive/recovery contributors. New tooling is substantially AI-assisted by Codex for @thakrarsagar. Historical sources, snapshots, completed evidence and accepted-result status are preserved.
