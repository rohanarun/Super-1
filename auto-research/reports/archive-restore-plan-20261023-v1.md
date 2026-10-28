# Experiment report: archive-restore-plan-20261023-v1

Status: completed stdlib metadata/planning contribution. No artifact transfer or model execution.

Approval: donor's 2026-10-23 contribution authorization and later multiple-focused-PR permission. [Claim #23](https://github.com/rohanarun/Super-1/issues/23). Base `847bb75042e01b2c879272f97e72746f285c8872`; seed `dcbb91d07e51cf13ee6fda1fefc14ca2`. This E00 / Stage 0 task follows the current reproduction priority and the adapter's instruction to approve large downloads; no seeded task picker exists.

Protocol: [PROTOCOL.json](../results/archive-restore-plan-20261023-v1/PROTOCOL.json), SHA-256 `f51c0e04eb254c8a1becaa9f242d1b8008b84c38410c6e7ef8012e142b61e300`, committed before implementation at `d880a914c5589f12c86034f5ddc72082ed94bdc6`. Eight parent files are pinned in [INPUTS.json](../results/archive-restore-plan-20261023-v1/INPUTS.json).

## Question and result

Can a donor estimate the archive restorer's actual payload/cache/output cost without downloading anything? Yes, for the pinned metadata and unchanged restorer. List mode reports selected path sizes; the new planner also accounts for compressed archive payloads, content aliases, cache deduplication and pending locations.

| Full selection | Current index | Historical original index |
|---|---:|---:|
| Selected paths | 30,885 | 30,885 |
| Unique selected objects | 16,867 | 16,867 |
| Unique selected content bytes | 33,983,273,743 | 33,983,273,743 |
| Output bytes including alias copies | 52,388,193,337 | 52,388,193,337 |
| Recorded fetch/cache payloads | 1,781 | 1,661 |
| Known download payload bytes | 33,201,100,068 | 8,524,996,534 |
| Pending unique objects | 0 | 120 |
| Pending content bytes | 0 | 24,676,103,534 |
| Complete cache plus output bytes | 85,589,293,405 | unavailable: blocked |

These are logical byte totals assuming empty cache/output, with no retries, HTTP or filesystem overhead. They are neither live availability checks nor transferred bytes. The historical/current selection digest is identical (`b31a6dc71aa368bd498916f83585585f78e89420ac94c630edbd1454db2c56a6`); recorded locations changed while the selected content identities remained stable. The historical 120-pending-object snapshot precedes the later reconciliation and final 58-object completion operation; it is not relabeled as current.

The deterministic example is the lexicographically first positive-byte archived `README.md`: `icloud-documents/trin/.w560_bench_pkg_attempt1/README.md`. Its 1,898-byte output requires one 41,631,983-byte evidence archive. The current saved release state is draft. This selection was planned from metadata only; its body was not read.

## Verification and failures

Both snapshots reconcile with their saved summary counts, unique bytes, index hashes and pending counts. Every current path is covered exactly once by 161 disjoint first-two-component selections, including standalone files. Their complete canonical plans hash to `94aa8ebdb2d7f156f383b9f652c5600062c3209f7632606464baa3e4986995cd`. The successful metadata batch took 49.9122 CPU / 49.9529 wall seconds on Python 3.13.5 (the exact recorded runtime/timings are in the result receipt).

Fourteen unit tests pass in normal and optimized Python. Thirteen malformed-metadata controls reject missing archive mappings, unsupported locations, duplicate paths, malformed digests, invalid byte counts, noncanonical paths, conflicting content/payload sizes, wrong member identity, invalid asset IDs, mutable Git references and direct-asset identity mismatches. Controlled tiny archives, raw Git-file bytes and synthetic checkpoint bytes are restored with the unchanged original tool while all network calls are stubbed. Predicted payload counts, payload/cache bytes, output counts/bytes and restored hashes agree; a second output restore reuses the warm cache without downloads. CLI checks use both real pinned snapshot readers: a current selection exits 0 and a historical pending selection exits 2.

The first full metadata attempt failed because the verification harness added a directory slash to 17 standalone receipt/log paths. Its traceback and original source are retained in [METADATA_ATTEMPT_01.json](../results/archive-restore-plan-20261023-v1/METADATA_ATTEMPT_01.json) and `development/verify_plans_attempt_01.py`. The grouping now preserves file prefixes and rejects overlapping buckets; a regression fixture covers shallow files. No accounting expectation or byte total was relaxed. Early and final test outcomes are retained in `DEVELOPMENT_CHECKS.json`.

## Limits and decision

Unchecked: live GitHub release state, draft credentials/authorization, remote digest/body availability, actual downloads/cache/disk space, extraction of real archive members, checkpoint deserialization/reconstruction, original datasets, numerical/model replay and scientific claims. Missing saved release state remains unknown. Any real restore needs a separately approved exact transfer/storage scope and must respect the sealed official-test restriction.

The minimum useful effect is met: donors can see whole-archive transfer amplification and alias storage before approving a restore. No benchmark, model or accepted result changes. Next step: an authorized reviewer can select a small allowed prefix, approve its reported transfer/storage budget and perform an authenticated checksum-verified restore. That step was not run in this session.

Credit remains with Rohan Arun and the original archive/recovery contributors. This additive tool/report is substantially AI-assisted by Codex for @thakrarsagar. Original archive metadata, historical snapshots, source bytes and registry content remain unchanged.
