# For AI agents

1. Read [auto-research.md](auto-research.md) first, then [CONTRIBUTING.md](CONTRIBUTING.md), then the files listed under "Read, in order" in `auto-research.md`.
2. Start with a read-only inventory. Check open issues, open PRs and `auto-research/RUN_REGISTRY.json` so you don't duplicate completed work, then run `python3 scripts/verify_evidence.py`.
3. Do not start training, evaluations, cloud resources or spending without explicit approval of the exact experiment and budget. Never touch the sealed WikiText-2 official test.
4. Preregister with `auto-research/PROTOCOL_TEMPLATE.json`. Store every output under `auto-research/` with a new run ID, and register it in `RUN_REGISTRY.json`.
5. Report every result, including null and negative ones. Open a focused PR or a handoff issue, and disclose AI assistance.
