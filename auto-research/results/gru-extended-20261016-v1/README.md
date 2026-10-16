# gru-extended-20261016-v1

Read [the completed report](../../reports/gru-extended-20261016-v1.md). This is an independent, one-seed, approximately1M-parameter comparison, not an official Super-1 study or a reproduction of its historical120M benchmark.

Use the pinned versions in `requirements-lock.txt`. Models require PyTorch/NumPy; charts additionally need Matplotlib. The CPU PyTorch index is https://download.pytorch.org/whl/cpu.

From the repository root, with the companion archive's `data/train.bin` and `data/validation.bin` available:

```bash
python auto-research/results/gru-extended-20261016-v1/launch.py --data-dir /path/to/data
python auto-research/results/gru-extended-20261016-v1/analyze.py --data-dir /path/to/data
python auto-research/results/gru-extended-20261016-v1/report.py
```

The launcher skips completed arms; incomplete runs resume model/optimizer/RNG state. To deliberately retrain, use a separate checkout/output copy and move the existing `execution/` directory aside. Keep frozen protocols and offsets. Do not rerun `prepare.py` over this study. The committed `execution/<arm>/curves.json.gz` files contain complete per-step records; `analyze.py` uses the companion `checkpoint.pt` files and the committed RESULT.json files. The chart script reads either compressed or uncompressed curves. [PRESERVATION.json](PRESERVATION.json) identifies external artifacts and hashes.

`models.py` supports `forward`, `step`, `transition`, and `last_logits`; the latter two wrapper methods were completed for this contribution without changing the trained forward computation. The inherited native MAC estimate is deliberately unavailable for the replacement cells. New cores are standalone equation implementations, not copied external CUDA kernels. Unchanged native source and its license are under `vendor/`.

Primary source/protocol files are protected by `SOURCE_MANIFEST.json`; all new contribution files are covered by the repository manifest. `REGISTRATION.json` records the protocol freeze before launch; the registry was updated locally before execution and will be pushed with completed results as the user requested. This was not a public preregistration.
