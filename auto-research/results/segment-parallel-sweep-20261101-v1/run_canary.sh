#!/usr/bin/env bash
# Super-1 segment-parallel sweep canary (run segment-parallel-sweep-20261101-v1).
# Everything stays in this session folder. Project code runs under bwrap with no network,
# read-only everywhere except this folder, one CPU thread, inside `capped`.
set -euo pipefail
# `capped` is the donor machine's resource-limit wrapper (one shared job slot, two cores, 12 GiB); drop it elsewhere.
S="$(cd "$(dirname "$0")" && pwd)"
cd "$S"
RUN=segment-parallel-sweep-20261101-v1
SRC="$S/repo/auto-research/results/cloud-recovery-20261015/exact-historical-source"
CKPT="$S/ckpt/pure-hierarchy-final.pt"
CKPT_SHA=beb12f7a9fce10195c639365cc673200e04c97affa9557239b70edd28bd9c66f
OUT="$S/out"
mkdir -p "$OUT" "$S/ckpt"

# 1. Private CPU-only environment (no global installs).
if ! venv/bin/python -c "import torch, numpy" 2>/dev/null; then
  [ -x venv/bin/python ] || uv venv --python 3.12 venv
  capped uv pip install --python venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
  capped uv pip install --python venv/bin/python numpy==2.2.6
fi
venv/bin/python -c "import torch,numpy;print('torch',torch.__version__,'numpy',numpy.__version__)" | tee "$OUT/env.txt"

# 2. Trained historical checkpoint from the project's public Releases, hash-checked.
if [ ! -f "$CKPT" ]; then
  gh release download recovered-120m-checkpoints-20261015 -R rohanarun/Super-1 \
    -p pure-hierarchy-final.pt -D "$S/ckpt"
fi
echo "$CKPT_SHA  $CKPT" | sha256sum -c - | tee "$OUT/ckpt_sha256.txt"

SANDBOX=(bwrap --ro-bind / / --bind "$S" "$S" --dev /dev --proc /proc --tmpfs /tmp
         --unshare-net --unshare-pid --die-with-parent --chdir "$S")

# 3. Stored-evidence positive control (standard library only).
"${SANDBOX[@]}" python3 -I repo/scripts/verify_evidence.py | tee "$OUT/verify_evidence.txt"

# 4. The preregistered canary.
capped "${SANDBOX[@]}" env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  venv/bin/python -I work/sweep_canary.py \
    --source-root "$SRC" \
    --checkpoint "$CKPT" --checkpoint-sha256 "$CKPT_SHA" \
    --plan "repo/auto-research/results/$RUN/PLAN.json" \
    --out "$OUT/RESULTS.json" 2>&1 | tee "$OUT/canary.log"
