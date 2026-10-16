"""Preserve only non-Git checkpoint payloads, with the existing exact-data reference."""
import argparse
import json
import zipfile
from pathlib import Path
from run import ROOT,sha,save
parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);args=parser.parse_args()
assert not args.archive.exists(),'Do not replace an existing checkpoint archive'
manifest=json.loads((ROOT/'CHECKPOINT_MANIFEST.json').read_text())
for value in manifest.values():assert sha(ROOT/value['path'])==value['sha256']
args.archive.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for v in manifest.values():z.write(ROOT/v['path'],v['path'])
    z.writestr('CHECKPOINTS.json',json.dumps(manifest,indent=2)+'\n')
    z.writestr('README.txt','Final checkpoints for gru-equal-cpu-20261016-v1. Extract under auto-research/results/gru-equal-cpu-20261016-v1/. Source/protocols/results live in JRowe47/Super-1. Exact data and original checkpoint are in gru_extended_checkpoints_20261016.zip, SHA256 c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6. Each checkpoint contains model, AdamW, Torch RNG and full training/monitor trace.\n')
with zipfile.ZipFile(args.archive) as z:
    assert z.testzip() is None
    import hashlib
    for v in manifest.values():assert hashlib.sha256(z.read(v['path'])).hexdigest()==v['sha256']
save(ROOT/'PRESERVATION.json',dict(archive=args.archive.name,sha256=sha(args.archive),bytes=args.archive.stat().st_size,
    included='Two complete checkpoints with optimizer/RNG/trace; no duplicate Git-backed code/results or data arrays',
    source_repository='JRowe47/Super-1',status='ARCHIVE_VERIFIED',checkpoint_manifest='CHECKPOINT_MANIFEST.json',
    exact_data_archive='gru_extended_checkpoints_20261016.zip',exact_data_sha256='c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6'))
print(json.dumps(json.loads((ROOT/'PRESERVATION.json').read_text())))
