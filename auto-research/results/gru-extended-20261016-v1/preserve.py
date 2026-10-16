"""Keep large checkpoint/data artifacts outside Git; publish loss traces compressed."""
import argparse,gzip,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--archive',type=Path,required=True);args=parser.parse_args()
s=json.loads((ROOT/'SUMMARY.json').read_text());p=json.loads((ROOT/'PROTOCOL.json').read_text())
checkpoints=json.loads((ROOT/'CHECKPOINT_MANIFEST.json').read_text())
artifact=args.archive.resolve();artifact.parent.mkdir(exist_ok=True,parents=True)
manifest={};branch='research/gru-extended-20261016-v1'
with zipfile.ZipFile(artifact,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for a in p['arms']:
  f=ROOT/'execution'/a/'checkpoint.pt';assert sha(f)==checkpoints[a]['sha256']
  dest=f'checkpoints/{a}/checkpoint.pt';z.write(f,dest)
  manifest[dest]=dict(sha256=sha(f),bytes=f.stat().st_size)
  raw=ROOT/'execution'/a/'curves.json';raw.with_suffix('.json.gz').write_bytes(gzip.compress(raw.read_bytes(),mtime=0))
 for name in ['train.bin','validation.bin']:
  f=args.data_dir/name;dest=f'data/{name}';z.write(f,dest)
  manifest[dest]=dict(sha256=sha(f),bytes=f.stat().st_size)
 instructions=f'''# Companion checkpoints and token arrays

Run: {ROOT.name}
Source/results: https://github.com/JRowe47/Super-1/tree/{branch}/auto-research/results/{ROOT.name}
Protocol SHA256: {s['protocol_sha256']}

This artifact deliberately excludes Git-backed code and reports. It contains three final model/optimizer/RNG checkpoints and the exact train/validation uint16 arrays. The code repository includes tokenizer.json and the data source/license metadata. Consult its README and PRESERVATION.json.

To replay, place checkpoints/<arm>/checkpoint.pt in the source run's execution/<arm>/ folder. Pass this archive's data/ directory to analyze.py --data-dir. SHA256 and sizes are in ARTIFACT_MANIFEST.json. Official tests and unrelated user files are absent.
'''
 z.writestr('README.md',instructions);z.writestr('ARTIFACT_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
receipt=dict(run_id=ROOT.name,archive_filename=artifact.name,bytes=artifact.stat().st_size,sha256=sha(artifact),
 protocol_sha256=s['protocol_sha256'],contents=manifest,location='Companion artifact retained by contributor outside Git; request from contributor for replay',
 public_download=False,git_exclusions=['checkpoint.pt','train.bin','validation.bin','uncompressed duplicate curves'],
 source_and_results=f'https://github.com/JRowe47/Super-1/tree/{branch}/auto-research/results/{ROOT.name}')
(ROOT/'PRESERVATION.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['archive_filename','bytes','sha256']}))
