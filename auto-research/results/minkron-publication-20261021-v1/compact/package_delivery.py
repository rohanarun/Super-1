"""Compact reproducibility bundle; omit redundant pilot weights and runtime."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
files=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file():continue
    rel=p.relative_to(ROOT)
    if '__pycache__' in rel.parts or p.suffix in ['.zip','.tmp']:continue
    if p.name=='delivery_manifest.json':continue
    if p.name=='final.pt' and 'pilot' in rel.parts:continue
    if p.name=='latest.pt':continue
    files.append(p)
data={str(p.relative_to(ROOT)):p.read_bytes() for p in files}
manifest={'status':'complete' if (ROOT/'pipeline_complete.json').exists() and (ROOT/'results/audit.json').exists() else 'partial_checkpoint',
 'files':[{'path':name,'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()} for name,value in data.items()],
 'exclusions':'Private PyTorch runtime, caches, duplicate pilot final checkpoints; all pilot metrics and main final optimizer/model checkpoints included.'}
mp=ROOT/'delivery_manifest.json';mp.write_text(json.dumps(manifest,indent=2));data[mp.name]=mp.read_bytes()
dst=ROOT/'Compact_Kron_budget_experiment_2026-10-21.zip'
with zipfile.ZipFile(dst,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name,value in data.items():z.writestr(name,value)
print(json.dumps({'file':str(dst),'bytes':dst.stat().st_size,'files':len(data),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest()},indent=2))
