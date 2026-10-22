"""Package the completed CPU exploration without the source parquet/runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/cpu-124m'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    run=ROOT/'research-scratch/cpu-124m/main'
    summary=json.loads((run/'summary.json').read_text())
    assessment=json.loads((run/'assessment.json').read_text())
    assert summary['status']=='COMPLETED'
    assert assessment['generation']['count']==10
    OUT.mkdir(parents=True,exist_ok=True)
    checkpoint=OUT/'minKron_124M_CPU_25min_Checkpoint.pt'
    if not checkpoint.exists():
        os.link(run/'checkpoint.pt',checkpoint)
    # The assessment already hashed the final checkpoint; a hardlink preserves bytes.
    assert checkpoint.stat().st_ino==(run/'checkpoint.pt').stat().st_ino
    patch=subprocess.check_output(['git','diff','--binary','3c8bf67','HEAD'],cwd=ROOT)
    (OUT/'cpu_scale_source_changes.patch').write_bytes(patch)
    paths={Path(p) for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p}
    for base in ('research-scratch/cpu-124m','data/fineweb_cpu','artifacts/cpu-124m'):
        for path in (ROOT/base).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or any(p.startswith('.') for p in path.relative_to(ROOT).parts):
                continue
            if path.suffix in ('.pt','.tmp','.zip','.parquet','.pyc'):
                continue
            if path.name=='delivery_manifest.json':
                continue
            paths.add(path.relative_to(ROOT))
    records=[]
    archive=OUT/'minKron_124M_CPU_25min_Source_and_Results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
        for relative in sorted(paths):
            path=ROOT/relative
            if not path.is_file() or path.suffix in ('.pt','.tmp','.zip','.parquet','.pyc'):
                continue
            output.write(path,arcname='minkron-nanogpt/'+str(relative))
            records.append(dict(path=str(relative),bytes=path.stat().st_size,sha256=sha(path)))
        output.writestr('minkron-nanogpt/archive_file_manifest.json',json.dumps(records,indent=2)+'\n')
    with zipfile.ZipFile(archive) as output:
        assert output.testzip() is None
    manifest=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files=[dict(name=path.name,path=str(path),bytes=path.stat().st_size,
                    sha256=assessment['checkpoint']['sha256'] if path==checkpoint else sha(path))
               for path in (OUT/'minKron_124M_CPU_25min_Report.html',
                            OUT/'minKron_124M_CPU_10_Generations.txt',archive,checkpoint)],
        archive_entries=len(records),checkpoint_restore_path='minkron-nanogpt/research-scratch/cpu-124m/main/checkpoint.pt')
    (OUT/'delivery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
