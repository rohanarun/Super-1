"""Verify this run's receipts and optional local bodies; never train or evaluate."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text())

def main(local):
    sources=read('SOURCE_MANIFEST.json')
    for file,digest in sources['files'].items():assert sha(ROOT/file)==digest,file
    assert sha(ROOT/'PROTOCOL.json')==sources['protocol_sha256']
    assert sha(ROOT/'PROFILE.json')==sources['profile_sha256']
    for file,digest in sources['addenda'].items():assert sha(ROOT/file)==digest,file
    fidelity=read('FIDELITY.json');assert fidelity['status']=='PASS' and len(fidelity['rows'])==24
    metrics=read('METRICS.json');terminal=read('TERMINAL.json')
    assert terminal['status']=='COMPLETED'
    assert len(metrics['screen'])==18 and len(metrics['extensions'])==3
    assert len({(r['arm'],r['seed']) for r in metrics['screen']})==18
    assert len(metrics['confirmation'])==21
    expected=metrics['budget']['screen_tokens_per_arm_seed']
    for row in metrics['screen']:assert row['completed'] and row['tokens']==expected
    for row in metrics['confirmation']:assert row['absolute_duplicate_difference']<=1e-7
    with gzip.open(ROOT/'ALL_STEPS.csv.gz','rt') as f:
        counts=Counter((r['stage'],r['arm'],int(r['seed'])) for r in csv.DictReader(f))
    for row in metrics['screen']+metrics['extensions']:
        assert counts[row['stage'],row['arm'],row['seed']]==row['steps']
    assert abs(sum(r['training_seconds'] for r in metrics['screen']+metrics['extensions'])-terminal['total_training_seconds'])<1e-6
    checked=[]
    if local:
        for item in read('CHECKPOINTS.json'):
            assert sha(ROOT/item['path'])==item['sha256'];checked.append(item['path'])
        manifest=read('data/manifest.json')
        for split,meta in manifest['splits'].items():
            assert sha(ROOT/'data'/meta['file'])==meta['sha256'];checked.append('data/'+meta['file'])
            assert sha(ROOT/'data'/meta['document_manifest'])==meta['document_manifest_sha256']
    result=dict(status='PASS',frozen_source_files=len(sources['files']),qualification_rows=24,
        completed_screen_runs=18,completed_extensions=3,confirmation_duplicates=21,
        checked_local_bodies=checked,official_test_accessed=False,
        mode='hashes and saved receipts only; no training or inference')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--local-artifacts',action='store_true')
    main(p.parse_args().local_artifacts)
