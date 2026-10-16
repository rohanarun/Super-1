"""Verify only this contribution's records and links; no model/data execution."""
import gzip,hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((ROOT/'PROTOCOL.json').read_text());s=json.loads((ROOT/'SUMMARY.json').read_text())
assert sha(ROOT/'PROTOCOL.json')==s['protocol_sha256']
for f,h in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()).items():assert sha(ROOT/f)==h,f
manifest=json.loads((ROOT/'CHECKPOINT_MANIFEST.json').read_text())
for row in s['ranking']:
 arm=row['arm'];folder=ROOT/'execution'/arm
 r=json.loads((folder/'RESULT.json').read_text())
 assert r['updates']==2000 and r['training_targets']==1024000 and r['seed']==101602
 assert abs(sum(r['final']['window_ce'])/32-row['ce'])<1e-12
 assert abs(math.exp(row['ce'])-row['perplexity'])<1e-9
 assert r['checkpoint_sha256']==manifest[arm]['sha256']
 raw=folder/'curves.json.gz'
 traces=json.loads(gzip.decompress(raw.read_bytes()))
 assert len(traces['train'])==2000 and traces['step']==2000
 assert traces['training_targets']==1024000
 assert traces['monitor'][-1]['step']==2000
 for i,point in enumerate(traces['train'],1):
  assert point['step']==i and point['training_targets']==i*512
  assert math.isfinite(point['ce']) and math.isfinite(point['gradient_norm'])
for path in [ROOT/'README.md',REPO/f'auto-research/reports/{ROOT.name}.md']:
 for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
  if not target.startswith(('https://','http://','#')):assert (path.parent/target).exists(),target
registry=json.loads((REPO/'auto-research/RUN_REGISTRY.json').read_text())
entry=[e for e in registry['runs'] if e['run_id']==ROOT.name];assert len(entry)==1
for key in ['protocol','source_manifest','metrics','report','terminal']:assert (REPO/entry[0][key]).is_file(),key
assert entry[0]['protocol_sha256']==s['protocol_sha256']
result=dict(status='PASS',scope='New contribution only; no full historical repository reevaluation',arms=3,updates_per_arm=2000,
 final_targets_per_arm=16384,source_hashes=True,all_metric_relations=True,complete_compressed_traces=True,local_report_links=True,
 frozen_protocol_sha256=s['protocol_sha256'])
(ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
