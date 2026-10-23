import bootstrap,hashlib,json
from pathlib import Path
import numpy as np
import torch
from model import ARMS,build,accounting
from experiment import ROOT,batch,atomic_json
from tasks import episode
torch.set_num_threads(1)
p=json.loads((ROOT/'protocol.json').read_text());checks=[];source=[]
for file,digest in p['source_sha256'].items():
 assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
 source.append(file)
for seed in p['seeds']:
 base=json.loads((ROOT/'results/main'/f'mingru_{seed}_{p["learning_rates"]["mingru"]:g}'/'terminal.json').read_text())
 terminals={a:json.loads((ROOT/'results/main'/f'{a}_{seed}_{p["learning_rates"][a]:g}'/'terminal.json').read_text()) for a in ARMS}
 for pi,bs in enumerate(base['phases']):
  limits={r['phases'][pi]['steps'] for r in terminals.values()};digests={};h=hashlib.sha256()
  for step in range(bs['steps']):
   x,y,_=batch(seed,pi,step);h.update(x.tobytes());h.update(y.tobytes())
   if step+1 in limits:digests[step+1]=h.hexdigest()
  for arm,r in terminals.items():assert digests[r['phases'][pi]['steps']]==r['phases'][pi]['data_sha256'],(arm,seed,pi)
 confirmation={}
 for arm in ARMS:
  run=ROOT/'results/main'/f'{arm}_{seed}_{p["learning_rates"][arm]:g}'
  r=json.loads((run/'terminal.json').read_text());assert r['status']=='completed'
  assert hashlib.sha256((ROOT/r['checkpoint']).read_bytes()).hexdigest()==r['checkpoint_sha256']
  for pi,stage in enumerate(r['phases']):
   assert stage['steps']<=base['phases'][pi]['steps']
   assert stage['training_cpu_seconds']<=base['phases'][pi]['training_cpu_seconds']+1e-9
  conf=json.loads((ROOT/'results/confirmation'/f'{arm}_{seed}'/'terminal.json').read_text())
  assert conf['checkpoint_sha256']==r['checkpoint_sha256']
  confirmation[arm]={f'{v["mode"]}_{v["age"]}':v['data_sha256'] for v in conf['records']}
  m=build(arm,seed);m.load_state_dict(torch.load(ROOT/r['checkpoint'],weights_only=False)['model']);m.eval()
  assert accounting(m)==r['accounting']
  with torch.no_grad():
   x,_,_=batch(seed,2,0);x=torch.tensor(x[:2]);a=m(x).logits;b=m(x).logits
   assert torch.equal(a,b)
   xx,_,_=episode(991011,batch=2,age=2048,mode='overwrite');xx=torch.from_numpy(xx)
   full=m(xx).logits[:,-1];token=m(xx,backend='tokenwise').logits[:,-1]
   torch.testing.assert_close(full,token,atol=2e-4,rtol=1e-3)
  checks.append({'arm':arm,'seed':seed,'cpu_and_data_caps':'PASS','training_data_regeneration':'PASS','checkpoint_and_state_accounting':'PASS','trained_long_sequence_tokenwise_max_delta':float((full-token).abs().max())})
 for arm in ARMS:assert confirmation[arm]==confirmation['mingru']
atomic_json({'status':'PASS','checks':checks,'frozen_sources':source,'confirmation_data_pairing':'PASS'},ROOT/'results/audit.json')
print('AUDIT PASS',flush=True)
