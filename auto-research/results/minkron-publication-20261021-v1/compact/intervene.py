"""First predeclared main seed only; memory interventions after 256 tokens."""
import bootstrap,copy,json,time
import numpy as np
import torch
from model import ARMS,build
from tasks import episode
from experiment import ROOT,atomic_json
torch.set_num_threads(1)
p=json.loads((ROOT/'protocol.json').read_text());seed=p['seeds'][0];rows=[]
with torch.no_grad():
 for arm in ARMS:
  lr=p['learning_rates'][arm];run=ROOT/'results/main'/f'{arm}_{seed}_{lr:g}'
  m=build(arm,seed);m.load_state_dict(torch.load(run/'final.pt',weights_only=False)['model']);m.eval()
  for mi,mode in enumerate(['competing','overwrite']):
   hits={k:0 for k in ['intact','zero_fast','zero_slow','zero_all']}
   for rep in range(8):
    x,y,_=episode(88000000+mi*10000+rep,batch=32,age=512,mode=mode);x=torch.from_numpy(x)
    state=m(x[:,:256]).final_state
    for kind in hits:
     edited={k:(v.clone() if torch.is_tensor(v) else v) for k,v in state.items()}
     for key,v in edited.items():
      fast=key in ['fast','kron_state'];slow=key in ['segment','document','segment_sum']
      if torch.is_tensor(v) and (kind=='zero_all' or kind=='zero_fast' and fast or kind=='zero_slow' and slow):v.zero_()
     pred=m(x[:,256:],edited).logits[:,-1].argmax(-1).numpy();hits[kind]+=int((pred==y).sum())
   for kind,h in hits.items():rows.append({'arm':arm,'seed':seed,'mode':mode,'intervention':kind,'correct':h,'n':256,'accuracy':h/256})
atomic_json({'rows':rows,'cut':256,'nominal_age':512,'note':'All target writes precede cut; counters preserved; independent diagnostic data; one seed only.'},ROOT/'results/interventions.json')
print(json.dumps(rows,indent=2))
