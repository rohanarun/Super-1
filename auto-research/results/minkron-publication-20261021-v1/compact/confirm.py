import bootstrap,argparse,hashlib,json,time
import numpy as np
import torch
from torch.nn import functional as F
from model import build
from tasks import episode,verify_oracle,MODES
from experiment import ROOT,atomic_json
torch.set_num_threads(1)
p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
protocol=json.loads((ROOT/'protocol.json').read_text());cfg=protocol['confirmation'];lr=protocol['learning_rates'][a.arm]
run=ROOT/'results/main'/f'{a.arm}_{a.seed}_{lr:g}';terminal=json.loads((run/'terminal.json').read_text())
assert terminal['status']=='completed';cp=ROOT/terminal['checkpoint']
assert hashlib.sha256(cp.read_bytes()).hexdigest()==terminal['checkpoint_sha256']
m=build(a.arm,a.seed);m.load_state_dict(torch.load(cp,weights_only=False)['model']);m.eval()
out=ROOT/'results/confirmation'/f'{a.arm}_{a.seed}';out.mkdir(parents=True,exist_ok=False)
seedbase=cfg['seed_base']+protocol['seeds'].index(a.seed)*1000000;rows=[];raw={};start=time.perf_counter()
with torch.no_grad():
 for mi,mode in enumerate(MODES):
  for age in cfg['ages']:
   n=cfg['primary_n'] if age==cfg['primary_age'] and mode in cfg['primary_modes'] else cfg['other_n']
   ys=[];preds=[];zs=[];actual=[];old=[];keys=[];stream=hashlib.sha256();loss=0.
   for rep in range(n//32):
    x,y,meta=episode(seedbase+mi*10000+age*10+rep,batch=32,age=age,mode=mode);verify_oracle(x,y)
    stream.update(x.tobytes());stream.update(y.tobytes());z=m(torch.from_numpy(x)).logits[:,-1]
    loss+=float(F.cross_entropy(z,torch.from_numpy(y),reduction='sum'))
    ys.append(y);preds.append(z.argmax(-1).numpy());zs.append(z.numpy());actual.append(meta['actual_age']);keys.append(meta['query_key'])
    old.append(meta['stale'] if mode=='overwrite' else np.full(32,-1))
   yy,pp,oo,aa,kk=[np.concatenate(v) for v in (ys,preds,old,actual,keys)]
   row={'arm':a.arm,'seed':a.seed,'mode':mode,'age':age,'actual_age':[int(aa.min()),int(aa.max())],
        'n':n,'correct':int((yy==pp).sum()),'accuracy':float((yy==pp).mean()),'loss':loss/n,
        'stale_rate':float((pp==oo).mean()) if mode=='overwrite' else None,'data_sha256':stream.hexdigest()}
   rows.append(row);tag=f'{mode}_{age}'
   for key,value in [('y',yy),('pred',pp),('logits',np.concatenate(zs)),('stale',oo),('age',aa),('key',kk)]:raw[tag+'_'+key]=value
   atomic_json(rows,out/'progress.json');print(json.dumps(row),flush=True)
np.savez_compressed(out/'predictions.npz',**raw)
atomic_json({'status':'completed','arm':a.arm,'seed':a.seed,'checkpoint_sha256':terminal['checkpoint_sha256'],
             'records':rows,'seconds':time.perf_counter()-start},out/'terminal.json')
