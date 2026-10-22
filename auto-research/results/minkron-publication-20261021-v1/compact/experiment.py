"""Paired, stage-wise CPU and data capped experiment; no test-based selection."""
import bootstrap,argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import ARMS,build,accounting
from tasks import episode,MODES
torch.set_num_threads(1)
ROOT=bootstrap.ROOT
PHASES=[('short',(4,8,16,32),10000),('medium',(32,64,128),1000),('long',(32,64,128,256,512),3000)]
SEEDS=[22101,22102,22103]

def atomic_json(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,path)

def optimizer(m,lr):
    yes=[];no=[]
    for name,p in m.named_parameters():
        (yes if p.ndim>=2 and name!='embedding.weight' and not getattr(p,'_no_weight_decay',False) else no).append(p)
    return torch.optim.AdamW([{'params':yes,'weight_decay':.1},{'params':no,'weight_decay':0}],lr=lr,betas=(.9,.95))

def batch(seed,phase,step):
    ages=PHASES[phase][1];age=ages[(step//4)%len(ages)];mode=MODES[step%4]
    return episode(seed*100000+phase*20000+step,batch=32,age=age,mode=mode)

@torch.no_grad()
def validation(m,seedbase,ages=(16,32),n=128):
    m.eval();out=[]
    for mi,mode in enumerate(MODES):
        for age in ages:
            hit=0
            for rep in range(n//32):
                x,y,_=episode(seedbase+mi*10000+age*10+rep,batch=32,age=age,mode=mode)
                pred=m(torch.from_numpy(x)).logits[:,-1].argmax(-1).numpy();hit+=int((pred==y).sum())
            out.append({'mode':mode,'age':age,'n':n,'correct':hit,'accuracy':hit/n})
    m.train();return out

def checkpoint(m,opt,path,meta):
    temp=path.with_suffix('.tmp');torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),
        'metadata':meta,'torch_rng':torch.get_rng_state()},temp);os.replace(temp,path)

def train(arm,seed,lr,pilot=False):
    assert json.loads((ROOT/'results/verification.json').read_text())['status']=='PASS'
    stage='pilot' if pilot else 'main';run=ROOT/'results'/stage/f'{arm}_{seed}_{lr:g}';run.mkdir(parents=True,exist_ok=False)
    m=build(arm,seed);opt=optimizer(m,lr);hist=[];phases=[];updates=0;tokens=0;total_cpu=0.;start=time.perf_counter()
    caps=None
    if not pilot and arm!='mingru':
        p=json.loads((ROOT/'protocol.json').read_text())
        base=ROOT/'results/main'/f'mingru_{seed}_{p["learning_rates"]["mingru"]:g}'/'terminal.json'
        caps=json.loads(base.read_text())['phases']
    for pi,(name,ages,maxsteps) in enumerate(PHASES[:1] if pilot else PHASES):
        cap=100. if pilot else caps[pi]['training_cpu_seconds'] if caps else None
        cpu=0.;costs=[];stream=hashlib.sha256();phase_tokens=0;done=0
        for step in range(maxsteps):
            reserve=max(1.,max(costs[-20:],default=0.)*1.5)
            if cap is not None and cpu+reserve>=cap:break
            t=time.process_time();x,y,meta=batch(seed,pi,step)
            for group in opt.param_groups:group['lr']=lr*min(1.,(updates+1)/50)
            opt.zero_grad(set_to_none=True);z=m(torch.from_numpy(x)).logits[:,-1]
            loss=F.cross_entropy(z,torch.from_numpy(y));loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            cost=time.process_time()-t;cpu+=cost;total_cpu+=cost;costs.append(cost)
            updates+=1;done=step+1;tokens+=x.size;phase_tokens+=x.size
            stream.update(x.tobytes());stream.update(y.tobytes())
            if done%50==0 or done==1:
                row={'phase':name,'phase_step':done,'updates':updates,'tokens':tokens,'loss':loss.item(),
                    'batch_accuracy':(z.argmax(-1)==torch.from_numpy(y)).float().mean().item(),
                    'age':meta['nominal_age'],'mode':meta['mode'],'cpu_seconds':total_cpu,'phase_cpu_seconds':cpu,
                    'phase_cpu_cap':cap,'grad_norm':float(gn)}
                hist.append(row);atomic_json(row,run/'progress.json')
                if done%500==0:print(json.dumps({'arm':arm,'seed':seed,**row}),flush=True)
            if done%500==0:
                atomic_json(hist,run/'history.json')
                checkpoint(m,opt,run/'latest.pt',{'arm':arm,'seed':seed,'lr':lr,'phase':pi,'phase_step':done,'cpu':total_cpu})
        phase={'phase':name,'steps':done,'tokens':phase_tokens,'training_cpu_seconds':cpu,'cpu_cap':cap,
               'data_sha256':stream.hexdigest(),'max_step_cpu_seconds':max(costs,default=0.)}
        assert cap is None or cpu<=cap,(phase,'CPU cap exceeded')
        phases.append(phase);atomic_json(phases,run/'phases.json')
        print(json.dumps({'phase_complete':phase,'arm':arm,'seed':seed}),flush=True)
    meta={'arm':arm,'seed':seed,'lr':lr,'phases':phases,'updates':updates,'tokens':tokens,
          'training_cpu_seconds':total_cpu,'training_wall_seconds':time.perf_counter()-start,'pilot':pilot}
    checkpoint(m,opt,run/'final.pt',meta)
    checks=validation(m,54000000,n=128) if pilot else []
    terminal={**meta,'status':'completed','accounting':accounting(m),'validation':checks,
              'checkpoint':str((run/'final.pt').relative_to(ROOT)),
              'checkpoint_sha256':hashlib.sha256((run/'final.pt').read_bytes()).hexdigest()}
    atomic_json(hist,run/'history.json');atomic_json(terminal,run/'terminal.json')
    (run/'latest.pt').unlink(missing_ok=True)
    print(json.dumps({'completed':terminal}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS),required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--lr',type=float,required=True)
    p.add_argument('--pilot',action='store_true');a=p.parse_args();train(a.arm,a.seed,a.lr,a.pilot)
