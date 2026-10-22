"""Deterministic training, pilot selection and complete checkpoint state."""
import bootstrap,argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import build,accounting
from tasks import episode,MODES
torch.set_num_threads(1)
ROOT=bootstrap.ROOT

def atomic_json(obj,p):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,p)

def optimizer(model,lr):
    decay=[];no=[]
    for name,p in model.named_parameters():
        (decay if p.ndim>=2 and name!='embedding.weight' and not getattr(p,'_no_weight_decay',False) else no).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':.1},{'params':no,'weight_decay':0}],lr=lr,betas=(.9,.95),eps=1e-8)

def batch_for(seed,step,batch=32,pilot=False):
    if step<200:ages=(4,8,16)
    elif step<500:ages=(32,64,128)
    else:ages=(32,64,128,256,512)
    age=ages[(step//4)%len(ages)];mode=MODES[step%4]
    return episode(seed*100000+step,batch=batch,age=age,mode=mode)

@torch.no_grad()
def evaluate(model,seedbase,ages=(32,128),nseq=128,primary_extra=False):
    training=model.training;model.eval();records=[]
    for mi,mode in enumerate(MODES):
        for age in ages:
            n=1024 if primary_extra and age==512 and mode in ('competing','overwrite') else nseq
            losses=[];correct=stale=0;count=0;age_min=10**9;age_max=0
            for rep in range((n+31)//32):
                B=min(32,n-count)
                x,y,meta=episode(seedbase+mi*10000+age*10+rep,batch=B,age=age,mode=mode)
                logits=model(torch.from_numpy(x)).logits[:,-1];pred=logits.argmax(-1).numpy()
                losses.append(float(F.cross_entropy(logits,torch.from_numpy(y)))*B)
                correct+=int((pred==y).sum());count+=B
                if mode=='overwrite':stale+=int((pred==meta['stale']).sum())
                age_min=min(age_min,int(meta['actual_age'].min()));age_max=max(age_max,int(meta['actual_age'].max()))
            records.append({'mode':mode,'nominal_age':age,'true_age_range':[age_min,age_max],
                            'correct':correct,'count':count,'accuracy':correct/count,'loss':sum(losses)/count,
                            'stale_rate':stale/count if mode=='overwrite' else None})
    model.train(training);return records

def checkpoint(model,opt,path,metadata):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                                            'metadata':metadata,'torch_rng':torch.get_rng_state()},temp);os.replace(temp,path)

def train(arm,seed,lr,steps,stage,eval_every=250):
    assert json.loads((ROOT/'results/verification.json').read_text())['status']=='PASS'
    runid=f'{arm}_s{seed}_lr{lr:g}';out=ROOT/'results'/stage/runid;out.mkdir(exist_ok=True,parents=True)
    if (out/'terminal.json').exists():raise FileExistsError(out)
    model=build(arm,seed);opt=optimizer(model,lr);history=[];stream=hashlib.sha256();timed=0;tokens=0
    start=time.perf_counter()
    for step in range(steps):
        t=time.perf_counter();x,y,meta=batch_for(seed,step)
        for group in opt.param_groups:group['lr']=lr*min(1.,(step+1)/50)
        opt.zero_grad(set_to_none=True);logits=model(torch.from_numpy(x)).logits[:,-1]
        loss=F.cross_entropy(logits,torch.from_numpy(y));loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
        timed+=time.perf_counter()-t;tokens+=x.size;stream.update(x.tobytes());stream.update(y.tobytes())
        if (step+1)%25==0 or step==0:
            row={'step':step+1,'loss':float(loss.detach()),'accuracy':float((logits.argmax(-1)==torch.from_numpy(y)).float().mean()),
                 'mode':meta['mode'],'nominal_age':meta['nominal_age'],'grad_norm':float(gn),'training_seconds':timed,'tokens':tokens}
            history.append(row)
            atomic_json({'runid':runid,'status':'running','latest':row,'elapsed_seconds':time.perf_counter()-start},out/'progress.json')
            if (step+1)%100==0:print(json.dumps({'run':runid,**row}),flush=True)
        if (step+1)%eval_every==0 or step+1==steps:
            metadata={'arm':arm,'seed':seed,'lr':lr,'steps':step+1,'tokens':tokens,'training_seconds':timed,
                      'data_sha256':stream.hexdigest(),'stage':stage,'width':64}
            checkpoint(model,opt,out/f'checkpoint_{step+1:05d}.pt',metadata)
            atomic_json(history,out/'history.json')
    validation=evaluate(model,91000000,ages=(32,128),nseq=128) if stage=='pilot' else []
    result={**metadata,'status':'completed','accounting':accounting(model),'validation':validation,
            'wall_seconds':time.perf_counter()-start,'checkpoint':str((out/f'checkpoint_{steps:05d}.pt').relative_to(ROOT))}
    result['checkpoint_sha256']=hashlib.sha256((ROOT/result['checkpoint']).read_bytes()).hexdigest()
    atomic_json(result,out/'terminal.json');print(json.dumps({'completed':result}),flush=True)
    return result

def overfit():
    records=[]
    for arm in ('mingru','kron'):
        m=build(arm,8301);opt=optimizer(m,.003);x,y,_=episode(8318,batch=16,age=32,mode='competing')
        x=torch.from_numpy(x);y=torch.from_numpy(y);start=time.perf_counter();passed=False
        for step in range(800):
            opt.zero_grad(set_to_none=True);logits=m(x).logits[:,-1];loss=F.cross_entropy(logits,y);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            if loss.item()<.03 and (logits.argmax(-1)==y).all():passed=True;break
        records.append({'arm':arm,'steps':step+1,'loss':float(loss.detach()),'accuracy':float((logits.argmax(-1)==y).float().mean()),
                        'passed':passed,'seconds':time.perf_counter()-start})
        print(json.dumps(records[-1]),flush=True)
    atomic_json(records,ROOT/'results/overfit.json')
    assert all(r['passed'] for r in records),records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--overfit',action='store_true');p.add_argument('--arm',choices=['mingru','kron'])
    p.add_argument('--seed',type=int,default=8901);p.add_argument('--lr',type=float,default=.003);p.add_argument('--steps',type=int,default=600)
    p.add_argument('--stage',default='pilot');a=p.parse_args()
    if a.overfit:overfit()
    else:train(a.arm,a.seed,a.lr,a.steps,a.stage)
