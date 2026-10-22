"""Bounded six-arm paired screen, followed by a fixed-time continuation."""
import argparse
import json
import math
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT, ARMS, SEEDS, make_model, config, sha, write_json
from model import optimizer

EOT=4095

def data():
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    result={}
    for split,meta in manifest['splits'].items():
        path=ROOT/'data'/meta['file'];assert sha(path)==meta['sha256']
        result[split]=np.load(path,mmap_mode='r')
    return result

class Windows:
    def __init__(self,tokens,seed,cursor=0):
        self.tokens=tokens;self.seed=seed;self.cursor=cursor;self.epoch=-1
        self.n=((len(tokens)-1)//128)//8*8
    def next(self):
        epoch=(self.cursor*8)//self.n;offset=(self.cursor*8)%self.n
        if epoch!=self.epoch:
            self.order=np.random.default_rng(self.seed+epoch*100000).permutation(self.n)
            self.epoch=epoch
        starts=self.order[offset:offset+8]*128
        values=torch.from_numpy(np.stack([self.tokens[s:s+129].astype(np.int64) for s in starts]))
        x,y=values[:,:128],values[:,1:];reset=x==EOT;reset[:,0]=True
        self.cursor+=1
        return x,y,reset

@torch.no_grad()
def evaluate(m,tokens,n):
    m.eval();total=0.;count=0;losses=[]
    for cursor in range(0,n,1024):
        values=torch.from_numpy(np.stack([tokens[s:s+129].astype(np.int64) for s in range(cursor,cursor+1024,128)]))
        x,y=values[:,:128],values[:,1:];reset=x==EOT;reset[:,0]=True
        logits,_=m(x,reset=reset)
        loss=F.cross_entropy(logits.reshape(-1,4096),y.reshape(-1),reduction='none')
        assert torch.isfinite(loss).all()
        total+=loss.double().sum().item();count+=loss.numel()
        losses.extend(loss.reshape(8,128).double().mean(1).tolist())
    m.train()
    return dict(ce=total/count,ppl=math.exp(total/count),tokens=count,window_ce=losses)

def checkpoint(m,opt,arm,seed,stage,steps,wall,dev):
    dest=ROOT/'checkpoints';dest.mkdir(exist_ok=True)
    path=dest/f'{stage}_{arm}_{seed}.pt'
    assert not path.exists()
    torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),arm=arm,seed=seed,
        stage=stage,steps=steps,training_seconds=wall,development=dev),path)
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)

def train_one(arm,seed,stage,arrays,budget,parent=None,global_remaining=900):
    m=make_model(arm,seed);opt=optimizer(m,.003)
    cursor=0;prior_seconds=0.
    if parent:
        saved=torch.load(ROOT/parent['checkpoint']['path'],weights_only=False,map_location='cpu')
        m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        cursor=saved['steps'];prior_seconds=saved['training_seconds']
    sampler=Windows(arrays['train'],seed,cursor)
    elapsed=0.;steps=0;log=[];limit=budget['screen_steps'] if stage=='screen' else 1000000
    wall_limit=min(global_remaining,300 if stage=='extension' else global_remaining)
    warm=min(20,max(4,limit//10))
    while steps<limit and elapsed<wall_limit:
        if stage=='screen':
            fraction=max(0,(steps-warm)/max(1,limit-warm))
            lr=.003*(steps+1)/warm if steps<warm else .0003+.0027*.5*(1+math.cos(math.pi*fraction))
        else:
            lr=.0003+.0012*.5*(1+math.cos(math.pi*min(1,elapsed/300)))
        t=time.perf_counter()
        x,y,reset=sampler.next()
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        logits,_=m(x,reset=reset)
        loss=F.cross_entropy(logits.reshape(-1,4096),y.reshape(-1))
        assert torch.isfinite(loss),f'Nonfinite loss {arm} {seed}'
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step();elapsed+=time.perf_counter()-t;steps+=1
        row=dict(stage=stage,arm=arm,seed=seed,step=steps,total_steps=cursor+steps,
            tokens=(cursor+steps)*1024,stage_training_seconds=elapsed,loss=float(loss.detach()),
            grad_norm=float(norm),lr=lr)
        log.append(row)
        if steps==1 or steps%25==0:
            with (ROOT/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    dev=evaluate(m,arrays['development'],16384)
    record=dict(arm=arm,seed=seed,stage=stage,steps=steps,total_steps=cursor+steps,
        tokens=steps*1024,total_tokens=(cursor+steps)*1024,training_seconds=elapsed,
        total_training_seconds=prior_seconds+elapsed,tokens_per_second=steps*1024/elapsed,
        parameters=sum(p.numel() for p in m.parameters()),development=dev,
        completed=steps==limit if stage=='screen' else elapsed>=wall_limit,
        final_training_loss=log[-1]['loss'],peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    record['checkpoint']=checkpoint(m,opt,arm,seed,stage,cursor+steps,prior_seconds+elapsed,dev)
    dest=ROOT/'runs';dest.mkdir(exist_ok=True)
    write_json(dest/f'{stage}_{arm}_{seed}.json',record)
    write_json(dest/f'{stage}_{arm}_{seed}_steps.json',log)
    print('FINISHED '+json.dumps({k:record[k] for k in ['stage','arm','seed','tokens','training_seconds','tokens_per_second','completed']}|{'dev_ce':dev['ce']}),flush=True)
    return record

def confirmation(record,arrays):
    m=make_model(record['arm'],record['seed'])
    saved=torch.load(ROOT/record['checkpoint']['path'],weights_only=False,map_location='cpu')
    m.load_state_dict(saved['model'])
    a=evaluate(m,arrays['confirmation'],32768);b=evaluate(m,arrays['confirmation'],32768)
    delta=abs(a['ce']-b['ce']);assert delta<=1e-7
    result=dict(arm=record['arm'],seed=record['seed'],stage=record['stage'],checkpoint=record['checkpoint'],
        first=a,duplicate_ce=b['ce'],absolute_duplicate_difference=delta)
    write_json(ROOT/'runs'/f"confirmation_{record['stage']}_{record['arm']}_{record['seed']}.json",result)
    return result

def main():
    started=datetime.now(timezone.utc).isoformat()
    assert not (ROOT/'TERMINAL.json').exists(),'Do not overwrite a completed run'
    assert json.loads((ROOT/'FIDELITY.json').read_text())['status']=='PASS'
    budget=json.loads((ROOT/'PROFILE.json').read_text())['budget'];torch.set_num_threads(budget['threads'])
    sources={p.name:sha(p) for p in ROOT.glob('*.py')}
    write_json(ROOT/'SOURCE_MANIFEST.json',dict(utc=started,files=sources,protocol_sha256=sha(ROOT/'PROTOCOL.json'),profile_sha256=sha(ROOT/'PROFILE.json'),addenda={p.name:sha(p) for p in ROOT.glob('PROTOCOL_ADDENDUM_*.json')}))
    arrays=data();screen=[];extensions=[];checks=[];spent=0.
    try:
        for i,seed in enumerate(SEEDS):
            order=ARMS[i:]+ARMS[:i]
            for arm in order:
                row=train_one(arm,seed,'screen',arrays,budget,global_remaining=900-spent)
                screen.append(row);spent+=row['training_seconds']
                write_json(ROOT/'PROGRESS.json',dict(stage='screen',screen=screen,spent_seconds=spent))
                if not row['completed']:raise RuntimeError('Screen time cap reached; no complete comparison')
        means={arm:sum(r['development']['ce'] for r in screen if r['arm']==arm)/3 for arm in ARMS}
        winner=min(ARMS[1:-1],key=lambda arm:(round(means[arm],6),arm))
        selected=dict(selected_recurrent_variant=winner,development_means=means,
            rule='Lowest mean development CE among four new recurrent variants; anchor and Transformer always continued.',
            checkpoints=[r['checkpoint'] for r in screen],utc=datetime.now(timezone.utc).isoformat())
        write_json(ROOT/'SELECTION_FREEZE.json',selected)
        # Confirmation never affects continuation selection.
        for row in screen:checks.append(confirmation(row,arrays))
        for arm in ['k8',winner,'transformer_lean']:
            parent=next(r for r in screen if r['arm']==arm and r['seed']==SEEDS[0])
            row=train_one(arm,SEEDS[0],'extension',arrays,budget,parent=parent,global_remaining=300)
            extensions.append(row);spent+=row['training_seconds'];checks.append(confirmation(row,arrays))
            write_json(ROOT/'PROGRESS.json',dict(stage='extension',screen=screen,extensions=extensions,spent_seconds=spent))
        write_json(ROOT/'METRICS.json',dict(screen=screen,extensions=extensions,confirmation=checks,selection=selected,budget=budget))
        write_json(ROOT/'CHECKPOINTS.json',[r['checkpoint'] for r in screen+extensions])
        write_json(ROOT/'TERMINAL.json',dict(status='COMPLETED',started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),
            total_training_seconds=spent,screen_runs=len(screen),extensions=len(extensions),
            all_finite=True,confirmation_duplicates_passed=len(checks),official_test_accessed=False))
    except BaseException as exc:
        write_json(ROOT/'TERMINAL.json',dict(status='FAILED_OR_INCOMPLETE',reason=repr(exc),started_utc=started,
            screen_runs=len(screen),extensions=len(extensions),training_seconds=spent,official_test_accessed=False))
        raise

if __name__=='__main__':main()
