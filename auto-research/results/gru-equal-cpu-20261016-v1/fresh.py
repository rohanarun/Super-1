"""Fresh equal-CPU comparison: spread warmup/cosine over the full clock budget."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from run import ROOT, PARENT, build, logits, optimizer, evaluate, save, sha

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--cpu',type=int,default=1)
    args=parser.parse_args()
    p=json.loads((ROOT/'FRESH_PROTOCOL.json').read_text())
    for name,digest in p['source_hashes'].items():
        assert sha(ROOT/name)==digest,name
    for name,digest in p['data_hashes'].items():
        assert sha(args.data_dir/name)==digest,name
    assert sha(ROOT/'OFFSETS.json')==p['offsets_sha256']
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    os.sched_setaffinity(0,{args.cpu})
    resource.setrlimit(resource.RLIMIT_CPU,(400,410))
    offsets=json.loads((ROOT/'OFFSETS.json').read_text())
    out=ROOT/'fresh_execution';out.mkdir(exist_ok=True)
    assert not (out/'RESULT.json').exists(),'Completed runs are immutable'
    cp=out/'checkpoint.pt'
    model=build('mingru_linear',p['config'],p['seed'])
    opt=optimizer(model,.002)
    state=dict(step=0,training_targets=0,train_cpu_seconds=0.,train_wall_seconds=0.,
               train=[],monitor=[],warmup_cpu_seconds=None)
    if cp.exists():
        old=torch.load(cp,map_location='cpu',weights_only=False)
        assert old['protocol_sha256']==sha(ROOT/'FRESH_PROTOCOL.json')
        model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer'])
        torch.set_rng_state(old['torch_rng']);state=old['state']
    train=np.memmap(args.data_dir/'train.bin',dtype=np.uint16,mode='r')
    val=np.memmap(args.data_dir/'validation.bin',dtype=np.uint16,mode='r')
    runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
                 threads=1,affinity=sorted(os.sched_getaffinity(0)),dtype='float32',device='cpu')
    def checkpoint():
        tmp=cp.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            state=state,arm='mingru_linear',config=p['config'],protocol_sha256=sha(ROOT/'FRESH_PROTOCOL.json')),tmp)
        tmp.replace(cp)
    def monitor():
        ev=evaluate(model,val,offsets['monitor_starts'])
        row=dict(step=state['step'],training_targets=state['training_targets'],train_cpu_seconds=state['train_cpu_seconds'],
                 train_wall_seconds=state['train_wall_seconds'],validation_ce=ev['ce'],validation_top1=ev['top1'])
        state['monitor'].append(row);save(out/'curves.json',state);print('FRESH',row,flush=True)
    if not cp.exists():
        save(out/'START.json',dict(protocol_sha256=sha(ROOT/'FRESH_PROTOCOL.json'),runtime=runtime,
             started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        monitor();checkpoint()
    begin=time.monotonic();recent=[.1]
    while state['step']<8000:
        if state['train_cpu_seconds']+max(.2,2*max(recent[-20:]))>=p['target_total_train_cpu_seconds']:
            break
        if time.monotonic()-begin>700:
            checkpoint();raise TimeoutError('Fresh wall budget reached')
        step=state['step']+1
        tick,cpu_tick=time.perf_counter(),time.process_time()
        model.train()
        xy=torch.from_numpy(np.stack([np.array(train[s:s+513],dtype=np.int64) for s in offsets['train_starts'][step-1]]))
        if step<=25:
            factor=step/25
        else:
            progress=(state['train_cpu_seconds']-state['warmup_cpu_seconds'])/(p['target_total_train_cpu_seconds']-state['warmup_cpu_seconds'])
            factor=.1+.9*.5*(1+math.cos(math.pi*min(1.,max(0.,progress))))
        lr=.002*factor
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        z=logits(model,'mingru_linear',xy[:,:-1].contiguous(),xy[:,1:].contiguous())
        ce=F.cross_entropy(z.flatten(0,1),xy[:,1:].flatten());ce.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(ce) or not torch.isfinite(norm):raise FloatingPointError(step)
        opt.step();model.constrain()
        wall,cpu=time.perf_counter()-tick,time.process_time()-cpu_tick
        recent.append(cpu);state['train_cpu_seconds']+=cpu;state['train_wall_seconds']+=wall
        state['step']=step;state['training_targets']+=512
        if step==25:state['warmup_cpu_seconds']=state['train_cpu_seconds']
        state['train'].append(dict(step=step,training_targets=state['training_targets'],ce=float(ce.detach()),
            lr=lr,gradient_norm=float(norm),train_cpu_seconds=state['train_cpu_seconds'],train_wall_seconds=state['train_wall_seconds']))
        if step in (25,50,100,200) or step%250==0:monitor()
        if step%100==0:checkpoint()
    if state['monitor'][-1]['step']!=state['step']:monitor()
    checkpoint();final=evaluate(model,val,offsets['final_starts'])
    unused=p['target_total_train_cpu_seconds']-state['train_cpu_seconds']
    result=dict(status='COMPLETE',arm='mingru_linear_fresh_clock',updates=state['step'],training_targets=state['training_targets'],
        parameters=p['parameters'],seed=p['seed'],final=final,monitor=state['monitor'],train_cpu_seconds=state['train_cpu_seconds'],
        train_wall_seconds=state['train_wall_seconds'],unused_budget_cpu_seconds=unused,
        budget_within_half_second=0<=unused<=.5,checkpoint_sha256=sha(cp),protocol_sha256=sha(ROOT/'FRESH_PROTOCOL.json'),runtime=runtime)
    save(out/'RESULT.json',result);save(out/'curves.json',state)
    print('FRESH COMPLETE',result['updates'],result['train_cpu_seconds'],final['ce'],flush=True)

if __name__=='__main__':main()
