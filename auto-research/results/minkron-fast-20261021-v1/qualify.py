"""Predeclared mechanics gate and training-only synthetic throughput probe."""
import copy
import json
import time
from dataclasses import asdict
import torch
from torch.nn import functional as F
from common import ROOT, ARMS, SEEDS, make_model, config, write_json, sha
from model import optimizer, tensor_bytes

def flat(state):
    if isinstance(state, torch.Tensor): return [state]
    out=[]
    for v in (state.values() if isinstance(state,dict) else state): out.extend(flat(v))
    return out

def check(a,b,atol,rtol):
    assert len(a)==len(b)
    worst=0.
    for x,y in zip(a,b):
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        torch.testing.assert_close(x,y,atol=atol,rtol=rtol)
        worst=max(worst,float((x-y).abs().max()))
    return worst

def main():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    torch.set_num_threads(1)
    rows=[]
    for dtype in [torch.float64,torch.float32]:
        tol=protocol['execution']['fidelity_tolerances'][str(dtype)]
        for arm in ARMS:
            for seed in [SEEDS[0],SEEDS[1]]:
                print(f'Checking {arm} {seed} {dtype}', flush=True)
                torch.manual_seed(seed)
                m=make_model(arm,seed).to(dtype)
                # Nonzero conv taps exercise history and cross-document masking.
                for block in m.blocks:
                    if getattr(block.mixer,'short_conv',0):
                        with torch.no_grad(): block.mixer.conv_weight[1:].normal_(0,.2)
                ref=copy.deepcopy(m)
                for block in ref.blocks:
                    if hasattr(block.mixer,'backend'): block.mixer.backend='recurrent'
                x=torch.randint(0,4095,(2,19)); reset=torch.zeros_like(x,dtype=torch.bool)
                reset[:,0]=True; reset[0,7]=True; reset[1,13]=True
                a,sa=m(x,reset=reset); b,sb=ref(x,reset=reset)
                forward=check([a]+flat(sa),[b]+flat(sb),**tol)
                target=torch.roll(x,-1,1)
                la=F.cross_entropy(a.reshape(-1,4096),target.flatten()); lb=F.cross_entropy(b.reshape(-1,4096),target.flatten())
                la.backward();lb.backward()
                grads=check([p.grad for p in m.parameters()],[p.grad for p in ref.parameters()],**tol)
                oa=optimizer(m,.001); ob=optimizer(ref,.001);oa.step();ob.step()
                step=check(list(m.parameters()),list(ref.parameters()),**tol)
                with torch.no_grad():
                    full,sfull=m(x,reset=reset)
                    left,sl=m(x[:,:6],reset=reset[:,:6]);right,sright=m(x[:,6:],sl,reset[:,6:])
                    streaming=check([full]+flat(sfull),[torch.cat((left,right),1)]+flat(sright),**tol)
                    altered=x.clone();altered[0,:7]=(altered[0,:7]+77)%4095
                    other,_=m(altered,reset=reset)
                    reset_error=check([full[0,7:]],[other[0,7:]],**tol)
                    # Future tokens must not affect past outputs.
                    changed=x.clone(); changed[:,12:]=(changed[:,12:]+101)%4095
                    future,_=m(changed,reset=reset)
                    causal=check([full[:,:12]],[future[:,:12]],**tol)
                rows.append(dict(arm=arm,seed=seed,dtype=str(dtype),forward_state_max_error=forward,
                    parameter_gradient_max_error=grads,optimizer_step_max_error=step,
                    streaming_max_error=streaming,reset_max_error=reset_error,causal_max_error=causal))
    write_json(ROOT/'FIDELITY.json',dict(status='PASS',rows=rows))
    print('Fidelity PASS: all arms, two seeds, FP64/FP32, outputs/states/gradients/AdamW/reset/causality/streaming',flush=True)
    timings=[]
    for threads in [1,2,4]:
        torch.set_num_threads(threads)
        for arm in ARMS:
            m=make_model(arm,SEEDS[0]); opt=optimizer(m,.001)
            x=torch.randint(0,4095,(8,128));reset=torch.zeros_like(x,dtype=torch.bool); reset[:,0]=True
            samples=[]
            for i in range(7):
                t=time.perf_counter();opt.zero_grad(set_to_none=True)
                logits,state=m(x,reset=reset)
                F.cross_entropy(logits.reshape(-1,4096),x.flatten()).backward();opt.step()
                if i>=2:samples.append(time.perf_counter()-t)
            samples.sort();seconds=samples[len(samples)//2]
            row=dict(arm=arm,threads=threads,median_step_seconds=seconds,tokens_per_second=1024/seconds,
                parameters=sum(p.numel() for p in m.parameters()),config=asdict(config(arm)),state_bytes_batch8=tensor_bytes(state)[0])
            timings.append(row);print(json.dumps({k:row[k] for k in ['arm','threads','tokens_per_second','parameters']}),flush=True)
    best_threads=max([1,2,4], key=lambda n:min(r['tokens_per_second'] for r in timings if r['threads']==n))
    minimum=min(r['tokens_per_second'] for r in timings if r['threads']==best_threads)
    # A 25% safety allowance plus 900-second screen leaves time for extension.
    steps=max(16,min(512,int(900*.75*minimum/(18*1024))))
    budget=dict(threads=best_threads,screen_steps=steps,tokens_per_update=1024,
        screen_tokens_per_arm_seed=steps*1024,screen_pairs=18,extension_seconds_per_arm=300,
        planned_total_training_walltime_ceiling_seconds=1800,phase1_walltime_ceiling_seconds=900,
        selected_by='Synthetic training throughput only; no evaluation or trained pilot inspected.')
    write_json(ROOT/'PROFILE.json',dict(timings=timings,budget=budget));print(json.dumps(budget),flush=True)

if __name__=='__main__':main()
