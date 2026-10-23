"""Forward/state/all-input gradient equivalence for the CPU GEMM factorization."""
import copy
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.nanogpt_memory.model import kron_chunk, kron_chunk_factored, Config, build


def main():
    torch.set_num_threads(4)
    rows = []
    for dtype in (torch.float32, torch.float64):
        for T, scale in [(1,.1),(17,.1),(32,1.),(33,5.)]:
            torch.manual_seed(621+T)
            q,k,v = [F.normalize(torch.randn(2,T,12,8,dtype=dtype),dim=-1) for _ in range(3)]
            if T>1:
                v[0,0]=0.;v[1,1]*=1e-12
            g = -scale*torch.rand_like(q)
            beta,c = [torch.rand(2,T,12,dtype=dtype) for _ in range(2)]
            s = torch.randn(2,12,8,8,dtype=dtype)*.1
            args = [a.requires_grad_() for a in (q,k,v,g,beta,c,s)]
            reset=torch.zeros(2,T,dtype=torch.bool)
            if T>16:
                reset[0,13]=True;reset[1,16]=True
            a=kron_chunk(*args,reset)
            b=kron_chunk_factored(*args,reset)
            for u,w in zip(a,b):torch.testing.assert_close(u,w,atol=3e-5 if dtype==torch.float32 else 1e-11,rtol=1e-4)
            weights=[torch.randn_like(x) for x in a]
            grads=[torch.autograd.grad(sum((u*w).sum() for u,w in zip(output,weights)),args,retain_graph=True) for output in (a,b)]
            for ga,gb in zip(*grads):torch.testing.assert_close(ga,gb,atol=3e-5 if dtype==torch.float32 else 1e-10,rtol=3e-4)
            rows.append(dict(dtype=str(dtype),length=T,decay_scale=scale,
                output_state_max_abs=max(float((u-w).abs().max().detach()) for u,w in zip(a,b)),
                gradient_max_abs=max(float((u-w).abs().max()) for u,w in zip(*grads))))
    # Multiple optimizer steps with all parameter gradients in the full decoder.
    cfg=Config(layers=2,width=64,kron_heads=4,enforce_equal_state=False,vocab_size=257)
    a=build(cfg,991)
    cfg2=copy.deepcopy(cfg);cfg2.kron_core='factored'
    b=build(cfg2,991)
    x=torch.randint(0,257,(2,65));y=torch.roll(x,-1,1)
    r=torch.zeros_like(x,dtype=torch.bool);r[0,17]=True;r[1,33]=True
    opts=[torch.optim.AdamW(m.parameters(),lr=.0003) for m in (a,b)]
    for step in range(3):
        losses=[]
        for m,opt in zip((a,b),opts):
            opt.zero_grad(set_to_none=True)
            logits,_=m(x,reset=r);loss=F.cross_entropy(logits.flatten(0,1),y.flatten());loss.backward();opt.step();losses.append(float(loss.detach()))
        assert abs(losses[0]-losses[1])<1e-4
    param_error=max(float((p-q).abs().max().detach()) for p,q in zip(a.parameters(),b.parameters()))
    assert param_error<1e-4
    result=dict(status='PASS',core=rows,three_step_parameter_max_abs=param_error,
        caveat='Timing tests are separate; fallback retains stable core for large decay ranges')
    (ROOT/'research-scratch/cpu-124m/factored_core_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
