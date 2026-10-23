"""Observation-axis sketches: renaming hidden units just renames columns."""
import time
import torch
from geometry import GROUPS,candidate_edges

class Capture:
    def __init__(self,model):
        self.enabled=False;self.values={};self.seconds=0.
        self.handles=[
          model.embedding.register_forward_hook(self.hook("embedding")),
          model.combine[0].register_forward_pre_hook(self.pre),
          model.combine[1].register_forward_hook(self.hook("ff")),
          model.combine[2].register_forward_hook(self.hook("output"))]
    def add(self,key,x):
        if self.enabled:
            t=time.perf_counter()
            self.values.setdefault(key,[]).append(x.detach().clone())
            self.seconds+=time.perf_counter()-t
    def hook(self,key):
        return lambda module,inputs,output:self.add(key,output)
    def pre(self,module,inputs):
        if self.enabled:
            for key,x in zip(("fast","segment","document"),inputs[0].chunk(3,-1)):self.add(key,x)
    def pop(self):
        result={k:torch.cat(v,dim=1) for k,v in self.values.items()}
        self.values={}
        return result
    def close(self):
        for h in self.handles:h.remove()

def normalized(x):return x/x.norm(dim=0,keepdim=True).clamp_min(1e-8)
def basic_stats(x):
    flat=x.flatten(0,1);mean=flat.mean(0);std=flat.std(0,unbiased=False).clamp_min(1e-5)
    c=x-mean
    lags=[]
    for lag in (1,4,16):
        lags.append((c[:,lag:]*c[:,:-lag]).mean((0,1))/(std.square()+1e-8))
    return torch.stack((mean/std,std.log(),*lags))

def individual(x,kind,k=64,seed=420):
    a=x.flatten(0,1);a=a-a.mean(0)
    rng=torch.Generator().manual_seed(seed)
    if kind=="exact":z=a
    elif kind=="countsketch":
        idx=torch.randint(k,(len(a),),generator=rng)
        sign=torch.randint(2,(len(a),),generator=rng).float().mul_(2).sub_(1)
        z=torch.zeros(k,a.shape[1],dtype=a.dtype)
        z.index_add_(0,idx,a*sign[:,None])
    elif kind=="hadamard":
        n=1<<(len(a)-1).bit_length()
        z=torch.zeros(n,a.shape[1],dtype=a.dtype);z[:len(a)]=a
        sign=torch.randint(2,(n,),generator=rng).float().mul_(2).sub_(1)
        z.mul_(sign[:,None]);h=1
        while h<n:
            v=z.reshape(-1,2,h,a.shape[1]);left=v[:,0].clone();right=v[:,1].clone()
            v[:,0]=left+right;v[:,1]=left-right;h*=2
        z=z[torch.randperm(n,generator=rng)[:k]]
    elif kind=="dsp":return normalized(basic_stats(x))
    elif kind=="fft_power":
        c=x-x.mean(1,keepdim=True)
        power=torch.fft.rfft(c,dim=1).abs().square().mean(0)
        edges=torch.linspace(1,len(power),9).long()
        z=torch.stack([power[edges[i]:max(int(edges[i])+1,int(edges[i+1]))].mean(0) for i in range(8)])
        return normalized(torch.cat((z.clamp_min(1e-9).log(),basic_stats(x))))
    else:raise ValueError(kind)
    if kind=="exact":return normalized(z)
    # Coarse magnitude/lag information is supplemental; centered trace carries most weight.
    s=basic_stats(x)
    s=torch.stack((s[0].clamp(-5,5)/5,s[1].clamp(-8,3)/8,*[v.clamp(-1,1) for v in s[2:]]))
    return normalized(torch.cat((normalized(z),.1*s)))

def describe(traces,kind="countsketch",k=64):
    d={name:individual(x,kind,k) for name,x in traces.items()}
    result={g:d[g] for g in ("fast","segment","document","ff")}
    result["res"]=normalized(torch.cat((d["embedding"],d["output"])))
    return result

def metric_probe(traces):
    exact=describe(traces,"exact")
    records=[]
    for kind,k in [("countsketch",32),("countsketch",64),("hadamard",32),("hadamard",64),("dsp",5),("fft_power",13),("exact",0)]:
        describe(traces,kind,k)
        times=[]
        for _ in range(5):
            t=time.perf_counter();z=describe(traces,kind,k);times.append(time.perf_counter()-t)
        group=[]
        t=time.perf_counter()
        for gi,g in enumerate(GROUPS):
            e,s=candidate_edges(z[g],900+gi)
            select=s.argsort(descending=True)[:16];pairs=e[select]
            truth=(exact[g][:,pairs[:,0]]*exact[g][:,pairs[:,1]]).sum(0)
            # Exact metric is evaluated on the same bounded candidate set, avoiding a hidden all-pairs solver.
            alltruth=(exact[g][:,e[:,0]]*exact[g][:,e[:,1]]).sum(0)
            oracle=alltruth.topk(min(16,len(alltruth))).values
            group.append({"group":g,"candidate_edges":len(e),"selected_exact_cosine":float(truth.mean()),
              "best_on_candidate_set_exact_cosine":float(oracle.mean()),
              "selected_cosine_rmse":float(((s[select]-truth)**2).mean().sqrt())})
        records.append({"kind":kind,"k":k,"build_median_seconds":sorted(times)[2],
          "build_times":times,"shortlist_seconds":time.perf_counter()-t,
          "fingerprint_bytes":sum(x.numel()*x.element_size() for x in z.values()),"groups":group})
    return records
