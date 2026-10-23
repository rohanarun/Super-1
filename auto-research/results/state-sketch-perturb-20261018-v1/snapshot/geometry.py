"""Exact legal hidden-coordinate permutations and function-changing secants."""
import torch

GROUPS=("res","fast","segment","document","ff")
def identities(width):
    return {g:torch.arange(width*(2 if g=="ff" else 1)) for g in GROUPS}
def packed(p,n):return torch.cat([p+i*len(p) for i in range(n)])
def transform_tensor(name,x,p):
    d=len(p["res"]); r,f,s,q,u=(p[k] for k in GROUPS)
    fast_in=torch.cat((r,s+d,q+2*d))
    combine_in=torch.cat((f,s+d,q+2*d))
    if name=="embedding.weight":return x[:,r]
    if name.startswith("final_norm."):return x[r]
    if name=="combine.0.weight":return x[u][:,combine_in]
    if name=="combine.0.bias":return x[u]
    if name=="combine.2.weight":return x[r][:,u]
    if name=="combine.2.bias":return x[r]
    if name=="fast_cell.A":return x[:,f]
    if name.startswith("fast_cell.projection."):
        rows=packed(f,x.shape[0]//d)
        return x[rows][:,fast_in] if x.ndim==2 else x[rows]
    for prefix,rows,cols in [
      ("fast_sequence.",f,fast_in),("fast_cell.",f,fast_in),
      ("segment_cell.",s,f),("document_cell.",q,s)]:
        if name.startswith(prefix):
            suffix=name[len(prefix):];rp=packed(rows,3)
            if suffix.startswith("weight_ih"):return x[rp][:,cols]
            if suffix.startswith("weight_hh"):return x[rp][:,rows]
            if suffix.startswith("bias_"):return x[rp]
    raise KeyError(name)
def transform(weights,p):
    return {n:transform_tensor(n,x,p) for n,x in weights.items()}
def transform_state(state,p):
    return {k:(v[:,p["fast" if k=="segment_sum" else k]] if torch.is_tensor(v) else v)
            for k,v in state.items()}

def candidate_edges(z,seed=19):
    """Four value-based sorted projections; at most ~8 candidate edges/unit."""
    rng=torch.Generator().manual_seed(seed)
    axes=torch.randn(z.shape[0],4,generator=rng,dtype=z.dtype)
    candidates=[]
    for a in (z.T@axes).T:
        order=a.argsort()
        candidates.append(torch.stack((order[:-1],order[1:]),1))
        candidates.append(torch.stack((order[:-2],order[2:]),1))
    pairs=torch.unique(torch.sort(torch.cat(candidates),dim=1).values,dim=0)
    sim=(z[:,pairs[:,0]]*z[:,pairs[:,1]]).sum(0)
    return pairs,sim

def select_permutation(desc,width,seed,mode):
    rng=torch.Generator().manual_seed(seed)
    p=identities(width);details={}
    for gi,g in enumerate(GROUPS):
        n=len(p[g])
        if mode=="random_pair":
            order=torch.randperm(n,generator=rng)[:4]
            pairs=order.reshape(2,2);scores=None
        else:
            z=desc[g]
            if mode=="scrambled":
                z=z[:,torch.randperm(n,generator=rng)]
            edges,sim=candidate_edges(z,seed=900+gi)
            ranked=sim.argsort(descending=True)
            top=ranked[:min(16,len(sim))]
            top=torch.cat((top[torch.randperm(len(top),generator=rng)],ranked[len(top):]))
            chosen=[];used=set();scores=[]
            for e in top:
                i,j=edges[e].tolist()
                if i not in used and j not in used:
                    chosen.append([i,j]);used.update([i,j]);scores.append(float(sim[e]))
                if len(chosen)==2:break
            pairs=torch.tensor(chosen,dtype=torch.long)
        for i,j in pairs.tolist():
            p[g][i]=j;p[g][j]=i
        details[g]={"pairs":pairs.tolist(),"shortlist_scores":scores}
    return p,details

def proposal(weights,desc,width,seed,mode,relative_radius):
    if mode=="iid":
        rng=torch.Generator().manual_seed(seed)
        delta={n:torch.randn(x.shape,generator=rng,dtype=x.dtype) for n,x in weights.items()}
        detail={}
    else:
        p,detail=select_permutation(desc,width,seed,mode)
        permuted=transform(weights,p)
        delta={n:permuted[n]-x for n,x in weights.items()}
    norm=torch.sqrt(sum(x.double().square().sum() for x in delta.values()))
    base_norm=torch.sqrt(sum(x.double().square().sum() for x in weights.values()))
    scale=float(relative_radius*base_norm/norm.clamp_min(1e-20))
    return {n:x*scale for n,x in delta.items()},detail
