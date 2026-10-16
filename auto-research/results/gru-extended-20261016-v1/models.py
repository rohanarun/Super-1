"""Three fast-cell adaptations in the pinned Super-1 slow hierarchy.

Independent equation implementations, not full minGRU/ParaRNN language models.
Sequence batching respects every 32-token slow-state feedback boundary.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor'))
from representation_experiments.predictive_models import HierarchicalDistributionStudent

LABELS = {'native': 'Ordinary GRU', 'mingru_linear': 'Linear minGRU', 'paragru': 'ParaGRU'}

def affine_scan(a, b, h0):
    """Inclusive Hillis-Steele scan of h[t]=a[t]*h[t-1]+b[t].

    Vectorizes over time at each log2(T) stage; O(T log T) work, not a fused
    work-efficient GPU kernel. Signed coefficients and nonzero h0 supported.
    """
    distance = 1
    while distance < a.shape[1]:
        b = torch.cat((b[:, :distance], b[:, distance:] + a[:, distance:]*b[:, :-distance]), 1)
        a = torch.cat((a[:, :distance], a[:, distance:]*a[:, :-distance]), 1)
        distance *= 2
    return b + a*h0[:, None]

def diagonal_terms(p, A, prev):
    z = torch.sigmoid(p[..., 0, :] + A[0]*prev)
    r = torch.sigmoid(p[..., 1, :] + A[1]*prev)
    c = torch.tanh(p[..., 2, :] + A[2]*prev*r)
    f = (1-z)*prev + z*c
    j = (1-z) + (c-prev)*z*(1-z)*A[0] + z*(1-c*c)*A[2]*(r+prev*r*(1-r)*A[1])
    return f, j, z, r, c

def diagonal_sequential(p, A, h0):
    values=[]
    for pt in p.unbind(1):
        h0=diagonal_terms(pt, A, h0)[0]
        values.append(h0)
    return torch.stack(values, 1)

class DiagonalNewton(torch.autograd.Function):
    """Converged diagonal Newton forward; exact recurrence adjoint by scan.

    Three iterations minimum, eight maximum, fail loudly if residual > tolerance.
    Backward differentiates the converged recurrence, not a truncated solver.
    First-order derivatives only are needed by this experiment.
    """
    @staticmethod
    def forward(ctx, p, A, h0):
        h=diagonal_terms(p, A, torch.zeros_like(p[..., 0, :]))[0]
        tolerance=1e-12 if p.dtype==torch.float64 else 5e-7
        for iteration in range(1, 9):
            prev=torch.cat((h0[:, None], h[:, :-1]), 1)
            f,j,*_=diagonal_terms(p,A,prev)
            h=affine_scan(j,f-j*prev,h0)
            if iteration>=3:
                prev=torch.cat((h0[:, None], h[:, :-1]), 1)
                residual=(diagonal_terms(p,A,prev)[0]-h).abs().max()
                if float(residual)<=tolerance:
                    break
        else:
            raise RuntimeError(f'ParaGRU Newton did not converge: residual={float(residual)}')
        ctx.save_for_backward(p,A,h0,h)
        stats=torch.tensor([iteration,float(residual)],device=p.device,dtype=torch.float64)
        ctx.mark_non_differentiable(stats)
        return h,stats

    @staticmethod
    def backward(ctx, grad, grad_stats):
        p,A,h0,h=ctx.saved_tensors
        prev=torch.cat((h0[:, None],h[:, :-1]),1)
        _,j,z,r,c=diagonal_terms(p,A,prev)
        next_j=torch.cat((j[:,1:],torch.zeros_like(j[:,:1])),1)
        adjoint=affine_scan(next_j.flip(1),grad.flip(1),torch.zeros_like(h0)).flip(1)
        dc=adjoint*z*(1-c*c)
        dz=adjoint*(c-prev)*z*(1-z)
        dr=dc*A[2]*prev*r*(1-r)
        dp=torch.stack((dz,dr,dc),-2)
        dA=torch.stack(((dz*prev).sum((0,1)),(dr*prev).sum((0,1)),(dc*prev*r).sum((0,1))))
        return dp,dA,adjoint[:,0]*j[:,0]

class MinCell(nn.Module):
    def __init__(self,width,positive=True):
        super().__init__()
        self.positive=positive
        self.projection=nn.Linear(3*width,2*width)
        nn.init.zeros_(self.projection.bias)

    def sequence(self,x,h0,backend):
        candidate,gate=self.projection(x).chunk(2,-1)
        if self.positive:
            candidate=torch.where(candidate>=0,candidate+.5,torch.sigmoid(candidate))
        z=torch.sigmoid(gate)
        a,b=1-z,z*candidate
        if backend=='parallel':
            return affine_scan(a,b,h0)
        values=[]
        for at,bt in zip(a.unbind(1),b.unbind(1)):
            h0=at*h0+bt;values.append(h0)
        return torch.stack(values,1)

class ParaCell(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.width=width
        self.projection=nn.Linear(3*width,3*width)
        self.A=nn.Parameter(torch.empty(3,width))
        nn.init.normal_(self.A,std=width**-.5)
        nn.init.zeros_(self.projection.bias)
        self.constrain()
        self.solver_histogram={}
        self.max_residual=0.

    @torch.no_grad()
    def constrain(self):
        # Each of the three recurrent diagonal vectors has L2 norm <= .5.
        self.A.mul_((.5/self.A.norm(dim=1,keepdim=True).clamp_min(1e-12)).clamp_max(1))

    def sequence(self,x,h0,backend):
        p=self.projection(x).reshape(*x.shape[:-1],3,self.width)
        if backend=='sequential':
            return diagonal_sequential(p,self.A,h0)
        h,stats=DiagonalNewton.apply(p,self.A,h0)
        count=int(stats[0]);self.solver_histogram[count]=self.solver_histogram.get(count,0)+1
        self.max_residual=max(self.max_residual,float(stats[1]))
        return h

class GRUHierarchy(HierarchicalDistributionStudent):
    def __init__(self,kind,**config):
        super().__init__(**config)
        self.variant=kind
        self.backend='parallel' if kind!='native' else 'batched'
        if kind=='native':
            # Preserve every original parameter and initializer exactly.
            with torch.random.fork_rng():
                sequence=nn.GRU(3*self.width,self.width,batch_first=True)
            with torch.no_grad():
                for suffix in ('weight_ih','weight_hh','bias_ih','bias_hh'):
                    getattr(sequence,suffix+'_l0').copy_(getattr(self.fast_cell,suffix))
            del self.fast_cell
            self.fast_sequence=sequence
        else:
            self.fast_cell=ParaCell(self.width) if kind=='paragru' else MinCell(self.width,positive=kind!='mingru_linear')

    def fast_values(self,x,h0,backend):
        if self.variant=='native':
            return self.fast_sequence(x,h0.unsqueeze(0))[0]
        return self.fast_cell.sequence(x,h0,backend)

    def forward(self,input_ids,state=None,backend=None):
        if input_ids.shape[1]==0:raise ValueError('At least one input token required')
        backend=backend or self.backend
        if state is None:state=self.initial_state(input_ids.shape[0],input_ids.device)
        fast=state['fast'];segment=state['segment'];document=state['document']
        segment_sum=state['segment_sum'];count=int(state['segment_tokens']);segments=int(state['segments'])
        embeddings=self.embedding(input_ids)
        features=[];position=0
        while position<input_ids.shape[1]:
            length=min(self.segment_length-count,input_ids.shape[1]-position)
            if backend=='tokenwise':length=1
            tokens=embeddings[:,position:position+length]
            old_segment=segment;old_document=document
            x=torch.cat((tokens,segment[:,None].expand(-1,length,-1),document[:,None].expand(-1,length,-1)),-1)
            values=self.fast_values(x,fast,'sequential' if backend=='tokenwise' else backend)
            fast=values[:,-1].clone() # state owns only one vector, not the sequence backing store
            segment_sum=segment_sum+values.sum(1);count+=length
            if count==self.segment_length:
                segment=self.segment_cell(segment_sum/self.segment_length,segment)
                segment_sum=torch.zeros_like(segment_sum);count=0;segments+=1
                if segments==self.document_segments:
                    document=self.document_cell(segment,document);segments=0
                seg_values=torch.cat((old_segment[:,None].expand(-1,length-1,-1),segment[:,None]),1)
                doc_values=torch.cat((old_document[:,None].expand(-1,length-1,-1),document[:,None]),1)
            else:
                seg_values=segment[:,None].expand(-1,length,-1)
                doc_values=document[:,None].expand(-1,length,-1)
            features.append(torch.cat((values,seg_values,doc_values),-1));position+=length
        hidden=self.combine(torch.cat(features,1))
        final_state=dict(fast=fast,segment=segment,document=document,segment_sum=segment_sum,
                         segment_tokens=count,segments=segments)
        return SimpleNamespace(logits=self.logits(hidden),hidden_states=hidden,final_state=final_state)

    def step(self,token_ids,state=None):
        out=self.forward(token_ids[:,None],state,backend='tokenwise')
        return out.logits[:,0],out.final_state,out.hidden_states[:,0]

    def transition(self,token_ids,state=None):
        out=self.forward(token_ids[:,None],state,backend='tokenwise')
        return out.hidden_states[:,0],out.final_state

    def last_logits(self,input_ids):
        return self.forward(input_ids).logits[:,-1:]

    def estimated_macs_per_token(self,context=512):
        if self.variant=='native':return super().estimated_macs_per_token(context)
        raise NotImplementedError('Use measured timing; native MAC formula does not apply to replacement cells')

    @torch.no_grad()
    def constrain(self):
        if self.variant=='paragru':self.fast_cell.constrain()

def build(arm,config,seed=101601):
    assert arm in LABELS,arm
    torch.manual_seed(seed)
    return GRUHierarchy(kind=arm,**config)

def logits(model,arm,x,targets):
    return model(x).logits

def optimizer(model,lr,weight_decay=.1):
    embeddings={id(m.weight) for m in model.modules() if isinstance(m,nn.Embedding) and m.num_embeddings==2048}
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        # ParaGRU diagonal vectors are stored together; treat as ndim1 vectors.
        (decay if p.ndim>=2 and id(p) not in embeddings and name!='fast_cell.A' else no_decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':weight_decay},
                             {'params':no_decay,'weight_decay':0.}],lr=lr,betas=(.9,.95),eps=1e-8)
