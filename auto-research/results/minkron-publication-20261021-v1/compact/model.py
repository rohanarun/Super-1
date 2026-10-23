"""Compact Kron fast memories in the identical existing Super-1 hierarchy.

The three small matrices use the authors' equations, unit-normalized q/k/v,
SiLU, separate write/erase gates, vector decay and gated RMSNorm. No convolution.
The optimized CPU core solves for update coefficients directly within each
hierarchy segment, without padding or forming a triangular inverse.
"""
import bootstrap
import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from previous_model import MinHierarchy, baseline
from fla.layers.kronlstm import KronLSTM
from fla.modules import FusedRMSNormGated
from fla.ops.kronlstm.naive import naive_recurrent_kronlstm, naive_chunk_kronlstm

ARMS={'mingru':None,'kron8':(1,8,8),'kron4h4':(4,4,4),'kron2h4':(2,4,4)}

def compact_chunk(q,k,v,g,beta,c,state=None):
    """Exact one-chunk recurrence; supports arbitrary head sizes, FP32/FP64."""
    orig=q.dtype;acc=torch.float64 if orig==torch.float64 else torch.float32
    q,k,v,g,beta,c=[x.transpose(1,2).to(acc) for x in (q,k,v,g,beta,c)]
    B,H,T,K=q.shape;V=v.shape[-1]
    S=q.new_zeros(B,H,K,V) if state is None else state.to(acc)
    if T==1:
        S=S*g[:,:,0].exp().unsqueeze(-1)
        norm=v[:,:,0].square().sum(-1)
        inv=norm.clamp_min(1e-18).reciprocal()
        score=((k[:,:,0].unsqueeze(-1)*S)*v[:,:,0].unsqueeze(-2)).sum((-2,-1))
        u=c[:,:,0]*norm*inv-beta[:,:,0]*inv*score
        S=S+u[...,None,None]*k[:,:,0,:,None]*v[:,:,0,None,:]
        o=(q[:,:,0,:,None]*S).sum(-2)*K**-.5
        return o.unsqueeze(1).to(orig),S
    gc=g.cumsum(-2);gam=gc.exp()
    mask=torch.ones(T,T,dtype=torch.bool,device=q.device).tril()
    L=(gc.unsqueeze(-2)-gc.unsqueeze(-3)).masked_fill(~mask.unsqueeze(-1),0).exp()
    Akk=(k.unsqueeze(-2)*k.unsqueeze(-3)*L).sum(-1)
    Aqk=(q.unsqueeze(-2)*k.unsqueeze(-3)*L).sum(-1)*mask*K**-.5
    vn=v.square().sum(-1);inv=vn.clamp_min(1e-18).reciprocal();bs=beta*inv
    M=(bs.unsqueeze(-1)*Akk*(v@v.transpose(-1,-2))).tril(-1)
    M=M+torch.eye(T,dtype=acc,device=q.device)
    p=(((k*gam)@S)*v).sum(-1)
    rhs=c*vn*inv-bs*p
    u=torch.linalg.solve_triangular(M,rhs.unsqueeze(-1),upper=False,unitriangular=True)
    uv=u*v
    out=K**-.5*((q*gam)@S)+Aqk@uv
    tail=k*(gc[...,-1:,:]-gc).exp()
    final=gam[...,-1,:,None]*S+tail.transpose(-1,-2)@uv
    return out.transpose(1,2).contiguous().to(orig),final

class CompactKron(nn.Module):
    def __init__(self,H,K,V,input_width=192,output_width=64):
        super().__init__();self.H=H;self.K=K;self.V=V
        self.num_heads=H;self.key_dim=H*K;self.value_dim=H*V
        self.qkv_proj=nn.Linear(input_width,H*(2*K+V),bias=False)
        self.b_proj=nn.Linear(input_width,H,bias=False)
        self.c_proj=nn.Linear(input_width,H,bias=False)
        self.f_proj=nn.Sequential(nn.Linear(input_width,V,bias=False),nn.Linear(V,H*K,bias=False))
        self.A_log=nn.Parameter(torch.empty(H));self.dt_bias=nn.Parameter(torch.empty(H*K))
        KronLSTM._reset_decay_parameters(self)
        self.g_proj=nn.Linear(input_width,H*V,bias=True)
        self.o_norm=FusedRMSNormGated(V,activation='sigmoid',eps=1e-5)
        self.o_proj=nn.Linear(H*V,output_width,bias=False)
        nn.init.zeros_(self.g_proj.weight);nn.init.constant_(self.g_proj.bias,3.)
        for name,p in self.named_parameters():
            if name in ('A_log','dt_bias') or name.endswith('.bias') or name.startswith('o_norm.'):
                p._no_weight_decay=True
        self.projection_backend='packed';self.core_backend='compact'

    def forward(self,x,state=None,backend=None):
        B,T,_=x.shape;H,K,V=self.H,self.K,self.V
        if self.projection_backend=='packed':
            weights=torch.cat([self.qkv_proj.weight,self.b_proj.weight,self.c_proj.weight,
                               self.f_proj[0].weight,self.g_proj.weight])
            qkv,b,c,f,og=F.linear(x,weights).split([H*(2*K+V),H,H,V,H*V],-1)
            f=F.linear(f,self.f_proj[1].weight);og=og+self.g_proj.bias
        else:
            qkv=self.qkv_proj(x);b=self.b_proj(x);c=self.c_proj(x);f=self.f_proj(x);og=self.g_proj(x)
        q,k,v=F.silu(qkv).split([H*K,H*K,H*V],-1)
        q,k,v=[F.normalize(a.reshape(B,T,H,D),dim=-1) for a,D in ((q,K),(k,K),(v,V))]
        acc=torch.float64 if x.dtype==torch.float64 else torch.float32
        g=-self.A_log.to(acc).exp()[None,None,:,None]*F.softplus(f.to(acc)+self.dt_bias).reshape(B,T,H,K)
        beta=b.sigmoid();c=c.to(acc).sigmoid()
        backend=backend or self.core_backend
        if backend=='compact':o,final=compact_chunk(q,k,v,g,beta,c,state)
        else:
            fn=naive_recurrent_kronlstm if backend=='reference' else naive_chunk_kronlstm
            kw={'chunk_size':T} if backend=='author_chunk' else {}
            o,final=fn(q,k,v,g,beta,initial_state=state,target_gate=c,output_final_state=True,**kw)
        o=self.o_norm(o,og.reshape(B,T,H,V)).reshape(B,T,H*V)
        return self.o_proj(o.to(x.dtype)),final

class CompactHierarchy(MinHierarchy):
    def __init__(self,arm,width=64):
        super().__init__(width=width);del self.fast_cell
        self.arm=arm;self.kron=CompactKron(*ARMS[arm],input_width=3*width,output_width=width)

    def initial_state(self,batch_size,device,dtype=None):
        state=super().initial_state(batch_size,device,dtype);del state['fast']
        H,K,V=ARMS[self.arm]
        state['kron_state']=torch.zeros(batch_size,H,K,V,device=device,dtype=dtype or self.embedding.weight.dtype)
        return state

    def forward(self,input_ids,state=None,backend=None):
        if input_ids.shape[1]==0:raise ValueError('At least one token required')
        if state is None:state=self.initial_state(len(input_ids),input_ids.device)
        segment=state['segment'];document=state['document'];segment_sum=state['segment_sum']
        count=int(state['segment_tokens']);segments=int(state['segments']);memory=state['kron_state']
        embeddings=self.embedding(input_ids);features=[];position=0
        while position<input_ids.shape[1]:
            length=min(self.segment_length-count,input_ids.shape[1]-position)
            if backend=='tokenwise':length=1
            tokens=embeddings[:,position:position+length];old_segment=segment;old_document=document
            x=torch.cat((tokens,segment[:,None].expand(-1,length,-1),document[:,None].expand(-1,length,-1)),-1)
            values,memory=self.kron(x,memory,backend=None if backend=='tokenwise' else backend)
            segment_sum=segment_sum+values.sum(1);count+=length
            if count==self.segment_length:
                segment=self.segment_cell(segment_sum/self.segment_length,segment)
                segment_sum=torch.zeros_like(segment_sum);count=0;segments+=1
                if segments==self.document_segments:
                    document=self.document_cell(segment,document);segments=0
                sv=torch.cat((old_segment[:,None].expand(-1,length-1,-1),segment[:,None]),1)
                dv=torch.cat((old_document[:,None].expand(-1,length-1,-1),document[:,None]),1)
            else:
                sv=segment[:,None].expand(-1,length,-1);dv=document[:,None].expand(-1,length,-1)
            features.append(torch.cat((values,sv,dv),-1));position+=length
        hidden=self.combine(torch.cat(features,1))
        return SimpleNamespace(logits=self.logits(hidden),hidden_states=hidden,final_state=dict(
            segment=segment,document=document,segment_sum=segment_sum,segment_tokens=count,
            segments=segments,kron_state=memory))

def build(arm,seed):
    torch.manual_seed(seed)
    m=MinHierarchy() if arm=='mingru' else CompactHierarchy(arm)
    m.arm=arm;return m

def state_vector(state):
    return torch.cat([v.flatten(1) for v in state.values() if torch.is_tensor(v)],-1)

def accounting(m):
    with torch.no_grad():state=m(torch.zeros(1,1,dtype=torch.long)).final_state
    values=[v for v in state.values() if torch.is_tensor(v)]
    result={'parameters':sum(p.numel() for p in m.parameters()),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),
            'persistent_tensor_state_bytes':sum(v.numel()*v.element_size() for v in values),
            'persistent_backing_storage_bytes':sum(v.untyped_storage().nbytes() for v in values),
            'state_shapes':{k:list(v.shape) if torch.is_tensor(v) else v for k,v in state.items()}}
    assert result['persistent_tensor_state_bytes']<=1024
    assert result['persistent_backing_storage_bytes']==result['persistent_tensor_state_bytes']
    assert result['parameters']<=111048
    return result
