"""Exact minGRU hierarchy baseline plus explicit Kron fast-mixer adaptation.

Shared hierarchy comes from the pinned existing Super-1 minGRU experiment.
It has no curve-generated weights. The separate curve models are not claimed
to be integrated. Slow recurrence, boundary timing and feedback are preserved.
Task adaptation: common 8-value head after the existing final LayerNorm.
Kron adaptation: input width 3w, output projection width w, H=2, K=V=16.
"""
import bootstrap
import importlib.util
import sys
from types import SimpleNamespace
import torch
from torch import nn
from fla.layers.kronlstm import KronLSTM

spec=importlib.util.spec_from_file_location('pinned_hierarchy',bootstrap.ROOT/'upstream/hierarchy/models.py')
baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)

class ValueHead:
    def logits(self,hidden):return self.value_head(self.final_norm(hidden))

class MinHierarchy(ValueHead,baseline.GRUHierarchy):
    def __init__(self,width=64,**kwargs):
        super().__init__(kind='mingru_linear',vocab_size=44,width=width,max_context=4096,
                         segment_length=32,document_segments=8)
        self.value_head=nn.Linear(width,8);self.arm='mingru'

class KronHierarchy(ValueHead,baseline.GRUHierarchy):
    def __init__(self,width=64,**kwargs):
        super().__init__(kind='mingru_linear',vocab_size=44,width=width,max_context=4096,
                         segment_length=32,document_segments=8)
        self.value_head=nn.Linear(width,8);self.arm='kron'
        del self.fast_cell
        self.kron=KronLSTM(hidden_size=3*width,k_head_dim=16,v_head_dim=16,
                          num_k_heads=2,num_v_heads=2,layer_idx=0,mode='chunk')
        # Rectangular projection adapts the official mixer to the fast-cell API.
        self.kron.o_proj=nn.Linear(32,width,bias=False)

    def initial_state(self,batch_size,device,dtype=None):
        state=super().initial_state(batch_size,device,dtype)
        state['kron_state']=None;state['conv_state']=None
        return state

    def forward(self,input_ids,state=None,backend=None):
        if input_ids.shape[1]==0:raise ValueError('At least one input token required')
        if state is None:state=self.initial_state(input_ids.shape[0],input_ids.device)
        fast=state['fast'];segment=state['segment'];document=state['document']
        segment_sum=state['segment_sum'];count=int(state['segment_tokens']);segments=int(state['segments'])
        cache={}
        if state.get('kron_state') is not None:
            cache[0]={'recurrent_state':state['kron_state'],'conv_state':state['conv_state']}
        embeddings=self.embedding(input_ids);features=[];position=0
        while position<input_ids.shape[1]:
            length=min(self.segment_length-count,input_ids.shape[1]-position)
            if backend=='tokenwise':length=1
            tokens=embeddings[:,position:position+length]
            old_segment=segment;old_document=document
            x=torch.cat((tokens,segment[:,None].expand(-1,length,-1),document[:,None].expand(-1,length,-1)),-1)
            values,_,cache=self.kron(x,past_key_values=cache,use_cache=True)
            fast=values[:,-1].clone()
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
            segment_tokens=count,segments=segments,kron_state=cache[0]['recurrent_state'],conv_state=cache[0]['conv_state'])
        return SimpleNamespace(logits=self.logits(hidden),hidden_states=hidden,final_state=final_state)

def build(arm,seed,width=64):
    torch.manual_seed(seed)
    return (MinHierarchy if arm=='mingru' else KronHierarchy)(width=width)

def state_vector(state):
    return torch.cat([v.flatten(1) for v in state.values() if torch.is_tensor(v)],-1)

def accounting(model):
    with torch.no_grad():state=model(torch.zeros(1,1,dtype=torch.long)).final_state
    return {'parameters':sum(p.numel() for p in model.parameters()),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'persistent_tensor_state_bytes':sum(v.numel()*v.element_size() for v in state.values() if torch.is_tensor(v)),
            'state_shapes':{k:list(v.shape) if torch.is_tensor(v) else v for k,v in state.items()},
            'exclusions':'Python counters/container overhead, weights, autograd activations and workspace'}
