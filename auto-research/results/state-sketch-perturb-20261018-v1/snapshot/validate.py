"""Mechanism checks before any language optimization."""
import copy,importlib.util,json,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"upstream"))
from models import build,optimizer
from geometry import GROUPS,identities,transform,transform_state,transform_tensor,proposal
from sketches import Capture,describe

def maxerr(a,b):
    if isinstance(a,dict):return max(maxerr(a[k],b[k]) for k in a)
    if torch.is_tensor(a):return float((a.detach()-b.detach()).abs().max())
    return float(abs(a-b))
def cfg(width=8,vocab=64):
    return dict(width=width,vocab_size=vocab,max_context=512,segment_length=32,document_segments=8)
def native_names(d):
    return {n.replace("fast_sequence.","fast_cell.").replace("_l0",""):v for n,v in d.items()}
def run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    rows=[]
    spec=importlib.util.spec_from_file_location("historical",ROOT/"upstream/exact_historical_predictive_models.py")
    historical=importlib.util.module_from_spec(spec);sys.modules[spec.name]=historical;spec.loader.exec_module(historical)
    for seed,length in [(11,33),(17,257)]:
        torch.manual_seed(seed)
        ref=historical.HierarchicalDistributionStudent(**cfg()).double()
        m=build("native",cfg(),seed).double();m.load_state_dict({n.replace("fast_cell.","fast_sequence.").replace("fast_sequence.weight_ih","fast_sequence.weight_ih_l0").replace("fast_sequence.weight_hh","fast_sequence.weight_hh_l0").replace("fast_sequence.bias_ih","fast_sequence.bias_ih_l0").replace("fast_sequence.bias_hh","fast_sequence.bias_hh_l0"):v for n,v in ref.state_dict().items()})
        x=torch.randint(64,(2,length));y=torch.randint(64,(2,length))
        state=ref.initial_state(2,torch.device("cpu"))
        for k,v in state.items():
            if torch.is_tensor(v):state[k]=torch.randn_like(v)*.2
        state["segment_tokens"]=31;state["segments"]=7
        s=copy.deepcopy(state);zs=[]
        for token in x.unbind(1):
            z,s,_=ref.step(token,s);zs.append(z)
        zref=torch.stack(zs,1);out=m(x,copy.deepcopy(state))
        error=max(maxerr(zref,out.logits),maxerr(s,out.final_state));assert error<1e-9,error
        F.cross_entropy(zref.flatten(0,1),y.flatten()).backward()
        F.cross_entropy(out.logits.flatten(0,1),y.flatten()).backward()
        grad=maxerr({n:p.grad for n,p in ref.named_parameters()},native_names({n:p.grad for n,p in m.named_parameters()}))
        assert grad<1e-9,grad
        o1=optimizer(ref,.002);o2=optimizer(m,.002);o1.step();o2.step()
        update=maxerr(ref.state_dict(),native_names(m.state_dict()));assert update<1e-9,update
        rows.append({"check":"exact_historical_vs_upstream_batched","seed":seed,"length":length,"forward_state_max_abs":error,"gradient_max_abs":grad,"adamw_step_max_abs":update})
    for kind in ("native","mingru_linear","paragru"):
        a=build(kind,cfg(12),103).double();b=copy.deepcopy(a)
        x=torch.randint(64,(2,257));y=torch.randint(64,(2,257))
        z1=a(x);z2=b(x,backend="tokenwise")
        err=max(maxerr(z1.logits,z2.logits),maxerr(z1.final_state,z2.final_state));assert err<2e-9,(kind,err)
        F.cross_entropy(z1.logits.flatten(0,1),y.flatten()).backward()
        F.cross_entropy(z2.logits.flatten(0,1),y.flatten()).backward()
        grad=maxerr({n:p.grad for n,p in a.named_parameters()},{n:p.grad for n,p in b.named_parameters()});assert grad<2e-8,(kind,grad)
        rows.append({"check":"upstream_parallel_vs_tokenwise","kind":kind,"forward_state_max_abs":err,"gradient_max_abs":grad})
        oa=optimizer(a,.002);oa.step();a.constrain();oa.zero_grad(set_to_none=True)
        b=copy.deepcopy(a);ob=optimizer(b,.002);ob.load_state_dict(copy.deepcopy(oa.state_dict()))
        g=torch.Generator().manual_seed(93);p=identities(12)
        for name in p:p[name]=torch.randperm(len(p[name]),generator=g)
        b.load_state_dict(transform(a.state_dict(),p))
        for name,param in b.named_parameters():
            for key,val in ob.state[param].items():
                if torch.is_tensor(val) and val.shape==param.shape:ob.state[param][key]=transform_tensor(name,val,p)
        ca=Capture(a);cb=Capture(b);ca.enabled=cb.enabled=True
        z1=a(x);z2=b(x);ca.enabled=cb.enabled=False
        err=max(maxerr(z1.logits,z2.logits),maxerr(transform_state(z1.final_state,p),z2.final_state))
        assert err<2e-9,(kind,err)
        ta,tb=ca.pop(),cb.pop()
        da=describe(ta);db=describe(tb)
        sketch=max(maxerr(da[k][:,p[k]],db[k]) for k in GROUPS);assert sketch<1e-7,(kind,sketch)
        d1,_=proposal(a.state_dict(),da,12,81,"sketch",.003)
        d2,_=proposal(b.state_dict(),db,12,81,"sketch",.003)
        direction=maxerr(transform(d1,p),d2);assert direction<1e-8,(kind,direction)
        F.cross_entropy(z1.logits.flatten(0,1),y.flatten()).backward()
        F.cross_entropy(z2.logits.flatten(0,1),y.flatten()).backward()
        for mod,opt in [(a,oa),(b,ob)]:
            torch.nn.utils.clip_grad_norm_(mod.parameters(),1.);opt.step();mod.constrain()
        optim=maxerr(transform(a.state_dict(),p),b.state_dict());assert optim<1e-8,(kind,optim)
        rows.append({"check":"full_network_permutation_and_optimizer","kind":kind,"forward_state_max_abs":err,"sketch_max_abs":sketch,"proposal_max_abs":direction,"optimizer_step_max_abs":optim})
        ca.close();cb.close()
    dest=ROOT/"auto-research/results/state-sketch-perturb-20261018-v1";dest.mkdir(parents=True,exist_ok=True)
    receipt={"status":"PASS","tests":rows,"tolerances":{"float64_reference":1e-9,"parallel_gradient":2e-8,"equivariant_sketch":1e-7,"equivariant_proposal_and_optimizer":1e-8},"torch":torch.__version__}
    (dest/"PREFLIGHT.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)
if __name__=="__main__":run()
