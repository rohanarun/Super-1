import bootstrap,copy,json,time
import torch
from torch.nn import functional as F
from model import ARMS,build,compact_chunk,state_vector,accounting
from fla.ops.kronlstm.naive import naive_recurrent_kronlstm
from tasks import episode,verify_oracle,MODES

torch.set_num_threads(1)
def run():
    result={'status':'running','core':[],'hierarchy':[],'accounting':{},'gradients':{},'overfit':[]}
    for arm,shape in list(ARMS.items())[1:]:
        H,K,V=shape
        for dtype in (torch.float32,torch.float64):
            for T in (1,17,33):
                torch.manual_seed(912+T);B=2
                q,k,v=[F.normalize(torch.randn(B,T,H,D,dtype=dtype),dim=-1).requires_grad_() for D in (K,K,V)]
                g=(-torch.rand(B,T,H,K,dtype=dtype)*.1).requires_grad_()
                b=torch.rand(B,T,H,dtype=dtype,requires_grad=True);c=torch.rand_like(b,requires_grad=True)
                s=torch.randn(B,H,K,V,dtype=dtype,requires_grad=True);inputs=[q,k,v,g,b,c,s]
                a,sa=compact_chunk(q,k,v,g,b,c,s)
                z,sz=naive_recurrent_kronlstm(q,k,v,g,b,initial_state=s,target_gate=c,output_final_state=True)
                tol=dict(atol=3e-5,rtol=3e-4) if dtype==torch.float32 else dict(atol=2e-10,rtol=2e-9)
                torch.testing.assert_close(a,z,**tol);torch.testing.assert_close(sa,sz,**tol)
                wa=torch.randn_like(a);ws=torch.randn_like(sa)
                ga=torch.autograd.grad((a*wa).sum()+(sa*ws).sum(),inputs,retain_graph=True)
                gz=torch.autograd.grad((z*wa).sum()+(sz*ws).sum(),inputs)
                for x,y in zip(ga,gz):torch.testing.assert_close(x,y,**tol)
                result['core'].append({'arm':arm,'T':T,'dtype':str(dtype),
                    'forward_error':float((a-z).abs().max()),'state_error':float((sa-sz).abs().max()),
                    'gradient_error':max(float((x-y).abs().max()) for x,y in zip(ga,gz))})
    common=dict(build('mingru',991).named_parameters())
    for arm in ARMS:
        m=build(arm,991);result['accounting'][arm]=accounting(m)
        for name,p in m.named_parameters():
            if name in common and not name.startswith('fast_cell.'):
                assert torch.equal(p,common[name]),name
        for T in (33,289):
            torch.manual_seed(190);x=torch.randint(0,44,(2,T));a=build(arm,992);b=copy.deepcopy(a)
            out=a(x);state=None;parts=[];position=0
            for width in (7,24,1,31,70,1000):
                if position==T:break
                end=min(T,position+width);r=b(x[:,position:end],state);parts.append(r.logits)
                state=r.final_state;position=end
            y=torch.cat(parts,1);tol=dict(atol=4e-5,rtol=4e-4)
            torch.testing.assert_close(out.logits,y,**tol)
            torch.testing.assert_close(state_vector(out.final_state),state_vector(state),**tol)
            (out.logits.square().mean()+state_vector(out.final_state).square().mean()).backward()
            (y.square().mean()+state_vector(state).square().mean()).backward()
            for (n,p),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
                assert n==nb
                if p.grad is None:assert pb.grad is None;continue
                torch.testing.assert_close(p.grad,pb.grad,atol=1e-4,rtol=1e-3)
            with torch.no_grad():
                a.eval();full=a(x);seq=a(x,backend='tokenwise')
                torch.testing.assert_close(full.logits,seq.logits,**tol)
                if arm!='mingru':
                    ref=a(x,backend='reference');torch.testing.assert_close(full.logits,ref.logits,**tol)
                    a.kron.projection_backend='plain';plain=a(x)
                    torch.testing.assert_close(full.logits,plain.logits,**tol)
            result['hierarchy'].append({'arm':arm,'length':T,'outputs_states_gradients_tokenwise':'PASS'})
        m=build(arm,993);x,y,_=episode(1403,batch=8,age=512,mode='overwrite')
        F.cross_entropy(m(torch.tensor(x)).logits[:,-1],torch.tensor(y)).backward()
        groups=['embedding','segment_cell','document_cell','combine','value_head']
        groups+=['fast_cell'] if arm=='mingru' else ['kron.'+n for n in ['qkv_proj','b_proj','c_proj','f_proj','g_proj','o_norm','o_proj','A_log','dt_bias']]
        grads={g:sum(float(p.grad.square().sum()) for n,p in m.named_parameters() if n.startswith(g) and p.grad is not None)**.5 for g in groups}
        assert min(grads.values())>0;result['gradients'][arm]=grads
    for mode in MODES:
        for age in (4,32,512,2048):
            x,y,_=episode(8321,batch=13,age=age,mode=mode);verify_oracle(x,y)
    result['paired_common_initialization']='PASS';result['task_oracle']='PASS'
    for arm in ARMS:
        m=build(arm,9911);opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0)
        x,y,_=episode(9933,batch=16,age=32,mode='competing');x=torch.tensor(x);y=torch.tensor(y)
        for step in range(800):
            opt.zero_grad(set_to_none=True);z=m(x).logits[:,-1];loss=F.cross_entropy(z,y)
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            if loss.item()<.03 and (z.argmax(-1)==y).all():break
        row={'arm':arm,'steps':step+1,'loss':loss.item(),'accuracy':(z.argmax(-1)==y).float().mean().item()}
        result['overfit'].append(row);print(json.dumps(row),flush=True)
        assert row['accuracy']==1.,row
    result['status']='PASS'
    (bootstrap.ROOT/'results/verification.json').write_text(json.dumps(result,indent=2));print('VERIFICATION PASS',flush=True)

if __name__=='__main__':run()
