"""Serial timing and profiler live-tensor peak accounting, separate from training caps."""
import bootstrap,argparse,gc,json,statistics,time
import torch
from torch.nn import functional as F
from model import ARMS,build,accounting
from experiment import ROOT,atomic_json,optimizer
torch.set_num_threads(1)

def timing():
    models={a:build(a,9911).eval() for a in ARMS};rows=[]
    # Interleave models in rotated order; models retain no cross-episode state.
    for B in (1,32):
        x=torch.randint(0,44,(B,256));samples={a:[] for a in ARMS}
        for rep in range(7):
            order=list(ARMS);order=order[rep%4:]+order[:rep%4]
            for arm in order:
                m=models[arm];state=None;t=time.perf_counter()
                with torch.inference_mode():
                    for i in range(x.shape[1]):state=m(x[:,i:i+1],state).final_state
                if rep>=2:samples[arm].append((time.perf_counter()-t)/x.shape[1])
        for arm,values in samples.items():
            rows.append({'arm':arm,'kind':'decode','batch':B,'seconds_per_step_median':statistics.median(values),'samples':values})
    x=torch.randint(0,44,(32,513));y=torch.randint(0,8,(32,))
    for kind in ('prefill','train_step'):
        values={a:[] for a in ARMS};opts={a:optimizer(m,.003) for a,m in models.items()}
        for rep in range(9):
            order=list(ARMS);order=order[rep%4:]+order[:rep%4]
            for arm in order:
                m=models[arm];t=time.perf_counter()
                if kind=='prefill':
                    with torch.inference_mode():m(x)
                else:
                    opts[arm].zero_grad(set_to_none=True);z=m(x).logits[:,-1]
                    F.cross_entropy(z,y).backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opts[arm].step()
                if rep>=3:values[arm].append(time.perf_counter()-t)
        for arm,vs in values.items():rows.append({'arm':arm,'kind':kind,'batch':32,'length':513,'seconds_median':statistics.median(vs),'samples':vs})
    atomic_json({'rows':rows,'accounting':{a:accounting(m) for a,m in models.items()},
        'note':'One thread, serial interleaved arms, random model weights and input; excludes initialization; CPU wall clock.'},ROOT/'results/timing.json')
    print(json.dumps(rows,indent=2))

def memory(arm,kind):
    m=build(arm,9911);opt=optimizer(m,.003)
    x=torch.randint(0,44,(32,513));y=torch.randint(0,8,(32,))
    if kind=='train':
        F.cross_entropy(m(x).logits[:,-1],y).backward();opt.step();opt.zero_grad(set_to_none=True)
    gc.collect()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],profile_memory=True,record_shapes=True,with_stack=True) as p:
        if kind=='train':
            loss=F.cross_entropy(m(x).logits[:,-1],y);loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        else:
            m.eval()
            with torch.inference_mode():out=m(x)
    dst=ROOT/'results/memory';dst.mkdir(exist_ok=True)
    temp=dst/f'{arm}_{kind}_timeline.json';p.export_memory_timeline(str(temp),device='cpu')
    times,sizes=json.loads(temp.read_text());totals=[sum(v) for v in sizes]
    row={'arm':arm,'kind':kind,'batch':32,'length':513,'peak_live_tensor_bytes':max(totals),
         'initial_live_tensor_bytes':totals[0],'peak_increment_bytes':max(totals)-totals[0],
         'note':'PyTorch CPU profiler tensor allocations, includes observed preexisting tensors. Excludes Python/runtime and untracked library workspace. This is working memory, not persistent sequence state.'}
    atomic_json(row,dst/f'{arm}_{kind}.json');temp.unlink();print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm');p.add_argument('--kind',default='train');p.add_argument('--timing',action='store_true');a=p.parse_args()
    if a.timing:timing()
    else:memory(a.arm,a.kind)
