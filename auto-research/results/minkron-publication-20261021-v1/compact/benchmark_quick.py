import bootstrap,json,time,statistics
import torch
from torch.nn import functional as F
from model import ARMS,build
torch.set_num_threads(1)
records=[]
for arm in ARMS:
    m=build(arm,1)
    for length in (33,129,513):
        x=torch.randint(0,44,(32,length));y=torch.randint(0,8,(32,))
        for backend in (['default'] if arm=='mingru' else ['compact','author_chunk']):
            if arm!='mingru':m.kron.core_backend=backend
            timing=[]
            for rep in range(8):
                start=time.process_time();m.zero_grad(set_to_none=True)
                F.cross_entropy(m(x).logits[:,-1],y).backward()
                if rep>=3:timing.append(time.process_time()-start)
            row=dict(arm=arm,length=length,backend=backend,forward_backward_cpu_ms=statistics.median(timing)*1000)
            records.append(row);print(json.dumps(row),flush=True)
(bootstrap.ROOT/'results/backend_benchmark.json').write_text(json.dumps(records,indent=2))
