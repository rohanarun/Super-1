import bootstrap,datetime,hashlib,json,platform
import torch
from model import ARMS
from experiment import ROOT,SEEDS,PHASES,atomic_json
selection={};pilots=[]
for arm in ARMS:
    choices=[]
    for lr in [.001,.003]:
        path=ROOT/'results/pilot'/f'{arm}_21901_{lr:g}'/'terminal.json'
        r=json.loads(path.read_text());assert r['status']=='completed'
        score=sum(v['accuracy'] for v in r['validation'])/len(r['validation'])
        choices.append((score,lr));pilots.append({'arm':arm,'lr':lr,'score':score,'cpu':r['training_cpu_seconds'],'steps':r['updates']})
    selection[arm]=max(choices)[1]
protocol={'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'question':'Can compact Kron outperform minGRU at no greater persistent state, weight memory and training CPU budget?',
 'scope':'Synthetic retrieval in the existing fast/segment/document hierarchy. No curve-generated weights; CPU implementation only.',
 'arms':ARMS,'seeds':SEEDS,'learning_rates':selection,'pilots':pilots,
 'pilot_budget':'100 process CPU seconds per LR per arm, maximum 10000 short updates; common seed21901 and validation54000000.',
 'curriculum':[{'name':name,'ages':ages,'max_updates':steps} for name,ages,steps in PHASES],
 'resource_policy':'Fresh minGRU runs complete all stages. Each compact run is capped by the paired minGRU training process CPU time separately in each stage, and by its update count. A one-second or 1.5x recent-step reserve prevents budget overrun. Faster arms cannot spend more updates/data. Each stage uses paired deterministic prefixes; later-stage offsets are independent of previous update counts.',
 'cpu_clock':'time.process_time, one Torch/OpenMP/MKL/OpenBLAS thread per worker; includes generation, forward, backward, clipping, optimizer; excludes checkpoints, logging, validation, interpreter import and experiment engineering.',
 'initialization':'Shared embedding, hierarchy, combine and task head are byte-identical within seed. Compact memory is newly initialized by the same PyTorch default/author gate rules.',
 'training':{'batch':32,'optimizer':'AdamW','betas':[.9,.95],'weight_decay':.1,'warmup':50,'gradient_clip':1.,'dtype':'FP32','device':'CPU','checkpoint_every':500},
 'confirmation':{'seed_base':76000000,'ages':[32,128,512,2048],'primary_modes':['competing','overwrite'],'primary_age':512,'primary_n':1024,'other_n':256,'selection':'final checkpoint only; no confirmation-based tuning'},
 'resource_benchmarks':{'decode_batch':[1,32],'prefill_length':513,'training_batch':32,'training_length':513,'timing':'serial, interleaved arms, one thread; independent of confirmation selection'},
 'fidelity':json.loads((ROOT/'results/verification.json').read_text())['status'],
 'platform':{'python':platform.python_version(),'torch':torch.__version__,'cpu_quota_cores':8},
 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in str(p)}}
atomic_json(protocol,ROOT/'protocol.json');print(json.dumps({'selected_learning_rates':selection,'pilots':pilots},indent=2))
