"""Matched-window three-GRU pilot; run one independently resumable arm."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse, hashlib, json, math, platform, resource, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import build, logits, optimizer

ROOT=Path(__file__).resolve().parent
DATA=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(path,obj):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)

@torch.no_grad()
def evaluate(m,arm,data,starts):
 m.eval(); losses=[];correct=0
 for s in starts:
  xy=torch.from_numpy(np.array(data[s:s+513],dtype=np.int64))[None]
  z=logits(m,arm,xy[:,:-1].contiguous(),xy[:,1:].contiguous())
  losses.append(float(F.cross_entropy(z.flatten(0,1),xy[:,1:].flatten())))
  correct+=int((z.argmax(-1)==xy[:,1:]).sum())
 return dict(ce=float(np.mean(losses)),perplexity=math.exp(float(np.mean(losses))),
             top1=correct/(512*len(starts)),targets=512*len(starts),window_ce=losses)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('arm');parser.add_argument('--cpu',type=int)
 parser.add_argument('--data-dir',type=Path,required=True)
 args=parser.parse_args();arm=args.arm;data_dir=args.data_dir.resolve()
 p=json.loads((ROOT/'PROTOCOL.json').read_text());offsets=json.loads((ROOT/'OFFSETS.json').read_text())
 assert sha(ROOT/'OFFSETS.json')==p['offsets_sha256']
 assert arm in p['arms']
 for path,digest in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()).items():
  assert sha(ROOT/path)==digest,path
 for name in ('train','validation'):
  assert sha(data_dir/f'{name}.bin')==p['data']['split_metadata']['splits'][name]['sha256'],name
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 torch.use_deterministic_algorithms(True)
 if args.cpu is not None:os.sched_setaffinity(0,{args.cpu})
 resource.setrlimit(resource.RLIMIT_CPU,(p['resource_limits']['cpu_seconds_per_arm'],p['resource_limits']['cpu_seconds_per_arm']+10))
 out=ROOT/'execution'/arm;out.mkdir(parents=True,exist_ok=True)
 cp=out/'checkpoint.pt';protocol_sha=sha(ROOT/'PROTOCOL.json')
 m=build(arm,p['configs'][arm],p['seed']);opt=optimizer(m,p['optimizer']['peak_lr'])
 assert sum(v.numel() for v in m.parameters())==p['parameters'][arm]
 state=dict(step=0,training_targets=0,train_wall_seconds=0.,train_cpu_seconds=0.,train=[],monitor=[])
 runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
              threads=1,affinity=sorted(os.sched_getaffinity(0)),device='cpu',dtype='float32')
 if cp.exists():
  old=torch.load(cp,map_location='cpu',weights_only=False)
  assert old['protocol_sha256']==protocol_sha
  m.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);state=old['state']
  torch.set_rng_state(old['torch_rng'])
  print('RESUME',arm,state['step'],flush=True)
 def checkpoint():
  tmp=cp.with_suffix('.tmp')
  torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
                  state=state,arm=arm,config=p['configs'][arm],protocol_sha256=protocol_sha),tmp);tmp.replace(cp)
 def monitor():
  ev=evaluate(m,arm,val,offsets['monitor_starts'])
  row=dict(step=state['step'],training_targets=state['training_targets'],
           train_wall_seconds=state['train_wall_seconds'],train_cpu_seconds=state['train_cpu_seconds'],
           validation_ce=ev['ce'],validation_top1=ev['top1'])
  state['monitor'].append(row)
  save(out/'progress.json',dict(arm=arm,**row,parameters=p['parameters'][arm],status='RUNNING'))
  save(out/'curves.json',state)
  print(arm,row,flush=True)
 train=np.memmap(data_dir/'train.bin',dtype=np.uint16,mode='r')
 val=np.memmap(data_dir/'validation.bin',dtype=np.uint16,mode='r')
 begin=time.monotonic(); cpu_begin=time.process_time()
 save(out/'START.json',dict(arm=arm,protocol_sha256=protocol_sha,runtime=runtime,
                          started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
 if not state['monitor']:monitor();checkpoint()
 for step in range(state['step']+1,p['updates']+1):
  if time.monotonic()-begin>p['resource_limits']['wall_seconds_per_arm']:
   checkpoint();raise TimeoutError('Explicit wall budget reached; checkpoint preserved')
  tick=time.perf_counter();cpu_tick=time.process_time();m.train()
  starts=offsets['train_starts'][step-1]
  xy=torch.from_numpy(np.stack([np.array(train[s:s+513],dtype=np.int64) for s in starts]))
  warmup=p['optimizer']['warmup_updates']
  factor=step/warmup if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/(p['updates']-warmup)))
  lr=p['optimizer']['peak_lr']*factor
  for group in opt.param_groups:group['lr']=lr
  opt.zero_grad(set_to_none=True)
  z=logits(m,arm,xy[:,:-1].contiguous(),xy[:,1:].contiguous())
  ce=F.cross_entropy(z.flatten(0,1),xy[:,1:].flatten());ce.backward()
  norm=torch.nn.utils.clip_grad_norm_(m.parameters(),p['optimizer']['clip_norm'])
  if not torch.isfinite(ce) or not torch.isfinite(norm):
   raise FloatingPointError(f'Nonfinite loss or gradient at step {step}')
  opt.step()
  m.constrain()
  state['train_wall_seconds']+=time.perf_counter()-tick
  state['train_cpu_seconds']+=time.process_time()-cpu_tick
  state['step']=step;state['training_targets']+=xy[:,1:].numel()
  state['train'].append(dict(step=step,training_targets=state['training_targets'],ce=float(ce.detach()),
                            lr=lr,gradient_norm=float(norm)))
  if step in p['monitor_steps']:monitor()
  if step%100==0:checkpoint()
 checkpoint()
 final=evaluate(m,arm,val,offsets['final_starts'])
 result=dict(arm=arm,status='COMPLETE',parameters=p['parameters'][arm],seed=p['seed'],
             config=p['configs'][arm],updates=state['step'],training_targets=state['training_targets'],
             final=final,monitor=state['monitor'],train_wall_seconds=state['train_wall_seconds'],
             train_cpu_seconds=state['train_cpu_seconds'],run_wall_seconds=time.monotonic()-begin,
             run_cpu_seconds=time.process_time()-cpu_begin,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             checkpoint_sha256=sha(cp),protocol_sha256=protocol_sha,runtime=runtime)
 if arm=='paragru':
  result['solver_histogram']=m.fast_cell.solver_histogram
  result['max_solver_residual']=m.fast_cell.max_residual
  result['diagonal_vector_norms']=m.fast_cell.A.detach().norm(dim=1).tolist()
 save(out/'RESULT.json',result);save(out/'curves.json',state)
 save(out/'progress.json',dict(arm=arm,status='COMPLETE',step=state['step'],final_ce=final['ce']))
 print('COMPLETE',arm,final['ce'],flush=True)

if __name__=='__main__':main()
