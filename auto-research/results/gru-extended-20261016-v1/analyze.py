"""Check final checkpoints and compare full-model CPU execution after training."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import build,LABELS
from run import evaluate
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);args=parser.parse_args()
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
p=json.loads((ROOT/'PROTOCOL.json').read_text());offsets=json.loads((ROOT/'OFFSETS.json').read_text())
assert sha(ROOT/'OFFSETS.json')==p['offsets_sha256']
for f,h in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()).items():assert sha(ROOT/f)==h,f
assert all((ROOT/'execution'/a/'RESULT.json').exists() for a in p['arms'])
val=np.memmap(args.data_dir/'validation.bin',dtype=np.uint16,mode='r')
assert sha(args.data_dir/'validation.bin')==p['data']['split_metadata']['splits']['validation']['sha256']
rows=[];checks={};timings={};checkpoints={};window_loss={}
for arm in p['arms']:
 folder=ROOT/'execution'/arm;r=json.loads((folder/'RESULT.json').read_text());cp=folder/'checkpoint.pt'
 assert r['updates']==2000 and r['training_targets']==1024000
 assert sha(cp)==r['checkpoint_sha256']
 old=torch.load(cp,map_location='cpu',weights_only=False);assert old['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
 m=build(arm,p['configs'][arm],p['seed']);m.load_state_dict(old['model']);m.eval()
 replay=evaluate(m,arm,val,offsets['final_starts'])
 replay_error=max(abs(a-b) for a,b in zip(replay['window_ce'],r['final']['window_ce']))
 assert replay_error<1e-6;window_loss[arm]=replay['window_ce']
 s=offsets['final_starts'][0];xy=torch.from_numpy(np.array(val[s:s+513],dtype=np.int64))[None]
 grads=[];outs=[]
 for backend in [m.backend,'tokenwise']:
  m.zero_grad(set_to_none=True);out=m(xy[:,:-1],backend=backend);outs.append(out.logits.detach())
  F.cross_entropy(out.logits.flatten(0,1),xy[:,1:].flatten()).backward()
  assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
  grads.append(torch.cat([v.grad.flatten() for v in m.parameters()]))
 output_error=float((outs[0]-outs[1]).abs().max());relative=float((grads[0]-grads[1]).norm()/grads[1].norm())
 assert output_error<5e-5 and relative<2e-4
 with torch.no_grad():
  full=m(xy[:,:-1]);first=m(xy[:,:257]);second=m(xy[:,257:-1],first.final_state)
  prefix=float((full.logits[:,:257]-first.logits).abs().max());carry=float((full.logits[:,257:]-second.logits).abs().max())
  assert max(prefix,carry)<5e-5
  state=list(v for v in full.final_state.values() if isinstance(v,torch.Tensor))
  storage=sum({v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in state}.values())
  logical=sum(v.numel() for v in state)
 checks[arm]=dict(replay_max_ce_error=replay_error,final_ce_replay=replay['ce'],logit_error=output_error,
 gradient_relative_l2=relative,prefix_error=prefix,carry_error=carry,state_scalars=logical,state_backing_bytes=storage)
 del out,outs,grads,full,first,second
 timings[arm]={}
 for backend in p['benchmark']['backends'][arm]:
  timings[arm][backend]={}
  for mode in ['forward','forward_backward']:
   times=[]
   for i in range(7):
    m.zero_grad(set_to_none=True);tick=time.perf_counter()
    if mode=='forward':
     with torch.no_grad():out=m(xy[:,:-1],backend=backend).logits
    else:
     out=m(xy[:,:-1],backend=backend).logits
     F.cross_entropy(out.flatten(0,1),xy[:,1:].flatten()).backward()
    duration=time.perf_counter()-tick
    if i>=2:times.append(duration)
    del out
   timings[arm][backend][mode]=dict(median_seconds=statistics.median(times),p95_seconds=float(np.percentile(times,95)),samples_seconds=times)
 with torch.no_grad():
  times=[]
  for i in range(7):
   state=None;tick=time.perf_counter()
   for k in range(128):_,state,_=m.step(xy[:,k],state)
   if i>=2:times.append(time.perf_counter()-tick)
 timings[arm]['streaming128']=dict(median_seconds=statistics.median(times),p95_seconds=float(np.percentile(times,95)),samples_seconds=times)
 weights=sum(v.numel()*v.element_size() for v in m.parameters())
 rows.append(dict(arm=arm,label=LABELS[arm],seed=p['seed'],parameters=r['parameters'],ce=r['final']['ce'],
  perplexity=r['final']['perplexity'],top1=r['final']['top1'],train_cpu_seconds=r['train_cpu_seconds'],
  train_wall_seconds=r['train_wall_seconds'],peak_training_rss_kib=r['peak_process_rss_kib'],
  state_scalars=logical,state_bytes=storage,weight_bytes=weights))
 checkpoints[arm]=dict(path=str(cp.relative_to(ROOT)),bytes=cp.stat().st_size,sha256=sha(cp),step=2000,
  protocol_sha256=old['protocol_sha256'],includes=['model','optimizer','RNG','all training/monitor metrics'])
 if arm=='paragru':checks[arm].update(solver_histogram=r['solver_histogram'],max_solver_residual=r['max_solver_residual'])
 print(arm,rows[-1],flush=True);del m,old
native=next(r for r in rows if r['arm']=='native')
for r in rows:
 r['ce_delta_vs_native']=r['ce']-native['ce'];r['training_cpu_speedup_vs_native']=native['train_cpu_seconds']/r['train_cpu_seconds']
 if r['arm']=='native':r['decision']='CONTROL'
 else:
  r['decision']='RETAIN_EXPLORATORY_CANDIDATE' if r['ce_delta_vs_native']<=.03 and r['training_cpu_speedup_vs_native']>=1.25 else 'DOES_NOT_MEET_EXPLORATORY_RULE'
  delta=np.array(window_loss[r['arm']])-np.array(window_loss['native'])
  r['better_windows_vs_native']=int((delta<0).sum())
save('CHECKS.json',checks);save('TIMING.json',dict(sequence_length=512,threads=1,dtype='FP32',batch_size=1,
  affinity=sorted(os.sched_getaffinity(0)),timings=timings,note='All training workers exited; two warmups/five repeats; no GPU, no optimizer in microbenchmark'))
save('CHECKPOINT_MANIFEST.json',checkpoints)
save('SUMMARY.json',dict(run_id=p['run_id'],status='COMPLETE_EXPLORATORY',protocol_sha256=sha(ROOT/'PROTOCOL.json'),
 scientific_protocol_sha256=sha(ROOT/'SCIENTIFIC_PROTOCOL.json'),ranking=sorted(rows,key=lambda r:r['ce']),
 updates_per_arm=2000,training_targets_per_arm=1024000,seed=p['seed'],final_targets=16384,
 decision_rule=p['decision_rule'],total_training_cpu_seconds=sum(r['train_cpu_seconds'] for r in rows),
 scope='Single fresh seed with fivefold budget; reused validation; not a same-budget replication or scaling result'))
save('TERMINAL.json',dict(status='COMPLETED',arms=p['arms'],complete_arms=3,updates_per_arm=2000,
 all_checkpoints_replayed=True,all_sequential_parity_checks_passed=True,official_test_accessed=False,
 completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
save('CLEANUP.json',dict(owned_training_workers='All launch.py child processes exited0 before analysis',
 new_cloud_resources_created=False,gpu_used=False,followup_running=False,automations_created=False,
 note='No cloud provisioning was attempted; no global account/resource inventory claim'))
