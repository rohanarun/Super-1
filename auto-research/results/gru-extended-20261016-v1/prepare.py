"""Freeze the bounded longer-exposure comparison before launch."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,importlib.util,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from models import build,optimizer,HierarchicalDistributionStudent
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(ROOT/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True)
parser.add_argument('--previous-study',type=Path,required=True);args=parser.parse_args()
data=args.data_dir.resolve();old=args.previous_study.resolve()
assert not (ROOT/'execution').exists(),'Never re-freeze an active/completed run'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
p=json.loads((old/'PROTOCOL.json').read_text());old_offsets=json.loads((old/'OFFSETS.json').read_text())
configs={a:dict(vocab_size=2048,width=w,max_context=512,segment_length=32,document_segments=8)
         for a,w in [('native',147),('mingru_linear',160),('paragru',153)]}
for name in ('train','validation'):
 assert sha(data/f'{name}.bin')==p['data']['split_metadata']['splits'][name]['sha256']
assert sha(data/'tokenizer.json')==p['data']['split_metadata']['tokenizer_sha256']
train=np.memmap(data/'train.bin',dtype=np.uint16,mode='r')
rng=np.random.default_rng(101611)
starts=rng.integers(0,len(train)-512,size=(2000,1)).tolist()
assert starts[:400]==old_offsets['train_starts']
offsets=dict(train_starts=starts,monitor_starts=old_offsets['monitor_starts'],final_starts=old_offsets['final_starts'])
save('OFFSETS.json',offsets)
counts={};checks={};states={}
s=starts[0][0];xy=torch.from_numpy(np.array(train[s:s+513],dtype=np.int64))[None]
for arm,cfg in configs.items():
 m=build(arm,cfg,101602);opt=optimizer(m,.002)
 counts[arm]=sum(v.numel() for v in m.parameters());assert abs(counts[arm]-1e6)<5000
 states[arm]=m.stored_state_elements();grad=[];outputs=[]
 for backend in [m.backend,'tokenwise']:
  m.zero_grad(set_to_none=True);out=m(xy[:,:-1],backend=backend)
  outputs.append(out.logits.detach())
  F.cross_entropy(out.logits.flatten(0,1),xy[:,1:].flatten()).backward()
  assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
  grad.append(torch.cat([v.grad.flatten() for v in m.parameters()]))
 err=float((outputs[0]-outputs[1]).abs().max());rel=float((grad[0]-grad[1]).norm()/grad[1].norm())
 assert err<2e-5 and rel<2e-4,(arm,err,rel)
 torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();m.constrain()
 with torch.no_grad():
  full=m(xy[:,:-1]);first=m(xy[:,:257]);second=m(xy[:,257:-1],first.final_state)
  prefix=float((full.logits[:,:257]-first.logits).abs().max())
  carry=float((full.logits[:,257:]-second.logits).abs().max())
  assert max(prefix,carry)<2e-5
  last=float((m.last_logits(xy[:,:64])-m(xy[:,:64]).logits[:,-1:]).abs().max());assert last==0
 checks[arm]=dict(output_error=err,gradient_relative_l2=rel,prefix_error=prefix,carry_error=carry,
                   parameters=counts[arm],state_scalars=states[arm],last_logits_error=last)
 print(arm,checks[arm],flush=True)
save('PREFLIGHT.json',dict(models=checks,data_hashes_verified=True,preflight_weights_discarded=True))
approval='User 2026-10-16 15:02 America/Denver: excellent - go ahead with the next experiment, and push the updates and results when done, proceed with linear mingru and paragru'
p.update(study='gru-extended-20261016-v1',run_id='gru-extended-20261016-v1',seed=101602,
 scope='Independent exploratory text screen in user fork; separate from official Super-1 studies',
 arms=list(configs),configs=configs,parameters=counts,state_scalars=states,updates=2000,
 training_targets_per_arm=1024000,monitor_steps=[0,25,50,100,200,400,600,800,1000,1250,1500,1750,2000],
 offsets_sha256=sha(ROOT/'OFFSETS.json'),reused_controls=[],control_study='all three trained fresh',
 hypothesis='Linear minGRU or ParaGRU retains near-native held-out CE after fivefold exposure with lower CPU cost',
 primary_comparator='native',minimum_useful_effect=dict(max_ce_regression=.03,min_cpu_speedup=1.25),
 decision_rule='Exploratory candidate if CE<=native+0.03 and training CPU speedup>=1.25, all checks/budgets pass; otherwise no advance. Not confirmation.',
 initialization='Same per-architecture initialization as prior screen, fresh seed101602; training from scratch, no checkpoint continuation',
 comparability=['Three fresh arms matched within0.5% of1M parameters, ordered windows, optimizer and final checkpoint budget',
 '2000 updates and cosine horizon vs400 before; fresh seed101602 vs101601; cross-study changes are not a single-factor training-budget ablation',
 'First400 data windows match earlier study; next1600 are from the same predeclared RNG stream',
 'Same monitor/final validation regions reused; not a pristine test or independent seed replication at the same budget',
 'Reset512 episodes and retained32-token/8-segment slow feedback; no long-context learning claim',
 'Positive-candidate minGRU omitted based on preceding pilot; no new per-arm LR search',
 'Native dense nn.GRU recurrence; minGRU signed affine scan; Para diagonal nonlinear recurrence with Newton and implicit adjoint',
 'Architectures retain their unequal state/compute; historical rotating reference is not in this longer-budget ranking'])
p['optimizer']['exclusions']='Tied token table, ndim<2, and Para recurrent diagonal vectors stored together in A'
p['resource_limits'].update(max_workers=3,cpu_seconds_per_arm=900,wall_seconds_per_arm=1800)
p['benchmark']=dict(repeats=5,warmups=2,lengths=[512],threads=1,
 measure='After all training exits; full-model forward+loss+backward and no-grad forward',
 backends={'native':['batched'],'mingru_linear':['sequential','parallel'],'paragru':['sequential','parallel']})
p['authorization']=dict(approval=approval,scope='Bounded existing-workspace CPU exploration and push to own fork; existing hosted-workspace exception retained',
 resources='Agent selected 3 single-thread CPU arms; expected ~600 CPU seconds /5min wall; hard900 CPU seconds perarm, no new paid resources',
 fork='JRowe47/Super-1',branch='research/gru-extended-20261016-v1',upstream_pr='Not requested in this turn; no automatic upstream PR/merge')
p['created_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
save('PROTOCOL.json',p)
source=[ROOT/n for n in ['models.py','run.py','launch.py','prepare.py']]+list((ROOT/'vendor').rglob('*.py'))
save('SOURCE_MANIFEST.json',{str(f.relative_to(ROOT)):sha(f) for f in source})
# Bind the repository template without imposing the unrelated historical official-study schedule.
q=json.loads((REPO/'auto-research/PROTOCOL_TEMPLATE.json').read_text())
q.update(status='FROZEN_AUTHORIZED_EXPLORATORY',experiment_id='independent-gru-exposure',run_id=p['run_id'],
 parent_run_ids=['gru_parallel_v1','gru_parallel_v1_linear_followup'],hypothesis=p['hypothesis'],primary_comparator='native',
 primary_metric='Final CE in nats per2048-BPE token on16384 reused validation targets',minimum_useful_effect=p['minimum_useful_effect'],
 success_rule=p['decision_rule'],futility_rule='Abort nonfinite loss/gradients, failed convergence, hash mismatch or resource cap; preserve every failure')
q['authorization']=dict(approved=True,user_approval_reference=approval,approved_resources=p['authorization']['resources'],max_cost='No new paid resources',
 max_walltime='1800seconds perarm; expected5min total with3workers',local_cpu_budget='900seconds perarm hard cap; expected600seconds aggregate')
q['design']=dict(original_source_hashes={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'vendor/representation_experiments').glob('*.py')},
 mechanism_equations='models.py and SOURCES.md',arms=p['arms'],control_rationale='Unchanged native recurrence with batched processing; parameter matched',
 seeds=[101602],seed_selection_method='Next integer after prior101601; chosen before outcomes',learning_rate_trials_per_arm=1,
 parameter_counts=counts,matching_track='parameters and tokens',compute_accounting='Measured CPU/wall training time, not matched FLOPs')
meta=p['data']['split_metadata'];q['data']=dict(dataset_versions=p['data'],tokenizer_sha256=meta['tokenizer_sha256'],train_sha256=meta['splits']['train']['sha256'],
 selection_sha256=meta['splits']['validation']['sha256'],confirmation_sha256='NOT_APPLICABLE_no_independent_confirmation',
 split_boundaries='OFFSETS.json; monitoring first600k validation tokens, endpoint after650k',research_wide_region_reuse_audit='Same validation arrays/windows as prior text pilots; already repeatedly viewed',
 teacher_and_objective='Full-vocabulary next-token CE, no teacher',official_test_access_permitted=False)
q['training']=dict(updates=2000,target_exposures_per_trial=1024000,tokens_per_update=512,microbatch=1,gradient_accumulation=1,
 bptt_length=512,accessible_history='At most512 tokens per reset training/evaluation episode',state_carry_and_detach='Full gradient within episode, zero state each episode',
 document_reset_policy='EOS tokens remain in stream; no extra reset at EOS inside episode, same as prior reset-window comparator',optimizer_schedule=p['optimizer'],
 precision='FP32',checkpoint_selection_rule='Final2000 only, no best-checkpoint selection')
q['execution']=dict(hardware='Existing hosted CPU workspace, one thread perarm; no GPU',runtime_versions={'torch':torch.__version__,'numpy':np.__version__},
 resolved_writable_output=str(ROOT),source_ownership_and_concurrent_writers='Unique new run directory, one worker perarm; prior runs finished',
 fidelity_tolerances={'logits':2e-5,'gradient_relative_l2':2e-4,'trained_logits':5e-5,'checkpoint_ce':1e-6},canary_matrix='PREFLIGHT.json; full512, split257 crossing segment/document boundaries, all gradients',
 cost_estimate=p['authorization']['resources'])
q['evaluation']=dict(primary_endpoint=q['primary_metric'],confirmation_frozen_before_scoring=False,canonical_precision='FP32',
 duplicate_policy='Replay final-window CE and sequential/parallel logit/gradient checks at saved weights',uncertainty_method='One-seed exploratory descriptive gaps; no significance claim',
 multiple_comparison_policy='Show both candidates and native; no hidden arms or tuning',no_test_tuning=True)
q['deployment']=dict(weight_dtype='FP32',kv_or_state_dtype='FP32',quantization_metadata='none',batching='B1',prefill_decode_protocol=p['benchmark'],
 warmup_and_synchronization='2 warmups,5 repeats on single CPU thread; all workers stopped',memory_components='Weight bytes, persistent tensor storage, peak training process RSS; other deployment components unmeasured',
 latency_statistics='Median and p95 of small benchmark, not production tail estimate')
q['evidence']=dict(protocol_sha256='Stored in REGISTRATION.json to avoid self-reference',source_manifest='SOURCE_MANIFEST.json',checkpoint_manifest='CHECKPOINT_MANIFEST.json',
 progress_path='execution/<arm>/curves.json',terminal_receipt_path='TERMINAL.json',preservation_policy='Source/results/figures in fork; full checkpoints and token arrays in companion artifact outside Git',cleanup_receipt='CLEANUP.json')
q['decision']=dict(outcome='PENDING_EXECUTION',reason='Frozen before training',next_experiment='Decide from observed outcomes; nothing queued')
save('SCIENTIFIC_PROTOCOL.json',q)
registration=dict(run_id=p['run_id'],status='FROZEN_BEFORE_TRAINING',approval=approval,protocol_sha256=sha(ROOT/'PROTOCOL.json'),
 scientific_protocol_sha256=sha(ROOT/'SCIENTIFIC_PROTOCOL.json'),created_utc=p['created_utc'])
save('REGISTRATION.json',registration)
registry=json.loads((REPO/'auto-research/RUN_REGISTRY.json').read_text())
registry['runs'].append(dict(run_id=p['run_id'],status='FROZEN_BEFORE_TRAINING',kind='Independent explorer text pilot',approval=approval,
 protocol=str((ROOT/'PROTOCOL.json').relative_to(REPO)),protocol_sha256=registration['protocol_sha256'],
 source_manifest=str((ROOT/'SOURCE_MANIFEST.json').relative_to(REPO)),metrics=str((ROOT/'SUMMARY.json').relative_to(REPO)),
 report=f"auto-research/reports/{p['run_id']}.md",terminal=str((ROOT/'TERMINAL.json').relative_to(REPO)),updates_per_arm=2000,arm_seed_rows=3))
registry['independent_fork_authorization']='User-authorized bounded GRU pilot in this fork; historical upstream pause applies to its separate official studies'
(REPO/'auto-research/RUN_REGISTRY.json').write_text(json.dumps(registry,indent=2)+'\n')
print('FROZEN',registration,flush=True)
