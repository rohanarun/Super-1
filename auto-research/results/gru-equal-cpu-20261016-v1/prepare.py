"""Freeze the bounded continuation before observing any new training outcome."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'gru-extended-20261016-v1'
REPO = ROOT.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
save = lambda p,x: p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (ROOT/'PROTOCOL.json').exists(), 'Never re-freeze a run'
parent = json.loads((PARENT/'PROTOCOL.json').read_text())
offsets = json.loads((PARENT/'OFFSETS.json').read_text())
starts = np.random.default_rng(parent['data_seed']).integers(0, parent['data']['train_tokens']-512, size=(8000,1)).tolist()
assert starts[:2000] == offsets['train_starts']
offsets['train_starts'] = starts
save(ROOT/'OFFSETS.json', offsets)
native = json.loads((PARENT/'execution/native/RESULT.json').read_text())
linear = json.loads((PARENT/'execution/mingru_linear/RESULT.json').read_text())
source = '../gru-extended-20261016-v1/execution/mingru_linear/checkpoint.pt'
inputs = [source, '../gru-extended-20261016-v1/PROTOCOL.json',
          '../gru-extended-20261016-v1/OFFSETS.json',
          '../gru-extended-20261016-v1/execution/native/RESULT.json',
          '../gru-extended-20261016-v1/execution/mingru_linear/RESULT.json']
p = dict(run_id=ROOT.name, hypothesis='Spending its speed advantage on more updates can let linear minGRU beat native GRU at matched training CPU time',
    scope='Single-seed exploratory budget-extension diagnostic; not a from-scratch schedule-optimized benchmark',
    authorization='User requests equal-compute comparison; continuing prior authorization for bounded CPU experiments and fork updates',
    parent_run_id=PARENT.name, parent_commit='f68ba902387ac53d4765f652259dc20b1a7ac49e',
    parent_protocol_sha256=sha(PARENT/'PROTOCOL.json'), source_checkpoint=source,
    input_hashes={name:sha(ROOT/name) for name in inputs}, config=parent['configs']['mingru_linear'],
    parameters=linear['parameters'], seed=parent['seed'], data_seed=parent['data_seed'],
    data_hashes={f'{n}.bin':parent['data']['split_metadata']['splits'][n]['sha256'] for n in ('train','validation')},
    tokenizer_sha256=parent['data']['split_metadata']['tokenizer_sha256'], offsets_sha256=sha(ROOT/'OFFSETS.json'),
    initial_updates=2000, initial_train_cpu_seconds=linear['train_cpu_seconds'],
    target_total_train_cpu_seconds=native['train_cpu_seconds'], stop_margin_cpu_seconds=.2,
    stopping_rule='Stop when remaining CPU budget is at most max(0.2 seconds, twice largest of last20 update costs). Report actual time and any overshoot; target within0.5s. No validation-dependent stopping.',
    maximum_total_updates=8000, continuation_lr=.0002,
    optimizer='Resume exact AdamW moments; same decay/exclusions/betas/clip; hold final LR0.0002 constant without restart',
    inherited_schedule='First2000 updates use original warmup25 and cosine0.002 to0.0002; native remains its frozen2000-step endpoint',
    data_order='Continue same RNG window stream; first2000 offsets identical to parent;512-target reset episodes',
    final_metric='CE in nats per2048-BPE token on same32 final-region validation windows (16384targets)',
    monitoring='Same8 monitor windows, every250 additional steps and endpoint; monitor and final CE remain separate',
    decision='Descriptive endpoint CE vs frozen native; candidate if lower with budget within0.5CPU seconds. No significance claim.',
    compute_definition='Sum of one-thread process CPU seconds inside training updates, including batch construction, forward/backward, optimizer and constraints; excludes validation, checkpoint IO and startup. Not FLOPs or end-to-end speedrun time.',
    limitations=['Unequal training tokens are intentional at equal time', 'Continuation LR differs from a fresh full-budget cosine schedule',
                'One seed and repeatedly reused validation; no GPU/scaling/general-speedrun claim',
                'Inherited parent ranking text mentions400 steps in error; parent code/results actually used2000; original evidence left immutable'],
    resource_limits=dict(threads=1, device='cpu', process_cpu_seconds=360, wall_seconds=600, new_paid_resources=False),
    official_test_access_permitted=False, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
save(ROOT/'PROTOCOL.json',p)
sources=[ROOT/'run.py',ROOT/'prepare.py',PARENT/'models.py']+list((PARENT/'vendor').rglob('*.py'))
import os
save(ROOT/'SOURCE_MANIFEST.json',{os.path.relpath(f,ROOT):sha(f) for f in sources})
save(ROOT/'REGISTRATION.json',dict(run_id=ROOT.name,status='FROZEN_BEFORE_CONTINUATION',protocol_sha256=sha(ROOT/'PROTOCOL.json'),created_utc=p['created_utc']))
registry_path=REPO/'auto-research/RUN_REGISTRY.json'
registry=json.loads(registry_path.read_text())
assert all(r['run_id']!=ROOT.name for r in registry['runs'])
registry['runs'].append(dict(run_id=ROOT.name,status='FROZEN_BEFORE_CONTINUATION',kind='Independent equal-CPU diagnostic',
    protocol=str((ROOT/'PROTOCOL.json').relative_to(REPO)),protocol_sha256=sha(ROOT/'PROTOCOL.json'),
    source_manifest=str((ROOT/'SOURCE_MANIFEST.json').relative_to(REPO)),metrics=str((ROOT/'SUMMARY.json').relative_to(REPO)),
    report=f'auto-research/reports/{ROOT.name}.md',terminal=str((ROOT/'TERMINAL.json').relative_to(REPO))))
save(registry_path,registry)
print(json.dumps(dict(run_id=ROOT.name,target_cpu_seconds=p['target_total_train_cpu_seconds'],
                      additional_cpu_seconds=p['target_total_train_cpu_seconds']-p['initial_train_cpu_seconds'],
                      protocol_sha256=sha(ROOT/'PROTOCOL.json'))))
