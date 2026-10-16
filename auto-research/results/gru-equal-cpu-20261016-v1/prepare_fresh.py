"""Register the fresh-budget arm separately without editing the continuation protocol."""
from pathlib import Path
import json
import time
from run import ROOT, sha, save

assert not (ROOT/'FRESH_PROTOCOL.json').exists()
p=json.loads((ROOT/'PROTOCOL.json').read_text())
source=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
for name in ('fresh.py','prepare_fresh.py'):source[name]=sha(ROOT/name)
q={k:p[k] for k in ('run_id','parent_run_id','parent_commit','config','parameters','seed','data_seed',
                    'data_hashes','tokenizer_sha256','offsets_sha256','target_total_train_cpu_seconds')}
q.update(arm='mingru_linear_fresh_clock',source_hashes=source,
    motivation='A fresh full-budget schedule avoids penalizing the continuation for reaching the LR floor early',
    initialization='Build from scratch at original seed101602; no checkpoint/model/optimizer continuation',
    schedule='Warmup25updates to0.002; cosine by cumulative measured training CPU time after warmup to0.0002 at256.559082992CPU seconds',
    stopping_rule=p['stopping_rule'],maximum_total_updates=8000,
    optimizer='Same original AdamW betas0.9/0.95 eps1e-8 decay0.1 exclusions and clip1',
    final_metric=p['final_metric'],comparison='Frozen native2000-step endpoint (full-horizon step cosine); fresh linear uses full-horizon CPU-clock cosine',
    caveats=['Schedule parametrization changes to clock time; no per-arm LR search', 'Native measured in prior run; host load and clock speed may differ',
             'Single seed and repeatedly used validation; no FLOP match or GPU performance claim', 'Show continuation and fresh outcomes; do not hide a losing arm'],
    official_test_access_permitted=False,resource_limits=dict(process_cpu_seconds=400,wall_seconds=700,threads=1,new_paid_resources=False),
    created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
save(ROOT/'FRESH_PROTOCOL.json',q)
save(ROOT/'FRESH_REGISTRATION.json',dict(status='FROZEN_BEFORE_FRESH_TRAINING',protocol_sha256=sha(ROOT/'FRESH_PROTOCOL.json'),
    created_utc=q['created_utc'],continuation_complete_at_freeze=(ROOT/'execution/RESULT.json').exists()))
print('FRESH FROZEN',sha(ROOT/'FRESH_PROTOCOL.json'))
