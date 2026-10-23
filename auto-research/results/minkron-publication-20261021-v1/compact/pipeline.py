"""Bounded experiment stages; pollable progress, no model selection on test data."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
def call(cmd,logname):
    print('RUN',cmd,flush=True)
    with open(ROOT/'logs'/logname,'w') as log:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
while not (ROOT/'pilot_completion.json').exists():time.sleep(2)
call(['python','freeze_protocol.py'],'protocol_selection.log')
call(['python','run_queue.py','baseline'],'baseline_queue.log')
call(['python','run_queue.py','compact'],'compact_queue.log')
p=json.loads((ROOT/'protocol.json').read_text());jobs=[(a,s) for s in p['seeds'] for a in p['arms']]
running=[]
while jobs or running:
    while jobs and len(running)<6:
        arm,seed=jobs.pop(0);log=open(ROOT/'logs'/f'confirm_{arm}_{seed}.log','w')
        proc=subprocess.Popen(['python','confirm.py','--arm',arm,'--seed',str(seed)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        running.append((arm,seed,proc,log))
    for job in list(running):
        arm,seed,proc,log=job;code=proc.poll()
        if code is not None:
            running.remove(job);log.close();print('CONFIRMED',arm,seed,code,flush=True)
            if code:raise RuntimeError((arm,seed,code))
    time.sleep(1)
call(['python','resources.py','--timing'],'timing.log')
for arm in p['arms']:
    for kind in ['train','inference']:
        call(['python','resources.py','--arm',arm,'--kind',kind],f'memory_{arm}_{kind}.log')
(ROOT/'pipeline_complete.json').write_text(json.dumps({'status':'completed','timestamp':time.time()},indent=2))
print('PIPELINE COMPLETE',flush=True)
