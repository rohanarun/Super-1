import argparse,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('stage',choices=['pilot','baseline','compact']);a=p.parse_args()
arms=['mingru','kron8','kron4h4','kron2h4'];jobs=[]
if a.stage=='pilot':
    for lr in [.001,.003]:
        for arm in arms:jobs.append((arm,21901,lr,True))
    workers=4
else:
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for seed in protocol['seeds']:
        for arm in (arms[:1] if a.stage=='baseline' else arms[1:]):
            jobs.append((arm,seed,protocol['learning_rates'][arm],False))
    workers=3 if a.stage=='baseline' else 6
running=[];finished=[];start=time.time();review=0
while jobs or running:
    while jobs and len(running)<workers:
        arm,seed,lr,pilot=jobs.pop(0);tag=f'{a.stage}_{arm}_{seed}_{lr:g}'
        log=open(ROOT/'logs'/f'{tag}.log','w')
        cmd=['python','experiment.py','--arm',arm,'--seed',str(seed),'--lr',str(lr)]+(['--pilot'] if pilot else [])
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        running.append((tag,proc,log));print('START',tag,proc.pid,flush=True)
    for job in list(running):
        tag,proc,log=job;code=proc.poll()
        if code is not None:
            running.remove(job);log.close();finished.append({'run':tag,'code':code})
            print('DONE',tag,code,flush=True)
            if code:raise RuntimeError(f'{tag} failed; inspect its log')
    elapsed=time.time()-start
    if int(elapsed//1800)>review:
        review=int(elapsed//1800)
        (ROOT/f'{a.stage}_review_{review*30}min.json').write_text(json.dumps({'elapsed':elapsed,'running':[x[0] for x in running],'finished':finished},indent=2))
    time.sleep(1)
(ROOT/f'{a.stage}_completion.json').write_text(json.dumps({'seconds':time.time()-start,'runs':finished},indent=2))
