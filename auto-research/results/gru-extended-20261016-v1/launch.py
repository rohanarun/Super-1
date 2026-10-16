"""Three independent one-thread workers; checkpoint every 100 steps."""
from pathlib import Path
import argparse, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parent
p=json.loads((ROOT/'PROTOCOL.json').read_text())
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True,type=Path)
args=parser.parse_args();data_dir=args.data_dir.resolve()
cpus=sorted(os.sched_getaffinity(0));workers=[]
for i,arm in enumerate(p['arms']):
 out=ROOT/'execution'/arm;out.mkdir(parents=True,exist_ok=True)
 if (out/'RESULT.json').exists():
  print('ALREADY COMPLETE',arm,flush=True);continue
 log=(out/'worker.log').open('a')
 child=subprocess.Popen([sys.executable,str(ROOT/'run.py'),arm,'--cpu',str(cpus[i%len(cpus)]),'--data-dir',str(data_dir)],stdout=log,stderr=subprocess.STDOUT)
 workers.append((arm,child,log));print('STARTED',arm,child.pid,flush=True)
while workers:
 remaining=[]
 for arm,child,log in workers:
  code=child.poll()
  if code is None:remaining.append((arm,child,log))
  else:log.close();print('EXIT',arm,code,flush=True)
 workers=remaining
 if workers:time.sleep(5)
results={arm:json.loads((ROOT/'execution'/arm/'RESULT.json').read_text()) if (ROOT/'execution'/arm/'RESULT.json').exists() else None for arm in p['arms']}
(ROOT/'execution/ALL_RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(v is not None for v in results.values()), 'At least one arm incomplete; inspect its checkpoint and worker log'
