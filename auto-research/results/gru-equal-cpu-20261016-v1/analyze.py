"""Replay endpoints, verify the budget and publish both schedule outcomes."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from run import ROOT,PARENT,build,evaluate,save,sha

def curve(path):
    return json.loads(path.read_text()) if path.exists() else json.loads(gzip.decompress(path.with_suffix('.json.gz').read_bytes()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    os.sched_setaffinity(0,{2})
    p=json.loads((ROOT/'PROTOCOL.json').read_text());fresh_p=json.loads((ROOT/'FRESH_PROTOCOL.json').read_text())
    for name,digest in fresh_p['source_hashes'].items():assert sha(ROOT/name)==digest,name
    assert sha(ROOT/'OFFSETS.json')==p['offsets_sha256']
    for name,digest in p['data_hashes'].items():assert sha(args.data_dir/name)==digest,name
    offsets=json.loads((ROOT/'OFFSETS.json').read_text())
    val=np.memmap(args.data_dir/'validation.bin',dtype=np.uint16,mode='r')
    native=json.loads((PARENT/'execution/native/RESULT.json').read_text())
    prior=json.loads((PARENT/'execution/mingru_linear/RESULT.json').read_text())
    checks={};rows=[]
    for key,label,result in [('native','Ordinary GRU',native),('linear_2000','Linear minGRU: 2000 updates',prior)]:
        rows.append(dict(arm=key,label=label,updates=result['updates'],parameters=result['parameters'],
            training_targets=result['training_targets'],cpu_seconds=result['train_cpu_seconds'],ce=result['final']['ce'],
            perplexity=result['final']['perplexity'],ce_delta_vs_native=result['final']['ce']-native['final']['ce'],reused=True))
    cp_manifest={}
    for folder,protocol,label in [('execution','PROTOCOL.json','Linear minGRU: low-LR extension'),
                                  ('fresh_execution','FRESH_PROTOCOL.json','Linear minGRU: fresh full-budget decay')]:
        out=ROOT/folder;r=json.loads((out/'RESULT.json').read_text());cp=out/'checkpoint.pt'
        assert sha(cp)==r['checkpoint_sha256']
        assert r['protocol_sha256']==sha(ROOT/protocol)
        assert r['budget_within_half_second']
        assert 0<=native['train_cpu_seconds']-r['train_cpu_seconds']<=.5
        old=torch.load(cp,map_location='cpu',weights_only=False)
        assert old['protocol_sha256']==sha(ROOT/protocol)
        state=old['state'];c=curve(out/'curves.json')
        assert c==state
        assert len(c['train'])==r['updates'] and c['training_targets']==512*r['updates']
        assert all(t['step']==i+1 and math.isfinite(t['ce']) for i,t in enumerate(c['train']))
        if folder=='execution':
            assert c['train'][:2000]==curve(PARENT/'execution/mingru_linear/curves.json')['train']
            assert all(t['lr']==.0002 for t in c['train'][2000:])
        else:
            assert c['train'][24]['lr']==.002
            assert all(a['lr']>=b['lr'] for a,b in zip(c['train'][25:-1],c['train'][26:]))
            assert abs(c['train'][-1]['lr']-.0002)<1e-7
        m=build('mingru_linear',p['config'],p['seed']);m.load_state_dict(old['model'])
        replay=evaluate(m,val,offsets['final_starts'])
        error=max(abs(a-b) for a,b in zip(replay['window_ce'],r['final']['window_ce']))
        assert error<1e-6
        start=offsets['final_starts'][0];xy=torch.from_numpy(np.array(val[start:start+513],dtype=np.int64))[None]
        outputs=[];grads=[]
        for backend in ('parallel','tokenwise'):
            m.zero_grad(set_to_none=True);z=m(xy[:,:-1],backend=backend).logits
            outputs.append(z.detach());F.cross_entropy(z.flatten(0,1),xy[:,1:].flatten()).backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
            grads.append(torch.cat([v.grad.flatten() for v in m.parameters()]))
        logit_error=float((outputs[0]-outputs[1]).abs().max())
        grad_error=float((grads[0]-grads[1]).norm()/grads[1].norm())
        assert logit_error<5e-5 and grad_error<2e-4
        checks[folder]=dict(replay_max_ce_error=error,logit_error=logit_error,gradient_relative_l2=grad_error,
                            budget_under_native_cpu_seconds=r['unused_budget_cpu_seconds'],trace_and_schedule_passed=True)
        delta=np.array(r['final']['window_ce'])-np.array(native['final']['window_ce'])
        rows.append(dict(arm=r['arm'],label=label,updates=r['updates'],parameters=r['parameters'],
            training_targets=r['training_targets'],cpu_seconds=r['train_cpu_seconds'],ce=r['final']['ce'],
            perplexity=r['final']['perplexity'],ce_delta_vs_native=float(delta.mean()),
            perplexity_change_percent=100*(math.exp(float(delta.mean()))-1),better_windows=int((delta<0).sum()),
            budget_unused_seconds=r['unused_budget_cpu_seconds'],reused=False))
        cp_manifest[folder]=dict(path=f'{folder}/checkpoint.pt',sha256=sha(cp),bytes=cp.stat().st_size,updates=r['updates'])
    short_linear=json.loads((ROOT/'references/short_linear_RESULT.json').read_text())
    short_nano=json.loads((ROOT/'references/short_nanogpt_RESULT.json').read_text())
    shared_time=prior['train_cpu_seconds']
    old_native_monitor=native['monitor']
    interp=float(np.interp(shared_time,[x['train_cpu_seconds'] for x in old_native_monitor],[x['validation_ce'] for x in old_native_monitor]))
    summary=dict(run_id=ROOT.name,status='COMPLETE_EXPLORATORY',rows=rows,seed=p['seed'],final_targets=16384,
        target_cpu_seconds=p['target_total_train_cpu_seconds'],
        early_equal_time=dict(cpu_seconds=shared_time,linear_monitor_ce=prior['monitor'][-1]['validation_ce'],
            native_interpolated_monitor_ce=interp,note='Linear interpolation between saved native800/1000 checkpoints; descriptive, not an evaluated checkpoint'),
        short_nanogpt_context=dict(seed=101601,updates=400,targets=204800,nanogpt_ce=short_nano['final']['ce'],
            linear_ce=short_linear['final']['ce'],nanogpt_cpu_seconds=short_nano['train_cpu_seconds'],
            linear_cpu_seconds=short_linear['train_cpu_seconds'],throughput_ratio=short_nano['train_cpu_seconds']/short_linear['train_cpu_seconds'],
            scope='Earlier common-recipe1M CPU pilot, not optimized nanoGPT speedrun'),
        new_training_cpu_seconds=rows[2]['cpu_seconds']-prior['train_cpu_seconds']+rows[3]['cpu_seconds'],
        primary_comparison='Fresh full-budget linear vs reused native; continuation shown as a schedule diagnostic',
        caution='Time budget is measured one-thread training CPU, not theoretical FLOPs or end-to-end GPU speedrun time')
    save(ROOT/'SUMMARY.json',summary);save(ROOT/'CHECKS.json',checks);save(ROOT/'CHECKPOINT_MANIFEST.json',cp_manifest)
    for folder in ('execution','fresh_execution'):
        raw=(ROOT/folder/'curves.json').read_bytes()
        (ROOT/folder/'curves.json.gz').write_bytes(gzip.compress(raw,mtime=0))
    save(ROOT/'TERMINAL.json',dict(status='COMPLETED',checkpoints_replayed=True,parity_passed=True,
        official_test_accessed=False,new_paid_resources=False,training_running=False,
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    registry_path=ROOT.parents[2]/'auto-research/RUN_REGISTRY.json'
    registry=json.loads(registry_path.read_text())
    entry=next(x for x in registry['runs'] if x['run_id']==ROOT.name)
    entry.update(status='COMPLETED_EXPLORATORY',fresh_protocol=str((ROOT/'FRESH_PROTOCOL.json').relative_to(ROOT.parents[2])),
                 fresh_protocol_sha256=sha(ROOT/'FRESH_PROTOCOL.json'),native_ce=rows[0]['ce'],
                 continued_linear_ce=rows[2]['ce'],fresh_linear_ce=rows[3]['ce'],new_training_cpu_seconds=summary['new_training_cpu_seconds'])
    save(registry_path,registry)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
