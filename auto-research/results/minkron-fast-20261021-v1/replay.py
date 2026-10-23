"""Rebuild omitted weights using the recorded steps and LRs, not a new time budget."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from common import ROOT, ARMS, SEEDS, make_model
from model import optimizer
from train import data, Windows

def tensor_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def main(args):
    out=Path(args.output)
    assert not out.exists(),'Refuse to replace an existing artifact'
    profile=json.loads((ROOT/'PROFILE.json').read_text())
    torch.set_num_threads(profile['budget']['threads'])
    model=make_model(args.arm,args.seed);opt=optimizer(model,.003)
    tokens=data()['train'];sampler=Windows(tokens,args.seed)
    with gzip.open(ROOT/'ALL_STEPS.csv.gz','rt') as f:
        rows=[r for r in csv.DictReader(f) if r['arm']==args.arm and int(r['seed'])==args.seed and
              (r['stage']=='screen' or args.stage=='extension')]
    rows.sort(key=lambda r:int(r['total_steps']))
    assert rows and len(rows)==int(rows[-1]['total_steps'])
    for row in rows:
        x,y,reset=sampler.next()
        for group in opt.param_groups:group['lr']=float(row['lr'])
        opt.zero_grad(set_to_none=True)
        logits,_=model(x,reset=reset)
        loss=F.cross_entropy(logits.reshape(-1,4096),y.reshape(-1))
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
    digest=tensor_digest(model.state_dict())
    manifests=json.loads((ROOT/'CHECKPOINTS.json').read_text())
    original=next(r for r in manifests if r['path']==f'checkpoints/{args.stage}_{args.arm}_{args.seed}.pt')
    matches=digest==original['model_tensor_sha256']
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),arm=args.arm,seed=args.seed,
        stage=args.stage,steps=len(rows),model_tensor_sha256=digest,matches_original_model_tensors=matches),out)
    print(json.dumps(dict(output=str(out),model_tensor_sha256=digest,expected=original['model_tensor_sha256'],
        matches_original_model_tensors=matches,warning='Exact matching also depends on CPU/PyTorch numerics. Archive bytes need not match.')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True);p.add_argument('--stage',choices=['screen','extension'],required=True)
    p.add_argument('--output',required=True);main(p.parse_args())
