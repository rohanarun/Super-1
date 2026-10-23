"""Exploratory context-use and seen-token unigram checks; never used for selection."""
import json
import math
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT, make_model, write_json
from train import data, Windows

@torch.no_grad()
def main():
    metrics=json.loads((ROOT/'METRICS.json').read_text())
    torch.set_num_threads(metrics['budget']['threads']);arrays=data();records=[]
    for row in metrics['extensions']:
        sampler=Windows(arrays['train'],row['seed']);counts=np.ones(4096,dtype=np.int64)
        for _ in range(row['total_steps']):
            _,y,_=sampler.next();counts+=np.bincount(y.numpy().reshape(-1),minlength=4096)
        neglog=-np.log(counts/counts.sum())
        model=make_model(row['arm'],row['seed']);model.eval()
        saved=torch.load(ROOT/row['checkpoint']['path'],weights_only=False,map_location='cpu')
        model.load_state_dict(saved['model']);normal=0.;reset_all=0.;unigram=0.;n=4096
        for start in range(0,n,1024):
            v=torch.from_numpy(np.stack([arrays['confirmation'][s:s+129].astype(np.int64) for s in range(start,start+1024,128)]))
            x,y=v[:,:128],v[:,1:];reset=x==4095;reset[:,0]=True
            a,_=model(x,reset=reset);b,_=model(x,reset=torch.ones_like(reset))
            normal+=F.cross_entropy(a.reshape(-1,4096),y.reshape(-1),reduction='sum').item()
            reset_all+=F.cross_entropy(b.reshape(-1,4096),y.reshape(-1),reduction='sum').item()
            unigram+=float(neglog[y.numpy()].sum())
        records.append(dict(arm=row['arm'],checkpoint_sha256=row['checkpoint']['sha256'],tokens=n,
            normal_ce=normal/n,reset_before_every_token_ce=reset_all/n,context_gain_nats=(reset_all-normal)/n,
            unigram_ce=unigram/n,unigram_training_exposures=int(counts.sum()-4096),
            unigram_smoothing='Add one count to each of4096 token types, using only targets this model actually saw.',
            interpretation='Reset-every-token retains the current input token but removes earlier context, including convolution history. Positive context gain is short-window context-use evidence, not long-memory or reasoning ability.'))
    write_json(ROOT/'DIAGNOSTICS.json',dict(status='EXPLORATORY_NOT_SELECTION',records=records))
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
