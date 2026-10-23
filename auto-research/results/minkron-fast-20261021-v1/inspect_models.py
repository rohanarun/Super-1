"""Fixed prompt samples and batch-one deployment measurements; no selection."""
import json
import time
import numpy as np
import torch
from common import ROOT, make_model, write_json
from model import tensor_bytes
from prepare_data import tokenizer

PROMPTS=['The little dog', 'The capital of France is', 'To make a cup of tea,',
    'Water freezes at', 'In a quiet village,', 'Once upon a time,',
    'The computer program', 'Question: What is 2 + 2?\nAnswer:']

@torch.no_grad()
def main():
    metrics=json.loads((ROOT/'METRICS.json').read_text())
    torch.set_num_threads(metrics['budget']['threads'])
    enc=tokenizer();samples=[];deployment=[]
    probe=torch.arange(256).remainder(4095).reshape(1,-1)
    for row in metrics['extensions']:
        model=make_model(row['arm'],row['seed']);model.eval()
        checkpoint=torch.load(ROOT/row['checkpoint']['path'],weights_only=False,map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        for prompt_number,prompt in enumerate(PROMPTS):
            x=torch.tensor([enc.encode_ordinary(prompt)]);logits,state=model(x)
            generated=[];generator=torch.Generator().manual_seed(99100+prompt_number)
            for _ in range(96):
                scores=logits[0,-1]/.8
                scores=scores.masked_fill(scores<torch.topk(scores,40).values[-1],-float('inf'))
                token=int(torch.multinomial(scores.softmax(-1),1,generator=generator))
                if token==4095:break
                generated.append(token);logits,state=model(torch.tensor([[token]]),state)
            samples.append(dict(arm=row['arm'],seed=row['seed'],prompt=prompt,
                completion=enc.decode(generated),generated_tokens=len(generated),
                temperature=.8,top_k=40,sampling_seed=99100+prompt_number,
                checkpoint_sha256=row['checkpoint']['sha256']))
        prefill=[]
        for i in range(7):
            t=time.perf_counter();_,state=model(probe[:,:128]);elapsed=time.perf_counter()-t
            if i>=2:prefill.append(elapsed)
        state128=tensor_bytes(state)
        times=[]
        for i in range(64):
            t=time.perf_counter();_,state=model(probe[:,128+i:129+i],state);times.append(time.perf_counter()-t)
        _,state256=model(probe)
        deployment.append(dict(arm=row['arm'],batch=1,weight_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            persistent_state_logical_bytes_128=state128[0],persistent_state_storage_bytes_128=state128[1],
            persistent_state_logical_bytes_256=tensor_bytes(state256)[0],
            prefill128_median_ms=float(np.median(prefill)*1000),prefill128_p95_ms=float(np.percentile(prefill,95)*1000),
            decode_median_ms=float(np.median(times)*1000),decode_p95_ms=float(np.percentile(times,95)*1000),
            decode_tokens_per_second=64/sum(times),threads=metrics['budget']['threads'],
            precision='FP32 weights and projections; minKron recurrence/state FP64; convolution cache FP32',
            activation_peak='not separately measured',isolated_deployment_process_peak='not measured'))
    write_json(ROOT/'SAMPLES.json',dict(policy='All eight prompts retained for all three extended models; no selection or retries.',samples=samples))
    write_json(ROOT/'DEPLOYMENT.json',deployment)
    print(json.dumps(deployment,indent=2))

if __name__=='__main__':main()
