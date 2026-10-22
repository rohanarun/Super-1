import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import statistics
import time
import torch
from torch.nn import functional as F
from .common import configuration, configure_cpu, make_optimizer, autocast, state_bytes, build


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layout', default='h24')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--optimizer', choices=['foreach', 'fused', 'plain'], default='foreach')
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--core', choices=['stable','factored'], default='stable')
    p.add_argument('--reps', type=int, default=3)
    args = p.parse_args()
    configure_cpu(args.threads)
    start = time.perf_counter()
    cfg = configuration(args.layout, args.chunk, args.core)
    model = build(cfg, 24101)
    opt = make_optimizer(model, .0003, args.optimizer)
    torch.manual_seed(22)
    tokens = torch.randint(0, cfg.vocab_size, (args.batch, args.length+1))
    reset = torch.zeros_like(tokens[:, :-1], dtype=torch.bool)
    reset[:, 0] = True
    samples = []
    for i in range(args.reps+1):
        now = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with autocast(args.precision):
            logits, states = model(tokens[:, :-1], reset=reset)
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True)
        opt.step()
        elapsed = time.perf_counter()-now
        assert all(s.dtype == torch.float32 for s in states)
        if i:
            samples.append(elapsed)
    result = dict(**vars(args), params=sum(p.numel() for p in model.parameters()), config=asdict(cfg),
        state_bytes=state_bytes(cfg), output_channels_per_layer=cfg.kron_heads*cfg.value_dim,
        seconds=samples, median_step_seconds=statistics.median(samples),
        median_tokens_per_second=args.batch*args.length/statistics.median(samples),
        rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        all_seconds=time.perf_counter()-start, final_loss=float(loss.detach()), finite_gradient_norm=float(norm),
        source='random-token systems benchmark only; optimizer updates discarded')
    print(json.dumps(result), flush=True)
    dest=Path('research-scratch/cpu-124m/benchmarks.jsonl')
    with dest.open('a') as out:
        out.write(json.dumps(result)+'\n')


if __name__ == '__main__':
    main()
