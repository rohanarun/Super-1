"""Time-bounded ~124M CPU LM run, fully logged and resumable.

This exploratory run is explicitly authorized separately from the GPU-only
common pilot. It makes no paired-seed or Transformer quality comparison.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F
from experiments.nanogpt_memory.data import file_hash, canonical_hash
from .common import configuration, configure_cpu, make_optimizer, autocast, state_bytes, build, rng_state, restore_rng
from experiments.nanogpt_memory.model import kron_chunk_factored

PROMPTS = [
    'The history of the city began',
    'A computer network is',
    'Scientists have discovered that',
    'In a small village near the mountains,',
    'The best way to learn a new skill is',
    'During the winter, the river',
    'The purpose of this experiment is',
    'To make a simple meal,',
    'The engineer examined the problem and',
    'In the years ahead, technology',
]


class Windows:
    def __init__(self, tokens, length, seed):
        self.tokens, self.length = tokens, length
        self.generator = np.random.default_rng(seed)
        self.order = self.generator.permutation((len(tokens)-1)//length)
        self.cursor = self.epoch = 0

    def batch(self, size):
        if self.cursor+size > len(self.order):
            raise RuntimeError('Training stream exhausted; refuse unplanned repetition')
        indices = self.order[self.cursor:self.cursor+size]
        values = np.stack([self.tokens[int(i)*self.length:int(i)*self.length+self.length+1] for i in indices]).astype(np.int64)
        self.cursor += size
        data = torch.from_numpy(values)
        x, y = data[:, :-1], data[:, 1:]
        reset = x == 50256
        reset[:, 0] = True
        return x, y, reset

    def state_dict(self):
        return dict(length=self.length, order=self.order, cursor=self.cursor, epoch=self.epoch,
                    generator=self.generator.bit_generator.state)

    def load_state_dict(self, state):
        assert state['length'] == self.length
        self.order, self.cursor, self.epoch = state['order'], state['cursor'], state['epoch']
        self.generator.bit_generator.state = state['generator']


@torch.no_grad()
def evaluate(model, data, count, length, batch_size, precision):
    model.eval()
    start = time.perf_counter()
    cursor, loss_sum, correct = 0, 0., 0
    while cursor < count:
        n = min(batch_size, (count-cursor)//length)
        if not n:
            this_len, n = count-cursor, 1
        else:
            this_len = length
        values = np.stack([data[cursor+i*this_len:cursor+(i+1)*this_len+1] for i in range(n)]).astype(np.int64)
        token_ids = torch.from_numpy(values)
        x, y = token_ids[:, :-1], token_ids[:, 1:]
        reset = x == 50256
        reset[:, 0] = True
        with autocast(precision):
            logits, _ = model(x, reset=reset)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum')
        loss_sum += float(loss)
        correct += int((logits.argmax(-1) == y).sum())
        cursor += n*this_len
    return dict(nll=loss_sum/count, perplexity=math.exp(loss_sum/count),
                token_accuracy=correct/count, evaluation_tokens=count, seconds=time.perf_counter()-start)


def save(path, model, opt, loader, protocol, metrics, curves):
    start = time.perf_counter()
    payload = dict(format='minkron_cpu_scale_v1', config=asdict(model.cfg), model=model.state_dict(),
        optimizer=opt.state_dict(), scheduler=dict(kind='elapsed-wall-time cosine',
            budget_seconds=protocol['training_seconds'], elapsed_seconds=metrics['elapsed_seconds'],
            peak_lr=protocol['peak_lr'], warmup_fraction=.05, final_lr_fraction=.1),
        rng=rng_state(), data_cursor=loader.state_dict(), recurrent_state=None,
        protocol=protocol, protocol_sha256=canonical_hash(protocol), metrics=metrics, curves=curves)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)
    return time.perf_counter()-start


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layout', default='h48')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    p.add_argument('--optimizer', choices=['fused','foreach','plain'], default='fused')
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--length', type=int, default=256)
    p.add_argument('--accumulate', type=int, default=2)
    p.add_argument('--chunk', type=int, default=32)
    p.add_argument('--core', choices=['stable','factored'], default='stable')
    p.add_argument('--seconds', type=float, default=1500)
    p.add_argument('--lr', type=float, default=.0006)
    p.add_argument('--seed', type=int, default=24101)
    p.add_argument('--out', default='research-scratch/cpu-124m/main')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--max-steps', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--eval-tokens', type=int, default=8192)
    args = p.parse_args()
    configure_cpu(args.threads)
    root, out = Path('data/fineweb_cpu'), Path(args.out)
    out.mkdir(parents=True, exist_ok=args.resume)
    manifest = json.loads((root/'manifest.json').read_text())
    # Confirmation is not opened until the final assessment script.
    for split in ('train','development'):
        if file_hash(root/manifest['splits'][split]['file']) != manifest['splits'][split]['sha256']:
            raise ValueError('Data checksum mismatch')
    train = np.load(root/'train.npy', mmap_mode='r')
    development = np.load(root/'development.npy', mmap_mode='r')
    cfg = configuration(args.layout, args.chunk, args.core)
    source_files = list(Path('experiments/cpu_scale').glob('*.py'))+[Path('experiments/nanogpt_memory/model.py')]
    code_hashes = {str(f): file_hash(f) for f in sorted(source_files)}
    protocol = dict(name='minkron_124m_cpu_25min_v1', user_scope='Explicit CPU scale exploration; separate from original common GPU contract',
        config=asdict(cfg), seed=args.seed, training_seconds=args.seconds, pilot=args.pilot,
        max_steps=args.max_steps, batch=args.batch, sequence_length=args.length,
        gradient_accumulation=args.accumulate, targets_per_optimizer_step=args.batch*args.length*args.accumulate,
        peak_lr=args.lr, optimizer=args.optimizer, optimizer_algorithm='AdamW; betas(.9,.95), epsilon 1e-8, weight decay .1; no embedding/norm/bias/decay-parameter decay',
        schedule='5% wall-time warmup, cosine to 10% peak; pilots use step schedule if max_steps set',
        grad_clip=1., precision=args.precision, parameters_and_optimizer_dtype='FP32', recurrence_state_dtype='FP32',
        torch_version=torch.__version__, cpu_capability=torch.backends.cpu.get_cpu_capability(),
        threads=args.threads, interop_threads=1, flush_denormals=True, data_manifest_sha256=canonical_hash(manifest),
        code_sha256=code_hashes, data_sampling='one seeded permutation of independent windows; no repetition; reset at window starts and EOT inputs',
        evaluation=dict(initial_final_tokens=args.eval_tokens, monitor_tokens=2048, monitor_interval_seconds=300,
                        confirmation='opened only after final checkpoint', checkpoint_interval_seconds=300),
        timing='Budget includes full training loop, data loading, periodic development evaluation, checkpoint writes and logging; setup, initial/final evaluation and final save are separate',
        generation=dict(prompts=PROMPTS, seeds=list(range(25101,25111)), temperature=.8, top_k=50, max_new_tokens=64,
                        stop_on_eot=True, selection='exactly these ten prompts; no cherry-picking or regeneration'),
        started_utc=datetime.now(timezone.utc).isoformat())
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    setup_start = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = build(cfg, args.seed)
    opt = make_optimizer(model, args.lr, args.optimizer)
    loader = Windows(train, args.length, args.seed+800000)
    setup_seconds = time.perf_counter()-setup_start
    metrics = dict(steps=0, scored_tokens=0, elapsed_seconds=0., optimizer_seconds=0.,
        development_seconds=0., checkpoint_seconds=0., process_cpu_seconds=0.,
        gradient_norm_max=0., clipped_steps=0, nonfinite_count=0, setup_seconds=setup_seconds)
    curves = []
    if args.resume:
        payload = torch.load(out/'checkpoint.pt', map_location='cpu', weights_only=False)
        old = payload['protocol']
        for key in ('config','seed','batch','sequence_length','gradient_accumulation','peak_lr','precision','optimizer','data_manifest_sha256','code_sha256'):
            if protocol[key] != old[key]:
                raise ValueError(f'Resume mismatch: {key}')
        protocol = old
        (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
        model.load_state_dict(payload['model'])
        opt.load_state_dict(payload['optimizer'])
        loader.load_state_dict(payload['data_cursor'])
        restore_rng(payload['rng'])
        metrics, curves = payload['metrics'], payload['curves']
        del payload
    else:
        initial = evaluate(model, development, args.eval_tokens, args.length, args.batch, args.precision)
        curves.append(dict(step=0, scored_tokens=0, elapsed_seconds=0., split='development', **initial))
        print(json.dumps(dict(event='initial_evaluation', **curves[-1])), flush=True)
    prior_seconds = metrics['elapsed_seconds']
    prior_cpu_seconds = metrics['process_cpu_seconds']
    train_start, cpu_start = time.perf_counter(), time.process_time()
    next_eval = (int(prior_seconds//300)+1)*300
    next_log = prior_seconds+30
    status, failure = 'RUNNING', None
    recent_loss = []
    try:
        while True:
            elapsed = prior_seconds+time.perf_counter()-train_start
            if elapsed >= args.seconds or (args.max_steps is not None and metrics['steps'] >= args.max_steps):
                break
            fraction = metrics['steps']/args.max_steps if args.pilot and args.max_steps else elapsed/args.seconds
            factor = max(.05, fraction/.05) if fraction < .05 else .1+.9*.5*(1+math.cos(math.pi*(fraction-.05)/.95))
            lr = args.lr*factor
            for group in opt.param_groups:
                group['lr'] = lr
            model.train()
            opt.zero_grad(set_to_none=True)
            start = time.perf_counter()
            losses = []
            for _ in range(args.accumulate):
                x, y, reset = loader.batch(args.batch)
                with autocast(args.precision):
                    logits, states = model(x, reset=reset)
                    loss = F.cross_entropy(logits.flatten(0,1), y.flatten())
                if not torch.isfinite(loss):
                    metrics['nonfinite_count'] += 1
                    raise FloatingPointError('Nonfinite training loss')
                (loss/args.accumulate).backward()
                losses.append(float(loss.detach()))
                if metrics['steps'] == 0:
                    assert all(s.dtype == torch.float32 for s in states)
                del logits, states, loss
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=True))
            opt.step()
            metrics['optimizer_seconds'] += time.perf_counter()-start
            metrics['steps'] += 1
            metrics['scored_tokens'] += args.batch*args.length*args.accumulate
            metrics['gradient_norm_max'] = max(metrics['gradient_norm_max'], norm)
            metrics['clipped_steps'] += int(norm > 1.)
            metrics['elapsed_seconds'] = prior_seconds+time.perf_counter()-train_start
            metrics['process_cpu_seconds'] = prior_cpu_seconds+time.process_time()-cpu_start
            train_row = dict(step=metrics['steps'], scored_tokens=metrics['scored_tokens'],
                elapsed_seconds=metrics['elapsed_seconds'], training_nll=sum(losses)/len(losses),
                grad_norm=norm, lr=lr, rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            with (out/'training.jsonl').open('a') as f:
                f.write(json.dumps(train_row)+'\n')
            recent_loss.append(train_row['training_nll'])
            if metrics['elapsed_seconds'] >= next_log:
                print(json.dumps(dict(event='progress', **train_row,
                    recent_training_nll=sum(recent_loss)/len(recent_loss))), flush=True)
                recent_loss = []
                next_log = metrics['elapsed_seconds']+30
            if metrics['elapsed_seconds'] >= next_eval and metrics['elapsed_seconds'] < args.seconds-30:
                dev = evaluate(model, development, 2048, args.length, args.batch, args.precision)
                metrics['development_seconds'] += dev['seconds']
                metrics['elapsed_seconds'] = prior_seconds+time.perf_counter()-train_start
                row = dict(step=metrics['steps'], scored_tokens=metrics['scored_tokens'],
                           elapsed_seconds=metrics['elapsed_seconds'], split='development', **dev)
                curves.append(row)
                print(json.dumps(dict(event='development', **row)), flush=True)
                (out/'curves.json').write_text(json.dumps(curves, indent=2)+'\n')
                metrics['checkpoint_seconds'] += save(out/'checkpoint.pt', model, opt, loader, protocol, metrics, curves)
                next_eval += 300
            (out/'progress.json').write_text(json.dumps(dict(status='RUNNING', **metrics), indent=2)+'\n')
        status = 'COMPLETED'
    except BaseException as error:
        status, failure = 'FAILED', repr(error)
        raise
    finally:
        metrics['elapsed_seconds'] = prior_seconds+time.perf_counter()-train_start
        metrics['process_cpu_seconds'] = prior_cpu_seconds+time.process_time()-cpu_start
        metrics['rss_peak_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        metrics['status'], metrics['failure_reason'] = status, failure
        final_save_seconds = 0.
        if status == 'COMPLETED':
            final = evaluate(model, development, args.eval_tokens, args.length, args.batch, args.precision)
            metrics['final_development'] = final
            curves.append(dict(step=metrics['steps'], scored_tokens=metrics['scored_tokens'],
                elapsed_seconds=metrics['elapsed_seconds'], split='development', **final))
            if not args.pilot:
                final_save_seconds = save(out/'checkpoint.pt', model, opt, loader, protocol, metrics, curves)
        metrics['final_checkpoint_seconds'] = final_save_seconds
        metrics['parameters'] = sum(p.numel() for p in model.parameters())
        metrics['parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters())
        metrics['optimizer_bytes'] = sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v))
        metrics['state_bytes_per_sequence'] = state_bytes(cfg)
        metrics['throughput_full_loop_tokens_per_second'] = metrics['scored_tokens']/max(metrics['elapsed_seconds'], 1e-9)
        metrics['throughput_optimizer_tokens_per_second'] = metrics['scored_tokens']/max(metrics['optimizer_seconds'], 1e-9)
        metrics['source_sha256'] = code_hashes
        metrics['factored_core_calls'] = getattr(kron_chunk_factored, 'calls', 0)
        metrics['stable_fallback_calls'] = getattr(kron_chunk_factored, 'fallbacks', 0)
        (out/'curves.json').write_text(json.dumps(curves, indent=2)+'\n')
        (out/'summary.json').write_text(json.dumps(metrics, indent=2)+'\n')
        print(json.dumps(dict(event='finished', **metrics)), flush=True)


if __name__ == '__main__':
    main()
