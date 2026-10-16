"""Continue the frozen linear-minGRU checkpoint up to native GRU's CPU budget."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'gru-extended-20261016-v1'
sys.path.insert(0, str(PARENT))
from models import build, logits, optimizer

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

@torch.no_grad()
def evaluate(model, data, starts):
    model.eval()
    losses, correct = [], 0
    for start in starts:
        xy = torch.from_numpy(np.array(data[start:start+513], dtype=np.int64))[None]
        z = logits(model, 'mingru_linear', xy[:, :-1].contiguous(), xy[:, 1:].contiguous())
        losses.append(float(F.cross_entropy(z.flatten(0, 1), xy[:, 1:].flatten())))
        correct += int((z.argmax(-1) == xy[:, 1:]).sum())
    ce = float(np.mean(losses))
    return dict(ce=ce, perplexity=float(np.exp(ce)), top1=correct/(512*len(starts)),
                targets=512*len(starts), window_ce=losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cpu', type=int, default=0)
    args = parser.parse_args()
    p = json.loads((ROOT / 'PROTOCOL.json').read_text())
    for name, digest in json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text()).items():
        assert sha(ROOT / name) == digest, name
    for name, digest in p['input_hashes'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in p['data_hashes'].items():
        assert sha(args.data_dir / name) == digest, name
    assert sha(ROOT / 'OFFSETS.json') == p['offsets_sha256']
    offsets = json.loads((ROOT / 'OFFSETS.json').read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    os.sched_setaffinity(0, {args.cpu})
    resource.setrlimit(resource.RLIMIT_CPU, (360, 370))
    out = ROOT / 'execution'
    out.mkdir(exist_ok=True)
    cp = out / 'checkpoint.pt'
    assert not (out / 'RESULT.json').exists(), 'Completed runs are immutable'
    model = build('mingru_linear', p['config'], p['seed'])
    opt = optimizer(model, p['continuation_lr'])
    old = torch.load(cp if cp.exists() else ROOT / p['source_checkpoint'],
                     map_location='cpu', weights_only=False)
    assert old['protocol_sha256'] == (sha(ROOT / 'PROTOCOL.json') if cp.exists()
                                     else p['parent_protocol_sha256'])
    model.load_state_dict(old['model'])
    opt.load_state_dict(old['optimizer'])
    torch.set_rng_state(old['torch_rng'])
    state = old['state']
    for group in opt.param_groups:
        group['lr'] = p['continuation_lr']
    train = np.memmap(args.data_dir / 'train.bin', dtype=np.uint16, mode='r')
    val = np.memmap(args.data_dir / 'validation.bin', dtype=np.uint16, mode='r')
    runtime = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                   platform=platform.platform(), threads=1, affinity=sorted(os.sched_getaffinity(0)),
                   dtype='float32', device='cpu')
    if not cp.exists():
        replay = evaluate(model, val, offsets['final_starts'])
        prior = json.loads((PARENT / 'execution/mingru_linear/RESULT.json').read_text())
        error = max(abs(a-b) for a,b in zip(replay['window_ce'], prior['final']['window_ce']))
        assert error < 1e-6
        save(out / 'START.json', dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             protocol_sha256=sha(ROOT/'PROTOCOL.json'), source_replay_max_error=error, runtime=runtime))

    def checkpoint():
        tmp = cp.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
            torch_rng=torch.get_rng_state(), state=state, arm='mingru_linear', config=p['config'],
            protocol_sha256=sha(ROOT/'PROTOCOL.json')), tmp)
        tmp.replace(cp)

    def monitor():
        ev = evaluate(model, val, offsets['monitor_starts'])
        row = dict(step=state['step'], training_targets=state['training_targets'],
                   train_cpu_seconds=state['train_cpu_seconds'], train_wall_seconds=state['train_wall_seconds'],
                   validation_ce=ev['ce'], validation_top1=ev['top1'])
        state['monitor'].append(row)
        save(out / 'curves.json', state)
        print(row, flush=True)

    begin = time.monotonic()
    recent = [.1]
    while state['step'] < p['maximum_total_updates']:
        # Leave a small measured margin rather than knowingly crossing the budget.
        reserve = max(p['stop_margin_cpu_seconds'], 2*max(recent[-20:]))
        if state['train_cpu_seconds'] + reserve >= p['target_total_train_cpu_seconds']:
            break
        if time.monotonic() - begin > 600:
            checkpoint()
            raise TimeoutError('Continuation wall limit reached')
        step = state['step'] + 1
        tick, cpu_tick = time.perf_counter(), time.process_time()
        model.train()
        xy = torch.from_numpy(np.stack([np.array(train[s:s+513], dtype=np.int64)
                                       for s in offsets['train_starts'][step-1]]))
        opt.zero_grad(set_to_none=True)
        z = logits(model, 'mingru_linear', xy[:, :-1].contiguous(), xy[:, 1:].contiguous())
        ce = F.cross_entropy(z.flatten(0, 1), xy[:, 1:].flatten())
        ce.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(ce) or not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite training at step {step}')
        opt.step()
        model.constrain()
        wall, cpu = time.perf_counter()-tick, time.process_time()-cpu_tick
        recent.append(cpu)
        state['train_wall_seconds'] += wall
        state['train_cpu_seconds'] += cpu
        state['step'] = step
        state['training_targets'] += 512
        state['train'].append(dict(step=step, training_targets=state['training_targets'],
            ce=float(ce.detach()), lr=p['continuation_lr'], gradient_norm=float(norm),
            train_cpu_seconds=state['train_cpu_seconds'], train_wall_seconds=state['train_wall_seconds']))
        if step % 250 == 0:
            monitor()
        if step % 100 == 0:
            checkpoint()
    if state['monitor'][-1]['step'] != state['step']:
        monitor()
    checkpoint()
    final = evaluate(model, val, offsets['final_starts'])
    unused = p['target_total_train_cpu_seconds'] - state['train_cpu_seconds']
    result = dict(status='COMPLETE', arm='mingru_linear', updates=state['step'],
        training_targets=state['training_targets'], parameters=p['parameters'], seed=p['seed'],
        final=final, monitor=state['monitor'], train_cpu_seconds=state['train_cpu_seconds'],
        train_wall_seconds=state['train_wall_seconds'], unused_budget_cpu_seconds=unused,
        continuation_train_cpu_seconds=state['train_cpu_seconds']-p['initial_train_cpu_seconds'],
        checkpoint_sha256=sha(cp), protocol_sha256=sha(ROOT/'PROTOCOL.json'), runtime=runtime,
        budget_within_half_second=0 <= unused <= .5)
    save(out/'RESULT.json', result)
    save(out/'curves.json', state)
    print('COMPLETE', {k: v for k,v in result.items() if k not in ('monitor','runtime','final')}, final['ce'], flush=True)

if __name__ == '__main__':
    main()
