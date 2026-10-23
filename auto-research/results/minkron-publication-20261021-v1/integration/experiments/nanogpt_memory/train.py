"""Single-device, fixed-token training harness for the three shared arms.

Production CLI requires CUDA and a frozen hardware/data/verification manifest.
Uses the verified native PyTorch backend; it is not a fused-kernel speedrun.
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import canonical_hash, file_hash, load_split, next_window
from .model import Config, build, optimizer


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def lr_at(token_count, budget, peak):
    warmup = max(1, int(.05*budget))
    if token_count < warmup:
        return peak*max(token_count, 1)/warmup
    progress = min(1., (token_count-warmup)/max(1, budget-warmup))
    return peak*(.1+.9*.5*(1+math.cos(math.pi*progress)))


@torch.no_grad()
def evaluate(model, tokens, count, sequence_length, device):
    model.eval()
    cursor, total = 0, 0.
    while cursor < count:
        length = min(sequence_length, count-cursor)
        x, y, reset, cursor = next_window(tokens, cursor, length, device)
        logits, _ = model(x, reset=reset)
        total += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum'))
    return total/count


def save_checkpoint(path, model, opt, tokens, steps, peak_lr, budget, train_seconds, setup_seconds,
                    frozen_hash, data_hash, source_hash):
    payload = dict(config=asdict(model.cfg), model=model.state_dict(), optimizer=opt.state_dict(),
        scheduler=dict(kind='token_cosine_5pct_warmup_10pct_floor', peak_lr=peak_lr,
                       budget=budget, consumed_tokens=tokens),
        rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        data_cursor=tokens, recurrent_state=None, optimizer_steps=steps, train_seconds=train_seconds,
        setup_seconds=setup_seconds, frozen_sha256=frozen_hash, data_sha256=data_hash,
        source_sha256=source_hash)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def train_fixed(cfg, train_tokens, dev_tokens, *, budget, dev_count, sequence_length,
                global_tokens, peak_lr, seed, device, outdir, frozen_hash, data_hash,
                source_hash, resume=False, target_nll=None, evaluation_every=100):
    """Common testable training primitive. Main CLI separately enforces gates."""
    outdir = Path(outdir)
    if outdir.exists() and not resume:
        raise FileExistsError('Use a fresh run directory; never overwrite seed outcomes')
    outdir.mkdir(parents=True, exist_ok=resume)
    checkpoint = outdir/'checkpoint.pt'
    attempts_path = outdir/'attempts.json'
    attempts = json.loads(attempts_path.read_text()) if resume and attempts_path.exists() else []
    if resume and not checkpoint.exists():
        raise FileNotFoundError('Resume requires an existing complete checkpoint')
    setup_start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build(cfg, seed).to(device)
    opt = optimizer(model, peak_lr)
    cursor = steps = 0
    train_seconds = 0.
    prior_setup = 0.
    curves = []
    if resume:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        expected = (frozen_hash, data_hash, source_hash, asdict(cfg), budget, peak_lr)
        actual = (ckpt['frozen_sha256'], ckpt['data_sha256'], ckpt['source_sha256'],
                  ckpt['config'], ckpt['scheduler']['budget'], ckpt['scheduler']['peak_lr'])
        if expected != actual:
            raise ValueError('Resume contract/data/source/config mismatch')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        cursor, steps = ckpt['data_cursor'], ckpt['optimizer_steps']
        train_seconds, prior_setup = ckpt['train_seconds'], ckpt['setup_seconds']
        torch.set_rng_state(ckpt['rng']['torch'].cpu())
        np.random.set_state(ckpt['rng']['numpy'])
        random.setstate(ckpt['rng']['python'])
        if ckpt['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
        # A crash after a later curve write but before checkpoint replacement
        # must not duplicate progress or retain a checkpoint-ahead curve.
        curves = [row for row in json.loads((outdir/'curves.json').read_text())
                  if row['trained_tokens'] <= cursor]
    attempt_setup_seconds = time.perf_counter()-setup_start
    # Keep the spent compute of failed attempts, including work beyond the
    # recovered checkpoint. Resuming must not erase the cost of a failure.
    if attempts:
        train_seconds = sum(a['train_seconds'] for a in attempts)
        prior_setup = sum(a['setup_seconds'] for a in attempts)
    setup_seconds = prior_setup+attempt_setup_seconds
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    session_start = time.perf_counter()
    resumed_train_seconds = train_seconds
    initial_cursor, initial_steps = cursor, steps
    failure = None
    reached = None if target_nll is None else False
    time_to_target = None
    try:
        while cursor < budget:
            # Wall-clock covers data transfer, optimizer, logging, checkpoint I/O
            # and periodic development evaluation. Setup and final confirmation
            # are separately reported. Only synchronizations define boundaries.
            model.train()
            opt.zero_grad(set_to_none=True)
            this_step = min(global_tokens, budget-cursor)
            selected_lr = lr_at(cursor+this_step, budget, peak_lr)
            for group in opt.param_groups:
                group['lr'] = selected_lr
            step_loss = 0.
            end_cursor = cursor+this_step
            while cursor < end_cursor:
                length = min(sequence_length, end_cursor-cursor)
                x, y, reset, next_cursor = next_window(train_tokens, cursor, length, device)
                logits, _ = model(x, reset=reset)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='sum')
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss')
                (loss/this_step).backward()
                step_loss += float(loss.detach())
                cursor = next_cursor
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            steps += 1
            if steps % evaluation_every == 0 or cursor == budget:
                nll = evaluate(model, dev_tokens, dev_count, sequence_length, device)
                synchronize(device)
                elapsed = resumed_train_seconds+time.perf_counter()-session_start
                row = dict(optimizer_steps=steps, trained_tokens=cursor, train_seconds=elapsed,
                           development_nll=nll, training_step_nll=step_loss/this_step,
                           gradient_norm_before_clip=float(grad_norm), lr=selected_lr)
                curves.append(row)
                (outdir/'curves.json').write_text(json.dumps(curves, indent=2)+'\n')
                print(json.dumps(row), flush=True)
                save_checkpoint(checkpoint, model, opt, cursor, steps, peak_lr, budget,
                    elapsed, setup_seconds, frozen_hash, data_hash, source_hash)
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        synchronize(device)
        train_seconds = resumed_train_seconds+time.perf_counter()-session_start
        attempt = dict(attempt=len(attempts)+1, resumed=resume,
            restored_token_cursor=initial_cursor, final_token_cursor=cursor,
            processed_tokens=cursor-initial_cursor, optimizer_steps=steps-initial_steps,
            train_seconds=train_seconds-resumed_train_seconds,
            setup_seconds=attempt_setup_seconds, failure_reason=failure)
        attempts.append(attempt)
        attempts_path.write_text(json.dumps(attempts, indent=2)+'\n')
        if target_nll is not None:
            crossings = [r for r in curves if r['development_nll'] <= target_nll]
            reached = bool(crossings)
            time_to_target = crossings[0]['train_seconds'] if crossings else None
        summary = dict(arm=cfg.arm, backend='native_torch_bounded_chunk_or_scan_or_sdpa_fp32', seed=seed,
            contract_sha256=frozen_hash, source_sha256=source_hash, data_manifest_sha256=data_hash,
            trained_tokens=cursor, optimizer_steps=steps, train_seconds=train_seconds, setup_seconds=setup_seconds,
            development_nll=curves[-1]['development_nll'] if curves else None, validation_nll=None,
            target_reached=reached, time_to_target=time_to_target, target_censored=(reached is False),
            failure_reason=failure, selected_peak_lr=peak_lr,
            failed_attempt_count=sum(a['failure_reason'] is not None for a in attempts),
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            optimizer_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)),
            state_bytes_per_sequence=4096 if cfg.width == 256 and cfg.layers == 4 and cfg.arm != 'transformer' else None,
            peak_device_allocated_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
            peak_device_reserved_bytes=torch.cuda.max_memory_reserved(device) if str(device).startswith('cuda') else None,
            prefill_tokens_per_second=None, decode_tokens_per_second=None)
        (outdir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return model, summary


def source_hash(root):
    return canonical_hash({str(p.relative_to(root)): file_hash(p) for p in sorted((root/'experiments').rglob('*.py'))})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frozen', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['minkron_8x8', 'mingru_linear', 'transformer'], required=True)
    parser.add_argument('--phase', choices=['pilot', 'confirmation'], required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    frozen = json.loads(Path(args.frozen).read_text())
    contract = json.loads((root/'comparison_contract.json').read_text())
    if not torch.cuda.is_available():
        raise RuntimeError('Research runs require CUDA; CPU is for preparation and verification only')
    if frozen.get('status') != 'frozen_after_gpu_and_data_qualification':
        raise ValueError('Hardware/data/verification contract is not frozen')
    if frozen['original_contract_sha256'] != canonical_hash(contract):
        raise ValueError('Original comparison contract mismatch')
    if frozen['device_name'] != torch.cuda.get_device_name(0) or frozen['torch_version'] != torch.__version__:
        raise ValueError('Requalify changed GPU/runtime')
    actual_source = source_hash(root)
    if frozen['source_sha256'] != actual_source:
        raise ValueError('Source changed after qualification')
    train, manifest = load_split(args.data, 'train')
    dev, _ = load_split(args.data, 'development')
    if canonical_hash(manifest) != frozen['data_manifest_sha256']:
        raise ValueError('Data manifest mismatch')
    if args.phase == 'pilot':
        if args.seed != 22901 or args.lr not in [.0003, .001, .003]:
            raise ValueError('Pilot must use shared seed and predeclared LR grid')
        budget, target = 2000000, None
    else:
        if args.seed not in [23101, 23102, 23103, 23104, 23105]:
            raise ValueError('Confirmation seed not predeclared')
        selection = frozen.get('selection')
        if not selection or args.lr != selection['learning_rates'][args.arm]:
            raise ValueError('Freeze development-only LR selection and target before confirmation')
        budget, target = 20000000, selection['development_target_nll']
    cfg = Config(arm=args.arm)
    model, summary = train_fixed(cfg, train, dev, budget=budget, dev_count=1000000,
        sequence_length=512, global_tokens=16384, peak_lr=args.lr, seed=args.seed, device='cuda',
        outdir=args.output, frozen_hash=canonical_hash(frozen), data_hash=canonical_hash(manifest),
        source_hash=actual_source, resume=args.resume, target_nll=target)
    if args.phase == 'confirmation':
        # Confirmation is opened only after the fixed-token final checkpoint.
        confirmation, _ = load_split(args.data, 'confirmation')
        start = time.perf_counter()
        summary['validation_nll'] = evaluate(model, confirmation, 1000000, 512, 'cuda')
        synchronize('cuda')
        summary['final_confirmation_seconds'] = time.perf_counter()-start
        (Path(args.output)/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
