"""Fixed-batch real-text overfit diagnostic; NOT the FineWeb pilot."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import time
import numpy as np
import tiktoken
import torch
from torch.nn import functional as F
from .model import Config, build, optimizer, tensor_bytes
from .data import file_hash, canonical_hash

ROOT = Path(__file__).resolve().parents[2]


def checkpoint_payload(model, opt, step, contract, data_hash):
    return dict(config=asdict(model.cfg), model=model.state_dict(), optimizer=opt.state_dict(),
                scheduler=dict(kind='constant_diagnostic', step=step, lr=opt.param_groups[0]['lr']),
                rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                         cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                data_cursor=dict(fixed_batch=True, completed_presentations=step), recurrent_state=None,
                contract_sha256=canonical_hash(contract), data_sha256=data_hash)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    outdir = ROOT/'research-scratch/nanogpt-memory/smoke'
    outdir.mkdir(exist_ok=False)
    fixture = ROOT/'data/fixture/handoff_prose.txt'
    cfg = Config()
    enc = tiktoken.get_encoding('gpt2')
    ids = enc.encode_ordinary(fixture.read_text())
    B, T, steps, seed, lr = 2, 64, 100, 22901, .003
    assert len(ids) >= B*(T+1)
    batch = torch.tensor([ids[i*(T+1):(i+1)*(T+1)] for i in range(B)], device=args.device)
    x, target = batch[:, :-1], batch[:, 1:]
    reset = torch.zeros_like(x, dtype=torch.bool)
    reset[:, 0] = True
    protocol = dict(status='frozen_before_training', purpose='real-text memorization/integration diagnostic only',
        config=asdict(cfg), seed=seed, steps=steps, lr=lr, optimizer='AdamW betas=(0.9,0.95), decay=0.1; exclusions match contract',
        device=args.device, threads=args.threads, sequence_length=T, batch_size=B,
        tokenizer='GPT-2', tokenizer_version=tiktoken.__version__, fixture_sha256=file_hash(fixture),
        token_ids=batch.cpu().tolist(), acceptance='finite gradients throughout; final batch NLL < 0.15 and next-token accuracy > 0.95',
        checkpoint_gate='serialize at step 50; restore model/optimizer/RNG/scheduler/data cursor; compare step 51',
        caveat='Repeated 128 scored tokens from the handoff, not FineWeb; no validation split, no generalization or speed comparison')
    (outdir/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    setup = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build(cfg, seed).to(args.device)
    opt = optimizer(model, lr)
    setup_seconds = time.perf_counter()-setup

    def one_step(m, o):
        m.train()
        o.zero_grad(set_to_none=True)
        logits, state = m(x, reset=reset)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target.flatten())
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite loss')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        o.step()
        return float(loss.detach()), float(grad_norm), state

    @torch.no_grad()
    def evaluate():
        model.eval()
        logits, state = model(x, reset=reset)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), target.flatten())
        return dict(nll=float(loss), accuracy=float((logits.argmax(-1) == target).float().mean()),
                    state_bytes=tensor_bytes(state)[0]//B)

    curves = [dict(step=0, presented_tokens=0, train_seconds=0., **evaluate())]
    train_seconds = 0.
    resumed_model = resumed_opt = None
    checkpoint_result = None
    status = 'RUNNING'
    try:
        with tempfile.TemporaryDirectory(prefix='minkron-checkpoint-') as tmp:
            for step in range(1, steps+1):
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                loss, grad_norm, state = one_step(model, opt)
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                train_seconds += time.perf_counter()-start
                if step == 50:
                    checkpoint = Path(tmp)/'resume.pt'
                    torch.save(checkpoint_payload(model, opt, step, protocol, file_hash(fixture)), checkpoint)
                    payload = torch.load(checkpoint, map_location=args.device, weights_only=False)
                    resumed_model = build(cfg, seed+1).to(args.device)
                    resumed_model.load_state_dict(payload['model'])
                    resumed_opt = optimizer(resumed_model, lr)
                    resumed_opt.load_state_dict(payload['optimizer'])
                    torch.set_rng_state(payload['rng']['torch'].cpu())
                    np.random.set_state(payload['rng']['numpy'])
                    random.setstate(payload['rng']['python'])
                    if payload['rng']['cuda'] is not None:
                        torch.cuda.set_rng_state_all(payload['rng']['cuda'])
                    assert payload['scheduler']['step'] == payload['data_cursor']['completed_presentations'] == 50
                    assert payload['contract_sha256'] == canonical_hash(protocol)
                if step == 51:
                    resumed_loss, _, _ = one_step(resumed_model, resumed_opt)
                    error = max(float((a-b).abs().max().detach()) for a, b in zip(model.parameters(), resumed_model.parameters()))
                    torch.testing.assert_close(torch.tensor(resumed_loss), torch.tensor(loss), atol=1e-7, rtol=1e-6)
                    assert error <= 1e-7
                    checkpoint_result = dict(status='PASS', step=51, parameter_max_abs=error,
                                             loss_delta=abs(resumed_loss-loss), serialized_bytes=checkpoint.stat().st_size,
                                             retained=False, reason='Regenerable diagnostic; avoid storing a 160 MB non-research checkpoint')
                    del resumed_model, resumed_opt, payload
                if step % 10 == 0:
                    row = dict(step=step, presented_tokens=step*B*T, train_seconds=train_seconds,
                               gradient_norm_before_clip=grad_norm, **evaluate())
                    curves.append(row)
                    print(json.dumps(row), flush=True)
                    (outdir/'curves.json').write_text(json.dumps(curves, indent=2)+'\n')
            final = curves[-1]
            status = 'PASS' if final['nll'] < .15 and final['accuracy'] > .95 else 'FAIL_OVERFIT_TARGET'
    except Exception as error:
        status = 'FAILED'
        (outdir/'failure.json').write_text(json.dumps(dict(error=repr(error), step=step), indent=2)+'\n')
        raise
    finally:
        result = dict(status=status, experiment_type='CPU real-text overfit diagnostic, not FineWeb pilot',
            arm=cfg.arm, backend='torch_bounded_chunk_fp32', config=asdict(cfg), seed=seed,
            diagnostic_contract_sha256=canonical_hash(protocol), original_comparison_contract_sha256=canonical_hash(json.loads((ROOT/'comparison_contract.json').read_text())),
            data_manifest_sha256=canonical_hash(protocol['token_ids']), trained_tokens=step*B*T,
            unique_scored_positions=B*T, optimizer_steps=step, train_seconds=train_seconds, setup_seconds=setup_seconds,
            validation_nll=None, training_batch_initial_nll=curves[0]['nll'], training_batch_final_nll=curves[-1]['nll'],
            training_batch_final_accuracy=curves[-1]['accuracy'], target_reached=None, time_to_target=None,
            failure_reason=None if status=='PASS' else status, parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            optimizer_bytes=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if torch.is_tensor(v)),
            state_bytes_per_sequence=4096, peak_device_allocated_bytes=None, peak_device_reserved_bytes=None,
            prefill_tokens_per_second=None, decode_tokens_per_second=None, checkpoint_reload=checkpoint_result,
            torch_version=torch.__version__, dtype='FP32', precision_gate='GPU BF16 untested')
        (outdir/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
