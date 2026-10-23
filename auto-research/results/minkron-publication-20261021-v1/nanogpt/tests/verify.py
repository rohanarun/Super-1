"""Run correctness gates; writes measurements even if a gate fails."""
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.nanogpt_memory.model import (Config, MinKron, build, optimizer,
    kron_chunk, kron_recurrent, tensor_bytes)
sys.path.insert(0, str(ROOT/'vendor/compact_kron_20261021'))
from model import CompactKron
from fla.ops.kronlstm.naive import naive_recurrent_kronlstm

torch.set_num_threads(int(os.environ.get('CHECK_THREADS', '2')))
DEVICE = os.environ.get('CHECK_DEVICE', 'cpu')
RESULT = {'status': 'RUNNING', 'device': DEVICE, 'torch': torch.__version__, 'tests': []}
RESULT['model_sha256'] = hashlib.sha256((ROOT/'experiments/nanogpt_memory/model.py').read_bytes()).hexdigest()
RESULT['device_name'] = torch.cuda.get_device_name(0) if DEVICE.startswith('cuda') else platform.processor()


def compare(a, b, atol=2e-5, rtol=2e-4):
    error = float((a.detach()-b.detach()).abs().max())
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return error


def record(name, **values):
    RESULT['tests'].append(dict(name=name, status='PASS', **values))
    print(json.dumps(RESULT['tests'][-1]), flush=True)


def make_terms(T, dtype, tiny=False):
    torch.manual_seed(491+T)
    shape = (2, T, 4, 8)
    q, k, v = [F.normalize(torch.randn(*shape, dtype=dtype, device=DEVICE), dim=-1) for _ in range(3)]
    if tiny:
        v[:, 0] = 0
        if T > 1:
            v[:, 1] *= 1e-12
    g = -.05*torch.rand(*shape, dtype=dtype, device=DEVICE)
    beta, c = [torch.rand(2, T, 4, dtype=dtype, device=DEVICE) for _ in range(2)]
    state = torch.randn(2, 4, 8, 8, dtype=dtype, device=DEVICE)*.1
    return [x.requires_grad_() for x in (q, k, v, g, beta, c, state)]


def chunk_sequence(args, reset):
    *terms, state = args
    outputs = []
    for start in range(0, terms[0].shape[1], 32):
        out, state = kron_chunk(*[a[:, start:start+32] for a in terms], state,
                                None if reset is None else reset[:, start:start+32])
        outputs.append(out)
    return torch.cat(outputs, 1), state


def authors_with_reset(args, reset):
    q, k, v, g, beta, c, state = args
    # Separate per-example document segments: independent upstream oracle.
    outputs, states = [], []
    for b in range(len(q)):
        starts = [0]+[int(t) for t in torch.where(reset[b])[0].tolist() if t > 0]+[q.shape[1]]
        seq, carry = [], state[b:b+1]
        for lo, hi in zip(starts[:-1], starts[1:]):
            if reset[b, lo]:
                carry = carry*0
            out, carry = naive_recurrent_kronlstm(q[b:b+1, lo:hi], k[b:b+1, lo:hi],
                v[b:b+1, lo:hi], g[b:b+1, lo:hi], beta[b:b+1, lo:hi],
                initial_state=carry, target_gate=c[b:b+1, lo:hi], output_final_state=True)
            seq.append(out)
        outputs.append(torch.cat(seq, 1))
        states.append(carry)
    return torch.cat(outputs, 0), torch.cat(states, 0)


def core_checks():
    for dtype in (torch.float64, torch.float32):
        for T in (1, 17, 33, 65, 513):
            args = make_terms(T, dtype)
            reset = torch.zeros(2, T, dtype=torch.bool, device=DEVICE)
            if T > 17:
                reset[0, 13] = True
                reset[1, 17] = True
            if T > 64:
                reset[0, 32] = True
                reset[1, 64] = True
            actual = chunk_sequence(args, reset)
            expected = authors_with_reset(args, reset)
            tol = 1e-10 if dtype == torch.float64 else 2e-5
            oe, se = [compare(a, b, atol=tol) for a, b in zip(actual, expected)]
            torch.manual_seed(19)
            weights = [torch.randn_like(a) for a in actual]
            grads = [torch.autograd.grad(sum((x*w).sum() for x, w in zip(output, weights)), args,
                                         retain_graph=True) for output in (actual, expected)]
            ge = max(compare(a, b, atol=tol*5) for a, b in zip(*grads))
            record('core_vs_authors', dtype=str(dtype), length=T, resets=True,
                   output_max_abs=oe, state_max_abs=se, all_inputs_and_initial_state_gradient_max_abs=ge)
    # Tiny and zero values exercise the exact norm floor, without normalizing away the case.
    args = make_terms(17, torch.float64, tiny=True)
    out = chunk_sequence(args, None)
    ref = kron_recurrent(*args[:-1], args[-1])
    errors = [compare(a, b, atol=1e-10) for a, b in zip(out, ref)]
    grad = torch.autograd.grad(out[0].square().sum()+out[1].square().sum(), args)
    assert all(torch.isfinite(g).all() for g in grad)
    record('zero_and_tiny_values', max_abs=max(errors), all_gradients_finite=True)
    # Numerical differentiation independent of either analytic implementation.
    torch.manual_seed(78)
    terms = [torch.randn(1, 4, 1, 2, dtype=torch.float64, device=DEVICE)*.1 for _ in range(3)]
    terms += [-torch.rand(1, 4, 1, 2, dtype=torch.float64, device=DEVICE)*.1]
    terms += [torch.rand(1, 4, 1, dtype=torch.float64, device=DEVICE)*.5 for _ in range(2)]
    terms += [torch.randn(1, 1, 2, 2, dtype=torch.float64, device=DEVICE)*.1]
    terms = [t.requires_grad_() for t in terms]
    assert torch.autograd.gradcheck(kron_chunk, tuple(terms), eps=1e-6, atol=1e-5, rtol=1e-3)
    record('finite_difference_gradcheck')


def source_checks():
    torch.manual_seed(712)
    source = CompactKron(4, 8, 8, input_width=256, output_width=256).to(DEVICE).double()
    source.core_backend = 'reference'
    torch.manual_seed(712)
    target = MinKron(Config()).to(DEVICE).double()
    mapping = {'forget1.weight': 'f_proj.1.weight', 'gate_bias': 'g_proj.bias',
               'A_log': 'A_log', 'dt_bias': 'dt_bias', 'out_norm.weight': 'o_norm.weight',
               'out_proj.weight': 'o_proj.weight'}
    packed_names = ['qkv_proj.weight', 'b_proj.weight', 'c_proj.weight', 'f_proj.0.weight', 'g_proj.weight']
    source_params, target_params = dict(source.named_parameters()), dict(target.named_parameters())
    for name, other in mapping.items():
        assert torch.equal(target_params[name], source_params[other]), name
    assert torch.equal(target.packed_weight, torch.cat([source_params[n] for n in packed_names]))
    x = torch.randn(2, 65, 256, dtype=torch.float64, device=DEVICE, requires_grad=True)
    s = torch.randn(2, 4, 8, 8, dtype=torch.float64, device=DEVICE, requires_grad=True)*.01
    a, b = target(x, s), source(x, s)
    err = max(compare(u, v, atol=1e-10) for u, v in zip(a, b))
    (a[0].square().mean()+a[1].square().mean()).backward(retain_graph=True)
    (b[0].square().mean()+b[1].square().mean()).backward()
    ge = max(compare(target_params[n].grad, source_params[o].grad, atol=1e-10) for n, o in mapping.items())
    ge = max(ge, compare(target.packed_weight.grad, torch.cat([source_params[n].grad for n in packed_names]), atol=1e-10))
    record('permanent_projection_packing', initializer='exact', output_state_max_abs=err,
           all_parameter_gradient_max_abs=ge)
    optimizers = [optimizer(m, .001) for m in (target, source)]
    for _ in range(3):
        for m, opt in zip((target, source), optimizers):
            opt.zero_grad(set_to_none=True)
            out, state = m(x.detach(), s.detach())
            (out.square().mean()+state.square().mean()).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
    post_error = max(compare(target_params[n], source_params[o], atol=1e-9) for n, o in mapping.items())
    post_error = max(post_error, compare(target.packed_weight, torch.cat([source_params[n] for n in packed_names]), atol=1e-9))
    record('packed_adamw_three_steps', parameter_max_abs=post_error)


def lm_checks():
    common = None
    for arm in ('minkron_8x8', 'mingru_linear', 'transformer'):
        cfg = Config(arm=arm, vocab_size=257)
        model = build(cfg, 902).to(DEVICE)
        shared = {n: p for n, p in model.named_parameters() if '.mixer.' not in n}
        if common is None:
            common = {n: p.detach().clone() for n, p in shared.items()}
        else:
            assert all(torch.equal(p, common[n]) for n, p in shared.items())
        tokens = torch.randint(0, 257, (2, 67), device=DEVICE)
        reset = torch.zeros_like(tokens, dtype=torch.bool)
        reset[0, 13] = True
        reset[1, 39] = True
        with torch.no_grad():
            full, fs = model(tokens, reset=reset)
            pieces, state = [], None
            for lo, hi in ((0, 1), (1, 18), (18, 34), (34, 67)):
                out, state = model(tokens[:, lo:hi], state, reset[:, lo:hi])
                pieces.append(out)
            split_err = compare(torch.cat(pieces, 1), full)
            changed = tokens.clone()
            changed[:, 30:] = (changed[:, 30:]+12) % 257
            future, _ = model(changed, reset=reset)
            causal_err = compare(full[:, :30], future[:, :30], atol=2e-6)
            reset_err = 0.
            for b, cut in ((0, 13), (1, 39)):
                independent, _ = model(tokens[b:b+1, cut:])
                reset_err = max(reset_err, compare(full[b:b+1, cut:], independent))
            values, backing = tensor_bytes(fs)
            if arm != 'transformer':
                assert values == backing == 8192, (values, backing)
        model.zero_grad(set_to_none=True)
        # Prefix token unique to row 0, suffix uses no such ID: reset must cut gradient.
        tx = torch.tensor([[1]*9+[2, 3, 4, 5, 6, 7]], device=DEVICE)
        rr = torch.zeros_like(tx, dtype=torch.bool)
        rr[:, 9] = True
        # Use hidden-state sensitivity through a hook: tied head otherwise gives
        # every embedding row a legitimate output gradient.
        captured = []
        hook = model.embedding.register_forward_hook(lambda m, i, o: (o.retain_grad(), captured.append(o)) and None)
        logits, _ = model(tx, reset=rr)
        logits[:, 9:].square().mean().backward()
        hook.remove()
        prefix_grad = float(captured[0].grad[:, :9].abs().max())
        # Cumulative-decay subtraction has FP32 cancellation roundoff. The
        # diagnostic measured 4.3e-9 in FP32 and 3.0e-18 in FP64, versus .149
        # on the suffix. Bound numerical residue rather than demand bitwise 0.
        suffix_grad = float(captured[0].grad[:, 9:].abs().max())
        assert prefix_grad <= 1e-7 and prefix_grad <= 1e-6*suffix_grad
        record('common_lm_causal_reset_split', arm=arm, split_max_abs=split_err,
               causality_max_abs=causal_err, reset_max_abs=reset_err,
               reset_prefix_gradient_max_abs=prefix_grad, state_tensor_bytes_batch2=values,
               state_backing_bytes_batch2=backing)
    record('identical_common_initialization_all_arms')


def full_shape_check():
    model = build(Config(), 22901).to(DEVICE)
    x = torch.randint(0, 50257, (1, 512), device=DEVICE)
    target = torch.roll(x, -1, 1)
    start = time.perf_counter()
    logits, state = model(x)
    loss = F.cross_entropy(logits.reshape(-1, 50257), target.flatten())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert tensor_bytes(state) == (4096, 4096)
    record('full_contract_shape_forward_backward', layers=4, width=256, vocab=50257,
           sequence_length=512, loss=float(loss.detach()), state_bytes=4096,
           parameters=sum(p.numel() for p in model.parameters()), elapsed_seconds=time.perf_counter()-start,
           note='Random-token integration check; not an LM quality measurement')


def main():
    start = time.perf_counter()
    try:
        core_checks()
        source_checks()
        lm_checks()
        full_shape_check()
        RESULT['status'] = 'PASS'
    except Exception as error:
        RESULT['status'] = 'FAIL'
        RESULT['error'] = repr(error)
        RESULT['traceback'] = traceback.format_exc()
        raise
    finally:
        RESULT['elapsed_seconds'] = time.perf_counter()-start
        dest = ROOT/'research-scratch/nanogpt-memory'/f'verification_{DEVICE}.json'
        dest.write_text(json.dumps(RESULT, indent=2)+'\n')


if __name__ == '__main__':
    main()
