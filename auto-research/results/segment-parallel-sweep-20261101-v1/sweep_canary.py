"""Segment-parallel Jacobi (multiple-shooting) sweeps for the exact historical hierarchy.

Mechanics canary only: CPU forward passes on synthetic token streams. No training, no optimizer
step, no corpus, no official test. The reference is HierarchicalDistributionStudent.transition from
the recovered byte-exact source, imported unchanged.

Each sweep guesses the state entering every 32-token segment, runs all segments' fast GRUs as one
batch, then recomputes the slow chains exactly from the fast means (segment_cell's input is only the
segment mean; document_cell's input is only the segment state). Segment 0 always starts from the exact
initial state, so after sweep k segments 0..k-1 equal the reference (the wavefront), and N sweeps
reproduce the whole sequence. The canary measures how many sweeps a given model actually needs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

DECLARED = {
    "fp64_parity_tolerance": 1e-9,      # max |hidden - reference|, as the project's parity64.py
    "fp32_parity_tolerance": 1e-5,      # as the project's FP32 parity.py
    "residual_tolerances": [1e-4, 1e-6, 1e-9, 1e-12],
    "primary_residual_tolerance": 1e-6,
    "falsifier": "trained checkpoint needs K(1e-6) >= 16 sweeps at T=1024 (break-even vs 32 sequential segments)",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_reference_module(source_root: Path):
    sys.path.insert(0, str(source_root))
    from representation_experiments import predictive_models  # noqa: E402
    return predictive_models


@torch.no_grad()
def reference_run(model, ids):
    """Token-by-token reference using the class's own transition (hidden = combine output)."""
    state = None
    hidden, fast, seg, doc = [], [], [], []
    for t in range(ids.shape[1]):
        h, state = model.transition(ids[:, t], state)
        hidden.append(h)
        fast.append(state["fast"])
        seg.append(state["segment"])
        doc.append(state["document"])
    return (torch.stack(hidden, 1), torch.stack(fast, 1),
            torch.stack(seg, 1), torch.stack(doc, 1))


def gru_step(gi, h, w_hh, b_hh):
    """torch.nn.GRUCell equations (gate order r, z, n) with a precomputed input projection."""
    gh = torch.addmm(b_hh, h, w_hh.t())
    i_r, i_z, i_n = gi.chunk(3, -1)
    h_r, h_z, h_n = gh.chunk(3, -1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (h - n) * z + n


@torch.no_grad()
def combine_hidden(model, out, S, Dd, broken_last_token=False):
    """combine([fast, segment, document]); a segment's last token sees the updated slow states.

    broken_last_token=True is a deliberate defect (failure control): it feeds the pre-update slow
    states to the last token, which the verifier must reject."""
    B, N, L, w = out.shape
    Dn = model.document_segments
    j = torch.arange(N)
    segc = S[:, j][:, :, None, :].expand(B, N, L, w).clone()
    docc = Dd[:, j // Dn][:, :, None, :].expand(B, N, L, w).clone()
    if not broken_last_token:
        segc[:, :, L - 1] = S[:, j + 1]
        docc[:, :, L - 1] = Dd[:, (j + 1) // Dn]
    x = torch.cat((out, segc, docc), -1).reshape(B, N * L, 3 * w)
    return model.combine(x)


@torch.no_grad()
def sweep_run(model, ids, ref, max_sweeps, stop_tol, parity_tol, broken_last_token=False):
    ref_hidden, ref_fast, ref_seg, ref_doc = ref
    B, T = ids.shape
    L, Dn, w = model.segment_length, model.document_segments, model.width
    assert T % L == 0
    N = T // L
    n_doc = N // Dn
    used_doc = (N - 1) // Dn + 1          # document states used as fast-GRU inputs
    fc, sc, dc = model.fast_cell, model.segment_cell, model.document_cell
    w_tok, w_seg, w_doc = fc.weight_ih[:, :w], fc.weight_ih[:, w:2 * w], fc.weight_ih[:, 2 * w:]
    emb = model.embedding(ids)
    gi_tok = (emb @ w_tok.t() + fc.bias_ih).reshape(B, N, L, 3 * w)

    # Reference slow states after i segment updates / m document updates.
    S_ref = torch.cat((torch.zeros_like(ref_seg[:, :1]), ref_seg[:, L - 1::L]), 1)        # (B, N+1, w)
    D_ref = torch.cat((torch.zeros_like(ref_doc[:, :1]), ref_doc[:, Dn * L - 1::Dn * L]), 1)  # (B, n_doc+1, w)
    ref_fast_seg = ref_fast.reshape(B, N, L, w)

    zeros = emb.new_zeros(B, w)
    Hin = emb.new_zeros(B, N, w)          # cold start: every guess is the zero state
    S = emb.new_zeros(B, N + 1, w)
    Dd = emb.new_zeros(B, n_doc + 1, w)
    doc_index = torch.arange(N) // Dn
    history, saved = [], {}
    tols = sorted(DECLARED["residual_tolerances"], reverse=True)
    for k in range(1, max_sweeps + 1):
        gi = gi_tok + (S[:, :N] @ w_seg.t())[:, :, None, :] + (Dd[:, doc_index] @ w_doc.t())[:, :, None, :]
        gi = gi.reshape(B * N, L, 3 * w)
        h = Hin.reshape(B * N, w)
        ssum = torch.zeros_like(h)
        outs = []
        for p in range(L):
            h = gru_step(gi[:, p], h, fc.weight_hh, fc.bias_hh)
            ssum = ssum + h
            outs.append(h)
        out = torch.stack(outs, 1).reshape(B, N, L, w)
        mean = (ssum / L).reshape(B, N, w)
        Hin_new = torch.cat((zeros[:, None], h.reshape(B, N, w)[:, :-1]), 1)
        S_list = [zeros]
        for i in range(N):
            S_list.append(sc(mean[:, i], S_list[-1]))
        S_new = torch.stack(S_list, 1)
        D_list = [zeros]
        for m in range(1, n_doc + 1):
            D_list.append(dc(S_new[:, m * Dn], D_list[-1]))
        D_new = torch.stack(D_list, 1)

        residual = max((Hin_new - Hin).abs().max().item(),
                       (S_new[:, :N] - S[:, :N]).abs().max().item(),
                       (D_new[:, :used_doc] - Dd[:, :used_doc]).abs().max().item())
        seg_fast_err = (out - ref_fast_seg).abs().amax(dim=(0, 2, 3))   # per segment
        exact_prefix = 0
        while exact_prefix < N and seg_fast_err[exact_prefix].item() <= parity_tol:
            exact_prefix += 1
        wavefront_err = seg_fast_err[:min(k, N)].max().item()
        history.append({
            "sweep": k, "residual": residual,
            "fast_err": seg_fast_err.max().item(),
            "seg_err": (S_new - S_ref).abs().max().item(),
            "doc_err": (D_new - D_ref).abs().max().item(),
            "wavefront_err": wavefront_err,
            "wavefront_ok": wavefront_err <= parity_tol,
            "exact_prefix_segments": exact_prefix,
        })
        for tol in tols:
            if tol not in saved and residual <= tol:
                hid = combine_hidden(model, out, S_new, D_new, broken_last_token)
                saved[tol] = {"K": k, "hidden_err": (hid - ref_hidden).abs().max().item()}
        Hin, S, Dd = Hin_new, S_new, D_new
        if residual <= stop_tol:
            break
    return history, {str(t): saved.get(t) for t in tols}


def find_state_dict(obj):
    if isinstance(obj, dict):
        keys = list(obj.keys())
        if any(isinstance(k, str) and k.endswith("fast_cell.weight_ih") for k in keys):
            return obj
        for v in obj.values():
            found = find_state_dict(v)
            if found is not None:
                return found
    return None


_TRAINED = {}


def build_model(pm, args, condition, seed, dtype, width):
    if condition.startswith("trained"):
        if "raw" not in _TRAINED:
            _TRAINED["raw"] = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        raw = _TRAINED["raw"]
        sd = find_state_dict(raw)
        if sd is None:
            raise RuntimeError("no hierarchy state_dict in checkpoint")
        prefix = next(k for k in sd if k.endswith("fast_cell.weight_ih"))[:-len("fast_cell.weight_ih")]
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        V, w = sd["embedding.weight"].shape
        model = pm.HierarchicalDistributionStudent(V, w, args.max_context, 32, 8)
        model.load_state_dict(sd, strict=True)    # class identity check
        info = {"top_level_keys": sorted(map(str, raw.keys()))[:20] if isinstance(raw, dict) else None,
                "state_dict_prefix": prefix, "vocab": V, "width": w}
    else:
        torch.manual_seed(seed)
        model = pm.HierarchicalDistributionStudent(args.synthetic_vocab, width, args.max_context, 32, 8)
        with torch.no_grad():
            if condition == "scale3":
                for n, p in model.named_parameters():
                    if "cell" in n or "combine" in n:
                        p.mul_(3.0)
            elif condition == "longmem":
                model.fast_cell.bias_hh[width:2 * width] += 4.0   # update gate z ~ 0.98
        info = {"vocab": args.synthetic_vocab, "width": width}
    return model.to(dtype).eval(), info


def token_streams(kind, vocab, batch, T, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        return torch.randint(0, vocab, (batch, T), generator=g)
    if kind == "periodic":
        phrase = torch.randint(0, vocab, (batch, 64), generator=g)
        return phrase.repeat(1, T // 64)
    if kind == "small_alphabet":
        alphabet = torch.randperm(vocab, generator=g)[:512]
        return alphabet[torch.randint(0, 512, (batch, T), generator=g)]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--checkpoint-sha256")
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=688)
    ap.add_argument("--synthetic-vocab", type=int, default=4096)
    ap.add_argument("--max-context", type=int, default=4096)
    args = ap.parse_args()
    torch.set_num_threads(1)
    pm = load_reference_module(args.source_root)
    plan = json.loads(args.plan.read_text())
    receipt = {"declared": DECLARED, "torch": torch.__version__, "threads": torch.get_num_threads(),
               "plan": plan, "rows": []}
    if args.checkpoint:
        digest = sha256_file(args.checkpoint)
        receipt["checkpoint_sha256"] = digest
        if args.checkpoint_sha256 and digest != args.checkpoint_sha256:
            raise RuntimeError("checkpoint hash mismatch: " + digest)
    cpu0 = time.process_time()
    for row in plan["rows"]:
        if time.process_time() - cpu0 > plan["cpu_seconds_ceiling"]:
            receipt["aborted"] = "cumulative CPU ceiling reached before row " + row["id"]
            break
        dtype = torch.float64 if row["dtype"] == "fp64" else torch.float32
        parity_tol = DECLARED["fp64_parity_tolerance" if row["dtype"] == "fp64" else "fp32_parity_tolerance"]
        stop_tol = row.get("stop_tol", 1e-12 if row["dtype"] == "fp64" else 0.0)
        model, info = build_model(pm, args, row["condition"], row.get("init_seed", 0), dtype,
                                  row.get("width", args.width))
        ids = token_streams(row["stream"], info["vocab"], row["batch"], row["T"], row["stream_seed"])
        t0 = time.process_time()
        ref = reference_run(model, ids)
        t1 = time.process_time()
        N = row["T"] // 32
        history, at_tol = sweep_run(model, ids, ref, min(row.get("max_sweeps", N + 1), N + 1), stop_tol,
                                    parity_tol, row.get("broken_last_token", False))
        t2 = time.process_time()
        result = dict(row, model_info=info, N_segments=N, history=history, at_tolerance=at_tol,
                      cpu_s_reference=round(t1 - t0, 2), cpu_s_sweeps=round(t2 - t1, 2),
                      wavefront_all_ok=all(h["wavefront_ok"] for h in history))
        receipt["rows"].append(result)
        print(json.dumps({k: result[k] for k in ("condition", "stream", "dtype", "T", "N_segments",
                                                  "at_tolerance", "wavefront_all_ok",
                                                  "cpu_s_reference", "cpu_s_sweeps")}), flush=True)
        args.out.write_text(json.dumps(receipt, indent=1))
        del model, ref
    receipt["total_cpu_s"] = round(time.process_time() - cpu0, 2)
    args.out.write_text(json.dumps(receipt, indent=1))
    print("TOTAL_CPU_S", receipt["total_cpu_s"])


if __name__ == "__main__":
    main()
