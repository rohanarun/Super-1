"""Post-hoc path-decomposition diagnostic for segment-parallel-sweep-20261101-v1 (PROTOCOL_ADDENDA.md, A1).

Reuses the frozen sweep_canary.py unchanged. Same sweep, but one coupling is pinned to the reference:
  fast_only  - segment/document states fixed at their reference values; only fast boundary states iterate.
  slow_only  - fast boundary states fixed at their reference values; only segment/document states iterate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_canary as sc_mod  # noqa: E402  (frozen; imported, not modified)


@torch.no_grad()
def pinned_sweep(model, ids, ref, mode, max_sweeps, stop_tol, parity_tol):
    ref_hidden, ref_fast, ref_seg, ref_doc = ref
    B, T = ids.shape
    L, Dn, w = model.segment_length, model.document_segments, model.width
    N = T // L
    n_doc = N // Dn
    used_doc = (N - 1) // Dn + 1
    fc, sc, dc = model.fast_cell, model.segment_cell, model.document_cell
    w_tok, w_seg, w_doc = fc.weight_ih[:, :w], fc.weight_ih[:, w:2 * w], fc.weight_ih[:, 2 * w:]
    emb = model.embedding(ids)
    gi_tok = (emb @ w_tok.t() + fc.bias_ih).reshape(B, N, L, 3 * w)
    S_ref = torch.cat((torch.zeros_like(ref_seg[:, :1]), ref_seg[:, L - 1::L]), 1)
    D_ref = torch.cat((torch.zeros_like(ref_doc[:, :1]), ref_doc[:, Dn * L - 1::Dn * L]), 1)
    ends = ref_fast[:, L - 1::L]
    H_ref = torch.cat((torch.zeros_like(ends[:, :1]), ends[:, :-1]), 1)
    ref_fast_seg = ref_fast.reshape(B, N, L, w)
    zeros = emb.new_zeros(B, w)
    Hin = H_ref.clone() if mode == "slow_only" else emb.new_zeros(B, N, w)
    S = S_ref.clone() if mode == "fast_only" else emb.new_zeros(B, N + 1, w)
    Dd = D_ref.clone() if mode == "fast_only" else emb.new_zeros(B, n_doc + 1, w)
    doc_index = torch.arange(N) // Dn
    history, saved = [], {}
    tols = sorted(sc_mod.DECLARED["residual_tolerances"], reverse=True)
    for k in range(1, max_sweeps + 1):
        gi = gi_tok + (S[:, :N] @ w_seg.t())[:, :, None, :] + (Dd[:, doc_index] @ w_doc.t())[:, :, None, :]
        gi = gi.reshape(B * N, L, 3 * w)
        h = Hin.reshape(B * N, w)
        ssum = torch.zeros_like(h)
        outs = []
        for p in range(L):
            h = sc_mod.gru_step(gi[:, p], h, fc.weight_hh, fc.bias_hh)
            ssum = ssum + h
            outs.append(h)
        out = torch.stack(outs, 1).reshape(B, N, L, w)
        mean = (ssum / L).reshape(B, N, w)
        if mode == "slow_only":
            Hin_new = H_ref
        else:
            Hin_new = torch.cat((zeros[:, None], h.reshape(B, N, w)[:, :-1]), 1)
        if mode == "fast_only":
            S_new, D_new = S_ref, D_ref
        else:
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
        seg_fast_err = (out - ref_fast_seg).abs().amax(dim=(0, 2, 3))
        history.append({"sweep": k, "residual": residual, "fast_err": seg_fast_err.max().item(),
                        "seg_err": (S_new - S_ref).abs().max().item(),
                        "doc_err": (D_new - D_ref).abs().max().item(),
                        "wavefront_ok": seg_fast_err[:min(k, N)].max().item() <= parity_tol})
        for tol in tols:
            if tol not in saved and residual <= tol:
                hid = sc_mod.combine_hidden(model, out, S_new, D_new)
                saved[tol] = {"K": k, "hidden_err": (hid - ref_hidden).abs().max().item()}
        Hin, S, Dd = Hin_new, S_new, D_new
        if residual <= stop_tol:
            break
    return history, {str(t): saved.get(t) for t in tols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=688)
    ap.add_argument("--synthetic-vocab", type=int, default=4096)
    ap.add_argument("--max-context", type=int, default=4096)
    args = ap.parse_args()
    torch.set_num_threads(1)
    pm = sc_mod.load_reference_module(args.source_root)
    bases = ({"base": "trained_uniform_T1024", "condition": "trained", "stream": "uniform", "stream_seed": 11, "batch": 3},
             {"base": "default_w688_s1", "condition": "default", "init_seed": 1, "stream": "uniform", "stream_seed": 101, "batch": 2})
    receipt = {"addendum": "A1", "declared": sc_mod.DECLARED, "torch": torch.__version__,
               "threads": torch.get_num_threads(), "rows": []}
    cpu0 = time.process_time()
    for base in bases:
        model, info = sc_mod.build_model(pm, args, base["condition"], base.get("init_seed", 0), torch.float64, args.width)
        ids = sc_mod.token_streams(base["stream"], info["vocab"], base["batch"], 1024, base["stream_seed"])
        t0 = time.process_time()
        ref = sc_mod.reference_run(model, ids)
        ref_cpu = round(time.process_time() - t0, 2)
        for mode in ("fast_only", "slow_only"):
            if time.process_time() - cpu0 > 60:
                receipt["aborted"] = "60 CPU-second ceiling reached before " + base["base"] + "/" + mode
                break
            t1 = time.process_time()
            history, at_tol = pinned_sweep(model, ids, ref, mode, 1024 // 32 + 1, 1e-12,
                                           sc_mod.DECLARED["fp64_parity_tolerance"])
            result = dict(base, mode=mode, T=1024, dtype="fp64", history=history, at_tolerance=at_tol,
                          cpu_s_reference_shared=ref_cpu, cpu_s_sweeps=round(time.process_time() - t1, 2))
            receipt["rows"].append(result)
            print(json.dumps({k: result[k] for k in ("base", "mode", "at_tolerance", "cpu_s_sweeps")}), flush=True)
            args.out.write_text(json.dumps(receipt, indent=1))
        del model, ref
        if "aborted" in receipt:
            break
    receipt["total_cpu_s"] = round(time.process_time() - cpu0, 2)
    args.out.write_text(json.dumps(receipt, indent=1))
    print("TOTAL_CPU_S", receipt["total_cpu_s"])


if __name__ == "__main__":
    main()
