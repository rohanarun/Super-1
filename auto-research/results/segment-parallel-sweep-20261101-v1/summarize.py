"""Summarize RESULTS.json from sweep_canary.py into SUMMARY.json and a markdown table (standard library only)."""
import json
import sys
from pathlib import Path

results = json.loads(Path(sys.argv[1]).read_text())
out_json, out_md = Path(sys.argv[2]), Path(sys.argv[3])
D = results["declared"]
rows, primary = [], []
for r in results["rows"]:
    tol = D["fp64_parity_tolerance"] if r["dtype"] == "fp64" else D["fp32_parity_tolerance"]
    at = r["at_tolerance"]
    final = r["history"][-1]
    conv = at.get("1e-12") if r["dtype"] == "fp64" else None
    row = {
        "id": r["id"], "condition": r["condition"], "stream": r["stream"], "dtype": r["dtype"],
        "T": r["T"], "N": r["N_segments"], "batch": r["batch"], "width": r["model_info"]["width"],
        "K": {k: (v["K"] if v else None) for k, v in at.items()},
        "hidden_err_at_K": {k: (v["hidden_err"] if v else None) for k, v in at.items()},
        "sweeps_run": len(r["history"]), "final_residual": final["residual"],
        "final_fast_err": final["fast_err"],
        "wavefront_all_ok": r["wavefront_all_ok"],
        "exact_prefix_by_sweep": [h["exact_prefix_segments"] for h in r["history"]],
        "residual_by_sweep": [h["residual"] for h in r["history"]],
        "fast_err_by_sweep": [h["fast_err"] for h in r["history"]],
        "parity_tolerance": tol,
        "cpu_s_reference": r["cpu_s_reference"], "cpu_s_sweeps": r["cpu_s_sweeps"],
        "role": r["role"],
    }
    if r["dtype"] == "fp64":
        row["converged_parity_pass"] = bool(conv) and conv["hidden_err"] <= tol
    else:
        best = min((v["hidden_err"] for v in at.values() if v), default=None)
        row["best_hidden_err"] = best
        row["converged_parity_pass"] = best is not None and best <= tol
    rows.append(row)
    if r["role"].startswith("PRIMARY"):
        primary.append(row)

prim_tol = str(D["primary_residual_tolerance"])
prim_K = [p["K"][prim_tol] for p in primary]
summary = {
    "run_id": results["plan"]["run_id"], "declared": D, "torch": results["torch"],
    "threads": results["threads"], "checkpoint_sha256": results.get("checkpoint_sha256"),
    "total_cpu_s": results.get("total_cpu_s"), "aborted": results.get("aborted"),
    "primary_K_1e-6_by_row": {p["id"]: p["K"][prim_tol] for p in primary},
    "primary_K_1e-6_max": (None if (not prim_K or None in prim_K) else max(prim_K)),
    "rows": rows,
}
mk = summary["primary_K_1e-6_max"]
summary["falsified"] = (mk is None) or (mk >= 16)
pos = next(r for r in rows if r["id"].startswith("smoke"))
neg = next(r for r in rows if r["id"].startswith("failctl"))
summary["positive_control_pass"] = pos["converged_parity_pass"] and pos["wavefront_all_ok"]
summary["failure_control_fails_as_required"] = not neg["converged_parity_pass"]
summary["all_wavefront_ok"] = all(r["wavefront_all_ok"] for r in rows)
out_json.write_text(json.dumps(summary, indent=1) + "\n")

fmt = lambda x: "—" if x is None else (f"{x:.1e}" if isinstance(x, float) else str(x))
lines = ["| Row | Condition / stream | dtype | T (N) | K(1e-4) | K(1e-6) | K(1e-9) | K(1e-12) | hidden err at K(1e-6) | wavefront | parity at convergence |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---|---|"]
for r in rows:
    lines.append("| `{}` | {} / {} | {} | {} ({}) | {} | {} | {} | {} | {} | {} | {} |".format(
        r["id"], r["condition"], r["stream"], r["dtype"], r["T"], r["N"],
        fmt(r["K"]["0.0001"]), fmt(r["K"]["1e-06"]), fmt(r["K"]["1e-09"]), fmt(r["K"]["1e-12"]),
        fmt(r["hidden_err_at_K"]["1e-06"]), "pass" if r["wavefront_all_ok"] else "FAIL",
        "pass" if r["converged_parity_pass"] else "fail"))
out_md.write_text("\n".join(lines) + "\n")
print(json.dumps({k: summary[k] for k in ("primary_K_1e-6_by_row", "primary_K_1e-6_max", "falsified",
                                          "positive_control_pass", "failure_control_fails_as_required",
                                          "all_wavefront_ok", "total_cpu_s", "aborted")}, indent=1))
