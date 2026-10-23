"""Post-run descriptive analysis. Does not select or change experiment settings."""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
NAMES = {"native": "GRU", "mingru_linear": "minGRU", "paragru": "ParaGRU"}
LABELS = {"backprop": "Backprop", "iid": "IID noise", "random_pair": "Random pairs",
          "sketch": "State sketch", "scrambled": "Scrambled sketch", "replay_backprop": "Extra replay backprop"}

def read(path):
    return json.loads(path.read_text())

def mean(values):
    return float(np.mean(values))

def main():
    terminal = read(OUT / "TERMINAL.json")
    assert terminal["status"] == "COMPLETE", terminal
    protocol = read(OUT / "PROTOCOL.json")
    assert terminal["source_seal_verified"]
    methods, seeds = protocol["methods"], protocol["seeds"]
    rows, arch, fingerprints = [], {}, {}
    for kind in NAMES:
        byseed = {}
        for seed in seeds:
            folder = OUT / f"{kind}-{seed}"
            byseed[seed] = {m: read(folder / f"{m}.json") for m in methods}
            rows.extend(byseed[seed].values())
            for probe in read(folder / "METRICS.json"):
                key = f"{probe['kind']}-{probe['k']}"
                item = fingerprints.setdefault(key, {"build_ms": [], "bytes": [], "selected_cosine": [],
                                                     "fast_cosine": [], "oracle_gap": [], "rmse": []})
                item["build_ms"].append(1000 * probe["build_median_seconds"])
                item["bytes"].append(probe["fingerprint_bytes"])
                for g in probe["groups"]:
                    item["selected_cosine"].append(g["selected_exact_cosine"])
                    item["oracle_gap"].append(g["best_on_candidate_set_exact_cosine"] - g["selected_exact_cosine"])
                    item["rmse"].append(g["selected_cosine_rmse"])
                    if g["group"] == "fast":
                        item["fast_cosine"].append(g["selected_exact_cosine"])
        contrasts = {}
        for control in ("backprop", "iid", "random_pair", "scrambled", "replay_backprop"):
            delta = [byseed[s]["sketch"]["followup"]["ce"] - byseed[s][control]["followup"]["ce"] for s in seeds]
            contrasts[control] = {"paired_seed_deltas_nats": delta, "mean_delta_nats": mean(delta),
                                  "min_delta_nats": min(delta), "max_delta_nats": max(delta),
                                  "wins": sum(d < 0 for d in delta)}
        metric = [100 * (byseed[s]["sketch"]["metric_seconds"] + byseed[s]["sketch"]["capture_seconds"]) /
                  byseed[s]["backprop"]["intervention_wall_seconds"] for s in seeds]
        wall = [100 * (byseed[s]["sketch"]["intervention_wall_seconds"] /
                      byseed[s]["backprop"]["intervention_wall_seconds"] - 1) for s in seeds]
        directly_timed = [100 * (byseed[s]["sketch"]["event_wall_seconds"] + byseed[s]["sketch"]["capture_seconds"]) /
                          byseed[s]["backprop"]["intervention_wall_seconds"] for s in seeds]
        checks = {
            "effect_vs_scrambled": contrasts["scrambled"]["mean_delta_nats"] <= -.005,
            "effect_vs_random_pair": contrasts["random_pair"]["mean_delta_nats"] <= -.005,
            "wins_vs_scrambled": contrasts["scrambled"]["wins"] >= 2,
            "wins_vs_random_pair": contrasts["random_pair"]["wins"] >= 2,
            "no_mean_harm_vs_backprop": contrasts["backprop"]["mean_delta_nats"] <= 0,
            "metric_overhead": mean(metric) <= 1,
            "total_wall_overhead": mean(wall) <= 10,
        }
        arch[kind] = {
            "parameters": byseed[seeds[0]]["backprop"]["parameters"],
            "mean_final_ce": {m: mean([byseed[s][m]["followup"]["ce"] for s in seeds]) for m in methods},
            "mean_intervention_ce": {m: mean([byseed[s][m]["intervention"]["ce"] for s in seeds]) for m in methods},
            "contrasts": contrasts, "metric_capture_overhead_percent": metric,
            "observed_training_wall_overhead_percent": wall, "direct_event_capture_fraction_percent": directly_timed,
            "guided_accepts": sum(byseed[s]["sketch"]["accepted"] for s in seeds),
            "replay_updates": [byseed[s]["replay_backprop"]["extra_replay_updates"] for s in seeds],
            "gate_checks": checks, "advance": all(checks.values()),
        }
    assert len(rows) == 54 and all(r["duplicate_exact"] for r in rows)
    fsummary = {key: {f"mean_{k}": mean(v) for k, v in value.items()} for key, value in fingerprints.items()}
    analysis = {"run_id": protocol["run_id"], "protocol_sha256": terminal["protocol_sha256"],
                "architectures": arch, "fingerprints": fsummary,
                "timing_aggregation": "Arithmetic mean of within-seed overhead ratios during the 96-update intervention; wall readings are descriptive, not replicated microbenchmarks.",
                "status": "ADVANCE" if any(a["advance"] for a in arch.values()) else "DO_NOT_ADVANCE_THIS_CONFIGURATION",
                "n_seeds_per_architecture": 3, "inferential_tests": "None; exploratory paired descriptive results only."}
    (OUT / "ANALYSIS.json").write_text(json.dumps(analysis, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})
    colors = {"native": "#2878A5", "mingru_linear": "#B56825", "paragru": "#7C5898"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), gridspec_kw={"width_ratios": [1.25, 1]})
    controls = ["backprop", "random_pair", "scrambled", "replay_backprop"]
    for i, (kind, a) in enumerate(arch.items()):
        shift = (i - 1) * .21
        for x, c in enumerate(controls):
            d = np.asarray(a["contrasts"][c]["paired_seed_deltas_nats"]) * 1000
            axes[0].scatter(x + shift + np.linspace(-.035, .035, len(d)), d, color=colors[kind], s=22, alpha=.55)
            axes[0].scatter(x + shift, d.mean(), marker="D", color=colors[kind], s=52,
                            label=NAMES[kind] if x == 0 else None)
        vals = [mean(a["metric_capture_overhead_percent"]), mean(a["direct_event_capture_fraction_percent"]),
                mean(a["observed_training_wall_overhead_percent"])]
        axes[1].bar(np.arange(3) + shift, vals, width=.19, color=colors[kind], label=NAMES[kind], alpha=.9)
    axes[0].axhline(0, color="#686868", lw=1)
    axes[0].set_xticks(range(4), ["Backprop", "Random\npairs", "Scrambled\nsketch", "Extra replay\nbackprop"])
    axes[0].set_ylabel("State-sketch CE − control CE (millinats/token)")
    axes[0].set_title("Held-out effect after 32 follow-up updates\nBelow zero favors the state sketch", loc="left", fontsize=12, pad=14)
    axes[0].legend(frameon=False, loc="best")
    axes[0].grid(axis="y", alpha=.15)
    axes[1].set_xticks(range(3), ["Sketch +\ncapture", "All timed\nextra work", "Observed\ntraining wall\ndifference"])
    axes[1].set_xlim(-.5, 2.5)
    axes[1].axhline(0, color="#686868", lw=1)
    axes[1].set_ylabel("% of paired backprop intervention time")
    axes[1].set_title("CPU cost, averaged over three seeds\nObserved wall difference includes timing noise", loc="left", fontsize=12, pad=14)
    axes[1].grid(axis="y", alpha=.15)
    fig.suptitle("Super-1: cheap structural anchors did not establish a useful perturbation gain" if not any(a["advance"] for a in arch.values()) else
                 "Super-1: state-anchored perturbation pilot", x=.055, ha="left", fontsize=16, weight="bold", y=.99)
    fig.text(.055, .025, "3 seeds per architecture • 173–178k parameters • WikiText-2 training documents split into train/held-out\nDots: individual seeds; diamonds: means. Predeclared useful effect: −5 millinats vs both structural controls.", fontsize=9, color="#555555")
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.79, wspace=.37)
    fig.savefig(OUT / "summary.png", dpi=180)
    fig.savefig(OUT / "summary.pdf")
    plt.close(fig)

    md = ["# Super-1: state-anchored perturbation pilot", "", f"Run: `{protocol['run_id']}`. Status: **{analysis['status']}**.", "",
          "Completed 54 paired branches: three seeds × three architectures × six conditions. This is a small CPU screening experiment, not a reproduction of the original full-scale Super-1 result. No settings were retuned after confirmation results.", "",
          "## Outcome", "", "The frozen advancement gate required at least 0.005 nats/token improvement over both scrambled anchors and random pairs, wins on at least two of three seeds against each, no mean harm versus backprop, sketch/capture overhead at most 1%, and total training wall overhead at most 10%.", "",
          "| Architecture | Parameters | Backprop CE | State-sketch CE | Δ vs backprop | Δ vs random pairs | Δ vs scrambled | Δ vs replay backprop | Gate |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for kind, a in arch.items():
        vals = [a["contrasts"][c]["mean_delta_nats"] for c in ("backprop", "random_pair", "scrambled", "replay_backprop")]
        md.append(f"| {NAMES[kind]} | {a['parameters']:,} | {a['mean_final_ce']['backprop']:.6f} | {a['mean_final_ce']['sketch']:.6f} | " + " | ".join(f"{v:+.6f}" for v in vals) + f" | {'PASS' if a['advance'] else 'FAIL'} |")
    if not any(a["advance"] for a in arch.values()):
        md += ["", "**No architecture passed the advancement gate.** This configuration does not justify a larger perturbation run. The passive fingerprint comparison and the downstream optimization test answer different questions: accurately measuring structural similarity is insufficient evidence that the proposed step improves learning."]
    md += ["", "CE is reserved document-held-out cross-entropy in nats/token after 96 intervention updates and 32 ordinary-backprop follow-up updates. Negative deltas favor the state sketch. Architecture comparisons are not the primary test: widths and parameter counts differ slightly. The primary comparisons are paired within each architecture and seed.", "",
           "All six conditions, mean final CE:", "", "| Condition | GRU | minGRU | ParaGRU |", "|---|---:|---:|---:|"]
    for m in methods:
        md.append(f"| {LABELS[m]} | " + " | ".join(f"{arch[k]['mean_final_ce'][m]:.6f}" for k in NAMES) + " |")
    md += ["",
           "![Paired effects and CPU costs](summary.png)", "", "## What was included from Rohan's repo", "",
           "Pinned upstream: [rohanarun/Super-1@492a87a5e263e806b0a2581c57259e81f1a000e6](https://github.com/rohanarun/Super-1/tree/492a87a5e263e806b0a2581c57259e81f1a000e6). Source files were copied byte-for-byte and checked against Git blob hashes in `UPSTREAM_PROVENANCE.json`.", "",
           "- Ordinary hierarchical GRU, using the merged batched `nn.GRU` fast path; historical tokenwise equivalence checked.",
           "- Linear minGRU with its parallel affine scan fast layer.",
           "- ParaGRU with the diagonal Newton solve/custom backward and upstream constraints.",
           "- The original segment/document GRU hierarchy, tied embedding/output, final normalization, and two-layer readout.",
           "- The merged AdamW parameter grouping, betas (0.9, 0.95), weight decay 0.1, and clipping. Peak LR was 0.002 with a declared short-run cosine schedule.",
           "- The repository's existing FineWeb-Edu BPE tokenizer with 2,048 vocabulary entries.", "",
           "These are the repository's fast-cell adaptations within the Super-1 slow hierarchy, not standalone reproductions of the full minGRU or ParaRNN language models. The pilot uses the merged reset-window training style. It does not include continuous learning, persistent full-document state, the historical 120M model, unrelated function-weight Transformer branches, teacher distillation, or Rohan's unpublished perturbation implementation. It tests the requested perturbation idea on the merged recurrent variants.", "",
           "## Mechanism and controls", "",
           "A consistent hidden-unit permutation is an exact reparameterization: it preserves the function. The experiment instead takes a small function-changing step along a coupled secant `P(theta) − theta`. Related units are proposed from cached activation relationships; success is never assumed from symmetry alone.", "",
           "Five coordinate groups cover the tied residual/embedding coordinates, fast state, segment state, document state, and readout intermediate units. Every connected weight, gate row, bias, recurrent column, and normalization parameter is transformed consistently. Two disjoint swaps are proposed per group. Activations include embedding, recurrent states, and readout activations; individual gate activations are not fingerprinted.", "",
           "The main treatment is CountSketch-64 over the observation axis, supplemented with bounded mean/variance/autocorrelation features. For observation-by-unit matrix X, observation sketch S, and hidden-coordinate permutation P, S(XP) = (SX)P. Renaming units therefore renames the descriptor columns; it does not invent a new geometry from their arbitrary indices. This is the structural anchor being tested.", "",
           "Descriptors are taken from four recent training windows, refreshed at intervention steps 32 and 96, and reused at step 64. These are cached activations from adjacent optimizer states, not a new full-model evaluation. Four random sorted projections build bounded pair shortlists; no full permutation group or all-pairs assignment is enumerated.", "",
           "At each event, two directions are tried with both signs, plus the unchanged parent. Each candidate has nominal global parameter L2 radius 0.003 relative to the current weights; ParaGRU constraints can adjust the actual final displacement. The best candidate is retained only if it improves CE on two sequences from the already-seen current training batch. Optimizer moments are retained because this is a parameter perturbation, not a coordinate relabeling. Four candidates are evaluated per event, three events per branch.", "",
           "| Condition | Purpose |", "|---|---|",
           "| Backprop | Identical nominal training schedule, no intervention |",
           "| IID noise | Same relative parameter norm and candidate evaluation budget |",
           "| Random pairs | Coupled permutation secants with random unit pairs |",
           "| State sketch | Current activation relationships choose nearby unit pairs |",
           "| Scrambled sketch | Same sketch algorithm with descriptors reassigned to different units |",
           "| Extra replay backprop | Reuses the same already-seen batch, spending approximately the guided event's wall budget on gradient updates |", "",
           "The replay control can overshoot by one replay optimizer step. Every realized replay count, overshoot, wall/CPU duration, and example exposure is recorded. This is approximate equal-time comparison on this CPU, not exact equal FLOPs or a GPU speed claim. 'No new tokens' means reusing examples: perturbation scoring still consumes forward-pass compute (15,360 target exposures per intervention branch).", "",
           "## Cheap metric benchmark", "",
           "Passive probes measured seven fingerprints on each of nine warmed parents. The table averages the nine parents. Build time excludes activation capture and candidate scoring. Exact trace cosine is the diagnostic target; it is not a measure of useful learning.", "",
           "| Fingerprint | Build ms | Descriptor KiB | Mean selected true cosine | Fast-state true cosine | Mean oracle gap on shortlist |",
           "|---|---:|---:|---:|---:|---:|"]
    for key, a in fsummary.items():
        md.append(f"| {key} | {a['mean_build_ms']:.2f} | {a['mean_bytes']/1024:.1f} | {a['mean_selected_cosine']:.3f} | {a['mean_fast_cosine']:.3f} | {a['mean_oracle_gap']:.3f} |")
    md += ["", "Each method produces its own bounded candidate set; the oracle gap compares its selected pairs to the best exact pairs within that same set. This is not full nearest-neighbor recall. The stored `shortlist_seconds` diagnostic includes exact-oracle scoring, so it is not used as an online overhead estimate. Exact descriptors are quick to materialize at this small size but retain much more data; the benchmark does not prove sketches are faster at every width.", "",
           "The FFT comparator uses log power in eight temporal frequency bands plus basic statistics. It discards phase and is not a test of cross-spectral coherence, phase-preserving FFT sketches, or every DSP approach. The DSP comparator uses mean/std, log std, and lag-1/4/16 autocorrelation. Only CountSketch-64 received an optimization intervention arm; passive metric rankings must not be described as downstream training gains for all seven methods.", "",
           "## Cost and acceptance", "",
           "| Architecture | Sketch + capture / baseline | All directly timed extra work / baseline | Observed training wall difference | Accepted guided events | Extra replay updates per seed |",
           "|---|---:|---:|---:|---:|---|"]
    for kind, a in arch.items():
        md.append(f"| {NAMES[kind]} | {mean(a['metric_capture_overhead_percent']):.2f}% | {mean(a['direct_event_capture_fraction_percent']):.2f}% | {mean(a['observed_training_wall_overhead_percent']):+.2f}% | {a['guided_accepts']}/9 | {a['replay_updates']} |")
    md += ["", "Ratios use the 96-update intervention period and are averaged within paired seeds. All directly timed extra work includes capture, descriptors, pair selection, candidate construction and forward scoring. Observed wall differences additionally reflect runtime variance in ordinary training. Fixed method order and only one timing per branch limit performance claims. The cheap metric and the complete perturbation search are distinct costs.", "",
           "## Reproducibility and limits", "",
           f"- Completed in {terminal['wall_seconds']/60:.2f} minutes main wall time, {terminal['cpu_seconds']/60:.2f} CPU minutes, on one CPU thread. Peak process RSS: {terminal['peak_rss_kib']/1024:.1f} MiB. Runtime: Python 3.12.14, PyTorch 2.8.0+cpu, NumPy 2.3.5.",
           "- Three independent parent seeds (73, 191, 307); 192 common warmup updates, then paired 96-update intervention and 32-update follow-up branches. Batch 4 × context 512. Roughly 173–178k parameters; early training only.",
           "- Only WikiText-2's training text is downloaded from a pinned commit. Exact documents are deduplicated; the first 16 hexadecimal digits of SHA256(document), modulo 100, assign train/monitor/confirmation. This yields train (482 documents; 3,161,532 tokens), monitor (65; 427,844), and confirmation (62; 464,356). Near-duplicate documents are not screened. No official validation or test split is read. Confirmation evaluates 32 fixed document-local windows (16,384 targets), not every held-out token. Sampling selects documents uniformly, so this is a window-sampling metric rather than corpus-token-weighted perplexity.",
           "- Confirmation never selects proposals, radii, dimensions, or checkpoints. Monitor is recorded at the parent only. Three seeds on one small held-out partition do not establish statistical significance or generality; no inferential p-values are reported.",
           "- All eight FP64 preflight cases passed: historical/batched forward, state, gradients and AdamW updates; parallel/sequential equivalence for every variant; whole-network permutation, sketch, proposal, and optimizer equivariance. All 54 final repeated evaluations were exactly identical in-process. A final audit reloaded one saved guided checkpoint per architecture in a fresh process: all three state hashes and held-out evaluations matched exactly. These checks establish numerical consistency for the exercised cases, not a formal proof for all states or tie cases.",
           "- Slow segment/document states have few updates per reset window. Refresh cadence was fixed; cached pair stability between refreshes was not measured. Autocorrelation of concatenated captured windows includes artificial joins. Similarity is diagnostic and does not by itself establish a favorable loss direction.",
           "- A null result here rejects advancement of this exact radius, pairing, cadence and budget. It does not disprove state-conditioned perturbations at another scale or after more pretraining. No radius sweep, phase-preserving FFT intervention, or GPU experiment was run.",
           "- The compact archive preserves source, tokenizer, document hashes, protocol, every branch's outcomes/candidates/loss trace, and final-state hashes. It omits re-downloadable corpus text, installed packages, and tensor checkpoints. Exact checkpoint recovery requires deterministic rerunning and a matching runtime; cross-platform bit identity is not promised.", "",
           f"Protocol SHA256: `{terminal['protocol_sha256']}`. See `ANALYSIS.json` for every paired delta and gate check, `PREFLIGHT.json` for numerical errors, and branch JSON files for all raw measurements.", "",
           "## Next decision", "",
           "This configuration failed its frozen advancement gate; the present pilot does not justify scaling it up. The descriptor may be useful as a cheap structural measurement even when its proposed perturbations add no measurable learning benefit. Any later phase-preserving or curvature-sensitive proposal should be registered as a separate experiment with fresh confirmation data, rather than tuned against this reserved result.", ""]
    (OUT / "REPORT.md").write_text("\n".join(md))
    print(json.dumps({"status": analysis["status"], "architectures": arch, "fingerprints": fsummary}, indent=2))

if __name__ == "__main__":
    main()
