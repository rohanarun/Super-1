"""Render monitor curves and report exact endpoint values without mixing the two."""
import gzip
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
PARENT=ROOT.parent/'gru-extended-20261016-v1'
def load_curve(path):
    return json.loads(path.read_text()) if path.exists() else json.loads(gzip.decompress(path.with_suffix('.json.gz').read_bytes()))

s=json.loads((ROOT/'SUMMARY.json').read_text());native,old,continued,fresh=s['rows']
graph=REPO/'auto-research/graphs'/ROOT.name;graph.mkdir(parents=True,exist_ok=True)
series=[('Ordinary GRU',load_curve(PARENT/'execution/native/curves.json'),'#293e65','-'),
        ('minGRU: early decay + extension',load_curve(ROOT/'execution/curves.json'),'#a4783d','--'),
        ('minGRU: full-budget decay',load_curve(ROOT/'fresh_execution/curves.json'),'#07816c','-')]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
                     'axes.titleweight':'bold','svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={'width_ratios':[1.5,1]})
for label,c,color,style in series:
    x=[r['train_cpu_seconds'] for r in c['monitor']];y=[r['validation_ce'] for r in c['monitor']]
    for ax in axes:ax.plot(x,y,label=label,color=color,linestyle=style,marker='o',markersize=3,lw=2)
for ax in axes:
    ax.grid(alpha=.18);ax.set_xlabel('Training CPU seconds · one thread');ax.set_ylabel('Monitoring cross-entropy · lower is better')
    ax.axvline(s['target_cpu_seconds'],color='#5c6570',ls=':',alpha=.65)
axes[0].set(xlim=(0,265),ylim=(4.3,8.4),title='Full training trajectory')
axes[0].legend(frameon=False,fontsize=9,loc='upper right')
tail=[r['validation_ce'] for _,c,_,_ in series for r in c['monitor'] if r['train_cpu_seconds']>=110]
axes[1].set(xlim=(110,262),ylim=(min(tail)-.08,max(tail)+.08),title='Later training · zoom')
axes[1].axvline(old['cpu_seconds'],color='#a4783d',ls=':',alpha=.6)
fig.suptitle('Equal CPU budget: linear minGRU versus ordinary GRU',fontsize=15,fontweight='bold')
fig.text(.5,.018,'~1M parameters · same seed and data stream · 4,096 monitoring targets; endpoint ranking uses separate 16,384 validation targets',ha='center',fontsize=9,color='#535b65')
fig.tight_layout(rect=(0,.055,1,.93))
fig.savefig(graph/'loss_curves.png',dpi=180);fig.savefig(graph/'loss_curves.svg');plt.close(fig)
table='\n'.join(f"| {r['label']} | {r['updates']:,} | {r['training_targets']:,} | {r['cpu_seconds']:.2f} | {r['ce']:.6f} | {r['perplexity']:.2f} |" for r in s['rows'])
outcome='lower' if fresh['ce']<native['ce'] else 'higher'
nano=s['short_nanogpt_context'];early=s['early_equal_time']
body=f'''# Equal-CPU GRU comparison

Linear minGRU with a fresh full-budget learning-rate schedule finishes at **{fresh['ce']:.6f} CE**, {abs(fresh['ce_delta_vs_native']):.6f} nats {outcome} than ordinary GRU at **{native['ce']:.6f}**. Both spend approximately **{s['target_cpu_seconds']:.2f} single-thread training CPU seconds**. The simpler continuation reaches **{continued['ce']:.6f}**, showing that the schedule materially affects how much benefit the extra updates provide.

This is an independent, single-seed explorer experiment in the contributor's fork, separate from official Super-1 studies. No official test data, GPU allocation or paid resources were used.

## Endpoint results

| Model / schedule | Updates | Target exposures | CPU seconds | Final CE | Perplexity |
|---|---:|---:|---:|---:|---:|
{table}

All models have approximately 1M parameters: native {native['parameters']:,}, linear {fresh['parameters']:,}. CE is in nats per 2048-BPE token. All endpoint rows use the same 32 held-out windows /16,384 targets; no best-checkpoint selection. The 2000-step linear row is an equal-token reference, not an equal-time endpoint. Native and that reference are immutable reused results from [the previous study](gru-extended-20261016-v1.md).

Fresh linear sees **{fresh['training_targets']/native['training_targets']:.2f}×** as many target exposures and changes perplexity by **{fresh['perplexity_change_percent']:+.2f}%** relative to native. It has lower CE on **{fresh['better_windows']}/32** final windows; continuation has lower CE on **{continued['better_windows']}/32**. Those are descriptive comparisons, not confidence intervals or independent replications. The budgets are within{fresh['budget_unused_seconds']:.3f}s and{continued['budget_unused_seconds']:.3f}s of the native budget, respectively, and both stop below it.

## Loss versus time

![Loss curves](../graphs/{ROOT.name}/loss_curves.png)

The graph shows the separate 8-window monitoring set, not the endpoint-ranking set. Joined markers interpolate visually between recorded observations. At{early['cpu_seconds']:.2f}s, the earlier linear checkpoint has monitor CE{early['linear_monitor_ce']:.6f}; linearly interpolating native's saved800/1000-step observations gives{early['native_interpolated_monitor_ce']:.6f}. That interpolated value is not a directly evaluated checkpoint. Final monitoring and final endpoint CE can rank a near tie differently; the full 32-window endpoint is the declared ranking metric.

## What ran

1. **Budget extension.** Load the exact linear step 2000 model, AdamW moments and Torch RNG. Continue the same predeclared random-window stream at its terminal LR 0.0002. Count the original{old['cpu_seconds']:.6f}s in the total budget. Freeze this procedure before continuation.
2. **Fresh full-budget schedule.** Initialize the same model at seed 101602 from scratch. Keep AdamW, parameter count, objective, input order and 25-update warmup unchanged. Spread cosine decay from0.002 to0.0002 across measured CPU time remaining after warmup. Freeze this additional arm before its training; the continuation had not completed when the fresh arm was registered.
3. **Native control.** Reuse the previous fresh seed 101602 run:2000 updates, warmup 25 then step-based cosine over its full 2000 updates. Its measured 256.559083s defines the time target. Fresh linear's clock-based cosine is approximately the same full-budget shape, not an identical update-indexed schedule.

The timer includes batch construction, model forward/backward, clipping, AdamW and constraints. It excludes validation, checkpoint writes, startup and plotting. It measures training CPU time, not theoretical FLOPs or end-to-end speedrun time. Workers use one CPU thread and FP32; historical native and new runs have the same software/host but were not simultaneous, so load, clock and cache effects remain possible. Stop decisions depend on time, never validation loss. Reproduction will take a different number of updates if host speed changes.

The model is unchanged: input-only sigmoid gates and signed linear candidates permit an affine scan in the fast cell; slow segment/document GRUs still provide feedback every 32 tokens / every 8 segments. Scans therefore cover 32-token chunks. The previous final-weight benchmark found no CPU speed gain from linear minGRU's scan versus its own sequential implementation. Its measured advantage over ordinary GRU mainly reflects the cheaper cell; a fused GPU scan and longer parallel chunks remain untested.

## nanoGPT implications

The earlier common-recipe 400-update CPU pilot used the same 1M scale and seed 101601. Linear minGRU: CE{nano['linear_ce']:.6f} in{nano['linear_cpu_seconds']:.2f}s; nanoGPT: CE{nano['nanogpt_ce']:.6f} in{nano['nanogpt_cpu_seconds']:.2f}s. That is **{nano['throughput_ratio']:.2f}×** training throughput in this small harness, with better endpoint loss. These short results are context, not a new equal-time nanoGPT result or a tuned-best-recipe comparison. Their source receipts are included under `references/`.

The [public modded-nanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) defines a FineWeb validation target on 8 H100 GPUs (README checked 2026-10-16). Our 1M CPU/FineWeb-Edu/2048-BPE results do not establish performance against that target. The practical next speed experiment is matched time-to-target-loss with the same data, tokenizer, hardware and timing boundaries, followed by GPU kernel work if resources are authorized. Ordinary GRU remains the stronger equal-token reference; linear minGRU is a promising time-budget candidate under this recipe.

## Reproduce and inspect

Keep this folder beside `gru-extended-20261016-v1`; its model/vendor files are hash-bound in the frozen protocols. Obtain the contributor-held `gru_extended_checkpoints_20261016.zip` for exact original checkpoints and train/validation arrays (SHA256 `c90bb924b3b69e49bf217582776e7add78292d8c72e8a2fa5e82bd28a6db80a6`). The tokenizer and source data recipe are in the parent study. No data arrays or checkpoints are embedded in Git.

The new `gru_equal_cpu_checkpoints_20261016.zip` preserves both final model/optimizer/RNG states; see [PRESERVATION.json](../results/{ROOT.name}/PRESERVATION.json) for its hash. Place its checkpoint files at the specified relative paths. The saved final windows replay within1e-6; parallel/sequential logits and all parameter gradients pass the parent tolerances. Protocol, input/source hashes, offset-prefix continuity, complete traces, learning-rate schedule and actual time budgets are checked. Native replay is inherited from the unchanged parent study.

```bash
python auto-research/results/{ROOT.name}/analyze.py --data-dir /path/to/exact/data
python auto-research/results/{ROOT.name}/report.py
```

To rerun training, use a separate copy/output directory and remove only that copy's `execution/` and `fresh_execution/` artifacts. Keep the frozen protocols and code. Run `run.py --data-dir ... --cpu 0` for continuation and `fresh.py --data-dir ... --cpu 1` for the fresh arm. Do not rerun preparation on this completed study or overwrite its evidence. Hardware-time scheduling prevents bit-identical fresh-run reproduction from just the seed; the exact observed update/LR trace is preserved in `curves.json.gz`, and checkpoint evaluation is deterministic here.

The parent protocol's inherited prose mistakenly says the ranking uses step 400; its runner, traces and checkpoints used 2000. This report uses the actual2000-step control and leaves the historical evidence unchanged.

Raw evidence: [summary](../results/{ROOT.name}/SUMMARY.json), [continuation protocol](../results/{ROOT.name}/PROTOCOL.json), [fresh protocol](../results/{ROOT.name}/FRESH_PROTOCOL.json), [checks](../results/{ROOT.name}/CHECKS.json). One seed, small models, repeatedly reused validation and short reset 512 episodes limit broader scaling or long-context conclusions.
'''
report=REPO/'auto-research/reports'/f'{ROOT.name}.md';report.write_text(body)
print(report)
