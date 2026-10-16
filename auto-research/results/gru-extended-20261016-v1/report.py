"""Render only verified run metrics and write the repository report."""
import gzip,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2];RUN=ROOT.name
GRAPH=REPO/'auto-research/graphs'/RUN;GRAPH.mkdir(exist_ok=True,parents=True)
s=json.loads((ROOT/'SUMMARY.json').read_text());p=json.loads((ROOT/'PROTOCOL.json').read_text())
checks=json.loads((ROOT/'CHECKS.json').read_text());timing=json.loads((ROOT/'TIMING.json').read_text())['timings']
rows=s['ranking'];by={r['arm']:r for r in rows};native=by['native']
def read_curves(a):
 path=ROOT/'execution'/a/'curves.json'
 text=path.read_text() if path.exists() else gzip.decompress(path.with_suffix('.json.gz').read_bytes()).decode()
 return json.loads(text)['monitor']
curves={a:read_curves(a) for a in p['arms']}
(GRAPH/'inputs.json').write_text(json.dumps(dict(summary=s,monitor_curves=curves),indent=2)+'\n')
colors={'native':'#2452cb','mingru_linear':'#0d947d','paragru':'#d97706'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5.5))
for arm in p['arms']:
 vals=curves[arm]
 for ax,key in zip(axes,['training_targets','train_cpu_seconds']):
  x=[v[key]/1e6 if key=='training_targets' else v[key] for v in vals]
  ax.plot(x,[v['validation_ce'] for v in vals],color=colors[arm],lw=2,marker='o',ms=3,label=by[arm]['label'])
for ax in axes:
 ax.grid(alpha=.18);ax.set_ylabel('Monitoring cross-entropy · lower is better')
axes[0].set_xlabel('Training targets (millions)');axes[1].set_xlabel('Training CPU seconds · evaluation excluded')
axes[0].set_title('Same training exposure',loc='left',fontweight='bold');axes[1].set_title('Measured CPU cost',loc='left',fontweight='bold')
fig.suptitle('GRU follow-up · five times the training exposure',x=.065,ha='left',fontweight='bold',fontsize=16)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.055))
fig.text(.065,.02,'~1M parameters · 2,000 updates · seed 101602 · reset 512-token episodes · validation reused',fontsize=9,color='#555')
fig.subplots_adjust(left=.065,right=.98,top=.82,bottom=.25,wspace=.23)
fig.savefig(GRAPH/'loss_curves.png',dpi=180);fig.savefig(GRAPH/'loss_curves.svg');plt.close(fig)
table='\n'.join(f"| {r['label']} | {r['parameters']:,} | {r['ce']:.6f} | {r['perplexity']:.2f} | {r['ce_delta_vs_native']:+.4f} | {r['train_cpu_seconds']:.2f} | {r['training_cpu_speedup_vs_native']:.2f}× |" for r in rows)
bench=[]
for a in p['arms']:
 fast='batched' if a=='native' else 'parallel'
 fastms=timing[a][fast]['forward_backward']['median_seconds']*1000
 seqms=timing[a].get('sequential',{}).get('forward_backward',{}).get('median_seconds')
 speed=f'{seqms*1000/fastms:.2f}×' if seqms else 'N/A'
 bench.append(f"| {by[a]['label']} | {fastms:.2f} | {seqms*1000:.2f} | {speed} |" if seqms else f"| {by[a]['label']} | {fastms:.2f} | — | {speed} |")
decisions='\n'.join(f"- **{r['label']}: {r['decision']}**; CE gap {r['ce_delta_vs_native']:+.4f}, CPU speedup {r['training_cpu_speedup_vs_native']:.2f}×." for r in rows if r['arm']!='native')
mem='\n'.join(f"| {r['label']} | {r['weight_bytes']:,} | {r['state_bytes']:,} | {r['peak_training_rss_kib']/1024:.1f} |" for r in rows)
solver=checks['paragru'];prior=json.loads((ROOT/'prior_screen/SUMMARY.json').read_text())
oldby={r['arm']:r for r in prior['ranking']}
prior_table='\n'.join(f"| {by[a]['label']} | {oldby[a]['ce']:.4f} | {by[a]['ce']:.4f} |" for a in p['arms'])
report=f'''# Experiment report: {RUN}

Status: **completed exploratory comparison**, not confirmed architecture superiority.

Approval: user request on 2026-10-16 at15:02 America/Denver to continue linear minGRU/ParaGRU experiments and push code/results to their fork. This independent courtesy study is separate from upstream's official research program. The earlier hosted-workspace execution exception remains in force; no paid resources were provisioned.

Protocol SHA256: `{s['protocol_sha256']}`. [Frozen protocol](../results/{RUN}/PROTOCOL.json), [scientific template binding](../results/{RUN}/SCIENTIFIC_PROTOCOL.json), [registration](../results/{RUN}/REGISTRATION.json), [source hashes](../results/{RUN}/SOURCE_MANIFEST.json).

## Question and controls

Does linear minGRU or ParaGRU remain close to native GRU after fivefold training exposure while using less CPU? The competing explanation is that the small earlier gaps masked a larger advantage from dense hidden-to-hidden mixing as learning proceeds. Native GRU is the fresh control. All three retain the same learned slow hierarchy, tied output embedding and32-token/8-segment clocks, with widths147/native,160/minGRU,153/ParaGRU to match approximately1M parameters.

Predeclared exploratory rule: final CE no more than0.03 nats worse than native **and** at least1.25× lower training CPU time, with complete budgets and numerical checks. This threshold identifies a candidate for another small experiment; it is not a statistical confirmation or promotion gate.

## What actually ran

Three fresh runs, seed101602, **2,000 updates and1,024,000 target exposures per arm**. Batch1, context512, zero state at each episode; no carry across random windows. EOS remains a token, with no additional within-window reset. All use the same ordered windows, optimizer and final-checkpoint selection. AdamW peak0.002, warmup25, cosine over2,000 updates to0.0002, clip1, FP32. No teacher, no distillation, no auxiliary objective. Recurrent diagonal vectors are excluded from decay and projected to L2 norm<=0.5 in ParaGRU.

The initialization seed and cosine horizon differ from the earlier400-update screen. The first400 data windows are the same;1,600 more follow from the same RNG stream. Consequently, cross-study gains are not a controlled estimate of changing only training duration. No arm received extra tuning.

The preserved FineWeb-Edu sample and train-only2,048-token byte-BPE tokenizer are unchanged and hash-checked. Exact train/validation/tokenizer hashes, source revision and split rules are in [data metadata](../results/{RUN}/data_metadata/) and [offsets](../results/{RUN}/OFFSETS.json). Monitoring uses4,096 targets; final ranking uses16,384 targets in a distinct validation region. Both regions were already used in earlier research. No official test was opened.

Execution used the existing CPU workspace, three concurrent workers with one pinned CPU thread each, PyTorch2.14.1+cpu and NumPy2.5.4. Aggregate measured training CPU time was **{s['total_training_cpu_seconds']:.2f}seconds**. All workers finished normally within900CPU-second/1,800wall-second per-arm caps. Setup, evaluation, analysis and serialization are excluded from training-time columns. All checkpoints include model, optimizer, RNG and full training metrics. [Checkpoint hashes](../results/{RUN}/CHECKPOINT_MANIFEST.json), [terminal receipt](../results/{RUN}/TERMINAL.json).

## Results

Final update2,000, CE in nats per BPE token; lower is better. PPL is exp(CE) for this tokenizer and cannot be compared to unrelated word-level benchmarks.

| Model | Parameters | CE ↓ | PPL ↓ | CE gap vs native | Training CPU seconds ↓ | Native/model CPU ratio |
|---|---:|---:|---:|---:|---:|---:|
{table}

![Learning curves](../graphs/{RUN}/loss_curves.png)

Curves show monitoring windows, so their endpoints differ from the final-ranking table. This single-seed comparison has no inferential error bars; paired window counts in [SUMMARY.json](../results/{RUN}/SUMMARY.json) are descriptive, not seed replication.

Historical context only:

| Model | Earlier400-update CE, seed101601 | Current2,000-update CE, seed101602 |
|---|---:|---:|
{prior_table}

The previous rotating-bank/curve score5.7008 is from the shorter budget and is **not** a control for this ranking. It was not retrained here. Positive-candidate minGRU was omitted after its preceding poor result. Historical summary/protocol/checks are retained under [prior_screen](../results/{RUN}/prior_screen/), with no claim that its checkpoints were retrained or re-evaluated.

## Deployment measurements

One-thread CPU benchmarks ran after all training workers exited, with two warmups and five repeats at the final weights. Full-model forward+cross-entropy+backward at512 tokens is reported below; it excludes optimizer updates. New-cell sequential/parallel paths share all projections and readout, so their pair measures recurrence execution including its backward implementation. Native already uses batched `nn.GRU` and readout; its dense recurrence remains sequential.

| Model | Batched/scan median ms ↓ | Same-cell sequential median ms | Scan speedup |
|---|---:|---:|---:|
{chr(10).join(bench)}

[Raw timings](../results/{RUN}/TIMING.json) also include no-grad forward,128 stateful one-token calls, all samples and empirical p95. Five repeats are an exploratory latency sample, not a production tail estimate. Streaming uses supplied known tokens and computes all logits; sampling/tokenization is excluded.

| Model | FP32 parameter bytes | Persistent state backing bytes | Peak training process RSS, MiB |
|---|---:|---:|---:|
{mem}

Persistent state excludes two Python counters and object overhead. Training process RSS includes Python/runtime/optimizer/activations and is not an inference-memory estimate. Dedicated deployment peak activation/workspace and allocator-reserved memory were not measured. State bytes must not be presented as total model memory.

Slow-state feedback limits both new scans to32-token chunks; this is not full-sequence parallel training. The direct affine tree uses O(T log T) work. ParaGRU uses diagonal recurrent matrices, nonlinear gates, Newton iterations and a reverse scan adjoint, and is a changed architecture from dense GRU. No GPU, large-model scaling, quantization, long-context training or retrieval result is established.

## Decision

{decisions}

Ordinary GRU is the comparator, not a newly promoted model. These decisions apply only to the recorded budget/seed/wrapper; different initializations, full published blocks or another schedule remain alternative explanations. The next useful small architecture test is a4-token causal input mixer for linear minGRU, with the no-mixer version and native GRU as controls at the same parameter budget. This would test whether missing local token interactions explain part of the loss gap. Expected cost is of the same order as this run (roughly600aggregate CPU seconds at2,000updates), to be remeasured before launch. No further experiment is running or queued.

## Integrity and cleanup

Initial and final sequential/parallel logits and gradients passed declared tolerances. Prefix/split continuation checks cross both slow-memory clocks. Every one of the32 final windows per arm replayed from the saved checkpoint within1e-6CE; maximum observed error was {max(v['replay_max_ce_error'] for v in checks.values()):.3g}. Final gradient relative L2 error was at most{max(v['gradient_relative_l2'] for v in checks.values()):.3g}. ParaGRU solver iteration histogram: `{solver['solver_histogram']}`; maximum residual{solver['max_solver_residual']:.3g}. No fallback, failed training arm or nonfinite result was hidden. [Checks](../results/{RUN}/CHECKS.json), [preflight](../results/{RUN}/PREFLIGHT.json).

Code, metrics, figures, logs and hashes are committed under `auto-research/`. Large checkpoints and token arrays are in a companion artifact outside Git; consult [preservation manifest](../results/{RUN}/PRESERVATION.json). That artifact is retained for the contributor, not exposed as a public dataset download. Anyone replaying the run needs the hash-matching arrays/checkpoints. [Reproduction commands](../results/{RUN}/README.md), [sources](../results/{RUN}/SOURCES.md), [cleanup](../results/{RUN}/CLEANUP.json).

This branch does not modify upstream model files or old evidence. The user requested a push to their fork; no upstream PR or merge was performed by this run.
'''
for before,after in {
 'at15:02':'at 15:02','and32-token/8-segment':'and 32-token/8-segment',
 'widths147/native,160/minGRU,153/ParaGRU':'widths 147/native, 160/minGRU, 153/ParaGRU',
 'approximately1M':'approximately 1M','than0.03':'than 0.03','least1.25':'least 1.25',
 'seed101602':'seed 101602','and1,024,000':'and 1,024,000','Batch1, context512':'Batch 1, context 512',
 'peak0.002, warmup25':'peak 0.002, warmup 25','over2,000':'over 2,000','to0.0002, clip1':'to 0.0002, clip 1',
 'earlier400-update':'earlier 400-update','first400':'first 400',';1,600':'; 1,600',
 'train-only2,048':'train-only 2,048','uses4,096':'uses 4,096','uses16,384':'uses 16,384',
 'PyTorch2.14.1':'PyTorch 2.14.1','NumPy2.5.4':'NumPy 2.5.4',
 'within900CPU-second/1,800wall-second':'within 900 CPU-second / 1,800 wall-second',
 'update2,000':'update 2,000','score5.7008':'score 5.7008','at512':'at 512',
 'forward,128':'forward, 128','to32-token':'to 32-token','is a4-token':'is a 4-token',
 'roughly600aggregate':'roughly 600 aggregate','at2,000updates':'at 2,000 updates',
 'the32 final':'the 32 final','within1e-6CE':'within 1e-6 CE','at most':'at most ',
 'maximum residual':'maximum residual ',
}.items():report=report.replace(before,after)
(REPO/f'auto-research/reports/{RUN}.md').write_text(report)
readme=f'''# {RUN}

Read [the completed report](../../reports/{RUN}.md). This is an independent, one-seed, approximately1M-parameter comparison, not an official Super-1 study or a reproduction of its historical120M benchmark.

Use the pinned versions in `requirements-lock.txt`. Models require PyTorch/NumPy; charts additionally need Matplotlib. The CPU PyTorch index is https://download.pytorch.org/whl/cpu.

From the repository root, with the companion archive's `data/train.bin` and `data/validation.bin` available:

```bash
python auto-research/results/{RUN}/launch.py --data-dir /path/to/data
python auto-research/results/{RUN}/analyze.py --data-dir /path/to/data
python auto-research/results/{RUN}/report.py
```

The launcher skips completed arms; incomplete runs resume model/optimizer/RNG state. To deliberately retrain, use a separate checkout/output copy and move the existing `execution/` directory aside. Keep frozen protocols and offsets. Do not rerun `prepare.py` over this study. The committed `execution/<arm>/curves.json.gz` files contain complete per-step records; `analyze.py` uses the companion `checkpoint.pt` files and the committed RESULT.json files. The chart script reads either compressed or uncompressed curves. [PRESERVATION.json](PRESERVATION.json) identifies external artifacts and hashes.

`models.py` supports `forward`, `step`, `transition`, and `last_logits`; the latter two wrapper methods were completed for this contribution without changing the trained forward computation. The inherited native MAC estimate is deliberately unavailable for the replacement cells. New cores are standalone equation implementations, not copied external CUDA kernels. Unchanged native source and its license are under `vendor/`.

Primary source/protocol files are protected by `SOURCE_MANIFEST.json`; all new contribution files are covered by the repository manifest. `REGISTRATION.json` records the protocol freeze before launch; the registry was updated locally before execution and will be pushed with completed results as the user requested. This was not a public preregistration.
'''
(ROOT/'README.md').write_text(readme)
print('WROTE',REPO/f'auto-research/reports/{RUN}.md')
