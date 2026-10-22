"""Derive the report and plots from completed receipts, without model execution."""
import csv
import gzip
import json
import statistics as stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, ARMS, SEEDS, write_json, sha

LABELS={'k8':'8x8 baseline','k16':'16x16','k32':'32x32','k16_lean':'16x16, half MLP',
    'k16_lean_conv':'16x16, half MLP + conv4','transformer_lean':'Transformer, half MLP'}

def main():
    m=json.loads((ROOT/'METRICS.json').read_text());term=json.loads((ROOT/'TERMINAL.json').read_text())
    assert term['status']=='COMPLETED'
    conf={(r['stage'],r['arm'],r['seed']):r['first']['ce'] for r in m['confirmation']}
    screen={(r['arm'],r['seed']):r for r in m['screen']}
    summary=[]
    for arm in ARMS:
        rows=[screen[arm,s] for s in SEEDS]
        ce=[conf['screen',arm,s] for s in SEEDS]
        delta=[conf['screen',arm,s]-conf['screen','k8',s] for s in SEEDS]
        dev_delta=[screen[arm,s]['development']['ce']-screen['k8',s]['development']['ce'] for s in SEEDS]
        summary.append(dict(arm=arm,parameters=rows[0]['parameters'],development_ce=stats.mean(r['development']['ce'] for r in rows),
            confirmation_ce=stats.mean(ce),confirmation_ce_sd=stats.stdev(ce),per_seed_confirmation_ce=ce,
            paired_confirmation_delta=delta,mean_paired_confirmation_delta=stats.mean(delta),
            paired_confirmation_delta_sd=stats.stdev(delta),mean_paired_development_delta=stats.mean(dev_delta),
            exploratory_development_advance_gate=stats.mean(dev_delta)<=-.02 and all(d<0 for d in dev_delta),
            confirmation_consistent=stats.mean(delta)<=-.02 and all(d<0 for d in delta),
            training_tokens_per_second=sum(r['tokens'] for r in rows)/sum(r['training_seconds'] for r in rows),
            mean_screen_seconds=stats.mean(r['training_seconds'] for r in rows)))
    extensions=[dict(arm=r['arm'],tokens=r['total_tokens'],additional_tokens=r['tokens'],
        total_training_seconds=r['total_training_seconds'],extension_seconds=r['training_seconds'],
        tokens_per_second=r['tokens_per_second'],development_ce=r['development']['ce'],
        confirmation_ce=conf['extension',r['arm'],r['seed']]) for r in m['extensions']]
    write_json(ROOT/'SUMMARY.json',dict(screen=summary,extensions=extensions,
        total_target_exposures=sum(r['tokens'] for r in m['screen']+m['extensions']),
        total_training_seconds=term['total_training_seconds'],selection=m['selection']['selected_recurrent_variant']))
    # Compact complete per-step scalar trajectories, including timed LR schedules.
    with gzip.open(ROOT/'ALL_STEPS.csv.gz','wt',newline='') as f:
        writer=None
        for path in sorted((ROOT/'runs').glob('*_steps.json')):
            for row in json.loads(path.read_text()):
                if writer is None: writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                writer.writerow(row)
    graphs=ROOT.parents[1]/'graphs'/ROOT.name;graphs.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    colors=['#64748b','#2563eb','#7c3aed','#0891b2','#059669','#ea580c']
    for i,row in enumerate(summary):
        axes[0].scatter([i-.13,i,i+.13],row['per_seed_confirmation_ce'],color=colors[i],s=28)
        axes[0].plot([i-.22,i+.22],[row['confirmation_ce']]*2,color=colors[i],linewidth=2.5)
    axes[0].set_xticks(range(6),['8x8','16x16','32x32','16 lean','16 lean\n+ conv','Transformer'])
    axes[0].set_ylabel('Confirmation cross-entropy (nat/token; lower is better)')
    axes[0].set_title('Equal exposure: 190,464 tokens per model\nThree seeds; points are individual results')
    for r in extensions:
        i=ARMS.index(r['arm']);before=conf['screen',r['arm'],SEEDS[0]]
        axes[1].plot([0,r['extension_seconds']/60],[before,r['confirmation_ce']],marker='o',color=colors[i],label=LABELS[r['arm']])
    axes[1].set_xlabel('Additional training minutes after the screen')
    axes[1].set_ylabel('Confirmation cross-entropy (nat/token)')
    axes[1].set_title('Equal additional time: 5 minutes\nSeed 25101 only; endpoints, not continuous evaluations')
    axes[1].legend(loc='best',fontsize=9)
    fig.savefig(graphs/'comparison.png',dpi=170);fig.savefig(graphs/'comparison.svg');plt.close(fig)
    write_json(graphs/'plot_inputs.json',dict(screen=summary,extensions=extensions))
    title=f'# Small minKron CPU experiments: {ROOT.name}\n\n'
    text=title+f'''Status: completed exploratory architecture screen and continuation. AI-assisted implementation, execution and analysis.

The user authorized smaller, faster minKron experiments on 2026-10-21. No GPU, external compute account, spending, sealed official test, automation or merge was used.

Protocol SHA256: `{sha(ROOT/'PROTOCOL.json')}`. Read the protocol with both numbered addenda; they were recorded before language training. Exact executable source hashes are in SOURCE_MANIFEST.json.

## Question and controls

Does increasing the matrix head size, halving the feed-forward width, or adding a short causal convolution improve a small all-recurrent language model's quality/cost tradeoff? The 8x8 baseline is the primary comparator. A RoPE Transformer with the lean MLP supplies a practical baseline. None of these are pretrained or instruction-tuned.

All arms use 3 layers, residual width192, vocabulary4096, tied embedding/head, RMSNorm, squared-ReLU MLP, 128-token training windows, batch8 and 1024 targets/update. Recurrent readout width is96: 12 heads of8x8, 6 of16x16, or3 of32x32. MLP hidden width is768 or384. Conv4 is depthwise on q/k/v, identity-initialized and reset at document boundaries. Parameters and FLOPs are not matched. Three paired seeds: 25101,25102,25103; one shared LR recipe with no per-arm tuning.

## What actually ran

Each of18 screen runs completed186 updates/190,464 targets. The baseline, development-selected recurrent variant `{m['selection']['selected_recurrent_variant']}`, and Transformer then continued for300 training seconds each on seed25101. Fixed-token endpoints and fixed-time continuations answer different questions. Continuations resume the next training window and optimizer, with a shared time-based cosine LR restart; total lifetime training seconds differ because screening times differ.

CPU: Linux, 8-core quota/8GiB memory,4 PyTorch threads, torch2.9.1+cpu, Python3.12, tiktoken0.14.0. Total timed training: {term['total_training_seconds']/60:.2f} minutes. Setup, synthetic qualification, evaluations, serialization and publication are additional. No CUDA kernels or GPU speed claim.

Two failed numerical checks are retained. The first caught Adam amplifying FP32 rounding error; epsilon was changed from1e-8 to1e-4 for every arm. The second caught a gradient mismatch. Recurrent normalization/decay/matrix updates/state/output normalization therefore run in FP64; projection weights, embedding, MLP and logits remain FP32. The Transformer is FP32. The original error tolerances were never loosened. All24 final qualification rows passed output/state/parameter-gradient/optimizer-step/streaming/reset/causality checks, including nonzero convolution taps. This precision choice adds recurrent compute and state bytes.

Data: the same document-disjoint FineWeb slice used in the124M pilot, decoded and reencoded losslessly with GPT-2's first4095 merge ranks plus EOT. No vocabulary fitting to heldout data. Training contains6,757,905 small-vocabulary tokens. Development and confirmation regions were already used in prior project work: this is internal ablation evidence, not a fresh transfer test. Loss/perplexity is not comparable with the prior50,257-token vocabulary. State resets at window starts and EOT; no long-context or persistent cross-window capability claim.

## Results

Development used16,384 targets; confirmation used32,768, scored twice from each frozen checkpoint. All confirmation duplicate differences are <=1e-7. Lower cross-entropy is better.

| Arm | Parameters | Dev CE mean | Confirmation CE mean ± sample SD | Training tokens/s | Mean screen seconds |
|---|---:|---:|---:|---:|---:|
'''
    for r in summary:text+=f"| {LABELS[r['arm']]} | {r['parameters']:,} | {r['development_ce']:.4f} | {r['confirmation_ce']:.4f} ± {r['confirmation_ce_sd']:.4f} | {r['training_tokens_per_second']:,.0f} | {r['mean_screen_seconds']:.2f} |\n"
    text+='\nEvery confirmation seed, in nat/token:\n\n| Arm | Seed25101 | Seed25102 | Seed25103 | Paired mean Δ vs8x8 | SD of paired Δ |\n|---|---:|---:|---:|---:|---:|\n'
    for r in summary:text+=f"| {LABELS[r['arm']]} | "+' | '.join(f'{v:.4f}' for v in r['per_seed_confirmation_ce'])+f" | {r['mean_paired_confirmation_delta']:+.4f} | {r['paired_confirmation_delta_sd']:.4f} |\n"
    text+='\nFive-minute continuations (seed25101 only):\n\n| Arm | Total tokens seen | Additional tokens | Total training seconds | Final dev CE | Final confirmation CE |\n|---|---:|---:|---:|---:|---:|\n'
    for r in extensions:text+=f"| {LABELS[r['arm']]} | {r['tokens']:,} | {r['additional_tokens']:,} | {r['total_training_seconds']:.2f} | {r['development_ce']:.4f} | {r['confirmation_ce']:.4f} |\n"
    text+='\nThree-seed uncertainty is descriptive. Six arms were screened and one variant selected using development; no significance or familywise-error claim. The continuation has only one seed and a shorter learning history than a serious capability run. SAMPLES.json preserves every fixed-prompt completion, with no cherry-picking or retries.\n'
    text+='\n## Deployment measurements\n\nSee DEPLOYMENT.json: FP32 weight bytes, actual persistent tensor/storage bytes, batch1 prefill128 and decode64 p50/p95. Two warmups; five measured prefills. Peak process RSS in run receipts includes training/runtime allocations. Isolated inference process peak and activation/workspace peaks were not measured; persistent state is not total memory.\n'
    text+='\n## Decision\n\nExploratory gate: >=0.02 nat/token mean development gain over8x8, with all three paired seeds agreeing. Confirmation is a reused-region consistency check.\n\n'
    for r in summary[1:]:text+=f"- {LABELS[r['arm']]}: development gate {'MET' if r['exploratory_development_advance_gate'] else 'NOT MET'}; confirmation consistency {'MET' if r['confirmation_consistent'] else 'NOT MET'}.\n"
    text+='\nNo result promotes a general architecture-superiority, reasoning, long-memory or SOTA claim. A stronger next comparison should use more equal-token training, more than three seeds if gaps remain small, fresh document-disjoint evaluation and a separately qualified FP32 kernel. A further run is not scheduled or running.\n'
    text+='\n## Integrity and cleanup\n\nTERMINAL.json records actual completion. CHECKPOINTS.json contains retained local checkpoint hashes; corpus arrays/checkpoint bodies are deliberately excluded from Git, with RECONSTRUCTION_REQUIRED.md. Full scalar trajectories including actual LR values are compacted in ALL_STEPS.csv.gz. Source, tokenizer and data hashes are preserved. The prior archived experiment is unchanged. No official test was opened. CLEANUP.json records process completion; no unrelated workload or artifact was removed.\n'
    report=ROOT.parents[1]/'reports'/f'{ROOT.name}.md';report.write_text(text)
    print(json.dumps(dict(screen=summary,extensions=extensions),indent=2))

if __name__=='__main__':main()
