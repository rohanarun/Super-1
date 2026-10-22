"""Build a standalone evidence report after the fixed CPU experiment finishes."""
import base64
from html import escape
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'research-scratch/cpu-124m/main'
OUT = ROOT/'artifacts/cpu-124m'


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, a, p = [read(RUN/name) for name in ('summary.json','assessment.json','protocol.json')]
    curves = read(RUN/'curves.json')
    train = [json.loads(line) for line in (RUN/'training.jsonl').read_text().splitlines()]
    gens = [json.loads(line) for line in (RUN/'generations.jsonl').read_text().splitlines()]
    benches = [json.loads(line) for line in (ROOT/'research-scratch/cpu-124m/benchmarks.jsonl').read_text().splitlines()]
    selection = read(ROOT/'research-scratch/cpu-124m/selection.json')
    environment = read(ROOT/'research-scratch/cpu-124m/environment.json')
    notes_path = ROOT/'research-scratch/cpu-124m/interpretation.json'
    notes = read(notes_path)
    assert len(gens) == 10 and s['status'] == 'COMPLETED'
    # Different evaluation sample sizes remain visually distinguishable.
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig, axes = plt.subplots(1, 3, figsize=(13,3.6), constrained_layout=True)
    x = np.array([r['elapsed_seconds']/60 for r in train])
    y = np.array([r['training_nll'] for r in train])
    axes[0].plot(x,y,color='#8dbbc2',alpha=.4,lw=.8,label='Training batch')
    window=min(10,len(y))
    if len(y)>=window:
        axes[0].plot(x[window-1:],np.convolve(y,np.ones(window)/window,'valid'),color='#177e89',label=f'{window}-step mean')
    endpoints=[r for r in curves if r['evaluation_tokens']==8192]
    monitors=[r for r in curves if r['evaluation_tokens']==2048]
    axes[0].plot([r['elapsed_seconds']/60 for r in endpoints],[r['nll'] for r in endpoints],ls='none',marker='s',color='#342b75',label='Development: 8,192 tokens')
    axes[0].plot([r['elapsed_seconds']/60 for r in monitors],[r['nll'] for r in monitors],ls='none',marker='o',color='#b75731',label='Development: 2,048 tokens')
    axes[0].set(xlabel='Main training time (minutes)',ylabel='Next-token NLL (nats)',title='Learning from scratch')
    axes[0].legend(fontsize=7,frameon=False)
    elapsed=np.array([r['elapsed_seconds'] for r in train])
    tokens=np.array([r['scored_tokens'] for r in train])
    axes[1].plot(elapsed/60,tokens/elapsed,color='#177e89',lw=2)
    axes[1].set(xlabel='Main training time (minutes)',ylabel='Training targets / second',title='Cumulative throughput')
    axes[1].text(.04,.08,'Includes periodic evaluation and saves',transform=axes[1].transAxes,fontsize=8,color='#556270')
    axes[2].plot(x,np.array([r['rss_peak_bytes']/2**30 for r in train]),color='#342b75',lw=2,label='Peak process RSS')
    axes[2].axhline(environment['memory_limit_bytes']/2**30,color='#b75731',ls='--',label='Container limit')
    axes[2].set(xlabel='Main training time (minutes)',ylabel='GiB',title='Training memory',ylim=(0,9))
    axes[2].legend(fontsize=8,frameon=False)
    for ax in axes:
        ax.grid(alpha=.15)
    image_path=OUT/'minKron_CPU_25min_training.png'
    fig.savefig(image_path,dpi=180)
    fig.savefig(OUT/'minKron_CPU_25min_training.svg')
    plt.close(fig)
    image='data:image/png;base64,'+base64.b64encode(image_path.read_bytes()).decode()
    c=a['confirmation']
    throughput=s['throughput_full_loop_tokens_per_second']
    total_training_tokens=s['scored_tokens']
    decode=a['generation']['generated_tokens']/a['generation']['decode_seconds']
    fallback=100*s['stable_fallback_calls']/s['factored_core_calls']
    metric_rows=[
      ('Actual parameters',f"{s['parameters']:,}"),
      ('Main training loop',f"{s['elapsed_seconds']/60:.2f} minutes; {s['steps']:,} updates"),
      ('Training targets',f"{total_training_tokens:,}; no repeated windows"),
      ('Full-loop / training-step throughput',f"{throughput:.1f} / {s['throughput_optimizer_tokens_per_second']:.1f} targets/s"),
      ('Peak process RSS',f"{s['rss_peak_bytes']/2**30:.2f} GiB of 8 GiB"),
      ('FP32 weights / AdamW state',f"{s['parameter_bytes']/1e6:.1f} / {s['optimizer_bytes']/1e6:.1f} MB"),
      ('Recurrent state per sequence',f"{s['state_bytes_per_sequence']/1024:.0f} KiB across all 12 layers"),
      ('Initial → final development NLL',f"{curves[0]['nll']:.4f} → {s['final_development']['nll']:.4f} on 8,192 targets"),
      ('Final confirmation NLL / perplexity',f"{c['nll']:.4f} / {c['perplexity']:.1f} on {c['evaluation_tokens']:,} targets"),
      ('Final confirmation token accuracy',f"{100*c['token_accuracy']:.2f}%"),
      ('Nonfinite failures / clipped updates',f"{s['nonfinite_count']} / {s['clipped_steps']} of {s['steps']}"),
      ('Stable-core fallbacks',f"{s['stable_fallback_calls']:,} / {s['factored_core_calls']:,} calls ({fallback:.1f}%, including evaluations)"),
      ('Generation decode throughput',f"{decode:.1f} tokens/s, batch 1, across ten samples"),
      ('Training targets per parameter',f"{total_training_tokens/s['parameters']:.6f}"),
    ]
    architecture=[]
    for name,mlp in [('h12',4352),('h24',4096),('h48',3584),('h72',3072)]:
        row=next(b for b in benches if b['layout']==name and b['precision']=='fp32' and b['threads']==4 and b['batch']==1 and b['length']==128 and b['optimizer']=='foreach' and b.get('core','stable')=='stable')
        architecture.append([name+(' — selected' if name=='h48' else ''),f"{row['params']:,}",mlp,row['output_channels_per_layer'],f"{row['state_bytes']/1024:.0f} KiB",f"{row['median_tokens_per_second']:.1f}"])
    pilot_table=table(['Peak LR','Updates','Dev NLL','Main-loop targets/s'],[[r['lr'],r['steps'],f"{r['development_nll']:.4f}",f"{r['throughput']:.1f}"] for r in selection['all_pilots']])
    benchmark_table=table(['Heads','Core','Type','Threads','AdamW','Batch × length','Chunk','Median targets/s','Peak GiB'],[[b['layout'],b.get('core','stable'),b['precision'],b['threads'],b['optimizer'],f"{b['batch']} × {b['length']}",b['chunk'],f"{b['median_tokens_per_second']:.1f}",f"{b['rss_peak_bytes']/2**30:.2f}"] for b in benches])
    samples=''.join('<article class="sample"><div class="sample-title">'+str(g['number'])+' · '+escape(g['prompt'])+'</div><pre>'+escape(g['continuation'])+'</pre><div class="meta">Seed '+str(g['seed'])+' · '+str(g['generated_tokens'])+' new tokens · '+escape(g['stop_reason'])+'</div></article>' for g in gens)
    d=a['recurrence_diagnostic']
    findings=''.join('<p>'+escape(note)+'</p>' for note in notes['thoughts'])
    core_check=a['trained_chunk_vs_recurrent']
    style='''body{margin:0;background:#f2f5f7;color:#20313e;font:16px/1.6 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:42px 28px 64px;background:white}h1{font-size:40px;line-height:1.12;letter-spacing:-1.3px;margin:12px 0 20px}h2{font-size:25px;margin-top:42px}h3{font-size:18px}.eyebrow{letter-spacing:1.5px;color:#177e89;text-transform:uppercase;font-size:12px;font-weight:750}.lead{font-size:20px;max-width:920px}.note,.meta{color:#5e6e79;font-size:13px}.callout{border-left:4px solid #177e89;background:#f0f7f8;padding:16px 22px;margin:26px 0}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #dbe3e8;vertical-align:top}th{color:#177e89;background:#f2f6f7;font-size:12px}td:first-child{font-weight:600}img{width:100%;height:auto}.sample{border:1px solid #dbe3e8;border-radius:8px;padding:18px 22px;margin:15px 0;break-inside:avoid}.sample-title{font-weight:700;color:#177e89}.sample pre{font:15px/1.65 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.meta{margin-top:8px}code{overflow-wrap:anywhere}a{color:#176d85}footer{border-top:1px solid #dbe3e8;margin-top:42px;padding-top:20px;font-size:12px;color:#5e6e79}@media print{body{background:white}main{padding:0}h2{break-after:avoid}.table{overflow:visible}h1{font-size:32px}}'''
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>minKron · 124M CPU scale experiment</title><style>{style}</style></head><body><main>
    <div class="eyebrow">Experiment record · 21 October 2026 · CPU only</div>
    <h1>123.38M parameters.<br>Twenty-five minutes on CPU.</h1>
    <p class="lead">{escape(notes['lead'])}</p>
    <div class="callout">{s['scored_tokens']:,} training targets · {throughput:.1f} targets/s end to end · {s['rss_peak_bytes']/2**30:.2f} GiB peak RAM · {c['perplexity']:.1f} held-out perplexity</div>
    <p>A NanoGPT-style decoder with minKron replacing attention: 12 layers, width 768, MLP width 3,584, tied GPT-2 embedding/head, pre-RMSNorm and ReLU². Each layer has 48 independent 8 × 8 minKron states. Training began from random weights.</p>
    <h2>Measured result</h2>{table(['Measurement','Result'],metric_rows)}
    <img src="{image}" alt="Training and held-out loss, cumulative training throughput, and peak memory over the 25-minute run">
    <p class="note">Training batches differ from held-out examples. Square markers use 8,192 development targets; round monitoring markers use only the first 2,048 and should not be treated as exactly comparable values. Throughput counts only training targets. Setup, pilots, initial/final evaluation, generation and final save are outside the 25-minute training-loop budget.</p>
    <h2>What the model has learned</h2>{findings}
    <p>The empirical unigram baseline, fitted only to targets actually seen by training with add-one smoothing, scored NLL {a['unigram_baseline']['nll']:.4f} on the same confirmation targets. On a separate 2,048-target diagnostic, preserving recurrence scored {d['normal_nll']:.4f}; clearing state before every token scored {d['reset_every_token_nll']:.4f}. The difference was {d['context_advantage_nats']:.4f} nats. This diagnostic measures a local contribution from history, not long-term memory.</p>
    <h2>Resource choices and CPU speedups</h2>
    {table(['Layout','Parameters','MLP width','Readout channels/layer','State/sequence','Probe targets/s'],architecture)}
    <p>The head spread changes recurrent capacity while keeping the parameter budget near 124M. The quoted row timings use the stable core, FP32, four threads and 1 × 128 tokens. They are short random-token systems probes, not language-quality comparisons. No other layout received a 25-minute training run.</p>
    <p>The host exposes a Xeon Platinum 8573C with an eight-core CPU quota and 8 GiB of RAM. PyTorch 2.9.1+cpu reports an AVX-512 backend. Four threads outperformed the tested two/eight-thread options. FP32 and foreach AdamW won the relevant probes. A factored GEMM implementation reduces pairwise temporary storage when safe, with an original-core fallback for large decay ranges. The chosen 2 × 256 layout briefly reached 203.4 targets/s on random tokens; that is not the sustained training result. Trained decays increased fallback frequency.</p>
    <p>Output/state/input-gradient checks covered FP32/FP64, resets, nonzero state, zero/tiny values and fallback cases. A three-update decoder check differed by at most 4.45e-6 in parameters. On the final trained model, chunked versus recurrent prompt evaluation passed, with maximum logit difference {core_check['logits_max_abs']:.3g}.</p>
    <details><summary>All 18 systems probes</summary>{benchmark_table}<p class="note">Each probe used one warmup and two or three measured updates in its own process. These are noisy microbenchmarks. Rows that change multiple settings do not isolate a causal speedup.</p></details>
    <h2>Learning-rate selection</h2>{pilot_table}
    <p>Each pilot used the same initialization, order, 8,192 training targets and 2,048 development targets. Lowest finite final development NLL selected 0.001. Main training restarted from scratch. Three earlier starts stopped before training because of a logging-key collision; their logs are retained. No pilot checkpoint or confirmation result was used to initialize or choose the main model.</p>
    <h2>Ten fixed generations</h2>
    <p>These are the exact continuations from the final timed checkpoint. Prompts and seeds were recorded before training. Temperature 0.8; top-k 50; maximum 64 new GPT-2 tokens; stop on EOT. There were no retries, filtering or selection of better-looking samples.</p>{samples}
    <h2>Data, scope and reproducibility</h2>
    <p>Five million training targets were prepared from the first shard of <a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/9bb295ddab0e05d785b879661af7260fed5140fc/sample/10BT">FineWeb sample-10BT at a pinned revision</a>. Document text hashes assign disjoint training/development/confirmation splits; exact text and IDs are deduplicated globally. Training uses a seeded permutation of independent 256-token windows, resetting at window starts and EOT inputs. Only {100*total_training_tokens/5000000:.2f}% of that available training pool was used. Near-duplicate overlap is not ruled out. This is a custom split, not the official speedrun validation set.</p>
    <p>The run uses one seed and contains no trained Transformer/minGRU comparator. It measures this implementation and early learning under this CPU budget. It does not establish architecture superiority, convergence, long-context quality or a scaling law. Holding this average throughput constant, 20M training targets would take about {20000000/throughput/3600:.1f} hours; this is a rough time extrapolation, not a quality prediction.</p>
    <p>The source archive contains code, tokenized data, document indices, every benchmark/pilot, source hashes, training logs and all sample token IDs. The separate checkpoint includes FP32 weights, AdamW state, scheduler, RNG states and data cursor. See <code>docs/CPU_124M.md</code> for reproduction. Wall-time scheduling prevents a claim of bitwise resume equivalence across different machine load.</p>
    <footer>Final checkpoint SHA256: <code>{a['checkpoint']['sha256']}</code><br>Checkpoint size: {a['checkpoint']['bytes']:,} bytes. Original GPU comparison contract remains unchanged. All numerical outcomes above come from this run's retained logs.</footer>
    </main></body></html>'''
    (OUT/'minKron_124M_CPU_25min_Report.html').write_text(html, encoding='utf-8')
    (OUT/'minKron_124M_CPU_10_Generations.txt').write_text((RUN/'generations.txt').read_text(),encoding='utf-8')
    print(json.dumps(dict(report=str(OUT/'minKron_124M_CPU_25min_Report.html'),chart=str(image_path),samples=len(gens)),indent=2))


if __name__=='__main__':
    main()
