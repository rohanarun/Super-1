"""Descriptive, paired analysis of the frozen phase-annealing experiment."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
NAMES={'native':'GRU','mingru_linear':'minGRU','paragru':'ParaGRU'}
PLACEMENTS={'pre':('sa_pre','extra_pre','baseline_keep'),
            'middle_keep':('sa_mid_keep','extra_mid_keep','baseline_keep'),
            'middle_reset':('sa_mid_reset','extra_mid_reset','baseline_reset'),
            'post':('sa_post','extra_post','baseline_keep')}
LABELS={'baseline_keep':'BP A → BP B, Adam retained','baseline_reset':'BP A → reset Adam → BP B',
        'sa_mid_keep':'BP A → SA → BP B, Adam retained','sa_mid_reset':'BP A → SA → reset Adam → BP B',
        'extra_mid_keep':'BP A → extra BP → BP B, Adam retained','extra_mid_reset':'BP A → extra BP → reset Adam → BP B',
        'greedy_mid_keep':'BP A → greedy search → BP B','sa_post':'BP A → BP B → SA',
        'extra_post':'BP A → BP B → extra BP','sa_pre':'SA → BP A → BP B','extra_pre':'Extra BP → clear Adam → BP A → BP B'}
def read(p):return json.loads(p.read_text())
def mean(x):return float(np.mean(x))
def contrast(values):return {'deltas_nats':values,'mean_nats':mean(values),'min_nats':min(values),'max_nats':max(values),'wins':sum(x<0 for x in values)}

def main():
    t=read(OUT/'TERMINAL.json');assert t['status']=='COMPLETE' and t['source_seal_verified']
    p=read(OUT/'PROTOCOL.json');seeds=p['seeds'];methods=p['methods'];summary={};raw={}
    for kind in NAMES:
        byseed={s:{m:read(OUT/f'{kind}-{s}'/(m+'.json')) for m in methods} for s in seeds};raw[kind]=byseed
        a={'parameters':byseed[seeds[0]]['baseline_keep']['parameters'],
           'mean_final_ce':{m:mean([byseed[s][m]['final']['ce'] for s in seeds]) for m in methods},'placements':{}}
        for placement,(sa,extra,base) in PLACEMENTS.items():
            values=[byseed[s][sa]['final']['ce']-byseed[s][extra]['final']['ce'] for s in seeds]
            baseline=[byseed[s][sa]['final']['ce']-byseed[s][base]['final']['ce'] for s in seeds]
            phases=[byseed[s][sa]['phase'] for s in seeds]
            a['placements'][placement]={
                'versus_extra_bp':contrast(values),'versus_baseline':contrast(baseline),
                'mean_sa_phase_wall_seconds':mean([r['wall_seconds'] for r in phases]),
                'mean_sa_phase_fraction_of_baseline_percent':mean([100*byseed[s][sa]['phase']['wall_seconds']/byseed[s][base]['total_training_wall_seconds'] for s in seeds]),
                'mean_observed_total_training_overhead_percent':mean([100*(byseed[s][sa]['total_training_wall_seconds']/byseed[s][base]['total_training_wall_seconds']-1) for s in seeds]),
                'mean_sa_vs_extra_total_wall_ratio':mean([byseed[s][sa]['total_training_wall_seconds']/byseed[s][extra]['total_training_wall_seconds'] for s in seeds]),
                'extra_bp_updates':[byseed[s][extra]['phase']['updates'] for s in seeds],
                'accepted':[r['accepted'] for r in phases],'uphill_accepted':[r['uphill_accepted'] for r in phases],
                'returned_initial_count':sum(r['returned_initial'] for r in phases),
                'unchanged_baseline_relative_win':all(r['returned_initial'] for r in phases) and mean(baseline)==0 and mean(values)<=-.005,
                'calibration_ce_changes':[r['returned_loss']-r['initial_loss'] for r in phases],
                'temperatures':[r['initial_temperature'] for r in phases],
                'advance':mean(values)<=-.005 and sum(x<0 for x in values)>=2 and mean(baseline)<=0}
        a['sa_vs_greedy_middle']=contrast([byseed[s]['sa_mid_keep']['final']['ce']-byseed[s]['greedy_mid_keep']['final']['ce'] for s in seeds])
        a['reset_effect_without_sa']=contrast([byseed[s]['baseline_reset']['final']['ce']-byseed[s]['baseline_keep']['final']['ce'] for s in seeds])
        a['reset_effect_with_sa']=contrast([byseed[s]['sa_mid_reset']['final']['ce']-byseed[s]['sa_mid_keep']['final']['ce'] for s in seeds])
        a['sa_middle_immediate_heldout_changes']=[byseed[s]['sa_mid_keep']['boundary']['ce']-byseed[s]['baseline_keep']['boundary']['ce'] for s in seeds]
        summary[kind]=a
    all_rows=[r for kinds in raw.values() for seed in kinds.values() for r in seed.values()]
    assert len(all_rows)==99 and all(r['duplicate_exact'] for r in all_rows)
    primary=any(a['placements']['middle_keep']['advance'] for a in summary.values())
    any_gate=any(v['advance'] for a in summary.values() for v in a['placements'].values())
    substantive_gate=any(v['advance'] and not v['unchanged_baseline_relative_win']
                         for a in summary.values() for v in a['placements'].values())
    result={'run_id':p['run_id'],'protocol_sha256':t['protocol_sha256'],'primary_passed_any_architecture':primary,
            'any_exploratory_gate_passed':any_gate,'architectures':summary,
            'status':'PRIMARY_ADVANCE' if primary else ('EXPLORATORY_GATE_ONLY' if any_gate else 'DO_NOT_ADVANCE_THIS_CONFIGURATION'),
            'practical_interpretation':('Replicate the qualifying effect before scaling.' if substantive_gate else
                'Do not scale this configuration: no qualifying advantage that improved the model rather than returning an unchanged baseline.'),
            'inference':'Paired descriptive results, n=3 per architecture. No p-values or claims of significance; placement comparisons beyond middle/keep are exploratory.'}
    (OUT/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'native':'#2878A5','mingru_linear':'#B56825','paragru':'#7C5898'}
    fig,axes=plt.subplots(1,2,figsize=(12,5.6))
    placements=['pre','middle_keep','post'];ticks=['Before both\nBP runs','Between\nBP runs','After both\nBP runs']
    for i,(kind,a) in enumerate(summary.items()):
        offset=(i-1)*.22
        for panel,comparison in enumerate(['versus_extra_bp','versus_baseline']):
            for j,placement in enumerate(placements):
                d=np.asarray(a['placements'][placement][comparison]['deltas_nats'])*1000
                axes[panel].scatter(j+offset+np.linspace(-.035,.035,3),d,s=25,alpha=.5,color=colors[kind])
                axes[panel].scatter(j+offset,d.mean(),s=55,marker='D',color=colors[kind],label=NAMES[kind] if j==0 else None)
    for ax,title in zip(axes,['Compared with extra backprop at the same boundary','Compared with no annealing at that boundary']):
        ax.axhline(0,lw=1,color='#666666');ax.axhline(-5,lw=1,ls=':',color='#999999')
        ax.set_xticks(range(3),ticks);ax.set_xlim(-.5,2.5);ax.set_ylabel('SA CE − control CE (millinats/token)')
        ax.set_title(title,loc='left',fontsize=11,pad=15);ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False)
    fig.suptitle('Super-1: simulated annealing as a separate training phase',x=.07,ha='left',fontsize=16,weight='bold',y=.98)
    fig.text(.07,.055,'Below zero favors annealing. Dots: paired seeds; diamonds: means. Dotted line: −5 millinat effect threshold.\n3 seeds × GRU / minGRU / ParaGRU • 173–178k parameters • Adam retained in the main sandwich comparison',fontsize=9,color='#555555')
    fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.24,wspace=.30)
    fig.savefig(OUT/'summary.png',dpi=180);plt.close(fig)

    md=['# Super-1: separate-phase simulated annealing','',
        '**Practical interpretation:** '+result['practical_interpretation'],'',
        f"Run `{p['run_id']}`. Frozen numerical gate status: **{result['status']}**.",'',
        'All 99 paired outcomes completed: three architectures × three seeds × eleven conditions. Annealing was a standalone, forward-only phase before all BP, between two complete short BP runs, or after both BP runs. No permutation descriptors or structural matching were used.','',
        '## Outcome','',
        'Primary test: BP A → SA → BP B with Adam state retained, versus spending approximately the same phase wall time on additional BP. The frozen usefulness gate requires at least 0.005 nats/token mean improvement over extra BP, wins in at least two of three seeds, and no mean harm versus the no-SA baseline. Other placements and reset comparisons are exploratory.','',
        '| Architecture | Placement | Δ CE vs extra BP | Wins / 3 | Δ CE vs no-SA | Gate |',
        '|---|---|---:|---:|---:|---|']
    for kind,a in summary.items():
        for placement,v in a['placements'].items():
            md.append(f"| {NAMES[kind]} | {placement} | {v['versus_extra_bp']['mean_nats']:+.6f} | {v['versus_extra_bp']['wins']} | {v['versus_baseline']['mean_nats']:+.6f} | {'PASS' if v['advance'] else 'FAIL'} |")
    nochange=[NAMES[k]+' / '+pl for k,a in summary.items() for pl,v in a['placements'].items() if v['advance'] and v['unchanged_baseline_relative_win']]
    if nochange:
        md+=['','**Relative gate without a model improvement:** '+', '.join(nochange)+'. These SA arms returned the unchanged baseline in all three seeds. Their apparent advantage comes from extra BP worsening held-out loss. This is not an annealing learning gain and does not justify scaling the SA phase.']
    md+=['','Negative deltas favor annealing. Every final score is evaluated on the same fixed 16,384 held-out targets. Comparisons are paired within an architecture and seed; absolute CE across architectures is not the primary test.','',
         '![Paired held-out effects](summary.png)','',
         'The primary sandwich gate '+('passed for at least one architecture.' if primary else 'did not pass for any architecture.')+
         (' At least one exploratory placement passed its descriptive gate with a model change; it requires a separately registered replication.' if substantive_gate and not primary else
          ' The exploratory numerical passes only returned unchanged baselines; they do not support scaling annealing.' if any_gate and not substantive_gate else ''),'',
         '## All conditions','',
         '| Condition | GRU mean CE | minGRU mean CE | ParaGRU mean CE |','|---|---:|---:|---:|']
    for m in methods:md.append('| '+LABELS[m]+' | '+' | '.join(f"{summary[k]['mean_final_ce'][m]:.6f}" for k in NAMES)+' |')
    md+=['','The no-op boundary is a checkpoint handoff, not idle sleep. Its identity was checked by reproducing the exact next Adam update. Every arm has the same scheduled A-to-B learning-rate restart. Only the explicit reset arms discard Adam moments and step counts. Thus a general learning-rate restart cannot by itself explain an SA-versus-matched-control difference.','',
         '## Annealing mechanism','',
         'Each SA phase uses a fixed batch of four 512-token training windows. The first A batch is used for pre-SA, the last A batch for middle SA, and the last B batch for post-SA. Parent and candidates are evaluated on exactly the same examples with recurrent state reset. The energy is mean training cross-entropy; it has no added weight regularizer.','',
         'The phase freezes a Gaussian proposal scale for each parameter tensor at max(entry RMS, 0.01). All eligible tensors receive independent zero-mean noise. Relative standard deviation cools geometrically from 0.03 to 0.003 over 96 proposals. Eight discarded proposals from the unchanged entry model calibrate the initial temperature: median positive loss increase divided by −log(0.7); if none are positive, median absolute change is used. Temperature cools to 1% of its initial value. The Metropolis acceptance rule is min(1, exp(−ΔCE/T)).','',
         'The chain can retain uphill moves. The returned endpoint is the lowest-loss accepted chain state, including the original entry. Temperature probes never select an endpoint. Both the final chain loss and returned-best loss are logged so exploration cannot be confused with the eventual selection. The T=0 middle control uses the same probes, proposal scales, schedule, random Gaussian draws and number of trials; it rejects every strictly uphill move.','',
         'For ParaGRU, only its constrained recurrent diagonal vectors `fast_cell.A` are held fixed during SA. All other trainable parameters remain eligible. This avoids a projection that would make the Gaussian proposal asymmetric at the constraint boundary. Ordinary BP continues to update and constrain A.','',
         'No backward call or optimizer update occurs during SA. Adam state is hashed before and after every phase and must remain exactly unchanged. Reset arms restart Adam only after the middle phase. Pre-extra-BP clears Adam before run A so its comparison with pre-SA concerns the learned starting weights rather than inherited optimizer memory.','',
         '## Did the chain actually explore?','',
         '| Architecture | Placement | Accepted / 288 trials | Uphill accepted | Returned original / 3 | Mean calibration CE change | Initial temperatures |',
         '|---|---|---:|---:|---:|---:|---|']
    for kind,a in summary.items():
        for placement in ['pre','middle_keep','post']:
            v=a['placements'][placement]
            md.append(f"| {NAMES[kind]} | {placement} | {sum(v['accepted'])} | {sum(v['uphill_accepted'])} | {v['returned_initial_count']} | {mean(v['calibration_ce_changes']):+.6f} | "+', '.join(f'{x:.4g}' for x in v['temperatures'])+' |')
    md+=['','The returned calibration loss cannot exceed the phase entry loss because the entry is an eligible fallback. That does not guarantee improved held-out loss or better subsequent BP. A returned-original result means this particular chain did not find a better calibration state; it is not evidence that all annealing schemes fail. The terminal chain endpoint followed by BP was not a separate arm: this experiment tests the best-calibration-checkpoint return rule. A relative win over extra BP caused only by extra-BP overfitting, with SA returning the unchanged baseline, must not be presented as an annealing learning gain.','',
         '## Optimizer resets and accepting uphill moves','',
         '| Architecture | SA − greedy after BP B | Reset − retain without SA | Reset − retain with SA | Immediate middle-SA held-out change |',
         '|---|---:|---:|---:|---:|']
    for kind,a in summary.items():
        md.append(f"| {NAMES[kind]} | {a['sa_vs_greedy_middle']['mean_nats']:+.6f} | {a['reset_effect_without_sa']['mean_nats']:+.6f} | {a['reset_effect_with_sa']['mean_nats']:+.6f} | {mean(a['sa_middle_immediate_heldout_changes']):+.6f} |")
    md+=['','These are descriptive paired differences. The SA/greedy comparison isolates the acceptance policy within the declared proposal family. It is not a replication of the earlier state-sketch perturbation experiment.','',
         '## Cost','',
         '| Architecture | Placement | SA phase seconds | Phase cost / baseline training | Observed total overhead | SA / extra-BP total wall ratio | Extra BP updates per seed |',
         '|---|---|---:|---:|---:|---:|---|']
    for kind,a in summary.items():
        for placement in ['pre','middle_keep','post']:
            v=a['placements'][placement]
            md.append(f"| {NAMES[kind]} | {placement} | {v['mean_sa_phase_wall_seconds']:.2f} | {v['mean_sa_phase_fraction_of_baseline_percent']:.1f}% | {v['mean_observed_total_training_overhead_percent']:+.1f}% | {v['mean_sa_vs_extra_total_wall_ratio']:.3f} | {v['extra_bp_updates']} |")
    md+=['','Each SA phase charges 106 forwards: entry, eight temperature probes, 96 chain proposals, and one returned-state verification. That is 217,088 target exposures on reused training examples. It consumes no new examples at the middle/post boundary, but is not token-free computation.','',
         'Extra BP repeatedly uses that identical phase batch until the measured SA wall budget is met. At most one update overshoot is expected and recorded. Extra learning rates are 0.002 before A, 0.0002 after A, and 0.0001 after B. This is approximate equal measured wall time, not exact equal FLOPs. Fixed execution order, shared parent timings, CPU variability and different numbers of repeated example exposures limit stronger efficiency claims. Evaluation and I/O time are excluded from training comparisons but included in the total main-run runtime.','',
         '## Protocol and scope','',
         '- Architectures: byte-exact ordinary GRU, linear minGRU and ParaGRU fast-cell adaptations within the pinned Super-1 slow hierarchy; widths 48/52/50, approximately 173–178k parameters. Upstream tokenizer and optimizer grouping are retained.',
         '- BP run A: 192 updates, LR warmup 16, peak 0.002, cosine decline to 0.0002. BP run B: 96 updates, warmup 8, peak 0.001, cosine decline to 0.0001. Every update uses batch 4 and context 512. Both are complete short scheduled runs; neither is claimed to reach convergence or represent full-scale pretraining.',
         '- Seeds 419, 563 and 701. Each architecture/seed shares exact initial weights and the same nominal batch sequences across conditions. Middle variants fork from identical run-A weights and moments. Post variants fork from the identical no-SA run-B endpoint.',
         '- Only pinned WikiText-2 TRAIN is used. Training has 482 documents / 3,161,532 tokens. Confirmation has 38 documents / 234,654 tokens, drawn from the previous monitor partition after excluding all 27 documents the old fixed monitor sampler evaluated. Those 38 documents were not trained on or evaluated by the previous pilot. The previous confirmation partition is entirely excluded. Exact document hashes and selection logic are preserved.',
         '- Confirmation uses 32 fixed document-local windows, 16,384 targets, sampled with a new RNG seed. It is a window-sampling CE measure, not corpus-token-weighted perplexity. Exact-document deduplication is applied; near-duplicate screening is not. No official validation or test data were read.',
         '- No confirmation score selects candidates, temperatures, checkpoint endpoints, radii, or training duration. Temperature calibration uses training data only. No settings were changed after the protocol was sealed or after confirmation results were viewed.',
         '- The inference unit is the parent seed, not each of the 99 branches. There are three seeds per architecture. Multiple exploratory comparisons and one small corpus preclude claims of statistical significance or generality.',
         '- This tests one finite Gaussian Metropolis policy, one temperature calibration rule, one radius schedule, one small calibration batch and a best-endpoint return rule. It is not SANTA, gradient-noise training, a theorem of global convergence, exact posterior sampling, or a claim about all simulated annealing.',
         '- Optimizing a repeatedly reused small batch can overfit it. Lower calibration loss is not itself learning progress. There is no sweep over phase duration, temperature, proposal geometry, or late-training saturation.',
         f"- Main runtime: {t['wall_seconds']/60:.2f} wall minutes, {t['cpu_seconds']/60:.2f} CPU minutes; one CPU thread; peak RSS {t['peak_rss_kib']/1024:.1f} MiB. Python 3.12.14, PyTorch 2.8.0+cpu, NumPy 2.3.5. The cap was 30 minutes.",
         '- Seven preflight checks passed: pinned source hashes, Metropolis formula, synthetic uphill-chain/best-state behavior, forward-only SA and Adam preservation for all three architectures, exact no-op next-update identity, and fresh confirmation separation. Every final in-process repeat evaluation matched exactly. See ARTIFACT_AUDIT.json for checkpoint reload verification.',
         '- The compact deliverable includes source, tokenizer, document hashes, frozen protocol, every outcome/loss trace/candidate decision, timing records and state hashes. It omits downloadable corpus text, token arrays, installed dependencies and tensor checkpoints. Equal-time extra-BP counts vary with machine speed; cross-platform bit identity is not promised.','',
         '## Sources and provenance','',
         '- [Pinned upstream Super-1](https://github.com/rohanarun/Super-1/tree/492a87a5e263e806b0a2581c57259e81f1a000e6). Source paths and Git blob hashes are in UPSTREAM_PROVENANCE.json.',
         '- [Embedding Simulated Annealing within Stochastic Gradient Descent](https://link.springer.com/chapter/10.1007/978-3-030-85672-4_1), Fischetti and Stringher, 2021: related hybrid-optimization precedent. Our separate-phase Gaussian experiment does not reproduce its gradient-informed neighborhood method.',
         '- [Bridging the Gap between Stochastic Gradient MCMC and Stochastic Optimization](https://proceedings.mlr.press/v51/chen16c.html), Chen et al., 2016: SANTA-related background. SANTA itself was not implemented here.','',
         f"Protocol SHA256: `{t['protocol_sha256']}`. See ANALYSIS.json for every paired seed delta and gate result; per-branch JSON files preserve all raw measurements.",'']
    (OUT/'REPORT.md').write_text('\n'.join(md))
    print(json.dumps({'status':result['status'],'architectures':summary},indent=2))

if __name__=='__main__':main()
