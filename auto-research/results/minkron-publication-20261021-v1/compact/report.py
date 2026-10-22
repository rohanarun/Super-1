import bootstrap,base64,html,json,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ARMS
from experiment import ROOT,atomic_json

p=json.loads((ROOT/'protocol.json').read_text())
names={'mingru':'minGRU','kron8':'Kron 1 × 8 × 8','kron4h4':'Kron 4 × 4 × 4','kron2h4':'Kron 2 × 4 × 4'}
colors={'mingru':'#586275','kron8':'#147eab','kron4h4':'#c27810','kron2h4':'#27905c'}
runs={};confirm={};summary={}
timing=json.loads((ROOT/'results/timing.json').read_text())
for arm in ARMS:
 runs[arm]=[];confirm[arm]=[]
 for seed in p['seeds']:
  path=ROOT/'results/main'/f'{arm}_{seed}_{p["learning_rates"][arm]:g}'
  runs[arm].append(json.loads((path/'terminal.json').read_text()))
  confirm[arm].append(json.loads((ROOT/'results/confirmation'/f'{arm}_{seed}'/'terminal.json').read_text()))
 metrics={}
 for mode in ['competing','overwrite']:
  vals=[next(v['accuracy'] for v in c['records'] if v['age']==512 and v['mode']==mode) for c in confirm[arm]]
  metrics[mode]={'values':vals,'mean':statistics.mean(vals),'sample_sd':statistics.stdev(vals)}
 times=[r for r in timing['rows'] if r['arm']==arm]
 decode={str(r['batch']):r['seconds_per_step_median'] for r in times if r['kind']=='decode'}
 memory={k:json.loads((ROOT/'results/memory'/f'{arm}_{k}.json').read_text()) for k in ['train','inference']}
 summary[arm]={'name':names[arm],'accounting':runs[arm][0]['accounting'],'primary':metrics,
  'cpu_seconds':[r['training_cpu_seconds'] for r in runs[arm]],'tokens':[r['tokens'] for r in runs[arm]],
  'phase_steps':[[z['steps'] for z in r['phases']] for r in runs[arm]],'decode_seconds':decode,
  'train_step_seconds':next(r['seconds_median'] for r in times if r['kind']=='train_step'),
  'prefill_seconds':next(r['seconds_median'] for r in times if r['kind']=='prefill'),'memory':memory}
base=summary['mingru']
for arm,s in summary.items():
 s['cpu_ratios']=[a/b for a,b in zip(s['cpu_seconds'],base['cpu_seconds'])]
 s['decode_ratios']={b:s['decode_seconds'][b]/base['decode_seconds'][b] for b in ['1','32']}
 s['primary_deltas_pp']={m:100*(s['primary'][m]['mean']-base['primary'][m]['mean']) for m in ['competing','overwrite']}
 s['strict_resource_and_mean_accuracy_screen']=all(s['primary'][m]['mean']>=base['primary'][m]['mean'] for m in ['competing','overwrite']) and all(v<=1 for v in s['decode_ratios'].values()) and s['memory']['train']['peak_live_tensor_bytes']<=base['memory']['train']['peak_live_tensor_bytes'] and s['memory']['inference']['peak_live_tensor_bytes']<=base['memory']['inference']['peak_live_tensor_bytes']
atomic_json({'protocol':p,'arms':summary,'note':'Descriptive three-seed screen; resource/accuracy conjunction is not a statistical proof.'},ROOT/'summary.json')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(15,4.8));xx=np.arange(4)
for ax,mode in zip(axs[:2],['competing','overwrite']):
 vals=[summary[a]['primary'][mode]['mean']*100 for a in ARMS]
 allvalues=[v*100 for a in ARMS for v in summary[a]['primary'][mode]['values']]
 for i,a in enumerate(ARMS):
  yy=np.array(summary[a]['primary'][mode]['values'])*100
  ax.scatter(i+np.array([-.13,0,.13]),yy,color=colors[a],s=40,zorder=3)
  ax.plot([i-.24,i+.24],[vals[i],vals[i]],color='black',lw=2)
 ax.set_ylim(max(0,5*np.floor(min(allvalues)/5)-5),101);ax.set_ylabel('Accuracy (%)');ax.set_title(mode.capitalize()+' · age 509–512')
 ax.set_xticks(xx,['minGRU','1×8×8','4×4×4','2×4×4'])
 ax.grid(axis='y',alpha=.2)
for b,offset in [('1',-.18),('32',.18)]:
 axs[2].bar(xx+offset,[summary[a]['decode_ratios'][b] for a in ARMS],width=.34,label='Batch '+b,alpha=.8)
axs[2].axhline(1,color='black',ls='--',lw=1);axs[2].set_ylabel('Decode latency / minGRU');axs[2].set_title('Measured CPU inference cost')
axs[2].set_xticks(xx,['minGRU','1×8×8','4×4×4','2×4×4']);axs[2].legend();axs[2].grid(axis='y',alpha=.2)
fig.suptitle('Compact Kron under the minGRU state and training CPU budget',fontsize=15)
fig.tight_layout();fig.savefig(ROOT/'compact_kron_results.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8),sharey=True)
for ax,arm in zip(axs.flat,ARMS):
 for seed in p['seeds']:
  hist=json.loads((ROOT/'results/main'/f'{arm}_{seed}_{p["learning_rates"][arm]:g}'/'history.json').read_text())
  loss=np.array([v['loss'] for v in hist]);cpu=np.array([v['cpu_seconds'] for v in hist]);window=10
  smooth=np.convolve(loss,np.ones(window)/window,mode='valid')
  ax.plot(cpu[window-1:],smooth,label=str(seed),lw=1.3)
 ax.set_yscale('log');ax.set_title(names[arm]);ax.set_xlabel('Training CPU seconds');ax.set_ylabel('Loss · 10 logged points average');ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.tight_layout();fig.savefig(ROOT/'compact_kron_learning.png',dpi=140);plt.close(fig)

def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def img(name):return '<img src="data:image/png;base64,'+base64.b64encode((ROOT/name).read_bytes()).decode()+'">'
mainrows=[];costrows=[];seedrows=[];longrows=[]
for arm,s in summary.items():
 ac=s['accounting'];mainrows.append([names[arm],f'{ac["persistent_tensor_state_bytes"]:,} B',f'{ac["parameter_bytes"]/1024:.1f} KiB',*[f'{s["primary"][m]["mean"]*100:.2f}%' for m in ['competing','overwrite']],f'{statistics.mean(s["cpu_seconds"]):.1f} s'])
 costrows.append([names[arm],f'{s["decode_seconds"]["1"]*1000:.3f} ms',f'{s["decode_ratios"]["1"]:.2f}×',f'{s["decode_ratios"]["32"]:.2f}×',f'{s["memory"]["train"]["peak_live_tensor_bytes"]/2**20:.1f} MiB',f'{s["memory"]["inference"]["peak_live_tensor_bytes"]/2**20:.1f} MiB'])
 for j,seed in enumerate(p['seeds']):
  seedrows.append([names[arm],seed,*[f'{s["primary"][m]["values"][j]*100:.2f}%' for m in ['competing','overwrite']],'/'.join(map(str,s['phase_steps'][j])),f'{s["cpu_ratios"][j]*100:.2f}%',f'{s["tokens"][j]:,}'])
 for mode in ['quiet','repeat','competing','overwrite']:
  vals=[next(r['accuracy'] for r in c['records'] if r['age']==2048 and r['mode']==mode) for c in confirm[arm]]
  longrows.append([names[arm],mode,*[f'{v*100:.1f}%' for v in vals],f'{statistics.mean(vals)*100:.1f}%'])
passed=[names[a] for a in ARMS if a!='mingru' and summary[a]['strict_resource_and_mean_accuracy_screen']]
decision=('Candidates passing the descriptive mean-accuracy and measured-resource screen: '+', '.join(passed)+'. This remains a three-seed synthetic result.') if passed else 'No compact variant demonstrates an overall win across both primary accuracies, decode latency, and measured peak working memory in this CPU experiment.'
body='<h1>Compact Kron: equal-state, capped-compute experiment</h1><p class="lead">'+decision+'</p>'
body+='<p>The goal is better memory performance per byte and per unit of compute. Larger full-Kron results are not used as evidence of a resource-matched win here.</p>'
body+='<h2>Final unseen retrieval</h2><p>Means over three fresh paired training seeds; 1,024 held-out episodes per seed and primary task. Chance is 12.5%. All compact models meet the persistent-state and weight-memory ceilings. In the accuracy plots, colored points are individual seeds and black ticks are means; axes zoom in on the observed range.</p>'
body+=table(['Model','Persistent state','Weights','Competing, 512','Overwrite, 512','Mean train CPU'],mainrows)+img('compact_kron_results.png')
best=summary['kron8'];peak_ratio=best['memory']['train']['peak_live_tensor_bytes']/base['memory']['train']['peak_live_tensor_bytes']
body+='<p><strong>The one-head 8×8 model is the strongest compact candidate.</strong> It preserves retrieval at the state ceiling, but its small mean advantage comes primarily from the third minGRU seed; the other two minGRU seeds score 100% on both primary tasks. At batch 1, its measured decode latency is '+f'{best["decode_ratios"]["1"]:.2f}'+'× minGRU, and its peak training tensor allocation is '+f'{(peak_ratio-1)*100:.1f}'+'% higher. This is not an overall resource-efficiency win.</p>'
body+='<h2>Resources beyond persistent state</h2><p>Serial, interleaved, one-thread CPU timings after training and confirmation finished. Decode includes the entire hierarchy and task head. Batch-32 latency is per batch step. Peak live tensor allocation is measured at batch 32, length 513; it includes observed preexisting tensors but excludes Python/runtime memory and untracked library workspace. These figures are separate from the per-sequence state budget.</p>'
body+=table(['Model','Decode B1','B1 / minGRU','B32 / minGRU','Training peak','Inference peak'],costrows)
body+='<h2>Every seed and its training allowance</h2><p>The baseline completes 10,000 short, 1,000 medium and 3,000 long updates. Each compact model stops at the paired baseline CPU-time ceiling within each stage, with a reserve to prevent overruns, and never exceeds the baseline update count. Paired data are identical prefixes within stages; complete datasets differ when models complete fewer updates.</p>'
body+=table(['Model','Seed','Competing','Overwrite','Short / medium / long updates','CPU / paired minGRU','Tokens'],seedrows)
body+=img('compact_kron_learning.png')
body+='<h2>Longer-delay extrapolation</h2><p>Age 2,048 was not part of the training curriculum. These are secondary diagnostics, 256 episodes per task and seed, and did not select models.</p>'
body+='<p>The 8×8 variant has stronger mean competing-write and overwrite retrieval at this length, but weaker quiet and repeated-distractor retention. Its benefit is task-dependent. The four-head variant deteriorates sharply at this length under the same compute ceiling.</p>'
body+=table(['Model','Task',*p['seeds'],'Mean'],longrows)
ip=ROOT/'results/interventions.json'
if ip.exists():
 data=json.loads(ip.read_text());body+='<h2>Does the model use its memory?</h2><p>Independent diagnostic on the first fixed seed only, after 256 tokens; all target writes occur before the cut. Counters are preserved. Each entry uses 256 episodes; these are not the confirmation cases.</p>'
 rr=[]
 for arm in ARMS:
  for mode in ['competing','overwrite']:
   vals=[next(r['accuracy'] for r in data['rows'] if r['arm']==arm and r['mode']==mode and r['intervention']==k) for k in ['intact','zero_fast','zero_slow','zero_all']]
   rr.append([names[arm],mode,*[f'{v*100:.1f}%' for v in vals]])
 body+=table(['Model','Task','Intact','Fast zeroed','Slow zeroed','All zeroed'],rr)
body+='<h2>Design, checks and limitations</h2><ul><li>All arms use the same existing width-64 fast/segment/document model, shared initialization and final 8-class readout. This baseline contains no curve-generated weights.</li><li>Compact Kron retains unit-normalized SiLU queries/keys/values, vector decay, independent write/erase gates, gated RMSNorm and output projection. It removes convolution and the unused cached output, and reduces matrix size.</li><li>One 8×8 and four 4×4 matrices contain 64 floats. Two 4×4 matrices contain 32. Shared hierarchy state contains 192 floats. All state is FP32.</li><li>FP32/FP64 reference agreement includes state and input gradients. Whole-model tests cover streaming splits, parameter gradients, tokenwise execution, packed projections, live component gradients, task oracles, reset and tiny-batch fit.</li><li>The CPU core uses the exact author recurrence with a direct triangular solve and no token padding. Upstream fused GPU kernels were not run. No GPU speed prediction is made.</li><li>Each arm receives the same two pilot learning rates and 100-CPU-second ceiling per rate. The selected rates were frozen before main training: '+html.escape(str(p['learning_rates']))+'. Pilot data and confirmation data are separate.</li><li>Training CPU excludes setup, checkpoint I/O, validation and engineering. Baseline workers run three at once; compact workers run six. Actual shared-host CPU time is capped, not theoretical FLOPs. Timing benchmarks run serially.</li><li>Three seeds and a synthetic register task support a bounded feasibility result, not general language-model superiority. Differences between seeds must not be replaced by an episode-only confidence interval.</li></ul>'
body+='<p>Reproduction: README.md, protocol.json, source_manifest.json, results/verification.json, results/audit.json, raw predictions and final checkpoints accompany the source bundle.</p>'
body+='<p>A 32-value minGRU has not been tested here. The reduced-state Kron arm is compared with the established 64-value baseline, so this screen does not establish the best possible minGRU memory/accuracy frontier. Learning-curve points are periodic batch snapshots rather than averages over every training batch.</p>'
css='body{font:16px/1.55 system-ui,sans-serif;color:#172a3a;background:#f6f8fb;margin:0;padding:28px}main{max-width:1180px;margin:auto;background:white;padding:30px;border-radius:14px}h1{font-size:30px;line-height:1.2}h2{margin-top:34px;color:#154e68}.lead{font-size:19px;background:#eaf3f8;padding:18px;border-left:4px solid #147eab}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #dce4e9;white-space:nowrap}th{background:#edf3f7}.scroll{overflow-x:auto}img{width:100%;height:auto;margin:20px 0}li{margin-bottom:10px}@media(max-width:650px){body{padding:8px}main{padding:16px}h1{font-size:24px}}'
(ROOT/'Compact_Kron_budget_report_2026-10-21.html').write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compact Kron budget experiment</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
print(json.dumps({'decision':decision,'primary':{a:summary[a]['primary'] for a in ARMS},'resource_ratios':{a:summary[a]['decode_ratios'] for a in ARMS}},indent=2))
