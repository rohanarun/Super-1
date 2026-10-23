"""Frozen, local CPU pilot; no official validation/test or cloud training."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]="1"
import argparse,copy,hashlib,json,math,platform,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"upstream"))
from models import build,optimizer
from geometry import proposal
from sketches import Capture,describe,metric_probe

RUN="state-sketch-perturb-20261018-v1"
OUT=ROOT/"auto-research/results"/RUN
METHODS=["backprop","iid","random_pair","sketch","scrambled","replay_backprop"]
CONFIG={
 "run_id":RUN,"upstream_commit":"492a87a5e263e806b0a2581c57259e81f1a000e6",
 "authorization":"User: Excellent, go ahead with that set of experiments; Include mingru and the other changes that made it into rohan's repo.",
 "seeds":[73,191,307],"widths":{"native":48,"mingru_linear":52,"paragru":50},
 "vocab_size":2048,"context":512,"batch":4,"segment_length":32,"document_segments":8,
 "warmup_training_updates":192,"intervention_updates":96,"followup_updates":32,
 "perturb_every":32,"anchor_refresh_every":64,"anchor_training_windows":4,"sketch_dimensions":64,
 "relative_l2_radius":.003,"directions_per_event":2,"signs":[-1,1],"selection_batch_size":2,
 "methods":METHODS,
 "primary_hypothesis":"Current activation relationships improve function-changing, coupled permutation-secant perturbations over scrambled-anchor and random-pair controls.",
 "primary_metric":"Reserved document-held-out CE after 96 intervention updates and 32 ordinary-backprop followup updates; compare within architecture/seed.",
 "useful_effect_nats":.005,
 "advance_rule":"Exploratory only: mean sketch CE at least .005 below both scrambled and random_pair, wins in at least2/3 paired seeds against each, no mean harm vs backprop; metric amortized overhead <=1%; total training overhead <=10%. Equal-time replay control must also be reported; n=3 is not a confirmation claim.",
 "optimizer":{"name":"upstream AdamW","peak_lr":.002,"betas":[.9,.95],"weight_decay":.1,"clip":1.,"warmup":16,"schedule":"cosine to10% over320 nominal updates; replay keeps current nominal-step LR"},
 "state_semantics":"512-token reset windows, sampled within individual training documents; matches merged minGRU comparison style; not a persistent-document training experiment.",
 "data_semantics":"Only WikiText2 TRAIN is downloaded; articles are hash-partitioned into new train/monitor/confirmation sets. Existing FineWeb-Edu BPE2048 is reused. No teacher/distillation. Not a reproduction of the120M or FineWeb quality benchmark.",
 "candidate_semantics":"Each structured direction is P(theta)-theta using two disjoint unit swaps per hidden group, normalized to fixed global relative L2 norm. Small +/- secant steps change the function; endpoint permutation is not claimed to improve it. Every group includes all connected gates/weights, tied embedding/output, and FFN. ParaGRU constraints apply to every candidate.",
 "scramble_semantics":"Same fingerprints and shortlist algorithm; descriptors randomly reassigned to units independently per group per direction. Counts/radius/evaluations match guided.",
 "replay_control":"Replay already-seen selection examples using backprop at each event until guided event wall time is met, counting capture+metric+search. At most one optimizer-step overshoot; report measured total wall/CPU and extra token exposures. No claim of exact equal-FLOPs.",
 "selection":"Current recent training batch only; confirmation scores never select candidates, radii, dimensions, checkpoints, or methods.",
 "resource_limits":{"device":"CPU","torch_threads":1,"interop_threads":1,"max_main_wall_seconds":1800,"cloud_spend":0},
 "futility":"Stop on nonfinite values, source/protocol mismatch, failed preflight, or resource cap. Preserve incomplete status and results; do not tune after observing confirmation.",
 "publication":"Unpublished; no GitHub push authorized in this turn."
}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(".tmp")
 temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)
def state_hash(m):
 h=hashlib.sha256()
 for n,p in m.state_dict().items():h.update(n.encode());h.update(p.detach().contiguous().numpy().tobytes())
 return h.hexdigest()
def source_manifest():
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob("*.py")) if "auto-research" not in p.parts}
def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 assert json.loads((OUT/"PREFLIGHT.json").read_text())["status"]=="PASS"
 protocol=copy.deepcopy(CONFIG);protocol["data"]=json.loads((ROOT/"data/manifest.json").read_text())
 protocol["source_hashes"]=source_manifest()
 dest=OUT/"PROTOCOL.json"
 if dest.exists():assert json.loads(dest.read_text())==protocol
 else:save(dest,protocol)
 save(ROOT/"auto-research/RUN_REGISTRY.json",{"runs":[{"run_id":RUN,"status":"REGISTERED","protocol_sha256":sha(dest),"authorization":CONFIG["authorization"]}]})
 print("REGISTERED",sha(dest),flush=True)

class Corpus:
 def __init__(self,split):
  z=np.load(ROOT/"data"/(split+".npz"));self.tokens=z["tokens"];self.offsets=z["offsets"]
 def batch(self,seed,index,batch=4):
  rng=np.random.default_rng(np.random.SeedSequence([seed,index,8721]));windows=[]
  for _ in range(batch):
   doc=int(rng.integers(len(self.offsets)-1));lo,hi=self.offsets[doc:doc+2]
   start=int(rng.integers(lo,hi-512))
   windows.append(self.tokens[start:start+513].astype(np.int64))
  t=torch.from_numpy(np.stack(windows));return t[:,:-1].contiguous(),t[:,1:].contiguous()

def learning_rate(step):
 o=CONFIG["optimizer"];warm=o["warmup"];total=320
 return o["peak_lr"]*(step/warm if step<=warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/(total-warm))))
def training_step(m,opt,x,y,step):
 m.train()
 for group in opt.param_groups:group["lr"]=learning_rate(step)
 opt.zero_grad(set_to_none=True)
 loss=F.cross_entropy(m(x).logits.flatten(0,1),y.flatten());loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
 if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError("nonfinite training")
 opt.step();m.constrain()
 return float(loss.detach())
@torch.no_grad()
def loss_value(m,x,y):
 m.eval();z=m(x).logits
 if not torch.isfinite(z).all():return float("inf")
 return float(F.cross_entropy(z.flatten(0,1),y.flatten()))
@torch.no_grad()
def evaluate(m,corpus):
 values=[]
 for index in range(8):
  x,y=corpus.batch(91823,index,4);values.append(loss_value(m,x,y))
 return {"ce":float(np.mean(values)),"batch_ce":values,"targets":16384}

@torch.no_grad()
def search(m,x,y,desc,method,seed,step):
 tick=time.perf_counter();cpu=time.process_time()
 base={n:p.clone() for n,p in m.state_dict().items()};base_loss=loss_value(m,x,y)
 best_loss=base_loss;best=None;rows=[]
 for d in range(CONFIG["directions_per_event"]):
  delta,detail=proposal(base,desc,m.width,seed*100000+step*17+d,method,CONFIG["relative_l2_radius"])
  for sign in CONFIG["signs"]:
   m.load_state_dict({n:p+sign*delta[n] for n,p in base.items()});m.constrain()
   value=loss_value(m,x,y)
   if not math.isfinite(value):raise FloatingPointError("nonfinite proposal")
   rows.append({"direction":d,"sign":sign,"loss":value,"pairs":detail})
   if value<best_loss-1e-7:
    best_loss=value;best={n:p.clone() for n,p in m.state_dict().items()}
 m.load_state_dict(base if best is None else best)
 return {"step":step,"base_loss":base_loss,"accepted_loss":best_loss,"accepted":best is not None,
   "candidates":rows,"search_wall_seconds":time.perf_counter()-tick,"search_cpu_seconds":time.process_time()-cpu,
   "selection_target_exposures":(1+len(rows))*y.numel()}

def main_run():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 protocol=json.loads((OUT/"PROTOCOL.json").read_text())
 assert protocol["source_hashes"]==source_manifest(),"Source seal mismatch"
 for split,meta in protocol["data"]["splits"].items():assert sha(ROOT/"data"/(split+".npz"))==meta["sha256"]
 train=Corpus("train");confirmation=Corpus("confirmation");monitor=Corpus("monitor")
 clock=time.monotonic();started_cpu=time.process_time();results=[]
 runtime={"python":sys.version,"torch":torch.__version__,"numpy":np.__version__,"platform":platform.platform(),"threads":1,"device":"cpu","dtype":"float32"}
 save(OUT/"START.json",{"status":"RUNNING","runtime":runtime,"protocol_sha256":sha(OUT/"PROTOCOL.json"),"pid":os.getpid()})
 def deadline():
  if time.monotonic()-clock>CONFIG["resource_limits"]["max_main_wall_seconds"]:raise TimeoutError("Frozen main wall budget exhausted")
 try:
  for kind,width in CONFIG["widths"].items():
   config=dict(vocab_size=2048,width=width,max_context=512,segment_length=32,document_segments=8)
   for seed in CONFIG["seeds"]:
    deadline();name=f"{kind}-{seed}";folder=OUT/name;folder.mkdir(exist_ok=True)
    m=build(kind,config,seed);opt=optimizer(m,.002);warmtrace=[];t=time.perf_counter();c=time.process_time()
    for step in range(1,CONFIG["warmup_training_updates"]+1):
     x,y=train.batch(seed,step);loss=training_step(m,opt,x,y,step)
     if step%64==0:
      print("WARMUP",name,step,round(loss,5),flush=True);deadline()
     warmtrace.append(loss)
    warm_wall=time.perf_counter()-t;warm_cpu=time.process_time()-c
    parent_weights=copy.deepcopy(m.state_dict());parent_opt=copy.deepcopy(opt.state_dict())
    parent_eval=evaluate(m,confirmation);parent_monitor=evaluate(m,monitor)
    capture=Capture(m);capture.enabled=True
    with torch.no_grad():m(x)
    capture.enabled=False;traces=capture.pop();capture.close()
    probes=metric_probe(traces)
    save(folder/"METRICS.json",probes)
    save(folder/"PARENT.json",{"parameters":sum(p.numel() for p in m.parameters()),"config":config,
      "warmup_wall_seconds":warm_wall,"warmup_cpu_seconds":warm_cpu,"warmup_losses":warmtrace,
      "confirmation":parent_eval,"monitor":parent_monitor,"state_sha256":state_hash(m)})
    torch.save({"model":parent_weights,"optimizer":parent_opt},folder/"parent.pt")
    guided_events=None
    for method in METHODS:
     deadline()
     m=build(kind,config,seed);m.load_state_dict(parent_weights);opt=optimizer(m,.002);opt.load_state_dict(copy.deepcopy(parent_opt))
     capture=Capture(m);desc=None;events=[];traintrace=[];nominal_seconds=0.;nominal_cpu=0.;extra_wall=0.;extra_cpu=0.;replays=0
     total_updates=CONFIG["intervention_updates"]+CONFIG["followup_updates"]
     for step in range(1,total_updates+1):
      nominal=CONFIG["warmup_training_updates"]+step
      next_event=(step-1)//32+1
      capture.enabled=(method in ("sketch","scrambled") and step<=96 and next_event%2==1 and (step-1)%32>=28)
      x,y=train.batch(seed,nominal)
      t=time.perf_counter();c=time.process_time()
      value=training_step(m,opt,x,y,nominal)
      nominal_seconds+=time.perf_counter()-t;nominal_cpu+=time.process_time()-c
      capture.enabled=False;traintrace.append(value)
      if step<=96 and step%32==0 and method!="backprop":
       if method=="replay_backprop":
        target=guided_events[len(events)]["total_event_wall_seconds"]
        t=time.perf_counter();c=time.process_time();count=0;last=0.
        while time.perf_counter()-t<target:
         s=time.perf_counter();training_step(m,opt,x[:2],y[:2],nominal);last=time.perf_counter()-s;count+=1
        elapsed=time.perf_counter()-t;cpu=time.process_time()-c;replays+=count;extra_wall+=elapsed;extra_cpu+=cpu
        event={"step":step,"replay_updates":count,"guided_event_budget_seconds":target,"actual_seconds":elapsed,
          "overshoot_seconds":elapsed-target,"last_step_seconds":last,"selection_target_exposures":count*y[:2].numel()}
       else:
        t=time.perf_counter();c=time.process_time();metric_seconds=0.;refresh=False
        capture_delta=capture.seconds;capture.seconds=0.
        if method in ("sketch","scrambled") and next_event%2==1:
         begin=time.perf_counter();desc=describe(capture.pop(),"countsketch",64);metric_seconds=time.perf_counter()-begin;refresh=True
        event=search(m,x[:2],y[:2],desc,method,seed,step)
        elapsed=time.perf_counter()-t;cpu=time.process_time()-c
        extra_wall+=elapsed;extra_cpu+=cpu
        event.update(metric_seconds=metric_seconds,capture_seconds=capture_delta,anchor_refreshed=refresh,
          total_event_wall_seconds=elapsed+capture_delta)
       events.append(event)
      if step==96:
       intermediate=evaluate(m,confirmation)
       intervention_wall=nominal_seconds+extra_wall;intervention_cpu=nominal_cpu+extra_cpu
       print("INTERVENTION",name,method,round(intermediate["ce"],6),flush=True)
      if step%32==0:deadline()
     final=evaluate(m,confirmation);duplicate=evaluate(m,confirmation)
     assert final==duplicate,"Repeat evaluation drift"
     if method=="sketch":guided_events=events
     row={"architecture":kind,"seed":seed,"method":method,"parameters":sum(p.numel() for p in m.parameters()),
       "parent_ce":parent_eval["ce"],"intervention":intermediate,"followup":final,"duplicate_exact":True,
       "nominal_updates":total_updates,"extra_replay_updates":replays,"nominal_target_exposures":total_updates*4*512,
       "selection_target_exposures":sum(e["selection_target_exposures"] for e in events),
       "intervention_wall_seconds":intervention_wall,"intervention_cpu_seconds":intervention_cpu,
       "total_training_wall_seconds":nominal_seconds+extra_wall,"total_training_cpu_seconds":nominal_cpu+extra_cpu,
       "nominal_training_wall_seconds":nominal_seconds,"event_wall_seconds":extra_wall,
       "metric_seconds":sum(e.get("metric_seconds",0) for e in events),
       "capture_seconds":sum(e.get("capture_seconds",0) for e in events),
       "accepted":sum(e.get("accepted",False) for e in events),
       "final_state_sha256":state_hash(m),"training_losses":traintrace,"events":events}
     if kind=="paragru":row["solver_max_residual"]=m.fast_cell.max_residual
     save(folder/(method+".json"),row)
     # Keep final weights for numerical replay locally; omit large tensors from the compact deliverable.
     torch.save(m.state_dict(),folder/(method+".pt"))
     results.append(row);capture.close()
     save(OUT/"PROGRESS.json",{"status":"RUNNING","completed":len(results),"planned":54,"last":{"architecture":kind,"seed":seed,"method":method,"ce":final["ce"]},"elapsed_seconds":time.monotonic()-clock})
     print("COMPLETE",name,method,round(final["ce"],6),"seconds",round(row["total_training_wall_seconds"],2),flush=True)
    del parent_weights,parent_opt
  terminal={"status":"COMPLETE","rows":len(results),"wall_seconds":time.monotonic()-clock,"cpu_seconds":time.process_time()-started_cpu,
   "runtime":runtime,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"protocol_sha256":sha(OUT/"PROTOCOL.json"),
   "source_seal_verified":source_manifest()==protocol["source_hashes"],"official_test_reads":0,"official_validation_reads":0}
  save(OUT/"TERMINAL.json",terminal)
  save(ROOT/"auto-research/RUN_REGISTRY.json",{"runs":[{"run_id":RUN,**terminal}]})
  print("ALL_COMPLETE",json.dumps(terminal),flush=True)
 except BaseException as error:
  save(OUT/"TERMINAL.json",{"status":"FAILED_OR_INCOMPLETE","error":repr(error),"completed_rows":len(results),"elapsed_seconds":time.monotonic()-clock})
  raise
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("action",choices=["freeze","run"]);args=p.parse_args()
 freeze() if args.action=="freeze" else main_run()
