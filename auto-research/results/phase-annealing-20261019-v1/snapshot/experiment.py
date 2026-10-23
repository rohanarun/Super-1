"""Separate-phase simulated annealing around complete short BP runs."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,hashlib,json,math,platform,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'))
from models import build,optimizer
RUN='phase-annealing-20261019-v1'
OUT=ROOT/'auto-research/results'/RUN
METHODS=['baseline_keep','baseline_reset','sa_mid_keep','sa_mid_reset','extra_mid_keep',
         'extra_mid_reset','greedy_mid_keep','sa_post','extra_post','sa_pre','extra_pre']
CONFIG={
 'run_id':RUN,'upstream_commit':'492a87a5e263e806b0a2581c57259e81f1a000e6',
 'authorization':'User explicitly requested annealing before/after BP entirely and sandwiched between runs, then authorized these experiments.',
 'seeds':[419,563,701],'widths':{'native':48,'mingru_linear':52,'paragru':50},
 'context':512,'batch':4,'vocab_size':2048,'segment_length':32,'document_segments':8,
 'bp_runs':{'A':{'updates':192,'peak_lr':.002,'warmup':16},'B':{'updates':96,'peak_lr':.001,'warmup':8}},
 'lr_schedule':'Each BP run has its own warmup/cosine schedule ending at10% of its peak. The identical A-to-B LR restart occurs in every arm; retained/reset Adam is varied explicitly.',
 'optimizer':'Byte-exact upstream AdamW grouping, betas(.9,.95), eps1e-8, weight_decay.1; gradient clip1.',
 'annealing':{'steps':96,'temperature_probes':8,'initial_relative_sd':.03,'final_relative_sd':.003,
               'temperature_final_ratio':.01,'target_initial_uphill_acceptance':.7,'rms_floor':.01},
 'annealing_semantics':'A standalone phase with no backward or optimizer calls. Gaussian proposals have fixed per-tensor RMS scaling frozen at phase entry, with SD cooled geometrically over96 trials. Metropolis accept=min(1,exp(-deltaCE/T)). Temperature is calibrated on8 discarded training-only proposals; chain starts from the unmodified phase entry. Return best accepted chain state including entry; retain final chain separately in logs. This is a finite heuristic search, not a claim of global convergence or exact full-data posterior sampling.',
 'paragru_constraint':'Hold only fast_cell.A fixed during SA, preserving its upstream norm constraints and a symmetric proposal on the remaining coordinates. BP still updates and constrains A. All other trainable parameters are eligible.',
 'selection_data':'One fixed4x512 training batch per phase: first A batch for pre, final A batch for middle, final B batch for post. Parent and every candidate use exactly the same batch. No new corpus examples at middle/post phases. Training CE only, no additional regularizer.',
 'fresh_confirmation':'38 original monitor documents never evaluated by the prior pilot, and never trained on.32 fixed512-token windows with a new evaluation RNG. Previous confirmation documents entirely excluded.',
 'methods':METHODS,
 'primary_comparison':'SA sandwich with retained Adam vs matched extra-BP sandwich with retained Adam, evaluated after complete BP run B.',
 'secondary_comparisons':'Pre-SA vs pre-extra-BP; post-SA vs post-extra-BP; sandwich reset pairs; greedy T=0 middle search vs SA; keep/reset no-op controls.',
 'equal_time_control':'Extra BP uses the identical phase calibration batch until measured SA phase wall time is reached; one-step overshoot is recorded. Pre uses LR.002 then clears Adam before A; middle LR.0002; post LR.0001. All SA calibration/RNG/copy/forward work is charged. This is measured approximate equal wall time, not equal FLOPs.',
 'useful_effect_nats':.005,
 'advance_rule':'For a placement/architecture: mean SA CE at least.005 nats below matched extra-BP, wins>=2/3 seeds, and no mean harm vs the corresponding no-SA baseline. Sandwich/keep is primary; all other comparisons exploratory. Report complete-training wall ratios; n=3 does not establish significance.',
 'boundaries':'Model recurrent state resets at every512-token forward window. RunA ends before middle annealing begins; runB starts only after annealing ends. Pre annealing precedes all BP; post annealing follows both BP runs and has no gradient refinement.',
 'limits':{'device':'cpu','threads':1,'max_main_wall_seconds':1800,'cloud_spend':0},
 'futility':'Abort on nonfinite values, failed checks/source seal or time cap. Preserve incomplete status. No retuning after confirmation.',
 'scope':'Small fixed-width pilot, not converged/full-scale pretraining; no permutation matching, CountSketch, continuous-learning additions, SANTA or gradient-noise optimizer.',
 'publication':'Unpublished local research artifact; no GitHub push in this task.'
}

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tree_hash(value):
    h=hashlib.sha256()
    def walk(x):
        h.update(type(x).__name__.encode())
        if torch.is_tensor(x):h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):walk(k);walk(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:walk(v)
        else:h.update(repr(x).encode())
    walk(value);return h.hexdigest()
def weights(m):return {n:p.detach().clone() for n,p in m.state_dict().items()}
def source_manifest():return {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py')) if 'auto-research' not in p.parts}

class Corpus:
    def __init__(self,split):
        z=np.load(ROOT/'data'/(split+'.npz'));self.tokens=z['tokens'];self.offsets=z['offsets']
    def batch(self,seed,index,batch=4):
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,39117]));v=[]
        for _ in range(batch):
            doc=int(rng.integers(len(self.offsets)-1));lo,hi=self.offsets[doc:doc+2]
            start=int(rng.integers(lo,hi-512));v.append(self.tokens[start:start+513].astype(np.int64))
        t=torch.from_numpy(np.stack(v));return t[:,:-1].contiguous(),t[:,1:].contiguous()

def lr_for(run,step):
    c=CONFIG['bp_runs'][run];n=c['updates'];w=c['warmup']
    return c['peak_lr']*(step/w if step<=w else .1+.9*.5*(1+math.cos(math.pi*(step-w)/(n-w))))
def update(m,opt,x,y,lr):
    m.train()
    for g in opt.param_groups:g['lr']=lr
    opt.zero_grad(set_to_none=True)
    loss=F.cross_entropy(m(x).logits.flatten(0,1),y.flatten());loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('nonfinite BP')
    opt.step();m.constrain();return float(loss.detach())
@torch.no_grad()
def loss_value(m,x,y):
    m.eval();v=F.cross_entropy(m(x).logits.flatten(0,1),y.flatten())
    if not torch.isfinite(v):raise FloatingPointError('nonfinite forward')
    return float(v)
@torch.no_grad()
def evaluate(m,corpus):
    values=[loss_value(m,*corpus.batch(81931,i)) for i in range(8)]
    return {'ce':float(np.mean(values)),'batch_ce':values,'targets':16384}

def accept_move(delta,temperature,u):
    if delta<=0:return True
    return temperature>0 and math.log(max(u,1e-300)) < -delta/temperature

@torch.no_grad()
def anneal(m,opt,x,y,seed,greedy=False,steps=None,probes=None):
    tick=time.perf_counter();cpu=time.process_time();a=CONFIG['annealing']
    steps=a['steps'] if steps is None else steps;probes=a['temperature_probes'] if probes is None else probes
    initial=weights(m);initial_loss=loss_value(m,x,y);current=initial;best=initial
    current_loss=best_loss=initial_loss;best_step=0
    before_opt=tree_hash(opt.state_dict()) if opt is not None else None
    for p in m.parameters():p.grad=None
    scales={n:float(p.double().square().mean().sqrt().clamp_min(a['rms_floor'])) for n,p in initial.items() if n!='fast_cell.A'}
    rng=torch.Generator().manual_seed(seed);uniform=np.random.default_rng(seed+171)
    def proposed(base,relative_sd):
        return {n:(p+torch.randn(p.shape,generator=rng,dtype=p.dtype)*(relative_sd*scales[n]) if n in scales else p.clone()) for n,p in base.items()}
    pilot=[]
    for _ in range(probes):
        m.load_state_dict(proposed(initial,a['initial_relative_sd']));pilot.append(loss_value(m,x,y)-initial_loss)
    m.load_state_dict(initial)
    positives=[d for d in pilot if d>0]
    scale=float(np.median(positives if positives else [abs(d) for d in pilot])) if pilot else 1e-6
    initial_temperature=max(scale/-math.log(a['target_initial_uphill_acceptance']),1e-6)
    rows=[]
    for i in range(steps):
        progress=i/max(steps-1,1)
        temp=0. if greedy else initial_temperature*a['temperature_final_ratio']**progress
        sd=a['initial_relative_sd']*(a['final_relative_sd']/a['initial_relative_sd'])**progress
        candidate=proposed(current,sd);m.load_state_dict(candidate);value=loss_value(m,x,y)
        delta=value-current_loss;u=float(uniform.random());accept=accept_move(delta,temp,u)
        if accept:
            current=candidate;current_loss=value
            if value<best_loss:
                best={n:p.clone() for n,p in candidate.items()};best_loss=value;best_step=i+1
        else:m.load_state_dict(current)
        rows.append({'step':i+1,'temperature':temp,'relative_sd':sd,'proposal_loss':value,'delta':delta,
                     'uniform':u,'accepted':accept,'uphill_accepted':bool(accept and delta>0),
                     'chain_loss':current_loss,'best_loss':best_loss})
    chain_hash=tree_hash(current);m.load_state_dict(best)
    assert all(p.grad is None for p in m.parameters()),'Annealing accumulated gradients'
    if opt is not None:assert before_opt==tree_hash(opt.state_dict()),'SA changed Adam state'
    if 'fast_cell.A' in initial:assert torch.equal(initial['fast_cell.A'],m.state_dict()['fast_cell.A'])
    end_loss=loss_value(m,x,y);assert end_loss==best_loss
    return {'algorithm':'greedy_T0' if greedy else 'simulated_annealing','seed':seed,'steps':steps,
            'temperature_probes':pilot,'initial_temperature':initial_temperature,'initial_loss':initial_loss,
            'returned_loss':best_loss,'chain_final_loss':current_loss,'best_step':best_step,
            'accepted':sum(r['accepted'] for r in rows),'uphill_accepted':sum(r['uphill_accepted'] for r in rows),
            'returned_initial':tree_hash(initial)==tree_hash(best),'initial_state_sha256':tree_hash(initial),
            'returned_state_sha256':tree_hash(best),'chain_final_state_sha256':chain_hash,
            'adam_unchanged':True,'backward_calls':0,'scales':scales,'proposals':rows,
            'forward_calls':probes+steps+2,'target_exposures':(probes+steps+2)*y.numel(),
            'wall_seconds':time.perf_counter()-tick,'cpu_seconds':time.process_time()-cpu}

def train_run(m,opt,corpus,seed,run,deadline):
    tick=time.perf_counter();cpu=time.process_time();losses=[]
    offset=0 if run=='A' else CONFIG['bp_runs']['A']['updates']
    for step in range(1,CONFIG['bp_runs'][run]['updates']+1):
        x,y=corpus.batch(seed,offset+step);losses.append(update(m,opt,x,y,lr_for(run,step)))
        if step%32==0:deadline()
    return {'run':run,'updates':len(losses),'losses':losses,'target_exposures':len(losses)*2048,
            'wall_seconds':time.perf_counter()-tick,'cpu_seconds':time.process_time()-cpu}

def extra_bp(m,opt,x,y,budget,lr):
    tick=time.perf_counter();cpu=time.process_time();losses=[];last=0.
    while time.perf_counter()-tick<budget:
        start=time.perf_counter();losses.append(update(m,opt,x,y,lr));last=time.perf_counter()-start
    elapsed=time.perf_counter()-tick
    return {'algorithm':'extra_BP','updates':len(losses),'losses':losses,'lr':lr,'matched_sa_budget_seconds':budget,
            'wall_seconds':elapsed,'cpu_seconds':time.process_time()-cpu,'overshoot_seconds':elapsed-budget,
            'last_step_seconds':last,'target_exposures':len(losses)*y.numel()}

def freeze():
    assert json.loads((OUT/'PREFLIGHT.json').read_text())['status']=='PASS'
    p=copy.deepcopy(CONFIG);p['data']=json.loads((ROOT/'data/manifest.json').read_text());p['source_hashes']=source_manifest()
    dest=OUT/'PROTOCOL.json'
    if dest.exists():assert json.loads(dest.read_text())==p
    else:save(dest,p)
    save(ROOT/'auto-research/RUN_REGISTRY.json',{'runs':[{'run_id':RUN,'status':'REGISTERED','protocol_sha256':sha(dest)}]})
    print('REGISTERED',sha(dest),flush=True)

def main_run():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    p=json.loads((OUT/'PROTOCOL.json').read_text());assert p['source_hashes']==source_manifest()
    for name,meta in p['data']['splits'].items():assert sha(ROOT/'data'/(name+'.npz'))==meta['sha256']
    assert not (OUT/'START.json').exists(),'Refuse to overwrite a started run'
    runtime={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
             'threads':1,'device':'cpu','dtype':'float32'}
    save(OUT/'START.json',{'status':'RUNNING','runtime':runtime,'protocol_sha256':sha(OUT/'PROTOCOL.json')})
    train=Corpus('train');confirmation=Corpus('confirmation');clock=time.monotonic();cpu=time.process_time();completed=[]
    def deadline():
        if time.monotonic()-clock>CONFIG['limits']['max_main_wall_seconds']:raise TimeoutError('Frozen main time budget exhausted')
    try:
        for kind,width in CONFIG['widths'].items():
            config=dict(vocab_size=2048,width=width,max_context=512,segment_length=32,document_segments=8)
            for seed in CONFIG['seeds']:
                deadline();folder=OUT/f'{kind}-{seed}';folder.mkdir(parents=True,exist_ok=True)
                def make(w=None,o=None):
                    m=build(kind,config,seed);opt=optimizer(m,.002)
                    if w is not None:m.load_state_dict(w)
                    if o is not None:opt.load_state_dict(copy.deepcopy(o))
                    return m,opt
                m,opt=make();initial=weights(m)
                a=train_run(m,opt,train,seed,'A',deadline);parent=weights(m);parent_opt=copy.deepcopy(opt.state_dict())
                parent_eval=evaluate(m,confirmation)
                save(folder/'PARENT.json',{'parameters':sum(v.numel() for v in m.parameters()),'config':config,
                    'run_A':a,'confirmation':parent_eval,'state_sha256':tree_hash(parent),'adam_sha256':tree_hash(parent_opt)})
                print('PARENT',kind,seed,round(parent_eval['ce'],6),flush=True)
                def record(method,m,opt,runs,phase=None,boundary=None):
                    final=evaluate(m,confirmation);duplicate=evaluate(m,confirmation);assert final==duplicate
                    row={'architecture':kind,'seed':seed,'method':method,'parameters':sum(v.numel() for v in m.parameters()),
                         'final':final,'duplicate_exact':True,'boundary':boundary,'bp_runs':runs,'phase':phase,
                         'bp_updates':sum(r['updates'] for r in runs)+(phase.get('updates',0) if phase else 0),
                         'total_training_wall_seconds':sum(r['wall_seconds'] for r in runs)+(phase['wall_seconds'] if phase else 0),
                         'total_training_cpu_seconds':sum(r['cpu_seconds'] for r in runs)+(phase['cpu_seconds'] if phase else 0),
                         'target_exposures':sum(r['target_exposures'] for r in runs)+(phase['target_exposures'] if phase else 0),
                         'final_state_sha256':tree_hash(m.state_dict()),'final_adam_sha256':tree_hash(opt.state_dict())}
                    if kind=='paragru':row['solver_max_residual']=m.fast_cell.max_residual
                    save(folder/(method+'.json'),row)
                    if method=='sa_mid_keep' and seed==CONFIG['seeds'][0]:torch.save(m.state_dict(),folder/(method+'.pt'))
                    completed.append(row)
                    save(OUT/'PROGRESS.json',{'status':'RUNNING','completed':len(completed),'planned':99,
                         'last':{'architecture':kind,'seed':seed,'method':method,'ce':final['ce']},'elapsed_seconds':time.monotonic()-clock})
                    print('COMPLETE',kind,seed,method,round(final['ce'],6),'seconds',round(row['total_training_wall_seconds'],2),flush=True)
                    deadline();return row
                # No-op boundary with preserved Adam; its LR restart is shared by every arm.
                b=train_run(m,opt,train,seed,'B',deadline);base_end=weights(m);base_opt=copy.deepcopy(opt.state_dict())
                record('baseline_keep',m,opt,[a,b],boundary=parent_eval)
                m,opt=make(parent);b_reset=train_run(m,opt,train,seed,'B',deadline)
                record('baseline_reset',m,opt,[a,b_reset],boundary=parent_eval)
                # Full SA middle phase, reused endpoint for the keep/reset Adam comparison.
                xmid,ymid=train.batch(seed,CONFIG['bp_runs']['A']['updates'])
                m,opt=make(parent,parent_opt);mid=anneal(m,opt,xmid,ymid,seed*100+2);mid_w=weights(m);mid_eval=evaluate(m,confirmation)
                save(folder/'PHASE_MIDDLE.json',mid)
                for method,retain in [('sa_mid_keep',True),('sa_mid_reset',False)]:
                    m,opt=make(mid_w,parent_opt if retain else None);br=train_run(m,opt,train,seed,'B',deadline)
                    record(method,m,opt,[a,br],mid,mid_eval)
                m,opt=make(parent,parent_opt);extra=extra_bp(m,opt,xmid,ymid,mid['wall_seconds'],lr_for('A',192))
                extra_w=weights(m);extra_opt=copy.deepcopy(opt.state_dict());extra_eval=evaluate(m,confirmation)
                save(folder/'PHASE_MIDDLE_EXTRA.json',extra)
                for method,retain in [('extra_mid_keep',True),('extra_mid_reset',False)]:
                    m,opt=make(extra_w,extra_opt if retain else None);br=train_run(m,opt,train,seed,'B',deadline)
                    record(method,m,opt,[a,br],extra,extra_eval)
                m,opt=make(parent,parent_opt);greedy=anneal(m,opt,xmid,ymid,seed*100+2,greedy=True);greedy_eval=evaluate(m,confirmation)
                br=train_run(m,opt,train,seed,'B',deadline);record('greedy_mid_keep',m,opt,[a,br],greedy,greedy_eval)
                # Post phase is genuinely after both complete BP runs, with no later BP.
                xpost,ypost=train.batch(seed,192+96)
                m,opt=make(base_end,base_opt);post=anneal(m,opt,xpost,ypost,seed*100+3)
                record('sa_post',m,opt,[a,b],post)
                m,opt=make(base_end,base_opt);post_extra=extra_bp(m,opt,xpost,ypost,post['wall_seconds'],lr_for('B',96))
                record('extra_post',m,opt,[a,b],post_extra)
                # Pre phase precedes every backward call. Its extra-BP control clears Adam before A.
                xpre,ypre=train.batch(seed,1)
                m,opt=make(initial);pre=anneal(m,opt,xpre,ypre,seed*100+1)
                ar=train_run(m,opt,train,seed,'A',deadline);pre_boundary=evaluate(m,confirmation)
                br=train_run(m,opt,train,seed,'B',deadline);record('sa_pre',m,opt,[ar,br],pre,pre_boundary)
                m,opt=make(initial);pre_extra=extra_bp(m,opt,xpre,ypre,pre['wall_seconds'],.002);opt=optimizer(m,.002)
                ar=train_run(m,opt,train,seed,'A',deadline);extra_boundary=evaluate(m,confirmation)
                br=train_run(m,opt,train,seed,'B',deadline);record('extra_pre',m,opt,[ar,br],pre_extra,extra_boundary)
        assert len(completed)==99
        terminal={'status':'COMPLETE','rows':len(completed),'wall_seconds':time.monotonic()-clock,'cpu_seconds':time.process_time()-cpu,
                  'runtime':runtime,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'protocol_sha256':sha(OUT/'PROTOCOL.json'),'source_seal_verified':source_manifest()==p['source_hashes'],
                  'official_validation_reads':0,'official_test_reads':0}
        save(OUT/'TERMINAL.json',terminal);save(OUT/'PROGRESS.json',{'status':'COMPLETE','completed':99,'planned':99,'elapsed_seconds':terminal['wall_seconds']})
        save(ROOT/'auto-research/RUN_REGISTRY.json',{'runs':[{'run_id':RUN,**terminal}]});print('ALL_COMPLETE',json.dumps(terminal),flush=True)
    except BaseException as e:
        save(OUT/'TERMINAL.json',{'status':'FAILED_OR_INCOMPLETE','error':repr(e),'completed_rows':len(completed),'elapsed_seconds':time.monotonic()-clock})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.action=='freeze' else main_run()
