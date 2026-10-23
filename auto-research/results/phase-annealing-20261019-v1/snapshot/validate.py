"""Training-only preflight for standalone Metropolis phases and BP boundaries."""
import copy,json,time
from unittest.mock import patch
import torch
import experiment as e

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    tests=[]
    provenance=json.loads((e.ROOT/'UPSTREAM_PROVENANCE.json').read_text())
    import hashlib
    for row in provenance['files']:
        b=(e.ROOT/row['local']).read_bytes()
        assert hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==row['git_blob_sha1']
    tests.append({'check':'pinned_upstream_blob_hashes','status':'PASS','files':len(provenance['files'])})
    assert e.accept_move(-1,0,.999) and e.accept_move(0,0,.9)
    assert not e.accept_move(.1,0,.00001)
    assert e.accept_move(.1,.1,.1) and not e.accept_move(.1,.1,.9)
    tests.append({'check':'metropolis_acceptance_formula','status':'PASS'})
    # Deterministic 2D quadratic verifies best-state restoration and genuine uphill acceptance.
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.tensor([1.,-1.]))
    def energy(m,x,y):return float(((m.w-torch.tensor([.2,.2]))**2).sum())
    tiny=Tiny();op=torch.optim.AdamW(tiny.parameters());initial=e.weights(tiny)
    with patch.object(e,'loss_value',energy):
        r=e.anneal(tiny,op,torch.zeros(1),torch.zeros(1),9021,steps=64)
        tiny.load_state_dict(initial);g=e.anneal(tiny,op,torch.zeros(1),torch.zeros(1),9021,greedy=True,steps=64)
    assert r['returned_loss']<=r['initial_loss'] and g['returned_loss']<=g['initial_loss']
    assert r['uphill_accepted']>0 and g['uphill_accepted']==0
    assert r['returned_loss']==min([r['initial_loss']]+[x['chain_loss'] for x in r['proposals']])
    tests.append({'check':'quadratic_chain_and_best_restore','status':'PASS','sa_uphill_accepted':r['uphill_accepted'],
                  'sa_best_loss':r['returned_loss'],'greedy_uphill_accepted':g['uphill_accepted']})
    corpus=e.Corpus('train');x,y=corpus.batch(9907,1);timings={}
    for kind,width in e.CONFIG['widths'].items():
        config=dict(vocab_size=2048,width=width,max_context=512,segment_length=32,document_segments=8)
        m=e.build(kind,config,9907);opt=e.optimizer(m,.002);times=[]
        for i in range(8):
            t=time.perf_counter();e.update(m,opt,x,y,.001);times.append(time.perf_counter()-t)
        before=e.weights(m);state=copy.deepcopy(opt.state_dict())
        # Reconstructed no-op boundaries preserve the exact next Adam update.
        clone=e.build(kind,config,9907);co=e.optimizer(clone,.002)
        clone.load_state_dict(before);co.load_state_dict(copy.deepcopy(state))
        l1=e.update(m,opt,x,y,e.lr_for('B',1));l2=e.update(clone,co,x,y,e.lr_for('B',1))
        assert l1==l2 and e.tree_hash(m.state_dict())==e.tree_hash(clone.state_dict())
        old_adam=e.tree_hash(opt.state_dict())
        with patch('torch.autograd.backward',side_effect=AssertionError('Backward called during SA')):
            result=e.anneal(m,opt,x,y,9907,steps=8,probes=2)
        assert old_adam==e.tree_hash(opt.state_dict()) and result['returned_loss']<=result['initial_loss']
        fresh=e.optimizer(m,.002);assert len(fresh.state)==0 and len(opt.state)>0
        tests.append({'check':'no_backward_adam_preserved_and_boundary_identity','kind':kind,'status':'PASS',
                      'sa_initial_loss':result['initial_loss'],'sa_best_loss':result['returned_loss'],
                      'sa_uphill_accepted':result['uphill_accepted']})
        timings[kind]={'bp_step_mean_seconds':sum(times[4:])/4,
                       'sa_forward_and_proposal_mean_seconds':result['wall_seconds']/result['forward_calls']}
    meta=json.loads((e.ROOT/'data/manifest.json').read_text())
    training={r['sha256'] for r in meta['documents'] if r['split']=='train'}
    confirmation={r['sha256'] for r in meta['documents'] if r['split']=='confirmation'}
    assert not (training&confirmation)
    assert not (confirmation&set(meta['previously_evaluated_monitor_documents_excluded']))
    assert all(80<=int(h[:16],16)%100<90 for h in confirmation)
    tests.append({'check':'confirmation_document_separation','status':'PASS','fresh_documents':len(confirmation)})
    estimate=sum(3*(1440*t['bp_step_mean_seconds']+7*106*t['sa_forward_and_proposal_mean_seconds']) for t in timings.values())
    e.save(e.OUT/'PREFLIGHT.json',{'status':'PASS','tests':tests,'training_only_timings':timings,
          'rough_main_seconds_excluding_eval':estimate,'confirmation_evaluations_during_preflight':0})
    print(json.dumps({'status':'PASS','checks':len(tests),'timings':timings,'rough_main_seconds_excluding_eval':estimate},indent=2))

if __name__=='__main__':main()
