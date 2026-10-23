"""Audit raw outcomes and reload one SA sandwich checkpoint per architecture."""
from pathlib import Path
import json,sys
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2];sys.path.insert(0,str(ROOT))
import torch
import experiment as e

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());terminal=json.loads((OUT/'TERMINAL.json').read_text())
    assert terminal['status']=='COMPLETE' and terminal['rows']==99
    assert e.source_manifest()==protocol['source_hashes']
    assert e.sha(OUT/'PROTOCOL.json')==terminal['protocol_sha256']
    checks=[];rows=[]
    for kind,width in protocol['widths'].items():
        for seed in protocol['seeds']:
            folder=OUT/f'{kind}-{seed}'
            data={m:json.loads((folder/(m+'.json')).read_text()) for m in protocol['methods']}
            for method,row in data.items():
                assert row['duplicate_exact']
                assert [r['updates'] for r in row['bp_runs']]==[192,96]
                for run in row['bp_runs']:assert len(run['losses'])==run['updates']
                phase=row['phase']
                if phase and phase['algorithm'] in ('simulated_annealing','greedy_T0'):
                    assert phase['backward_calls']==0 and phase['adam_unchanged']
                    assert phase['forward_calls']==106 and phase['target_exposures']==217088
                    assert phase['returned_loss']<=phase['initial_loss']
                    assert phase['returned_loss']==min([phase['initial_loss']]+[r['chain_loss'] for r in phase['proposals']])
                    for proposal in phase['proposals']:
                        assert proposal['accepted']==e.accept_move(proposal['delta'],proposal['temperature'],proposal['uniform'])
                    if phase['algorithm']=='greedy_T0':assert phase['uphill_accepted']==0
                if phase and phase['algorithm']=='extra_BP':
                    assert len(phase['losses'])==phase['updates']
                    assert 0<=phase['overshoot_seconds']<=phase['last_step_seconds']+.002
                rows.append(row)
            assert data['sa_mid_keep']['phase']==data['sa_mid_reset']['phase']
            assert data['extra_mid_keep']['phase']==data['extra_mid_reset']['phase']
            if seed==protocol['seeds'][0]:
                model=e.build(kind,dict(vocab_size=2048,width=width,max_context=512,segment_length=32,document_segments=8),seed)
                model.load_state_dict(torch.load(folder/'sa_mid_keep.pt',map_location='cpu',weights_only=True))
                assert e.tree_hash(model.state_dict())==data['sa_mid_keep']['final_state_sha256']
                actual=e.evaluate(model,e.Corpus('confirmation'));assert actual==data['sa_mid_keep']['final']
                checks.append({'architecture':kind,'seed':seed,'state_hash_exact':True,'fresh_process_evaluation_exact':True,'ce':actual['ce']})
    assert len(rows)==99
    result={'status':'PASS','branches_audited':len(rows),'protocol_and_source_seals':True,
            'metropolis_decisions_recomputed':True,'best_checkpoint_rule_verified':True,'fresh_checkpoint_checks':checks}
    e.save(OUT/'ARTIFACT_AUDIT.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
