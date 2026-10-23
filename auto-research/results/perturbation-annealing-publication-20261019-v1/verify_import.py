"""Verify the two published snapshots without training, inference or downloads."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    publication = read(HERE / 'IMPORT.json')
    registry = read(ROOT / 'auto-research/RUN_REGISTRY.json')
    checks = []
    for item in publication['experiments']:
        snapshot = ROOT / item['snapshot']
        raw = ROOT / item['raw_results']
        protocol = read(raw / 'PROTOCOL.json')
        terminal = read(raw / 'TERMINAL.json')
        analysis = read(raw / 'ANALYSIS.json')
        audit = read(raw / 'ARTIFACT_AUDIT.json')
        assert terminal['status'] == 'COMPLETE' and terminal['source_seal_verified']
        assert sha(raw / 'PROTOCOL.json') == item['protocol_sha256'] == terminal['protocol_sha256']
        assert analysis['protocol_sha256'] == item['protocol_sha256']
        assert audit['status'] == 'PASS' and audit['branches_audited'] == item['completed_outcomes']
        for name, expected in protocol['source_hashes'].items():
            assert sha(snapshot / name) == expected, (item['run_id'], name)
        inventory = read(snapshot / 'ARTIFACT_MANIFEST.json')['files']
        if isinstance(inventory, dict):
            inventory = [dict(path=k, **v) for k,v in inventory.items()]
        for entry in inventory:
            file = snapshot / entry['path']
            assert file.stat().st_size == entry['bytes'] and sha(file) == entry['sha256']
        rows = 0
        for architecture in protocol['widths']:
            for method in protocol['methods']:
                values = []
                for seed in protocol['seeds']:
                    record = read(raw / f'{architecture}-{seed}' / (method + '.json'))
                    assert record['duplicate_exact']
                    assert record['architecture'] == architecture and record['seed'] == seed and record['method'] == method
                    final = record.get('final', record.get('followup'))
                    assert final['targets'] == 16384 and math.isfinite(final['ce'])
                    assert abs(statistics.mean(final['batch_ce']) - final['ce']) < 1e-12
                    assert len(record['final_state_sha256']) == 64
                    values.append(final['ce']); rows += 1
                expected = analysis['architectures'][architecture]['mean_final_ce'][method]
                assert abs(statistics.mean(values) - expected) < 1e-12
        assert rows == item['completed_outcomes'] == terminal['rows']
        assert terminal['official_test_reads'] == 0 and terminal['official_validation_reads'] == 0
        entry = [r for r in registry['runs'] if r.get('run_id') == item['run_id']]
        assert len(entry) == 1 and entry[0]['protocol_sha256'] == item['protocol_sha256']
        for key in ['protocol', 'terminal', 'metrics', 'report', 'audit']:
            assert (ROOT / item[key]).is_file()
        for name in ['summary.png', 'ANALYSIS.json']:
            assert sha(raw / name) == sha(ROOT / 'auto-research/graphs' / item['run_id'] / name)
        assert sha(raw / 'run.log') == sha(ROOT / 'auto-research/logs' / item['run_id'] / 'run.log')
        checks.append({'run_id':item['run_id'], 'outcomes':rows,
                       'original_artifact_hashes':True, 'protocol_and_source_seals':True,
                       'reported_means_recomputed':True, 'duplicate_receipts_preserved':True,
                       'charts_and_logs_identical':True})
    return {'status':'PASS', 'scope':'Stored evidence verification only; no new model execution',
            'completed_outcomes':sum(x['outcomes'] for x in checks), 'experiments':checks,
            'new_training':False, 'new_model_evaluation':False, 'official_test_access':False}

if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
