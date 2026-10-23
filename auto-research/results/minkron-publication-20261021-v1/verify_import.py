"""Audit compact minKron evidence using the standard library; no model execution."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def near(a, b, name):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), name)


def main():
    inventory = read(ROOT / 'IMPORT.json')
    included = excluded = python_files = 0
    for archive in inventory['archives']:
        files = {f['path']: f for f in archive['files']}
        for f in files.values():
            if 'published_path' not in f:
                require(bool(f.get('excluded_reason')), 'Unexplained exclusion')
                excluded += 1
                continue
            path = ROOT / f['published_path']
            require(path.resolve().is_relative_to(ROOT), 'Unsafe import path')
            require(path.is_file(), f'Missing {path}')
            require(path.stat().st_size == f['bytes'] and sha(path) == f['sha256'], f'Changed {path}')
            included += 1
            if path.suffix == '.py':
                compile(path.read_bytes(), str(path), 'exec')
                python_files += 1
        original = read(ROOT / archive['original_manifest'])
        rows = original if isinstance(original, list) else original['files']
        require(len(rows) == archive['original_manifest_entries_verified'], 'Manifest count')
        for f in rows:
            require(f['sha256'] == files[f['path']]['sha256'], 'Original seal mismatch')
            require(f['bytes'] == files[f['path']]['bytes'], 'Original size mismatch')

    compact = ROOT / 'compact'
    protocol = read(compact / 'protocol.json')
    summary = read(compact / 'summary.json')
    for path, digest in protocol['source_sha256'].items():
        require(sha(compact / path) == digest, f'Compact source seal {path}')
    source_entries = {f['path']: f for f in inventory['archives'][0]['files']}
    metrics = {}
    pairing = {}
    raw_cases = confirmation_rows = 0
    for arm, values in summary['arms'].items():
        primary = {mode: [] for mode in protocol['confirmation']['primary_modes']}
        for i, seed in enumerate(protocol['seeds']):
            name = f'{arm}_{seed}_{protocol["learning_rates"][arm]:g}'
            train = read(compact / 'results/main' / name / 'terminal.json')
            confirmation = read(compact / 'results/confirmation' / f'{arm}_{seed}' / 'terminal.json')
            require(train['status'] == confirmation['status'] == 'completed', 'Incomplete run')
            require(train['checkpoint_sha256'] == confirmation['checkpoint_sha256'], 'Checkpoint association')
            require(train['checkpoint_sha256'] == source_entries[train['checkpoint']]['sha256'], 'Checkpoint archive seal')
            require(train['updates'] == sum(p['steps'] for p in train['phases']), 'Update accounting')
            require(train['tokens'] == sum(p['tokens'] for p in train['phases']) == values['tokens'][i], 'Token accounting')
            near(train['training_cpu_seconds'], sum(p['training_cpu_seconds'] for p in train['phases']), 'CPU accounting')
            near(train['training_cpu_seconds'], values['cpu_seconds'][i], 'Summary CPU')
            baseline = read(compact / 'results/main' / f'mingru_{seed}_0.003/terminal.json')
            for phase, base in zip(train['phases'], baseline['phases']):
                require(phase['phase'] == base['phase'], 'Stage association')
                require(phase['steps'] <= base['steps'], 'Update cap')
                require(phase['training_cpu_seconds'] <= base['training_cpu_seconds'] + 1e-8, 'CPU cap')
            for row in confirmation['records']:
                require(0 <= row['correct'] <= row['n'], 'Invalid confirmation counts')
                near(row['correct'] / row['n'], row['accuracy'], 'Accuracy mismatch')
                key = (seed, row['mode'], row['age'])
                if key in pairing:
                    require(pairing[key] == row['data_sha256'], 'Unpaired confirmation data')
                pairing[key] = row['data_sha256']
                if row['age'] == 512 and row['mode'] in primary:
                    primary[row['mode']].append(row['accuracy'])
                raw_cases += row['n']
                confirmation_rows += 1
        for mode, rows in primary.items():
            require(rows == values['primary'][mode]['values'], 'Per-seed summary mismatch')
            near(statistics.mean(rows), values['primary'][mode]['mean'], 'Mean mismatch')
            near(statistics.stdev(rows), values['primary'][mode]['sample_sd'], 'SD mismatch')
        if arm != 'mingru':
            require(values['strict_resource_and_mean_accuracy_screen'] is False, 'Preserve failed overall screen')
        metrics[arm] = {m: statistics.mean(v) for m, v in primary.items()}
    require(confirmation_rows == 192, 'Expected four arms, three seeds, sixteen task/age cells')
    require(len(list((compact / 'results/pilot').glob('*/terminal.json'))) == 8, 'Retain all pilots')

    nano = ROOT / 'nanogpt'
    run = nano / 'research-scratch/cpu-124m/main'
    scale = read(run / 'summary.json')
    plan = read(run / 'protocol.json')
    assessment = read(run / 'assessment.json')
    for path, digest in plan['code_sha256'].items():
        require(sha(nano / path) == digest == scale['source_sha256'][path], f'Scale source seal {path}')
    data_manifest = read(nano / 'data/fineweb_cpu/manifest.json')
    canonical_data_hash = hashlib.sha256(json.dumps(data_manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(canonical_data_hash == plan['data_manifest_sha256'], 'Canonical data manifest seal')
    require(sha(nano / 'experiments/cpu_assess.py') == assessment['assessment_script_sha256'], 'Assessment source seal')
    require(scale['steps'] == 262, 'Training update count')
    require(scale['steps'] * plan['targets_per_optimizer_step'] == scale['scored_tokens'] == 134144, 'Final exposure')
    require(scale['status'] == 'COMPLETED' and scale['nonfinite_count'] == 0, 'Scale completion')
    near(scale['scored_tokens'] / scale['elapsed_seconds'], scale['throughput_full_loop_tokens_per_second'], 'Throughput')
    near(math.exp(assessment['confirmation']['nll']), assessment['confirmation']['perplexity'], 'Confirmation perplexity')
    curves = read(run / 'curves.json')
    near(curves[-1]['nll'], scale['final_development']['nll'], 'Final development curve')
    require(curves[0]['evaluation_tokens'] == curves[-1]['evaluation_tokens'] == 8192, 'Development comparison target count')
    generations = [json.loads(line) for line in (run / 'generations.jsonl').read_text().splitlines()]
    require(len(generations) == assessment['generation']['count'] == 10, 'All ten generations')
    for i, row in enumerate(generations):
        require(row['number'] == i + 1 and row['seed'] == plan['generation']['seeds'][i], 'Generation seed/order')
        require(row['prompt'] == plan['generation']['prompts'][i], 'Generation prompt')
        require(len(row['generated_ids']) == row['generated_tokens'], 'Generation token count')
    require(sum(g['generated_tokens'] for g in generations) == assessment['generation']['generated_tokens'], 'Generation total')
    require(len(read(nano / 'research-scratch/cpu-124m/startup_failures.json')) == 3, 'Keep failed pilot startups')
    require(assessment['checkpoint']['sha256'] == inventory['external_scale_checkpoint']['sha256'], 'External checkpoint association')
    initial = ROOT / 'integration'
    initial_status = read(initial / 'research-scratch/nanogpt-memory/status.json')
    require(sha(initial / 'experiments/nanogpt_memory/model.py') == initial_status['source_model_sha256'], 'Original integration model identity')
    require(initial_status['status'] == 'CPU_INTEGRATION_PASS_GPU_FINEWEB_PENDING', 'Do not relabel proposed GPU work completed')
    result = {
        'status': 'PASS', 'kind': 'Stored-artifact audit; no model execution',
        'included_file_hashes': included, 'explicitly_excluded_members': excluded,
        'python_sources_compiled_without_execution': python_files,
        'compact_confirmation_rows': confirmation_rows, 'compact_cases_in_count_receipts': raw_cases,
        'compact_primary_means': metrics, 'compact_main_runs': 12, 'compact_lr_pilots': 8,
        'scale_updates': scale['steps'], 'scale_scored_tokens': scale['scored_tokens'],
        'scale_confirmation_nll': assessment['confirmation']['nll'], 'generations': len(generations),
        'new_training': False, 'new_model_evaluation': False,
        'external_checkpoint_reloaded': False, 'raw_predictions_checked_in_this_audit': False,
        'omitted_artifacts': 'See RECONSTRUCTION_REQUIRED.md and IMPORT.json. Raw predictions and full training trace were checked locally before slimming; see PREPARATION_AUDIT.json.',
        'original_checks': 'Preserved receipts, not newly repeated numerical fidelity tests.'
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
