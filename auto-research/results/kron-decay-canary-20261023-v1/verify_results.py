"""Check saved canary coverage, gates, analytic tiny references and resource receipts.

Standard library only. This does not replay tensors or restore gradient arrays.
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1] / 'results/minkron-fast-20261021-v1'
CORE_GATES = (2, 4, 6, 8, 10, 12)
MODEL_CHECKS = {'forward_state', 'loss', 'gradients', 'optimizer_step',
                'optimizer_state', 'streaming', 'reset', 'causality'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_cases(protocol):
    design = protocol['design']
    cases, tiny = set(), {}
    for dtype in design['tiny_core_cases']['parameter_dtypes']:
        for gates in design['tiny_core_cases']['g_sequences']:
            for initial in design['tiny_core_cases']['initial_states']:
                for reset in (False, True):
                    label = f'tiny/torch.{dtype}/{tuple(gates)}/S{initial}/R{int(reset)}'
                    cases.add(label)
                    tiny[label] = dtype, gates, initial, reset
    for dtype in design['tiny_core_cases']['parameter_dtypes']:
        for seed in design['random_core_cases']['seeds']:
            for shape in design['random_core_cases']['shapes_B_T_H_K_V']:
                cases.add(f'random/torch.{dtype}/{seed}/' + ','.join(map(str, shape)))
    return cases, tiny


def validate(result, protocol, resources):
    require(protocol['authorization']['approved'] is True, 'Missing exact approval')
    require(result['status'] == 'COMPLETED_SYNTHETIC_MECHANICS_CANARY', 'Incomplete attempt')
    require(result['threads'] == result['interop_threads'] == 1, 'Thread limit')
    require(result['torch_version'] == '2.9.1+cpu' and result['cuda_build'] is None, 'CPU build')
    require(result['corpus_training'] is False and result['official_test_access'] is False
            and result['checkpoints_saved'] is False, 'Scope flags')
    variants = protocol['design']['variants']
    cases, tiny = expected_cases(protocol)
    rows = result['core_rows']
    lookup = {(row[0], row[1]): row for row in rows}
    require(len(lookup) == len(rows), 'Duplicate core row')
    require(set(lookup) == {(case, variant) for case in cases for variant in variants}, 'Core coverage')
    for row in rows:
        require(len(row) == len(result['core_columns']) == 15, 'Core columns')
        require(all(type(row[i]) is bool for i in CORE_GATES), 'Core gate type')
        dtype = 'torch.float64' if '/torch.float64/' in row[0] else 'torch.float32'
        require(row[13] == dtype, 'Core accumulation dtype')
        require(row[4] == (not row[14]), 'Gradient failure names')
    primary = all(all(lookup[case, 'interval_direct'][i] for i in CORE_GATES) for case in cases)
    require(primary == result['primary_core_gate_passed'], 'Primary gate mismatch')
    model_variants = variants + ['source_factored_fp64_core']
    model_lookup = {row['variant']: row for row in result['model_rows']}
    require(len(model_lookup) == len(result['model_rows']) and set(model_lookup) == set(model_variants),
            'Model coverage')
    model_gates = {}
    for name, row in model_lookup.items():
        require(row['recurrence_fp64'] is (name == 'source_factored_fp64_core'), 'Model precision policy')
        checks = {key: value for key, value in row.items() if isinstance(value, dict)}
        require('forward_state' in checks and set(checks) <= MODEL_CHECKS, 'Model check inventory')
        for check in checks.values():
            require(type(check['passed']) is bool and type(check['finite']) is bool, 'Model gate type')
            require(check['passed'] == (not check['failed']), 'Model failure names')
        model_gates[name] = set(checks) == MODEL_CHECKS and all(check['passed'] for check in checks.values())
    require(model_gates == result['model_gate_passed_by_variant'], 'Model gate mismatch')
    tiny_lookup = {(row['case'], row['variant']): row for row in result['tiny_outputs']}
    require(len(tiny_lookup) == len(result['tiny_outputs'])
            and set(tiny_lookup) == {(case, variant) for case in tiny for variant in variants},
            'Tiny diagnostic coverage')
    for (case, variant), record in tiny_lookup.items():
        row = lookup[case, variant]
        require(record['forward']['passed'] == row[2]
                and record['forward']['max_abs'] == row[3]
                and record['gradients']['passed'] == row[4]
                and record['gradients']['max_abs'] == row[5], 'Tiny/table disagreement')
        dtype, gates, initial, reset = tiny[case]
        state = 0.0 if reset else float(initial)
        analytic = []
        for gate in gates:
            state = .5 * math.exp(gate) * state + .5
            analytic.append(state)
        tolerance = protocol['tolerances'][dtype]
        require(len(record['reference_outputs']) == len(analytic)
                and len(record['reference_state']) == 1, 'Tiny reference shape')
        for saved, expected in zip(record['reference_outputs'] + record['reference_state'], analytic + [state]):
            require(math.isfinite(saved) and abs(saved - expected)
                    <= tolerance['atol'] + tolerance['rtol'] * abs(expected), 'Analytic tiny reference')
    metadata = {row['case']: row for row in result['cumulative_metadata']}
    require(len(metadata) == len(result['cumulative_metadata']) and set(metadata) == set(tiny),
            'Cumulative-log coverage')
    numerical = [row for row in resources if row['phase'] == 'numerical']
    require(len(numerical) == 1, 'Expected one completed numerical attempt')
    require(all(row['stopped_by_guard'] is None for row in resources), 'Resource guard stopped a process')
    require(sum(row['charged_cpu_seconds'] for row in numerical) <= 600
            and sum(row['charged_wall_seconds'] for row in numerical) <= 900, 'Cumulative numerical cap')
    require(all(row['peak_sampled_environment_temp_results_bytes'] < 1024**3 for row in resources),
            'Additional disk cap')
    require(numerical[0]['exit_code'] == (0 if primary else 1), 'Primary-only CLI exit status')
    require(0 <= result['cpu_seconds'] <= numerical[0]['charged_cpu_seconds'] <= 600,
            'Harness CPU accounting')
    summary = {'status': 'PASS_SAVED_RECEIPT_COHERENCE_NOT_NUMERICAL_REPLAY',
               'core_cases_per_variant': len(cases), 'core_rows': len(rows),
               'analytic_tiny_reference_records': len(tiny_lookup),
               'primary_core_gate_passed': primary, 'model_gates': model_gates,
               'core_summary': {}, 'model_summary': {},
               'charged_numerical_cpu_seconds': sum(row['charged_cpu_seconds'] for row in numerical),
               'charged_numerical_wall_seconds': sum(row['charged_wall_seconds'] for row in numerical),
               'peak_sampled_environment_temp_results_bytes': max(
                   row['peak_sampled_environment_temp_results_bytes'] for row in resources)}
    for name in variants:
        chosen = [row for row in rows if row[1] == name]
        summary['core_summary'][name] = {
            key: sum(row[index] for row in chosen) for key, index in
            [('forward_pass', 2), ('gradient_pass', 4), ('streaming_pass', 6),
             ('reset_pass', 8), ('causal_pass', 10), ('finite', 12)]}
        summary['core_summary'][name]['all_gates_pass'] = sum(
            all(row[index] for index in CORE_GATES) for row in chosen)
    for name, row in model_lookup.items():
        summary['model_summary'][name] = {
            'recurrence_fp64': row['recurrence_fp64'],
            'failed_gates': [key for key, value in row.items() if isinstance(value, dict) and not value['passed']],
            'gradients': row.get('gradients')}
    return summary


def verify():
    protocol = json.loads((HERE / 'CANARY_PROTOCOL.json').read_text())
    for name, expected in protocol['implementation_sha256'].items():
        require(hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected,
                'Frozen implementation changed: ' + name)
    inputs = json.loads((HERE / 'SOURCE_INPUTS.json').read_text())
    for name, expected in inputs['files'].items():
        require(hashlib.sha256((PARENT / name).read_bytes()).hexdigest() == expected,
                'Parent source changed: ' + name)
    result = json.loads((HERE / 'execution/attempt-01.stdout').read_text())
    resources = [json.loads(path.read_text()) for path in sorted((HERE / 'execution').glob('*.resource.json'))]
    summary = validate(result, protocol, resources)
    records = []
    for line in (HERE / 'execution/attempt-01.stderr').read_text().splitlines():
        if line.startswith('{'):
            records.append(json.loads(line))
    require([row['core_result'] for row in records if 'core_result' in row] == result['core_rows'],
            'Raw core diagnostics disagree')
    require([row['model_result'] for row in records if 'model_result' in row] == result['model_rows'],
            'Raw model diagnostics disagree')
    summary['raw_diagnostics_reconciled'] = True
    summary['unchecked'] = ['Full gradient arrays, independent tensor replay, wider model grid, '
                            'performance, corpus quality and historical failure root cause.']
    return summary


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True, allow_nan=False))
