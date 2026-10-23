"""Verify complete pinned archive inventories and every disjoint restore root."""
from datetime import datetime, timezone
import hashlib
import json
import platform
import time

from restore_plan import HERE, build_plan, load_snapshot, require


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def compact(plan):
    return {key: value for key, value in plan.items() if key != 'payloads'}


def root_prefixes(manifest):
    prefixes = set()
    for row in manifest['files']:
        parts = row['restore_path'].split('/')
        prefixes.add('/'.join(parts[:2]) + ('/' if len(parts) > 2 else ''))
    ordered = sorted(prefixes)
    require(not any(right.startswith(left) for left, right in zip(ordered, ordered[1:])),
            'Root selections overlap')
    return ordered


def verify():
    wall, cpu = time.perf_counter(), time.process_time()
    snapshots, current = {}, None
    for name in ('current', 'historical'):
        manifest, summary, sha = load_snapshot(name)
        require(sha == summary['files_manifest_sha256'], 'Summary/index seal mismatch')
        plan = build_plan(manifest, '', summary.get('release_states', ()))
        for metric, field in [('selected_paths', 'mapped_source_paths'),
                              ('selected_unique_content_objects', 'unique_content_objects'),
                              ('selected_unique_content_bytes', 'unique_content_bytes'),
                              ('pending_unique_objects', 'pending_objects')]:
            require(plan[metric] == summary[field], 'Snapshot accounting: ' + name + '/' + metric)
        require(plan['restorable_from_saved_locations'] == (summary['pending_objects'] == 0),
                'Pending status mismatch')
        snapshots[name] = {'manifest_sha256': sha, 'plan_sha256': hashlib.sha256(canonical(plan)).hexdigest(),
                           'plan': compact(plan)}
        if name == 'current':
            current = manifest, summary, plan
    manifest, summary, whole = current
    roots = root_prefixes(manifest)
    root_rows, root_hash = [], hashlib.sha256()
    for prefix in roots:
        plan = build_plan(manifest, prefix, summary.get('release_states', ()))
        require(plan['restorable_from_saved_locations'], 'Current root blocked: ' + prefix)
        root_hash.update(canonical(plan) + b'\n')
        root_rows.append([prefix, plan['selected_paths'], plan['selected_unique_content_objects'],
                          plan['known_payloads'], plan['known_download_payload_bytes'],
                          plan['selected_output_logical_bytes']])
    require(sum(row[1] for row in root_rows) == whole['selected_paths'], 'Root path partition mismatch')
    require(sum(row[5] for row in root_rows) == whole['selected_output_logical_bytes'],
            'Root output-byte partition mismatch')
    examples = sorted(row['restore_path'] for row in manifest['files']
                      if row['restore_path'].endswith('/README.md')
                      and row['location']['kind'] == 'archive_object' and row['bytes'] > 0)
    require(examples, 'Missing README example')
    example = build_plan(manifest, examples[0], summary.get('release_states', ()))
    result = {
        'status': 'PASS_PINNED_METADATA_PLANS_ONLY',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(),
        'implementation_sha256': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                  for name in ('restore_plan.py', 'test_restore_plan.py', 'verify_plans.py')},
        'snapshots': snapshots,
        'current_root_prefixes_checked': len(roots),
        'root_plan_hash_order': 'Canonical sorted-key JSON plus newline, in prefix lexical order',
        'root_plans_sha256': root_hash.hexdigest(),
        'root_columns': ['prefix', 'selected_paths', 'unique_content_objects', 'known_payloads',
                         'known_download_payload_bytes', 'selected_output_logical_bytes'],
        'root_plans': root_rows,
        'example_selection': 'Lexicographically first positive-byte archived README.md; no sampling or tuning',
        'readme_example': example,
        'wall_seconds': time.perf_counter() - wall,
        'cpu_seconds': time.process_time() - cpu,
        'artifact_transfers': False,
        'new_model_execution': False,
        'live_release_metadata_checked': False,
    }
    require(result['cpu_seconds'] <= 120, 'Single metadata batch CPU cap exceeded')
    return result


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True))
