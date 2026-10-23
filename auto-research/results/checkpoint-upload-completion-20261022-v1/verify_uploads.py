"""Verify the exact authorized pending-object set and uploaded checkpoint receipts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true')
    args = parser.parse_args()
    request = json.loads((HERE / 'INPUT.json').read_text())
    plan = json.loads((HERE / 'execution/PLAN.json').read_text())
    receipts = json.loads((HERE / 'execution/RECEIPTS.json').read_text())
    terminal = json.loads((HERE / 'execution/TERMINAL.json').read_text())
    original = subprocess.check_output(['git', 'show', request['input_index_commit'] + ':' + request['input_index_path']], cwd=ROOT)
    assert hashlib.sha256(original).hexdigest() == plan['manifest_sha256']
    index = json.loads(gzip.decompress(original))
    expected = {row['sha256']: row['bytes'] for row in index['files'] if row['location']['kind'] == 'upload_pending'}
    planned = {row['sha256']: row['bytes'] for row in plan['objects']}
    assert planned == expected
    assert len(expected) == request['expected_pending_objects']
    assert sum(expected.values()) == request['expected_pending_bytes']
    assert plan['repo'] == request['repo'] and plan['tag'] == request['release_tag'] and plan['release_id'] == request['release_id']
    assert set(receipts) == set(expected)
    assert len({row['asset_id'] for row in receipts.values()}) == len(expected)
    for sha, row in receipts.items():
        assert row['sha256'] == sha and row['bytes'] == expected[sha] and row['github_digest_verified'] is True
    assert terminal['status'] == 'ALL_PENDING_CHECKPOINTS_UPLOADED_AND_VERIFIED'
    assert terminal['objects'] == len(expected) and terminal['bytes'] == sum(expected.values())
    assert terminal['release_remained_draft'] is True
    assert terminal['local_sources_deleted'] is False and terminal['new_training_or_model_execution'] is False
    smoke = json.loads((HERE / 'ROUNDTRIP_SMOKE.json').read_text())
    assert smoke['status'] == 'PASS' and smoke['sha256'] in expected
    assert smoke['bytes'] == expected[smoke['sha256']]
    assert smoke['asset_id'] == receipts[smoke['sha256']]['asset_id']
    assert smoke['saved_locally'] is False and smoke['model_executed'] is False
    checked = 0
    if args.remote:
        assets = {}
        page = 1
        while True:
            rows = json.loads(subprocess.check_output(['gh', 'api', f"repos/{request['repo']}/releases/{request['release_id']}/assets?per_page=100&page={page}"]))
            assets.update({row['id']: row for row in rows})
            if len(rows) < 100:
                break
            page += 1
        for sha, receipt in receipts.items():
            asset = assets[receipt['asset_id']]
            assert asset['state'] == 'uploaded' and asset['name'] == receipt['name']
            assert asset['size'] == expected[sha] and asset['digest'] == 'sha256:' + sha
            checked += 1
    print(json.dumps({'status': 'PASS', 'authorized_objects_verified': len(expected),
                      'bytes_verified': sum(expected.values()), 'remote_assets_checked': checked,
                      'input_index_sha256': plan['manifest_sha256'], 'roundtrip_download_verified': True,
                      'new_model_execution': False}, sort_keys=True))


if __name__ == '__main__':
    main()
