"""Reconcile the preserved index with uploaded GitHub assets, without uploading.

The original FILES.json.gz, SUMMARY.json and historical receipts stay unchanged.
Authentication is delegated to gh; no model, corpus or checkpoint is executed.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
REPO = 'rohanarun/Super-1'


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path], text=True))


def main():
    original = HERE / 'FILES.json.gz'
    manifest = json.loads(gzip.decompress(original.read_bytes()))
    releases = api(f'repos/{REPO}/releases?per_page=100')
    available = {}
    release_states = []
    for release in releases:
        page = 1
        while True:
            assets = api(f'repos/{REPO}/releases/{release["id"]}/assets?per_page=100&page={page}')
            for asset in assets:
                if asset['state'] == 'uploaded' and asset.get('digest', '').startswith('sha256:'):
                    available.setdefault((asset['digest'][7:], asset['size']), asset)
            if len(assets) < 100:
                break
            page += 1
        release_states.append({'id': release['id'], 'tag': release['tag_name'],
                               'draft': release['draft'], 'published_at': release['published_at']})
    promoted = set()
    for row in manifest['files']:
        if row['location']['kind'] != 'upload_pending':
            continue
        asset = available.get((row['sha256'], row['bytes']))
        if asset is None:
            continue
        row['location'] = {
            'kind': 'checkpoint_asset', 'asset_id': asset['id'], 'name': asset['name'],
            'bytes': asset['size'], 'sha256': row['sha256'], 'state': asset['state'],
            'github_digest_verified': True, 'url': asset['browser_download_url']}
        promoted.add(row['sha256'])
    objects = {row['sha256']: row for row in manifest['files']}
    pending = [row for row in objects.values() if row['location']['kind'] == 'upload_pending']
    checkpoints = [row for row in objects.values() if row['location']['kind'] == 'checkpoint_asset']
    payload = gzip.compress((json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n').encode(), mtime=0)
    (HERE / 'FILES.current.json.gz').write_bytes(payload)
    summary = json.loads((HERE / 'SUMMARY.json').read_text())
    summary.update({
        'status': 'PARTIAL_ARCHIVE_UPLOADS_PAUSED' if pending else 'ALL_SELECTED_OBJECTS_LOCATED_AND_UPLOADED',
        'files_manifest_sha256': hashlib.sha256(payload).hexdigest(),
        'files_manifest': 'FILES.current.json.gz',
        'content_location_counts': dict(Counter(row['location']['kind'] for row in objects.values())),
        'pending_objects': len(pending), 'verified_large_checkpoints': len(checkpoints),
        'verified_large_checkpoint_bytes': sum(row['bytes'] for row in checkpoints),
        'new_release_assets': None, 'new_release_asset_bytes': None,
        'snapshot_note': 'Original summary is historical. Current locations reconciled by size/SHA-256 only; no transfers resumed.',
        'release_states': release_states})
    (HERE / 'SUMMARY.current.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    receipt = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'original_index_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
               'additional_checkpoint_objects_located': len(promoted), 'pending_objects': len(pending),
               'uploads_resumed': False, 'draft_release_published': False, 'new_model_execution': False,
               'verified_large_checkpoints': len(checkpoints), 'verified_large_checkpoint_bytes': sum(row['bytes'] for row in checkpoints),
               'remaining_pending_sha256': sorted(row['sha256'] for row in pending), 'release_states': release_states}
    (HERE / 'RECONCILIATION-20261022.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: receipt[key] for key in ['additional_checkpoint_objects_located', 'pending_objects', 'verified_large_checkpoints', 'verified_large_checkpoint_bytes']}, sort_keys=True))


if __name__ == '__main__':
    main()
