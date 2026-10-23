"""Verify the archive index and, optionally, GitHub's uploaded asset digests."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote', action='store_true', help='Read GitHub release metadata using gh')
    parser.add_argument('--require-complete', action='store_true', help='Fail if any selected object is still pending')
    args = parser.parse_args()
    current = (HERE / 'FILES.current.json.gz').exists()
    index_path = HERE / ('FILES.current.json.gz' if current else 'FILES.json.gz')
    summary_path = HERE / ('SUMMARY.current.json' if current else 'SUMMARY.json')
    with gzip.open(index_path, 'rt') as f:
        manifest = json.load(f)
    summary = json.loads(summary_path.read_text())
    require(hashlib.sha256(index_path.read_bytes()).hexdigest() == summary['files_manifest_sha256'], 'Index hash mismatch')
    paths, objects, locations = {}, {}, {}
    for row in manifest['files']:
        path = PurePosixPath(row['restore_path'])
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe restore path')
        require(row['bytes'] >= 0 and len(row['sha256']) == 64, 'Invalid file metadata')
        require(row['restore_path'] not in paths, 'Duplicate restore destination')
        paths[row['restore_path']] = row['sha256']
        require(row['sha256'] not in objects or objects[row['sha256']] == row['bytes'], 'Inconsistent object size')
        objects[row['sha256']] = row['bytes']
        loc = row['location']
        require(loc['kind'] in ['existing_git_file', 'existing_release_asset', 'archive_object', 'checkpoint_asset', 'upload_pending'], 'Missing content location')
        require(row['sha256'] not in locations or locations[row['sha256']] == loc['kind'], 'Inconsistent content location')
        locations[row['sha256']] = loc['kind']
        if loc['kind'] == 'archive_object':
            require(loc['asset'] in manifest['assets'], 'Missing archive asset')
            require(loc['member'] == 'objects/' + row['sha256'], 'Invalid archive object name')
        elif loc['kind'] in ['existing_release_asset', 'checkpoint_asset', 'upload_pending']:
            require(loc['sha256'] == row['sha256'] and loc['bytes'] == row['bytes'], 'Asset/object mismatch')
    require(len(paths) == summary['mapped_source_paths'], 'Source path count mismatch')
    require(len(objects) == summary['unique_content_objects'], 'Unique object count mismatch')
    require(sum(objects.values()) == summary['unique_content_bytes'], 'Unique byte count mismatch')
    pending = sum(kind == 'upload_pending' for kind in locations.values())
    require(pending == summary['pending_objects'], 'Pending object count mismatch')
    require(dict(Counter(locations.values())) == summary['content_location_counts'], 'Location count mismatch')
    require(not pending or summary['status'] != 'ALL_SELECTED_OBJECTS_LOCATED_AND_UPLOADED', 'Incomplete archive mislabeled complete')
    if args.require_complete:
        require(pending == 0, 'Archive incomplete: selected objects remain pending')
    remote_checked = 0
    if args.remote:
        all_assets = {}
        releases = json.loads(subprocess.check_output(['gh', 'api', 'repos/rohanarun/Super-1/releases?per_page=100'], text=True))
        for release in releases:
            page = 1
            while True:
                assets = json.loads(subprocess.check_output(['gh', 'api', f"repos/rohanarun/Super-1/releases/{release['id']}/assets?per_page=100&page={page}"], text=True))
                all_assets.update({a['id']: a for a in assets})
                if len(assets) < 100:
                    break
                page += 1
        expected = {a['asset_id']: a for a in manifest['assets'].values()}
        for row in manifest['files']:
            loc = row['location']
            if loc['kind'] in ['existing_release_asset', 'checkpoint_asset']:
                expected[loc['asset_id']] = loc
        for asset_id, item in expected.items():
            require(asset_id in all_assets, 'Recorded remote asset missing')
            actual = all_assets[asset_id]
            require(actual['state'] == 'uploaded', 'Incomplete remote asset')
            require(actual['size'] == item['bytes'], 'Remote size mismatch')
            require(actual['digest'] == 'sha256:' + item['sha256'], 'Remote digest mismatch')
            remote_checked += 1
    print(json.dumps({'status': 'PASS_PARTIAL_ARCHIVE' if pending else 'PASS_COMPLETE_ARCHIVE', 'pending_objects': pending, 'source_paths': len(paths), 'unique_objects': len(objects),
                      'unique_bytes': sum(objects.values()), 'remote_assets_checked': remote_checked,
                      'new_model_evaluation': False}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
