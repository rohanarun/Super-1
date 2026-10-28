"""Plan pinned archive restore bytes offline; never transfer or restore content."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import quote, unquote, urlsplit


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARCHIVE = REPO / 'auto-research/results/all-experiments-archive-20261018-v1'
HEX64 = re.compile(r'[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def byte_count(value):
    require(type(value) is int and value >= 0, 'Invalid byte count')
    return value


def digest(value):
    require(isinstance(value, str) and HEX64.fullmatch(value), 'Invalid SHA-256')
    return value


def relative_path(value):
    require(isinstance(value, str) and value and '\x00' not in value,
            'Invalid relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts
            and str(path) == value and value != '.', 'Noncanonical relative path')
    return value


def build_plan(manifest, prefix, release_states=()):
    """Match startswith selection and SHA-keyed payload caching in the restorer."""
    require(isinstance(prefix, str), 'Prefix must be a string')
    rows = [row for row in manifest['files']
            if relative_path(row['restore_path']).startswith(prefix)]
    require(rows, 'No restore paths match the prefix')
    paths, objects, pending, payloads = set(), {}, {}, {}
    states = {row['tag']: row['draft'] for row in release_states}
    selection = []
    for row in rows:
        path, sha, size = row['restore_path'], digest(row['sha256']), byte_count(row['bytes'])
        require(path not in paths, 'Duplicate restore path')
        paths.add(path)
        require(objects.setdefault(sha, size) == size, 'Conflicting content size')
        selection.append((path, sha, size))
        loc = row['location']
        kind = loc['kind']
        if kind == 'upload_pending':
            pending[sha] = size
            continue
        if kind == 'archive_object':
            require(loc['asset'] in manifest['assets'], 'Missing archive asset')
            require(loc['member'] == 'objects/' + sha, 'Archive member identity mismatch')
            source = manifest['assets'][loc['asset']]
            payload_sha, payload_size = digest(source['sha256']), byte_count(source['bytes'])
            locator = {'kind': 'github_asset', 'asset_id': source['asset_id'], 'url': source['url']}
        elif kind == 'existing_git_file':
            require(isinstance(loc['commit'], str)
                    and re.fullmatch(r'[0-9a-f]{40}', loc['commit']), 'Invalid pinned Git commit')
            git_path = relative_path(loc['path'])
            payload_sha, payload_size = sha, size
            locator = {'kind': 'pinned_git_file', 'url':
                       'https://raw.githubusercontent.com/rohanarun/Super-1/'
                       + loc['commit'] + '/' + quote(git_path)}
        elif kind in ('checkpoint_asset', 'existing_release_asset'):
            require(loc['sha256'] == sha and byte_count(loc['bytes']) == size,
                    'Direct asset identity mismatch')
            payload_sha, payload_size = sha, size
            locator = {'kind': 'github_asset', 'asset_id': loc['asset_id'], 'url': loc['url']}
        else:
            raise ValueError('Unsupported location kind: ' + str(kind))
        require(isinstance(locator['url'], str) and locator['url'], 'Missing payload URL')
        if locator['kind'] == 'github_asset':
            require(type(locator['asset_id']) is int and locator['asset_id'] > 0,
                    'Invalid asset id')
            url_path = urlsplit(locator['url']).path
            marker = '/releases/download/'
            tag = unquote(url_path.split(marker, 1)[1].split('/', 1)[0]) if marker in url_path else None
            locator['release_tag'] = tag
            locator['draft_in_saved_summary'] = states.get(tag)
        payload = payloads.setdefault(payload_sha, {
            'sha256': payload_sha, 'bytes': payload_size, 'locators': {},
            'selected_paths': 0, 'selected_output_bytes': 0, 'content': set()})
        require(payload['bytes'] == payload_size, 'Conflicting payload size')
        payload['locators'][json.dumps(locator, sort_keys=True)] = locator
        payload['selected_paths'] += 1
        payload['selected_output_bytes'] += size
        payload['content'].add(sha)
    fetches = []
    for sha in sorted(payloads):
        payload = payloads[sha]
        fetches.append({**{k: v for k, v in payload.items() if k not in ('locators', 'content')},
                        'locators': [payload['locators'][key] for key in sorted(payload['locators'])],
                        'selected_unique_content_objects': len(payload['content'])})
    download = sum(row['bytes'] for row in fetches)
    output = sum(row['bytes'] for row in rows)
    complete = not pending
    selection_bytes = json.dumps(sorted(selection), separators=(',', ':')).encode()
    return {
        'status': 'PLANNED_COMPLETE_SELECTION' if complete else 'BLOCKED_PENDING_OBJECTS',
        'restorable_from_saved_locations': complete,
        'prefix': prefix,
        'assumptions': 'Empty cache/output; unchanged restorer; no retries, HTTP or filesystem overhead. '
                       'No existing cache/output, live availability, credentials or free space inspected.',
        'selected_paths': len(paths),
        'selected_unique_content_objects': len(objects),
        'selected_unique_content_bytes': sum(objects.values()),
        'selected_output_logical_bytes': output,
        'selection_sha256': hashlib.sha256(selection_bytes).hexdigest(),
        'pending_unique_objects': len(pending),
        'pending_content_bytes': sum(pending.values()),
        'known_payloads': len(fetches),
        'known_download_payload_bytes': download,
        'fresh_cache_logical_bytes_if_complete': download if complete else None,
        'cache_plus_output_logical_bytes_if_complete': download + output if complete else None,
        'authenticated_gh_locators_present': any(loc['kind'] == 'github_asset'
            for row in fetches for loc in row['locators']),
        'draft_tags_in_saved_summary': sorted({loc['release_tag']
            for row in fetches for loc in row['locators']
            if loc.get('draft_in_saved_summary') is True}),
        'payloads': fetches,
        'artifact_transfers': False,
        'new_model_execution': False,
    }


def load_snapshot(snapshot):
    inputs = json.loads((HERE / 'INPUTS.json').read_text())
    for name, expected in inputs['files'].items():
        require(hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected,
                'Pinned input changed: ' + name)
    suffix = '.current' if snapshot == 'current' else ''
    source = ARCHIVE / ('FILES' + suffix + '.json.gz')
    manifest = json.loads(gzip.decompress(source.read_bytes()))
    summary = json.loads((ARCHIVE / ('SUMMARY' + suffix + '.json')).read_text())
    return manifest, summary, hashlib.sha256(source.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True,
                        help='Same startswith selection as archived restorer; empty string selects all')
    parser.add_argument('--snapshot', choices=('current', 'historical'), default='current')
    args = parser.parse_args(argv)
    try:
        manifest, summary, source_sha = load_snapshot(args.snapshot)
        plan = build_plan(manifest, args.prefix, summary.get('release_states', ()))
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print('PLAN_ERROR: ' + str(exc), file=sys.stderr)
        return 1
    plan['snapshot'] = args.snapshot
    plan['manifest_sha256'] = source_sha
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0 if plan['restorable_from_saved_locations'] else 2


if __name__ == '__main__':
    sys.exit(main())
