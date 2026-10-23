"""Upload only checkpoint objects marked pending in a frozen archive index.

No model execution, source deletion, corpus access, or release publication.
Uses gh authentication in memory; interruption recovery reconciles remote assets.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import threading
import urllib.parse


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', path], text=True))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.temporary')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', required=True, type=int)
    parser.add_argument('--socket-timeout', type=float, default=None)
    args = parser.parse_args()
    mount = Path('/Volumes/SP AI 01_16')
    assert mount.is_mount() and os.access(mount / 'CodexWorkspaces', os.W_OK)
    output = args.output.resolve()
    assert output.is_relative_to(mount / 'CodexWorkspaces')
    assert args.workers > 0
    output.mkdir(parents=True, exist_ok=True)
    raw = args.manifest.read_bytes()
    manifest = json.loads(gzip.decompress(raw))
    objects = {}
    for row in manifest['files']:
        if row['location']['kind'] == 'upload_pending':
            objects.setdefault(row['sha256'], []).append(row)
    releases = api(f'repos/{args.repo}/releases?per_page=100')
    release = next(r for r in releases if r['tag_name'] == args.tag)

    def remote_assets():
        result = []
        page = 1
        while True:
            batch = api(f'repos/{args.repo}/releases/{release["id"]}/assets?per_page=100&page={page}')
            result.extend(batch)
            if len(batch) < 100:
                return result
            page += 1

    plan = []
    for digest, aliases in sorted(objects.items()):
        paths = [Path(row['source_path']) for row in aliases]
        # Source evidence may be an existing iCloud file. Only new output must
        # live on the SSD; never copy or modify those historical source files.
        paths = [p for p in paths if p.is_file() and not p.is_symlink() and p.stat().st_size == aliases[0]['bytes']]
        paths.sort(key=lambda p: not p.is_relative_to(mount))
        assert paths, 'No available recorded source for ' + digest
        path = paths[0]
        stat = path.stat()
        assert sha256(path) == digest, 'Local SHA-256 mismatch: ' + str(path)
        assert stat.st_size == path.stat().st_size and stat.st_mtime_ns == path.stat().st_mtime_ns
        plan.append({'sha256': digest, 'bytes': stat.st_size, 'source': str(path),
                     'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns,
                     'name': 'checkpoint-' + digest + (path.suffix or '.pt')})
    frozen = {'repo': args.repo, 'tag': args.tag, 'release_id': release['id'],
              'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'objects': plan}
    plan_path = output / 'PLAN.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == frozen, 'Frozen plan changed'
    else:
        atomic_json(plan_path, frozen)
    print(json.dumps({'phase': 'PREFLIGHT_VERIFIED', 'objects': len(plan), 'bytes': sum(p['bytes'] for p in plan)}), flush=True)
    token = subprocess.check_output(['gh', 'auth', 'token'], text=True).strip()
    receipts_path = output / 'RECEIPTS.json'
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    lock = threading.Lock()
    stop = threading.Event()
    existing = {a['name']: a for a in remote_assets()}

    def check(asset, item):
        assert asset['state'] == 'uploaded' and asset['size'] == item['bytes']
        assert asset.get('digest') == 'sha256:' + item['sha256']

    def upload(item):
        if stop.is_set():
            return
        path = Path(item['source'])
        stat = path.stat()
        assert stat.st_ino == item['inode'] and stat.st_size == item['bytes'] and stat.st_mtime_ns == item['mtime_ns']
        asset = existing.get(item['name'])
        if asset is None:
            print('UPLOAD_START', item['name'], item['bytes'], flush=True)
            connection = http.client.HTTPSConnection('uploads.github.com', timeout=args.socket_timeout)
            upload_path = f'/repos/{args.repo}/releases/{release["id"]}/assets?' + urllib.parse.urlencode({'name': item['name']})
            digest = hashlib.sha256()
            sent = 0
            try:
                connection.putrequest('POST', upload_path)
                connection.putheader('Authorization', 'Bearer ' + token)
                connection.putheader('User-Agent', 'Super-1-pending-checkpoint-upload')
                connection.putheader('Content-Type', 'application/octet-stream')
                connection.putheader('Content-Length', str(item['bytes']))
                connection.endheaders()
                with path.open('rb') as source:
                    for chunk in iter(lambda: source.read(4 * 1024**2), b''):
                        digest.update(chunk)
                        sent += len(chunk)
                        connection.send(chunk)
                response = connection.getresponse()
                body = response.read()
                assert response.status == 201, 'Upload HTTP status ' + str(response.status)
                asset = json.loads(body)
                assert sent == item['bytes'] and digest.hexdigest() == item['sha256']
            except Exception:
                # Resolve an uncertain send by reading remote state, never by blind retry.
                asset = next((a for a in remote_assets() if a['name'] == item['name']), None)
                if asset is None:
                    raise
                assert sent in (0, item['bytes']), 'Source transfer interrupted; inspect remote asset'
            finally:
                connection.close()
        check(asset, item)
        receipt = {'asset_id': asset['id'], 'name': asset['name'], 'bytes': asset['size'],
                   'sha256': item['sha256'], 'github_digest_verified': True,
                   'url': asset['browser_download_url'], 'verified_utc': datetime.now(timezone.utc).isoformat()}
        with lock:
            receipts[item['sha256']] = receipt
            atomic_json(receipts_path, receipts)
            print(json.dumps({'phase': 'UPLOAD_VERIFIED', 'completed': len(receipts), 'total': len(plan), 'bytes_verified': sum(r['bytes'] for r in receipts.values()), 'name': item['name']}), flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(upload, item): item for item in plan}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                stop.set()
                failures.append({'sha256': futures[future]['sha256'], 'type': type(error).__name__, 'message': str(error)})
                atomic_json(output / 'FAILURES.json', failures)
                print('UPLOAD_FAILURE', failures[-1], flush=True)
    assert not failures, 'Upload failures require reconciliation; no blind retries performed'
    final_remote = {a['name']: a for a in remote_assets()}
    for item in plan:
        check(final_remote[item['name']], item)
    terminal = {'status': 'ALL_PENDING_CHECKPOINTS_UPLOADED_AND_VERIFIED', 'objects': len(plan),
                'bytes': sum(p['bytes'] for p in plan), 'completed_utc': datetime.now(timezone.utc).isoformat(),
                'release_remained_draft': api(f'repos/{args.repo}/releases/{release["id"]}')['draft'],
                'local_sources_deleted': False, 'new_training_or_model_execution': False}
    atomic_json(output / 'TERMINAL.json', terminal)
    print(json.dumps(terminal), flush=True)


if __name__ == '__main__':
    main()
