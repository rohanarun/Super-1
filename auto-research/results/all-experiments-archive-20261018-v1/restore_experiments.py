"""List archived experiments or restore selected paths without executing them."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.parse
import urllib.request


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('FILES.json.gz'))
    parser.add_argument('--prefix', default='', help='Restore-path prefix; list mode is the default')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--output', type=Path, help='Required restore directory; use a writable SSD')
    parser.add_argument('--cache', type=Path, help='Required download cache directory; use a writable SSD')
    args = parser.parse_args()
    with gzip.open(args.manifest, 'rt') as f:
        manifest = json.load(f)
    rows = [r for r in manifest['files'] if r['restore_path'].startswith(args.prefix)]
    if not args.restore:
        for row in rows:
            print(f"{row['bytes']:>12}  {row['restore_path']}")
        print(f'{len(rows)} paths selected. No files downloaded or restored.')
        return
    if args.output is None or args.cache is None:
        parser.error('--restore requires --output and --cache')
    output, cache = args.output.resolve(), args.cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    verified = set()
    for row in rows:
        dest = (output / row['restore_path']).resolve()
        if not dest.is_relative_to(output):
            raise RuntimeError('Unsafe restore path')
        if dest.exists():
            if sha256(dest) != row['sha256']:
                raise RuntimeError(f'Refusing to overwrite different content: {dest}')
            continue
        loc = row['location']
        if loc['kind'] == 'upload_pending':
            raise RuntimeError('This object has not finished uploading; obtain the completed index before restoring it')
        if loc['kind'] == 'existing_git_file':
            url = 'https://raw.githubusercontent.com/rohanarun/Super-1/' + loc['commit'] + '/' + urllib.parse.quote(loc['path'])
            expected, size = row['sha256'], row['bytes']
        elif loc['kind'] == 'archive_object':
            asset = manifest['assets'][loc['asset']]
            url, expected, size = asset['url'], asset['sha256'], asset['bytes']
        else:
            url, expected, size = loc['url'], row['sha256'], row['bytes']
        blob = cache / expected
        if expected not in verified:
            if not blob.exists():
                if shutil.disk_usage(cache).free < size:
                    raise RuntimeError('Insufficient cache space for next asset')
                part = blob.with_suffix('.partial')
                with urllib.request.urlopen(url) as source, part.open('wb') as target:
                    shutil.copyfileobj(source, target, 4 * 1024**2)
                if part.stat().st_size != size or sha256(part) != expected:
                    raise RuntimeError('Downloaded asset digest/size mismatch')
                part.replace(blob)
            elif blob.stat().st_size != size or sha256(blob) != expected:
                raise RuntimeError('Cached asset digest/size mismatch')
            verified.add(expected)
        if shutil.disk_usage(output).free < row['bytes']:
            raise RuntimeError('Insufficient output space for next file')
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_name(dest.name + '.restoring')
        if loc['kind'] == 'archive_object':
            with tarfile.open(blob, 'r:gz') as archive:
                member = archive.getmember(loc['member'])
                if not member.isfile() or member.size != row['bytes']:
                    raise RuntimeError('Invalid archive object')
                with archive.extractfile(member) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target, 4 * 1024**2)
        else:
            shutil.copyfile(blob, temporary)
        if temporary.stat().st_size != row['bytes'] or sha256(temporary) != row['sha256']:
            raise RuntimeError('Restored object digest/size mismatch')
        temporary.replace(dest)
        print('RESTORED', row['restore_path'])


if __name__ == '__main__':
    main()
