"""Functional planning tests; all restored bytes are tiny local test fixtures."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import restore_plan as planner


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture():
    content = {'a': b'print(1)\n', 'b': b'{"result": 0}\n',
               'git': b'pinned source\n', 'checkpoint': b'synthetic bytes, never deserialized\n'}
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as archive:
        for data in (content['a'], content['b'], bytes(range(256)) * 256):
            member = tarfile.TarInfo('objects/' + sha(data))
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    archive_bytes = gzip.compress(raw.getvalue(), mtime=0)
    tag = 'all-experiments-20261018'
    url = 'https://github.com/rohanarun/Super-1/releases/download/' + tag + '/evidence.tar.gz'
    asset = {'asset_id': 11, 'bytes': len(archive_bytes), 'sha256': sha(archive_bytes), 'url': url}
    rows = []
    for path, key in [('study/a.py', 'a'), ('study/copy/a.py', 'a'), ('study/b.json', 'b')]:
        rows.append({'restore_path': path, 'sha256': sha(content[key]), 'bytes': len(content[key]),
                     'location': {'kind': 'archive_object', 'asset': 'evidence.tar.gz',
                                  'member': 'objects/' + sha(content[key]), 'url': url}})
    for path in ['study/code.py', 'study/copy/code.py']:
        rows.append({'restore_path': path, 'sha256': sha(content['git']), 'bytes': len(content['git']),
                     'location': {'kind': 'existing_git_file', 'commit': '1' * 40, 'path': 'code.py'}})
    checkpoint = content['checkpoint']
    rows.append({'restore_path': 'study/final.pt', 'sha256': sha(checkpoint), 'bytes': len(checkpoint),
                 'location': {'kind': 'checkpoint_asset', 'asset_id': 12, 'bytes': len(checkpoint),
                              'sha256': sha(checkpoint), 'url': url.replace('evidence.tar.gz', 'final.pt')}})
    manifest = {'assets': {'evidence.tar.gz': asset}, 'files': rows, 'release': tag}
    return manifest, content, {11: archive_bytes, 12: checkpoint}


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.content, self.remote = fixture()
        self.states = [{'tag': self.manifest['release'], 'draft': True}]

    def plan(self, manifest=None, prefix='study/'):
        return planner.build_plan(self.manifest if manifest is None else manifest, prefix, self.states)

    def test_whole_archive_and_alias_costs(self):
        result = self.plan()
        self.assertEqual(result['selected_paths'], 6)
        self.assertEqual(result['selected_unique_content_objects'], 4)
        self.assertEqual(result['known_payloads'], 3)
        self.assertEqual(result['known_download_payload_bytes'],
                         sum(map(len, self.remote.values())) + len(self.content['git']))
        output = 2 * len(self.content['a']) + len(self.content['b']) \
            + 2 * len(self.content['git']) + len(self.content['checkpoint'])
        self.assertEqual(result['selected_output_logical_bytes'], output)
        self.assertEqual(result['cache_plus_output_logical_bytes_if_complete'],
                         result['known_download_payload_bytes'] + output)
        self.assertEqual(result['draft_tags_in_saved_summary'], [self.manifest['release']])

    def test_one_member_still_needs_full_archive(self):
        result = self.plan(prefix='study/a.py')
        self.assertEqual(result['selected_output_logical_bytes'], len(self.content['a']))
        self.assertEqual(result['known_download_payload_bytes'], len(self.remote[11]))
        self.assertGreater(result['known_download_payload_bytes'], result['selected_output_logical_bytes'])

    def test_existing_release_kind(self):
        self.manifest['files'][-1]['location']['kind'] = 'existing_release_asset'
        self.assertTrue(self.plan()['restorable_from_saved_locations'])

    def test_unknown_release_state_stays_unknown(self):
        result = planner.build_plan(self.manifest, 'study/', ())
        self.assertEqual(result['draft_tags_in_saved_summary'], [])
        self.assertTrue(result['authenticated_gh_locators_present'])
        asset = next(p for p in result['payloads'] if p['sha256'] == sha(self.remote[11]))
        self.assertIsNone(asset['locators'][0]['draft_in_saved_summary'])

    def test_order_does_not_change_plan(self):
        first = self.plan()
        self.manifest['files'].reverse()
        self.assertEqual(self.plan(), first)

    def test_zero_byte_git_object(self):
        row = copy.deepcopy(self.manifest['files'][3])
        row.update(restore_path='study/empty', sha256=sha(b''), bytes=0)
        row['location']['path'] = 'empty'
        self.manifest['files'].append(row)
        result = self.plan()
        self.assertEqual(result['known_payloads'], 4)
        self.assertEqual(result['selected_unique_content_objects'], 5)

    def test_pending_blocks_complete_budget(self):
        row = copy.deepcopy(self.manifest['files'][-1])
        row['location']['kind'] = 'upload_pending'
        self.manifest['files'][-1] = row
        result = self.plan()
        self.assertFalse(result['restorable_from_saved_locations'])
        self.assertEqual(result['status'], 'BLOCKED_PENDING_OBJECTS')
        self.assertEqual(result['pending_unique_objects'], 1)
        self.assertEqual(result['pending_content_bytes'], row['bytes'])
        self.assertIsNone(result['fresh_cache_logical_bytes_if_complete'])
        self.assertIsNone(result['cache_plus_output_logical_bytes_if_complete'])

    def test_empty_selection_fails(self):
        with self.assertRaisesRegex(ValueError, 'No restore paths'):
            self.plan(prefix='missing/')

    def test_root_partition_includes_shallow_files(self):
        from verify_plans import root_prefixes
        manifest = {'files': [{'restore_path': path} for path in
                             ['loose.txt', 'two/file.txt', 'deep/root/a', 'deep/root/b']]}
        self.assertEqual(root_prefixes(manifest), ['deep/root/', 'loose.txt', 'two/file.txt'])
        manifest['files'].append({'restore_path': 'two/file.txt-copy'})
        with self.assertRaisesRegex(ValueError, 'overlap'):
            root_prefixes(manifest)

    def test_metadata_corruption_controls(self):
        controls = []
        def case(label, mutate):
            manifest = copy.deepcopy(self.manifest)
            mutate(manifest)
            controls.append((label, manifest))
        case('missing archive', lambda m: m['assets'].clear())
        case('unsupported location', lambda m: m['files'][0]['location'].update(kind='unknown'))
        case('duplicate path', lambda m: m['files'].append(copy.deepcopy(m['files'][0])))
        case('malformed digest', lambda m: m['files'][0].update(sha256='bad'))
        case('bool bytes', lambda m: m['files'][0].update(bytes=True))
        case('negative bytes', lambda m: m['files'][0].update(bytes=-1))
        case('noncanonical path', lambda m: m['files'][0].update(restore_path='study//a.py'))
        case('content size conflict', lambda m: m['files'][1].update(bytes=m['files'][1]['bytes'] + 1))
        case('archive member', lambda m: m['files'][0]['location'].update(member='wrong'))
        case('asset id', lambda m: m['assets']['evidence.tar.gz'].update(asset_id=True))
        case('Git pin', lambda m: m['files'][3]['location'].update(commit='main'))
        case('direct identity', lambda m: m['files'][-1]['location'].update(sha256='0' * 64))
        def conflicting_payload(m):
            other = copy.deepcopy(m['assets']['evidence.tar.gz'])
            other['bytes'] += 1
            m['assets']['other.tar.gz'] = other
            m['files'][2]['location']['asset'] = 'other.tar.gz'
        case('payload size conflict', conflicting_payload)
        for label, manifest in controls:
            with self.subTest(control=label), self.assertRaises(ValueError):
                self.plan(manifest)

    def test_cli_statuses_and_errors(self):
        summary = {'release_states': self.states}
        for pending, expected in [(False, 0), (True, 2)]:
            manifest = copy.deepcopy(self.manifest)
            if pending:
                manifest['files'][-1]['location']['kind'] = 'upload_pending'
            out = io.StringIO()
            with patch.object(planner, 'load_snapshot', return_value=(manifest, summary, 'f' * 64)), \
                    redirect_stdout(out):
                self.assertEqual(planner.main(['--prefix', 'study/']), expected)
            self.assertEqual(json.loads(out.getvalue())['restorable_from_saved_locations'], not pending)
        with patch.object(planner, 'load_snapshot', return_value=(self.manifest, summary, 'f' * 64)), \
                redirect_stderr(io.StringIO()) as err:
            self.assertEqual(planner.main(['--prefix', 'absent/']), 1)
        self.assertIn('PLAN_ERROR', err.getvalue())

    def test_cli_rejects_changed_pinned_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'input').write_bytes(b'changed')
            (root / 'INPUTS.json').write_text(json.dumps({'files': {'input': sha(b'original')}}))
            with patch.object(planner, 'HERE', root), patch.object(planner, 'REPO', root):
                with self.assertRaisesRegex(ValueError, 'Pinned input changed'):
                    planner.load_snapshot('current')

    def load_restorer(self):
        source = planner.ARCHIVE / 'restore_experiments.py'
        spec = importlib.util.spec_from_file_location('frozen_archive_restorer', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def restore(self, module, root, prefix, output_name='out', forbid_download=False):
        source = root / 'manifest.json.gz'
        source.write_bytes(gzip.compress(json.dumps(self.manifest).encode(), mtime=0))
        downloads = []
        def fake_gh(argv, stdout, check):
            self.assertFalse(forbid_download, 'Unexpected download with warm cache')
            asset_id = int(argv[-1].rsplit('/', 1)[-1])
            self.assertEqual(argv, ['gh', 'api', '-H', 'Accept: application/octet-stream',
                                   'repos/rohanarun/Super-1/releases/assets/' + str(asset_id)])
            self.assertTrue(check)
            data = self.remote[asset_id]
            downloads.append(data)
            stdout.write(data)
        def fake_url(url):
            self.assertFalse(forbid_download, 'Unexpected download with warm cache')
            self.assertEqual(url, 'https://raw.githubusercontent.com/rohanarun/Super-1/' + '1' * 40 + '/code.py')
            data = self.content['git']
            downloads.append(data)
            return io.BytesIO(data)
        argv = ['restore', '--manifest', str(source), '--prefix', prefix, '--restore',
                '--output', str(root / output_name), '--cache', str(root / 'cache')]
        with patch.object(sys, 'argv', argv), patch.object(module.subprocess, 'run', side_effect=fake_gh), \
                patch.object(module.urllib.request, 'urlopen', side_effect=fake_url), redirect_stdout(io.StringIO()):
            module.main()
        return downloads

    def test_differential_against_unchanged_restorer(self):
        module = self.load_restorer()
        for prefix in ('study/', 'study/a.py', 'study/copy/'):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                result = self.plan(prefix=prefix)
                downloads = self.restore(module, root, prefix)
                self.assertEqual(len(downloads), result['known_payloads'])
                self.assertEqual(sum(map(len, downloads)), result['known_download_payload_bytes'])
                cached = [path for path in (root / 'cache').rglob('*') if path.is_file()]
                output = [path for path in (root / 'out').rglob('*') if path.is_file()]
                self.assertEqual(sum(path.stat().st_size for path in cached),
                                 result['fresh_cache_logical_bytes_if_complete'])
                self.assertEqual(len(output), result['selected_paths'])
                self.assertEqual(sum(path.stat().st_size for path in output),
                                 result['selected_output_logical_bytes'])
                for path in output:
                    row = next(row for row in self.manifest['files']
                               if row['restore_path'] == path.relative_to(root / 'out').as_posix())
                    self.assertEqual(sha(path.read_bytes()), row['sha256'])
                self.assertEqual(self.restore(module, root, prefix, 'warm-out', forbid_download=True), [])

    def test_pending_matches_restorer_without_download(self):
        self.manifest['files'] = [copy.deepcopy(self.manifest['files'][-1])]
        self.manifest['files'][0]['location']['kind'] = 'upload_pending'
        self.assertFalse(self.plan()['restorable_from_saved_locations'])
        module = self.load_restorer()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, 'not finished uploading'):
                self.restore(module, Path(folder), 'study/', forbid_download=True)


if __name__ == '__main__':
    unittest.main()
