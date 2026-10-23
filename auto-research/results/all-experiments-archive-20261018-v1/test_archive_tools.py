"""Archive status and restore tests: stdlib fixtures only; no model execution."""
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b'authenticated restore fixture\n'
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.row = {'restore_path': 'example/checkpoint.pt', 'bytes': len(self.payload), 'sha256': self.digest,
                    'location': {'kind': 'upload_pending', 'bytes': len(self.payload), 'sha256': self.digest}}

    def fixture(self, complete=False, label=None):
        row = json.loads(json.dumps(self.row))
        if complete:
            row['location'].update({'kind': 'checkpoint_asset', 'asset_id': 123, 'url': 'https://github.com/example'})
        manifest = {'files': [row], 'assets': {}}
        raw = gzip.compress(json.dumps(manifest).encode(), mtime=0)
        (self.root / 'FILES.json.gz').write_bytes(raw)
        summary = {'files_manifest_sha256': hashlib.sha256(raw).hexdigest(), 'mapped_source_paths': 1,
                   'unique_content_objects': 1, 'unique_content_bytes': len(self.payload),
                   'pending_objects': 0 if complete else 1,
                   'content_location_counts': {row['location']['kind']: 1},
                   'status': label or ('ALL_SELECTED_OBJECTS_LOCATED_AND_UPLOADED' if complete else 'PARTIAL_ARCHIVE_UPLOADS_PAUSED')}
        (self.root / 'SUMMARY.json').write_text(json.dumps(summary))
        return row

    def verify(self, *args):
        module = load('verify_archive')
        output = io.StringIO()
        with patch.object(module, 'HERE', self.root), patch.object(sys, 'argv', ['verify_archive.py', *args]), contextlib.redirect_stdout(output):
            module.main()
        return json.loads(output.getvalue())

    def test_partial_status_is_not_complete(self):
        self.fixture()
        result = self.verify()
        self.assertEqual(result['status'], 'PASS_PARTIAL_ARCHIVE')
        self.assertEqual(result['pending_objects'], 1)

    def test_complete_gate_rejects_pending(self):
        self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'remain pending'):
            self.verify('--require-complete')

    def test_false_completion_label_rejected(self):
        self.fixture(label='ALL_SELECTED_OBJECTS_LOCATED_AND_UPLOADED')
        with self.assertRaisesRegex(RuntimeError, 'mislabeled complete'):
            self.verify()

    def test_complete_archive_passes_strict_gate(self):
        self.fixture(complete=True)
        self.assertEqual(self.verify('--require-complete')['status'], 'PASS_COMPLETE_ARCHIVE')

    def test_remote_digest_mismatch_rejected(self):
        self.fixture(complete=True)
        replies = [json.dumps([{'id': 9}]), json.dumps([{'id': 123, 'state': 'uploaded', 'size': len(self.payload), 'digest': 'sha256:' + '0' * 64}])]
        with patch('subprocess.check_output', side_effect=replies), self.assertRaisesRegex(RuntimeError, 'digest mismatch'):
            self.verify('--remote')

    def test_authenticated_draft_asset_restore(self):
        self.fixture(complete=True)
        module = load('restore_experiments')
        def download(argv, stdout, check):
            self.assertEqual(argv, ['gh', 'api', '-H', 'Accept: application/octet-stream', 'repos/rohanarun/Super-1/releases/assets/123'])
            self.assertTrue(check)
            stdout.write(self.payload)
        argv = ['restore_experiments.py', '--manifest', str(self.root / 'FILES.json.gz'), '--restore',
                '--output', str(self.root / 'restored'), '--cache', str(self.root / 'cache')]
        with patch.object(sys, 'argv', argv), patch('subprocess.run', side_effect=download), contextlib.redirect_stdout(io.StringIO()):
            module.main()
        self.assertEqual((self.root / 'restored/example/checkpoint.pt').read_bytes(), self.payload)

    def test_pending_restore_does_not_download(self):
        self.fixture()
        module = load('restore_experiments')
        argv = ['restore_experiments.py', '--manifest', str(self.root / 'FILES.json.gz'), '--restore',
                '--output', str(self.root / 'restored'), '--cache', str(self.root / 'cache')]
        with patch.object(sys, 'argv', argv), patch('subprocess.run') as download:
            with self.assertRaisesRegex(RuntimeError, 'not finished uploading'):
                module.main()
            download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
