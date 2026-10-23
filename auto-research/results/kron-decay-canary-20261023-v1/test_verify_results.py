"""Corruption controls for saved receipts; never imports a tensor library."""
import copy
import json
from pathlib import Path
import unittest

import verify_results as audit


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.result = json.loads((audit.HERE / 'execution/attempt-01.stdout').read_text())
        self.protocol = json.loads((audit.HERE / 'CANARY_PROTOCOL.json').read_text())
        self.resources = [json.loads(path.read_text()) for path in
                          sorted((audit.HERE / 'execution').glob('*.resource.json'))]

    def check(self):
        return audit.validate(self.result, self.protocol, self.resources)

    def test_original_keeps_failed_model_gate(self):
        result = self.check()
        self.assertEqual(result['core_rows'], 250)
        self.assertTrue(result['primary_core_gate_passed'])
        self.assertFalse(result['model_gates']['interval_direct'])
        self.assertTrue(result['model_gates']['source_factored'])

    def test_missing_and_duplicate_core_rows_rejected(self):
        original = copy.deepcopy(self.result['core_rows'])
        self.result['core_rows'].pop()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.check()
        self.result['core_rows'] = original
        self.result['core_rows'][-1] = self.result['core_rows'][0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.check()

    def test_claimed_core_and_model_gates_rejected(self):
        self.result['primary_core_gate_passed'] = False
        with self.assertRaisesRegex(ValueError, 'Primary gate'):
            self.check()
        self.result['primary_core_gate_passed'] = True
        self.result['model_gate_passed_by_variant']['interval_direct'] = True
        with self.assertRaisesRegex(ValueError, 'Model gate'):
            self.check()

    def test_false_precision_label_rejected(self):
        self.result['core_rows'][0][13] = 'torch.float64'
        with self.assertRaisesRegex(ValueError, 'dtype'):
            self.check()

    def test_wrong_tiny_reference_rejected(self):
        self.result['tiny_outputs'][0]['reference_state'][0] += .1
        with self.assertRaisesRegex(ValueError, 'Analytic tiny'):
            self.check()

    def test_resource_overruns_rejected(self):
        index = next(i for i, row in enumerate(self.resources) if row['phase'] == 'numerical')
        original = copy.deepcopy(self.resources)
        self.resources[index]['charged_cpu_seconds'] = 601
        with self.assertRaisesRegex(ValueError, 'Cumulative'):
            self.check()
        self.resources = original
        self.resources[index]['peak_sampled_environment_temp_results_bytes'] = 1024**3
        with self.assertRaisesRegex(ValueError, 'disk cap'):
            self.check()

    def test_threads_and_missing_approval_rejected(self):
        self.result['threads'] = 2
        with self.assertRaisesRegex(ValueError, 'Thread'):
            self.check()
        self.result['threads'] = 1
        self.protocol['authorization']['approved'] = False
        with self.assertRaisesRegex(ValueError, 'approval'):
            self.check()

    def test_optimizer_gate_does_not_hide_gradient_failure(self):
        row = next(row for row in self.result['model_rows'] if row['variant'] == 'interval_direct')
        self.assertTrue(row['optimizer_step']['passed'])
        self.assertTrue(row['optimizer_state']['passed'])
        self.assertFalse(row['gradients']['passed'])
        self.assertFalse(self.check()['model_gates']['interval_direct'])


if __name__ == '__main__':
    unittest.main()
