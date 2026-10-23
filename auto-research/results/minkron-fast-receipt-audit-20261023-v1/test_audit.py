"""Corrupt in-memory receipts after hash checking to exercise semantic checks."""
import copy
import unittest

import audit


class ReceiptAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = audit.load_bundle()

    def test_archived_positive_control(self):
        result = audit.audit(self.original)
        self.assertEqual(result["status"], "PASS_STORED_RECEIPT_CONSISTENCY")
        self.assertEqual(result["selected_recurrent_variant"], "k16_lean")
        self.assertEqual(result["total_target_exposures"], 9307136)
        self.assertEqual(result["trajectory_rows"], 9089)
        self.assertFalse(result["screen"][2]["confirmation_consistent"])
        self.assertFalse(result["new_model_execution"])

    def test_corrupted_evidence_is_rejected(self):
        def summary_mean(bundle):
            bundle["SUMMARY.json"]["screen"][3]["confirmation_ce"] -= .1

        def promote_failed_gate(bundle):
            bundle["SUMMARY.json"]["screen"][2]["confirmation_consistent"] = True

        def duplicate_identity(bundle):
            bundle["METRICS.json"]["screen"][1]["arm"] = "k8"

        def misbound_checkpoint(bundle):
            bundle["METRICS.json"]["confirmation"][0]["checkpoint"]["sha256"] = "0" * 64

        def fake_duplicate_pass(bundle):
            row = bundle["METRICS.json"]["confirmation"][0]
            row["duplicate_ce"] += .01
            row["absolute_duplicate_difference"] = abs(row["duplicate_ce"] - row["first"]["ce"])

        def missing_window(bundle):
            bundle["METRICS.json"]["confirmation"][0]["first"]["window_ce"].pop()

        def altered_window_mean(bundle):
            bundle["METRICS.json"]["screen"][0]["development"]["window_ce"][0] += .1

        def broken_resume_cursor(bundle):
            bundle["steps"][0]["total_steps"] = "1"

        def altered_recorded_lr(bundle):
            bundle["steps"][0]["lr"] = "0.1"

        def missing_step(bundle):
            bundle["steps"].pop(0)

        def altered_lifetime_accounting(bundle):
            bundle["METRICS.json"]["extensions"][0]["total_training_seconds"] += 1

        def incomplete_terminal(bundle):
            bundle["TERMINAL.json"]["status"] = "INCOMPLETE"

        def fake_qualification_coverage(bundle):
            bundle["FIDELITY.json"]["rows"][0] = copy.deepcopy(bundle["FIDELITY.json"]["rows"][1])

        def fake_qualification_pass(bundle):
            bundle["FIDELITY.json"]["rows"][0]["parameter_gradient_max_error"] = .1

        def broken_data_binding(bundle):
            bundle["PREREGISTRATION_RECEIPT.json"]["data_manifest_sha256"] = "0" * 64

        def broken_tokenizer_binding(bundle):
            bundle["data/manifest.json"]["tokenizer_sha256"] = "0" * 64

        def fake_replay_pass(bundle):
            bundle["REPLAY_VERIFICATION.json"]["model_tensor_sha256"] = "0" * 64

        def altered_printed_result(bundle):
            bundle["report"] = bundle["report"].replace("6.3964 ±", "0.0000 ±", 1)

        def nonfinite_receipt(bundle):
            bundle["SUMMARY.json"]["screen"][0]["confirmation_ce"] = float("nan")

        cases = [(summary_mean, "summary"), (promote_failed_gate, "summary"),
                 (duplicate_identity, "screen identity coverage"),
                 (misbound_checkpoint, "confirmation checkpoint"),
                 (fake_duplicate_pass, "duplicate tolerance"),
                 (missing_window, "evaluation coverage"), (altered_window_mean, "window mean"),
                 (broken_resume_cursor, "trajectory cursor"),
                 (altered_recorded_lr, "trajectory learning rate"),
                 (missing_step, "trajectory step sequence"),
                 (altered_lifetime_accounting, "lifetime time"),
                 (incomplete_terminal, "terminal status"),
                 (fake_qualification_coverage, "qualification identity coverage"),
                 (fake_qualification_pass, "qualification saved absolute bound"),
                 (broken_data_binding, "data manifest binding"),
                 (broken_tokenizer_binding, "tokenizer binding"),
                 (fake_replay_pass, "saved replay binding"),
                 (altered_printed_result, "published result tables"),
                 (nonfinite_receipt, "nonfinite saved scalar")]
        for mutate, message in cases:
            with self.subTest(corruption=mutate.__name__):
                bundle = copy.deepcopy(self.original)
                mutate(bundle)
                with self.assertRaisesRegex(ValueError, message):
                    audit.audit(bundle)

    def test_input_hash_guard(self):
        contents = {"receipt.json": b'{"status":"PASS"}'}
        expected = {name: audit.sha(content) for name, content in contents.items()}
        audit.check_hashes(contents, expected)
        with self.assertRaisesRegex(ValueError, "input hash mismatch: receipt.json"):
            audit.check_hashes({"receipt.json": b'{"status":"FAIL"}'}, expected)
        with self.assertRaisesRegex(ValueError, "input inventory mismatch"):
            audit.check_hashes({}, expected)

    def test_irrelevant_report_table_does_not_change_results(self):
        bundle = copy.deepcopy(self.original)
        bundle["report"] += "\n| An unrelated table | 123 |\n"
        self.assertEqual(audit.audit(bundle), audit.audit(self.original))


if __name__ == "__main__":
    unittest.main()
