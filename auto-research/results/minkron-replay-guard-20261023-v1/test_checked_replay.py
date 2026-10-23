"""Synthetic filesystem/control-flow tests; no PyTorch or real model execution."""
import ast
import contextlib
import copy
import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import checked_replay as guard


SCRATCH = Path(__file__).resolve().parents[4]


class CheckedReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = guard.read_inputs()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="replay-guard-test-", dir=SCRATCH)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def plan(self):
        plan = guard.make_plan(self.inputs, "k16", 25101, "screen", self.root / "result.pt")
        body = self.root / "synthetic-array.bin"
        body.write_bytes(b"synthetic array for control flow only")
        plan["required_local_arrays"] = [dict(split="train", path=str(body), sha256=guard.file_sha256(body))]
        return plan

    def child(self, plan, mode="match"):
        def run(command, **kwargs):
            candidate = Path(command[command.index("--output") + 1])
            candidate.write_bytes(b"synthetic checkpoint; no model tensors")
            expected = plan["expected_model_tensor_sha256"]
            receipt = dict(output=str(candidate), expected=expected,
                           model_tensor_sha256=expected, matches_original_model_tensors=True)
            if mode == "mismatch":
                receipt.update(model_tensor_sha256="0" * 64, matches_original_model_tensors=False)
            elif mode == "false_flag":
                receipt["matches_original_model_tensors"] = False
            elif mode == "lying_flag":
                receipt["model_tensor_sha256"] = "0" * 64
            elif mode == "wrong_expected":
                receipt.update(expected="0" * 64, model_tensor_sha256="0" * 64)
            elif mode == "wrong_output":
                receipt["output"] = str(self.root / "somewhere-else.pt")
            elif mode == "missing_artifact":
                candidate.unlink()
            elif mode == "empty_artifact":
                candidate.write_bytes(b"")
            elif mode == "symlink_artifact":
                candidate.unlink()
                candidate.symlink_to(plan["required_local_arrays"][0]["path"])
            elif mode == "output_race":
                Path(plan["output"]).write_bytes(b"concurrent writer must survive")
            stdout = "not JSON" if mode == "bad_json" else json.dumps(receipt)
            if mode == "array_json":
                stdout = "[]"
            elif mode == "null_json":
                stdout = "null"
            if mode == "missing_field":
                del receipt["expected"]
                stdout = json.dumps(receipt)
            return subprocess.CompletedProcess(command, 7 if mode == "child_error" else 0,
                                               stdout=stdout, stderr="synthetic diagnostic")
        return run

    def test_all_21_recorded_targets_have_complete_plans(self):
        for checkpoint in self.inputs["CHECKPOINTS.json"]:
            name = Path(checkpoint["path"]).stem
            stage, rest = name.split("_", 1)
            arm, seed = rest.rsplit("_", 1)
            with self.subTest(target=name):
                plan = guard.make_plan(self.inputs, arm, int(seed), stage, self.root / (name + ".pt"))
                self.assertEqual(plan["status"], "PLAN_ONLY_NO_MODEL_EXECUTION")
                self.assertEqual(plan["expected_model_tensor_sha256"], checkpoint["model_tensor_sha256"])
                self.assertGreaterEqual(plan["recorded_steps"], 186)

    def test_unsupported_extension_targets_fail_before_execution(self):
        supported = {(row["arm"], row["seed"]) for row in self.inputs["METRICS.json"]["extensions"]}
        for arm in ("k8", "k16", "k32", "k16_lean", "k16_lean_conv", "transformer_lean"):
            for seed in (25101, 25102, 25103):
                if (arm, seed) in supported:
                    continue
                with self.subTest(arm=arm, seed=seed), self.assertRaisesRegex(ValueError, "unsupported"):
                    guard.make_plan(self.inputs, arm, seed, "extension", self.root / "result.pt")

    def test_broken_schedule_is_rejected_before_execution(self):
        for mode in ("missing", "duplicate", "cursor", "nonfinite_lr"):
            inputs = copy.deepcopy(self.inputs)
            indexes = [i for i, row in enumerate(inputs["steps"])
                       if row["arm"] == "k16" and int(row["seed"]) == 25101 and row["stage"] == "screen"]
            index = indexes[0]
            if mode == "missing":
                inputs["steps"].pop(index)
            elif mode == "duplicate":
                inputs["steps"].append(copy.deepcopy(inputs["steps"][index]))
            elif mode == "cursor":
                inputs["steps"][index]["tokens"] = "999999"
            else:
                inputs["steps"][index]["lr"] = "nan"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                guard.make_plan(inputs, "k16", 25101, "screen", self.root / "result.pt")

    def test_matching_synthetic_replay_publishes_once(self):
        plan = self.plan()
        result = guard.run_checked(plan, runner=self.child(plan))
        self.assertEqual(result["status"], "PASS_EXACT_MODEL_TENSOR_REPLAY")
        self.assertEqual(Path(plan["output"]).read_bytes(), b"synthetic checkpoint; no model tensors")
        self.assertEqual(result["checkpoint_sha256"], hashlib.sha256(Path(plan["output"]).read_bytes()).hexdigest())
        self.assertEqual(result["checkpoint_bytes"], Path(plan["output"]).stat().st_size)
        self.assertEqual(list(self.root.glob("minkron-replay-*")), [])

    def test_failed_replays_never_publish_and_keep_diagnostics(self):
        modes = ("mismatch", "false_flag", "lying_flag", "wrong_expected", "wrong_output",
                 "missing_artifact", "empty_artifact", "symlink_artifact", "bad_json", "array_json",
                 "null_json", "missing_field", "child_error")
        for mode in modes:
            with self.subTest(mode=mode):
                plan = self.plan()
                with self.assertRaisesRegex(ValueError, "diagnostics retained at"):
                    guard.run_checked(plan, runner=self.child(plan, mode))
                self.assertFalse(Path(plan["output"]).exists())
                diagnostics = list(self.root.glob("minkron-replay-*"))
                self.assertTrue(diagnostics)
                self.assertTrue(all((path / "stdout.txt").is_file() and (path / "stderr.txt").is_file()
                                    for path in diagnostics))

    def test_missing_or_changed_arrays_do_not_launch_a_child(self):
        for mode in ("missing", "changed"):
            plan = self.plan()
            body = Path(plan["required_local_arrays"][0]["path"])
            if mode == "missing":
                body.unlink()
            else:
                body.write_bytes(b"changed")
            calls = []
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "required local array"):
                guard.run_checked(plan, runner=lambda *args, **kwargs: calls.append(args))
            self.assertEqual(calls, [])
            self.assertFalse(Path(plan["output"]).exists())

    def test_existing_file_and_dangling_symlink_are_preserved(self):
        output = self.root / "result.pt"
        output.write_bytes(b"existing artifact")
        with self.assertRaisesRegex(ValueError, "output already exists"):
            guard.make_plan(self.inputs, "k16", 25101, "screen", output)
        self.assertEqual(output.read_bytes(), b"existing artifact")
        output.unlink()
        missing = self.root / "missing-target.pt"
        output.symlink_to(missing)
        with self.assertRaisesRegex(ValueError, "output already exists"):
            guard.make_plan(self.inputs, "k16", 25101, "screen", output)
        self.assertTrue(output.is_symlink())
        self.assertFalse(missing.exists())

    def test_destination_created_during_replay_is_never_overwritten(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "diagnostics retained at"):
            guard.run_checked(plan, runner=self.child(plan, "output_race"))
        self.assertEqual(Path(plan["output"]).read_bytes(), b"concurrent writer must survive")
        staged = list(self.root.glob("minkron-replay-*/checkpoint.pt"))
        self.assertEqual(len(staged), 1)
        self.assertEqual(staged[0].read_bytes(), b"synthetic checkpoint; no model tensors")

    def test_existing_output_is_rechecked_at_execution_boundary(self):
        plan = self.plan()
        output = Path(plan["output"])
        output.write_bytes(b"created after planning")
        with self.assertRaisesRegex(ValueError, "output already exists"):
            guard.run_checked(plan, runner=lambda *args, **kwargs: self.fail("child was launched"))
        self.assertEqual(output.read_bytes(), b"created after planning")

    def test_frozen_parent_cannot_be_an_output_directory(self):
        with self.assertRaisesRegex(ValueError, "frozen parent"):
            guard.make_plan(self.inputs, "k16", 25101, "screen", guard.PARENT / "new-checkpoint.pt")

    def test_source_hash_and_missing_source_fail_before_execution(self):
        fake_parent = self.root / "parent"
        fake_parent.mkdir()
        with mock.patch.object(guard, "PARENT", fake_parent):
            with self.assertRaises(OSError):
                guard.read_inputs()
            (fake_parent / "replay.py").write_text("changed source\n")
            with self.assertRaisesRegex(ValueError, "input hash mismatch: replay.py"):
                guard.read_inputs()

    def test_plan_cli_is_metadata_only_under_isolated_optimized_python(self):
        output = self.root / "planned.pt"
        command = [sys.executable, "-I", "-O", "-B", str(Path(guard.__file__)),
                   "--arm", "k16", "--seed", "25101", "--stage", "screen",
                   "--output", str(output), "--plan"]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "PLAN_ONLY_NO_MODEL_EXECUTION")
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob("minkron-replay-*")), [])

    def test_unsupported_target_cli_fails_without_artifacts(self):
        output = self.root / "unsupported.pt"
        command = [sys.executable, "-I", "-O", "-B", str(Path(guard.__file__)),
                   "--arm", "k32", "--seed", "25103", "--stage", "extension",
                   "--output", str(output)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(json.loads(result.stderr)["status"], "FAILED")
        self.assertIn("unsupported", result.stderr)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob("minkron-replay-*")), [])


class ArchivedControlFlowTests(unittest.TestCase):
    """Execute only the archived main AST against no-op computation stubs."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="replay-old-test-", dir=SCRATCH)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "PROFILE.json").write_text(json.dumps(dict(budget=dict(threads=1))))
        checkpoint = dict(path="checkpoints/screen_k8_25101.pt", model_tensor_sha256="a" * 64)
        (self.root / "CHECKPOINTS.json").write_text(json.dumps([checkpoint]))
        with gzip.open(self.root / "ALL_STEPS.csv.gz", "wt") as stream:
            writer = csv.DictWriter(stream, fieldnames=("arm", "seed", "stage", "total_steps", "lr"))
            writer.writeheader()
            writer.writerow(dict(arm="k8", seed=25101, stage="screen", total_steps=1, lr=.001))
        self.calls = dict(steps=0, models=0)

    def original_main(self, optimize=0):
        calls = self.calls

        class Tensor:
            def reshape(self, *args):
                return self

        class Loss:
            def backward(self):
                pass

        class Model:
            def __call__(self, *args, **kwargs):
                return Tensor(), None

            def parameters(self):
                return []

            def state_dict(self):
                return {}

        class Optimizer:
            param_groups = [dict(lr=0)]

            def zero_grad(self, **kwargs):
                pass

            def step(self):
                calls["steps"] += 1

            def state_dict(self):
                return {}

        def make_model(*args):
            calls["models"] += 1
            return Model()

        source = (guard.PARENT / "replay.py").read_text()
        main = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "main")
        module = ast.Module(body=[main], type_ignores=[])
        torch = types.SimpleNamespace(set_num_threads=lambda *args: None,
            nn=types.SimpleNamespace(utils=types.SimpleNamespace(clip_grad_norm_=lambda *args, **kwargs: None)),
            save=lambda value, path: Path(path).write_bytes(b"synthetic overwritten checkpoint"))
        sampler = types.SimpleNamespace(next=lambda: (Tensor(), Tensor(), None))
        namespace = dict(Path=Path, ROOT=self.root, json=json, csv=csv, gzip=gzip, torch=torch,
                         make_model=make_model, optimizer=lambda *args: Optimizer(),
                         data=lambda: dict(train=[]), Windows=lambda *args: sampler,
                         F=types.SimpleNamespace(cross_entropy=lambda *args: Loss()),
                         tensor_digest=lambda *args: "b" * 64)
        exec(compile(module, "archived_replay_main_only", "exec", optimize=optimize), namespace)
        return namespace["main"]

    def args(self, stage="screen"):
        return types.SimpleNamespace(arm="k8", seed=25101, stage=stage, output=str(self.root / "result.pt"))

    def test_archived_main_returns_normally_on_tensor_mismatch(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            returned = self.original_main()(self.args())
        self.assertIsNone(returned)
        self.assertFalse(json.loads(output.getvalue())["matches_original_model_tensors"])
        self.assertEqual(self.calls["steps"], 1)

    def test_archived_overwrite_guard_disappears_with_optimization(self):
        output = Path(self.args().output)
        output.write_bytes(b"existing synthetic artifact")
        with self.assertRaises(AssertionError):
            self.original_main(optimize=0)(self.args())
        self.assertEqual(output.read_bytes(), b"existing synthetic artifact")
        with contextlib.redirect_stdout(io.StringIO()):
            self.original_main(optimize=1)(self.args())
        self.assertEqual(output.read_bytes(), b"synthetic overwritten checkpoint")

    def test_archived_unsupported_target_is_checked_after_computation(self):
        with self.assertRaises(StopIteration):
            self.original_main()(self.args(stage="extension"))
        self.assertEqual(self.calls["models"], 1)
        self.assertEqual(self.calls["steps"], 1)


if __name__ == "__main__":
    unittest.main()
