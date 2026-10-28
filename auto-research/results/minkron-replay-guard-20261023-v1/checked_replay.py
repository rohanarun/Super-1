"""Check a recorded reconstruction before publishing it; never overwrite an output.

--plan reads metadata only. Running without --plan performs the archived training
replay and requires the separate experiment/compute authorization in AGENTS.md.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "minkron-fast-20261021-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inputs():
    expected = json.loads((HERE / "INPUTS.json").read_text())["files"]
    contents = {}
    for name, digest in expected.items():
        path = (PARENT / name).resolve()
        require(path.is_relative_to(PARENT.resolve()), "input escapes parent package")
        contents[name] = path.read_bytes()
        require(hashlib.sha256(contents[name]).hexdigest() == digest, "input hash mismatch: " + name)
    result = {name: json.loads(contents[name]) for name in
              ("METRICS.json", "CHECKPOINTS.json", "PROFILE.json", "data/manifest.json")}
    with gzip.open(io.BytesIO(contents["ALL_STEPS.csv.gz"]), "rt") as stream:
        result["steps"] = list(csv.DictReader(stream))
    return result


def make_plan(inputs, arm, seed, stage, output):
    require(stage in ("screen", "extension"), "unsupported stage")
    name = f"checkpoints/{stage}_{arm}_{seed}.pt"
    matches = [row for row in inputs["CHECKPOINTS.json"] if row["path"] == name]
    require(len(matches) == 1, "unsupported or ambiguous recorded target: " + name)
    checkpoint = matches[0]
    expected = checkpoint["model_tensor_sha256"]
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "invalid expected tensor digest")
    metrics = inputs["METRICS.json"]
    runs = metrics["screen"] if stage == "screen" else metrics["extensions"]
    records = [row for row in runs if row["arm"] == arm and row["seed"] == seed and row["stage"] == stage]
    require(len(records) == 1, "target has no unique terminal record")
    record = records[0]
    require(record["completed"] is True, "target is incomplete")
    require(record["checkpoint"]["sha256"] == checkpoint["sha256"], "checkpoint identity mismatch")
    rows = [row for row in inputs["steps"] if row["arm"] == arm and int(row["seed"]) == seed
            and (row["stage"] == "screen" or stage == "extension")]
    rows.sort(key=lambda row: int(row["total_steps"]))
    require([int(row["total_steps"]) for row in rows] == list(range(1, record["total_steps"] + 1)),
            "recorded replay trajectory is incomplete or duplicated")
    screen_steps = metrics["budget"]["screen_steps"]
    for index, row in enumerate(rows):
        want_stage = "screen" if index < screen_steps else "extension"
        require(row["stage"] == want_stage and int(row["tokens"]) == (index + 1) * 1024,
                "recorded replay cursor/stage mismatch")
        lr = float(row["lr"])
        require(math.isfinite(lr) and lr > 0, "invalid recorded learning rate")
    require(rows and rows[-1]["stage"] == stage and int(rows[-1]["tokens"]) == record["total_tokens"],
            "replay endpoint mismatch")
    requested = Path(output).expanduser()
    require(not os.path.lexists(requested), "output already exists")
    destination = requested.parent.resolve() / requested.name
    require(not destination.is_relative_to(PARENT.resolve()), "output must be outside the frozen parent package")
    require(not os.path.lexists(destination), "output already exists")
    arrays = []
    for split, meta in inputs["data/manifest.json"]["splits"].items():
        path = (PARENT / "data" / meta["file"]).resolve()
        require(path.is_relative_to((PARENT / "data").resolve()), "array escapes data directory")
        arrays.append(dict(split=split, path=str(path), sha256=meta["sha256"]))
    return dict(status="PLAN_ONLY_NO_MODEL_EXECUTION", arm=arm, seed=seed, stage=stage,
                output=str(destination), expected_model_tensor_sha256=expected,
                recorded_steps=len(rows), recorded_target_exposures=record["total_tokens"],
                threads=inputs["PROFILE.json"]["budget"]["threads"], required_local_arrays=arrays)


def check_receipt(receipt, plan, candidate):
    require(isinstance(receipt, dict), "replay did not return a JSON object")
    require(Path(receipt["output"]).resolve() == candidate, "replay output identity mismatch")
    require(receipt["expected"] == plan["expected_model_tensor_sha256"], "replay expected digest is not bound to target")
    require(receipt["matches_original_model_tensors"] is True, "replay reported a tensor mismatch")
    require(receipt["model_tensor_sha256"] == receipt["expected"], "replay actual digest does not match expected")
    require(candidate.is_file() and not candidate.is_symlink(), "replay checkpoint is missing or not a regular file")
    require(candidate.stat().st_size > 0, "replay checkpoint is empty")


def run_checked(plan, runner=None):
    if runner is None:
        runner = subprocess.run
    destination = Path(plan["output"])
    require(not os.path.lexists(destination), "output already exists")
    for array in plan["required_local_arrays"]:
        path = Path(array["path"])
        require(path.is_file(), "required local array is missing: " + str(path))
        require(file_sha256(path) == array["sha256"], "required local array hash mismatch: " + str(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="minkron-replay-", dir=destination.parent)).resolve()
    candidate = staging / "checkpoint.pt"
    command = [sys.executable, "-B", str(PARENT / "replay.py"), "--arm", plan["arm"],
               "--seed", str(plan["seed"]), "--stage", plan["stage"], "--output", str(candidate)]
    try:
        result = runner(command, cwd=PARENT, capture_output=True, text=True)
        (staging / "stdout.txt").write_text(result.stdout)
        (staging / "stderr.txt").write_text(result.stderr)
        require(result.returncode == 0, f"replay child failed with exit code {result.returncode}")
        receipt = json.loads(result.stdout)
        check_receipt(receipt, plan, candidate)
        container_sha256 = file_sha256(candidate)
        container_bytes = candidate.stat().st_size
        # Same-filesystem link creates the final path atomically and refuses any existing path.
        os.link(candidate, destination)
    except (ValueError, KeyError, TypeError, OSError) as error:
        raise ValueError(f"{error}; diagnostics retained at {staging}") from error
    candidate.unlink()
    (staging / "stdout.txt").unlink()
    (staging / "stderr.txt").unlink()
    staging.rmdir()
    return dict(status="PASS_EXACT_MODEL_TENSOR_REPLAY", output=str(destination),
                checkpoint_sha256=container_sha256, checkpoint_bytes=container_bytes,
                model_tensor_sha256=receipt["model_tensor_sha256"],
                expected=plan["expected_model_tensor_sha256"],
                recorded_steps=plan["recorded_steps"],
                recorded_target_exposures=plan["recorded_target_exposures"],
                replay_receipt=receipt, replay_stderr=result.stderr,
                warning="Exact model-tensor identity does not imply identical checkpoint-container bytes or a new quality result.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=("screen", "extension"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", action="store_true", help="inspect metadata without model execution or array reads")
    args = parser.parse_args(argv)
    try:
        plan = make_plan(read_inputs(), args.arm, args.seed, args.stage, args.output)
        result = plan if args.plan else run_checked(plan)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps(dict(status="FAILED", error=str(error))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
