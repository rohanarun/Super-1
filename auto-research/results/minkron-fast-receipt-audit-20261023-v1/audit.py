"""Reconcile pinned, saved minKron receipts. Standard library; no model execution."""
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARENT = "auto-research/results/minkron-fast-20261021-v1/"
REPORT = "auto-research/reports/minkron-fast-20261021-v1.md"
ORIGINAL_DATA = "auto-research/results/minkron-publication-20261021-v1/nanogpt/data/fineweb_cpu/manifest.json"
ARMS = ("k8", "k16", "k32", "k16_lean", "k16_lean_conv", "transformer_lean")
SEEDS = (25101, 25102, 25103)
LABELS = ("8×8 baseline", "16×16", "32×32", "16×16, half MLP",
          "16×16, half MLP + conv4", "Transformer, half MLP")
JSON_FILES = ("METRICS.json", "SUMMARY.json", "TERMINAL.json", "FIDELITY.json",
              "PROTOCOL.json", "PROTOCOL_ADDENDUM_01.json", "PROTOCOL_ADDENDUM_02.json",
              "PROFILE.json", "SOURCE_MANIFEST.json", "PREREGISTRATION_RECEIPT.json",
              "SELECTION_FREEZE.json", "SELECTED_CONFIG.json", "CHECKPOINTS.json",
              "REPLAY_VERIFICATION.json", "data/manifest.json", "data/tokenizer.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def check_hashes(contents, expected):
    require(contents.keys() == expected.keys(), "input inventory mismatch")
    for name, digest in expected.items():
        require(sha(contents[name]) == digest, f"input hash mismatch: {name}")


def same(actual, expected, label):
    if isinstance(expected, bool):
        require(actual is expected, label)
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key, value in expected.items():
            same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for index, value in enumerate(expected):
            same(actual[index], value, f"{label}[{index}]")
    else:
        require(actual == expected, label)


def finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite saved scalar")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def load_bundle():
    inventory = json.loads((HERE / "INPUTS.json").read_text())
    contents = {}
    for name in inventory["files"]:
        path = (REPO / name).resolve()
        require(path.is_relative_to(REPO), "input outside repository")
        contents[name] = path.read_bytes()
    check_hashes(contents, inventory["files"])
    bundle = {name: json.loads(contents[PARENT + name]) for name in JSON_FILES}
    bundle["receipts"] = {name[len(PARENT + "runs/"):]: json.loads(content)
                          for name, content in contents.items()
                          if name.startswith(PARENT + "runs/")}
    with gzip.open(io.BytesIO(contents[PARENT + "ALL_STEPS.csv.gz"]), "rt") as stream:
        bundle["steps"] = list(csv.DictReader(stream))
    bundle["report"] = contents[REPORT].decode()
    bundle["original_data"] = json.loads(contents[ORIGINAL_DATA])
    bundle["hashes"] = {name: sha(content) for name, content in contents.items()}
    registry = json.loads((REPO / "auto-research/RUN_REGISTRY.json").read_text())
    parents = [entry for entries in registry.values() if isinstance(entries, list)
               for entry in entries if isinstance(entry, dict)
               and entry.get("run_id") == "minkron-fast-20261021-v1"]
    require(len(parents) == 1, "parent registry identity coverage")
    canonical = json.dumps(parents[0], sort_keys=True, separators=(",", ":")).encode()
    require(sha(canonical) == inventory["parent_registry_entry_sha256"], "parent registry binding")
    bundle["registry"] = parents[0]
    return bundle


def keyed(rows, fields, expected, label):
    keys = [tuple(row[field] for field in fields) for row in rows]
    require(len(keys) == len(expected) and set(keys) == expected, label + " identity coverage")
    return dict(zip(keys, rows))


def check_score(score, targets, label):
    require(score["tokens"] == targets and len(score["window_ce"]) == targets // 128,
            label + " evaluation coverage")
    require(all(value >= 0 for value in score["window_ce"]), label + " negative CE")
    same(score["ce"], statistics.mean(score["window_ce"]), label + " window mean")
    same(score["ppl"], math.exp(score["ce"]), label + " perplexity")


def audit(bundle):
    finite(bundle)
    metrics, summary = bundle["METRICS.json"], bundle["SUMMARY.json"]
    terminal, protocol = bundle["TERMINAL.json"], bundle["PROTOCOL.json"]
    budget = metrics["budget"]
    same(budget, bundle["PROFILE.json"]["budget"], "profile budget")
    require(budget["screen_steps"] == 186 and budget["tokens_per_update"] == 1024
            and budget["screen_tokens_per_arm_seed"] == 190464, "frozen screen budget")
    screen_keys = {(arm, seed) for arm in ARMS for seed in SEEDS}
    screen = keyed(metrics["screen"], ("arm", "seed"), screen_keys, "screen")
    means = {arm: statistics.mean(screen[arm, seed]["development"]["ce"] for seed in SEEDS)
             for arm in ARMS}
    winner = min(ARMS[1:-1], key=lambda arm: (round(means[arm], 6), arm))
    selection = metrics["selection"]
    same(selection, bundle["SELECTION_FREEZE.json"], "selection freeze")
    same(selection["development_means"], means, "development selection means")
    require(selection["selected_recurrent_variant"] == winner, "development-only winner")
    same(selection["checkpoints"], [row["checkpoint"] for row in metrics["screen"]],
         "selection checkpoint identities")
    extension_keys = {(arm, SEEDS[0]) for arm in ("k8", winner, "transformer_lean")}
    extensions = keyed(metrics["extensions"], ("arm", "seed"), extension_keys, "extension")
    keys = {("screen", arm, seed) for arm, seed in screen_keys}
    keys |= {("extension", arm, seed) for arm, seed in extension_keys}
    confirmation = keyed(metrics["confirmation"], ("stage", "arm", "seed"), keys, "confirmation")
    checkpoints = {row["path"]: row for row in bundle["CHECKPOINTS.json"]}
    require(len(checkpoints) == len(bundle["CHECKPOINTS.json"]) == 21, "checkpoint identity coverage")
    groups = defaultdict(list)
    for step in bundle["steps"]:
        groups[step["stage"], step["arm"], int(step["seed"])].append(step)
    require(set(groups) == keys, "trajectory identity coverage")
    expected_receipts = set()
    for row in metrics["screen"] + metrics["extensions"]:
        stage, arm, seed = row["stage"], row["arm"], row["seed"]
        label = f"{stage}_{arm}_{seed}"
        require((stage, arm, seed) in keys and row["completed"] is True, label + " completion")
        offset = 0 if stage == "screen" else screen[arm, seed]["steps"]
        prior_seconds = 0 if stage == "screen" else screen[arm, seed]["training_seconds"]
        require(row["steps"] > 0 and row["total_steps"] == offset + row["steps"], label + " total steps")
        require(row["tokens"] == row["steps"] * 1024
                and row["total_tokens"] == row["total_steps"] * 1024, label + " target exposures")
        if stage == "screen":
            require(row["steps"] == 186, label + " screen steps")
        same(row["total_training_seconds"], prior_seconds + row["training_seconds"], label + " lifetime time")
        same(row["tokens_per_second"], row["tokens"] / row["training_seconds"], label + " throughput")
        require(row["parameters"] == screen[arm, seed]["parameters"], label + " parameter count")
        check_score(row["development"], 16384, label + " development")
        cp = row["checkpoint"]
        require(cp["path"] == f"checkpoints/{label}.pt", label + " checkpoint identity")
        for field in ("path", "bytes", "sha256"):
            same(cp[field], checkpoints[cp["path"]][field], label + " checkpoint " + field)
        confirm = confirmation[stage, arm, seed]
        same(confirm["checkpoint"], cp, label + " confirmation checkpoint")
        check_score(confirm["first"], 32768, label + " confirmation")
        difference = abs(confirm["first"]["ce"] - confirm["duplicate_ce"])
        same(confirm["absolute_duplicate_difference"], difference, label + " duplicate difference")
        require(difference <= 1e-7, label + " duplicate tolerance")
        for name, record in ((label + ".json", row), ("confirmation_" + label + ".json", confirm)):
            expected_receipts.add(name)
            same(bundle["receipts"][name], record, "receipt " + name)
        trace = groups[stage, arm, seed]
        require([int(step["step"]) for step in trace] == list(range(1, row["steps"] + 1)),
                label + " trajectory step sequence")
        previous_seconds = 0.0
        for index, step in enumerate(trace):
            total = offset + index + 1
            require(int(step["total_steps"]) == total and int(step["tokens"]) == total * 1024,
                    label + " trajectory cursor")
            elapsed = float(step["stage_training_seconds"])
            require(math.isfinite(elapsed) and elapsed > previous_seconds, label + " trajectory time")
            for field in ("loss", "grad_norm", "lr"):
                value = float(step[field])
                require(math.isfinite(value) and value >= 0, label + " trajectory " + field)
            if stage == "screen":
                warm = min(20, max(4, row["steps"] // 10))
                fraction = max(0, (index - warm) / max(1, row["steps"] - warm))
                lr = .003 * (index + 1) / warm if index < warm else .0003 + .0027 * .5 * (1 + math.cos(math.pi * fraction))
            else:
                lr = .0003 + .0012 * .5 * (1 + math.cos(math.pi * min(1, previous_seconds / 300)))
            same(float(step["lr"]), lr, label + " trajectory learning rate")
            if stage == "extension":
                require(previous_seconds < 300, label + " continuation stop")
            previous_seconds = elapsed
        same(previous_seconds, row["training_seconds"], label + " trajectory terminal time")
        same(float(trace[-1]["loss"]), row["final_training_loss"], label + " terminal loss")
        if stage == "extension":
            require(previous_seconds >= 300, label + " continuation duration")
    require(set(bundle["receipts"]) == expected_receipts, "per-run receipt coverage")
    require(set(checkpoints) == {row["checkpoint"]["path"] for row in metrics["screen"] + metrics["extensions"]},
            "checkpoint path coverage")

    recomputed = []
    for arm in ARMS:
        rows = [screen[arm, seed] for seed in SEEDS]
        ce = [confirmation["screen", arm, seed]["first"]["ce"] for seed in SEEDS]
        delta = [ce[index] - confirmation["screen", "k8", seed]["first"]["ce"] for index, seed in enumerate(SEEDS)]
        dev_delta = [screen[arm, seed]["development"]["ce"] - screen["k8", seed]["development"]["ce"] for seed in SEEDS]
        require(len({row["parameters"] for row in rows}) == 1, arm + " paired parameter counts")
        recomputed.append(dict(arm=arm, parameters=rows[0]["parameters"], development_ce=means[arm],
            confirmation_ce=statistics.mean(ce), confirmation_ce_sd=statistics.stdev(ce),
            per_seed_confirmation_ce=ce, paired_confirmation_delta=delta,
            mean_paired_confirmation_delta=statistics.mean(delta), paired_confirmation_delta_sd=statistics.stdev(delta),
            mean_paired_development_delta=statistics.mean(dev_delta),
            exploratory_development_advance_gate=statistics.mean(dev_delta) <= -.02 and all(value < 0 for value in dev_delta),
            confirmation_consistent=statistics.mean(delta) <= -.02 and all(value < 0 for value in delta),
            training_tokens_per_second=sum(row["tokens"] for row in rows) / sum(row["training_seconds"] for row in rows),
            mean_screen_seconds=statistics.mean(row["training_seconds"] for row in rows)))
    continuation = [dict(arm=row["arm"], tokens=row["total_tokens"], additional_tokens=row["tokens"],
        total_training_seconds=row["total_training_seconds"], extension_seconds=row["training_seconds"],
        tokens_per_second=row["tokens_per_second"], development_ce=row["development"]["ce"],
        confirmation_ce=confirmation["extension", row["arm"], row["seed"]]["first"]["ce"])
        for row in metrics["extensions"]]
    exposures = sum(row["tokens"] for row in metrics["screen"] + metrics["extensions"])
    seconds = sum(row["training_seconds"] for row in metrics["screen"] + metrics["extensions"])
    same(summary, dict(screen=recomputed, extensions=continuation, total_target_exposures=exposures,
                       total_training_seconds=seconds, selection=winner), "summary")
    require(terminal["status"] == "COMPLETED" and terminal["all_finite"] is True, "terminal status")
    for field, value in (("screen_runs", 18), ("extensions", 3), ("confirmation_duplicates_passed", 21),
                         ("official_test_accessed", False), ("total_training_seconds", seconds)):
        same(terminal[field], value, "terminal " + field)
    require(seconds <= budget["planned_total_training_walltime_ceiling_seconds"]
            and sum(row["training_seconds"] for row in metrics["screen"]) <= budget["phase1_walltime_ceiling_seconds"],
            "training budget ceilings")
    for field, value in (("completed_screen_runs", 18), ("completed_extensions", 3),
                         ("total_target_exposures", exposures), ("training_seconds", seconds),
                         ("selected_recurrent_variant", winner)):
        same(bundle["registry"][field], value, "registry " + field)

    fidelity = bundle["FIDELITY.json"]
    require(fidelity["status"] == "PASS", "qualification status")
    fidelity_keys = {(arm, seed, dtype) for arm in ARMS for seed in SEEDS[:2]
                     for dtype in ("torch.float32", "torch.float64")}
    keyed(fidelity["rows"], ("arm", "seed", "dtype"), fidelity_keys, "qualification")
    errors = ("forward_state_max_error", "parameter_gradient_max_error", "optimizer_step_max_error",
              "streaming_max_error", "reset_max_error", "causal_max_error")
    for row in fidelity["rows"]:
        atol = protocol["execution"]["fidelity_tolerances"][row["dtype"]]["atol"]
        require(all(0 <= row[field] <= atol for field in errors), "qualification saved absolute bound")

    hashes, sources = bundle["hashes"], bundle["SOURCE_MANIFEST.json"]
    for name, digest in sources["files"].items():
        require(hashes[PARENT + name] == digest, "source binding " + name)
    for name, digest in sources["addenda"].items():
        require(hashes[PARENT + name] == digest, "addendum binding " + name)
    prereg = bundle["PREREGISTRATION_RECEIPT.json"]
    require(hashes[PARENT + "PROTOCOL.json"] == sources["protocol_sha256"] == prereg["protocol_sha256"], "protocol binding")
    require(hashes[PARENT + "PROFILE.json"] == sources["profile_sha256"], "profile binding")
    require(hashes[PARENT + "data/manifest.json"] == prereg["data_manifest_sha256"], "data manifest binding")
    data, tokenizer = bundle["data/manifest.json"], bundle["data/tokenizer.json"]
    require(hashes[PARENT + "data/tokenizer.json"] == data["tokenizer_sha256"] == protocol["data"]["tokenizer_sha256"], "tokenizer binding")
    same(data["source"], bundle["original_data"], "original data manifest")
    for split, field in (("train", "train_sha256"), ("development", "selection_sha256"), ("confirmation", "confirmation_sha256")):
        require(data["splits"][split]["sha256"] == protocol["data"][field], split + " data binding")
    require(tokenizer["n_vocab"] == 4096 and tokenizer["eot"] == 4095
            and len(tokenizer["ranks"]) == 4095
            and [row[1] for row in tokenizer["ranks"]] == list(range(4095)), "tokenizer rank coverage")
    require(tokenizer["full_gpt2_ranks_sha256"] == data["source"]["tokenizer"]["ranks_sha256"], "source tokenizer binding")
    started = datetime.fromisoformat(terminal["started_utc"])
    require(datetime.fromisoformat(prereg["utc"]) <= started == datetime.fromisoformat(sources["utc"]), "preregistration chronology")
    for name in ("PROTOCOL_ADDENDUM_01.json", "PROTOCOL_ADDENDUM_02.json"):
        require(bundle[name]["before_language_training"] is True
                and datetime.fromisoformat(bundle[name]["utc"]) < started, "addendum chronology")
    require(started < datetime.fromisoformat(selection["utc"]) < datetime.fromisoformat(terminal["completed_utc"]), "selection chronology")
    config = bundle["SELECTED_CONFIG.json"]
    require(winner == "k16_lean" and config["parameters"] == screen[winner, SEEDS[0]]["parameters"]
            and config["config"]["key_dim"] == 16 and config["config"]["value_dim"] == 16
            and config["config"]["mlp_hidden"] == 384 and config["config"]["short_conv"] == 0
            and config["config"]["recurrence_fp64"] is True, "selected config binding")
    replay = bundle["REPLAY_VERIFICATION.json"]
    require(replay["matches_original_model_tensors"] is True
            and replay["model_tensor_sha256"] == replay["expected"]
            == checkpoints["checkpoints/screen_k16_25101.pt"]["model_tensor_sha256"], "saved replay binding")

    result_section = bundle["report"].partition("## Results\n")[2].partition("\n## Deployment measurements")[0]
    report_rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
                   for line in result_section.splitlines()
                   if line.startswith("| ") and not line.startswith("| Arm ")]
    expected_rows = []
    for label, row in zip(LABELS, recomputed):
        expected_rows.append([label, f"{row['parameters']:,}", f"{row['development_ce']:.4f}",
            f"{row['confirmation_ce']:.4f} ± {row['confirmation_ce_sd']:.4f}",
            f"{row['training_tokens_per_second']:,.0f}", f"{row['mean_screen_seconds']:.2f}"])
    for label, row in zip(LABELS, recomputed):
        expected_rows.append([label] + [f"{value:.4f}" for value in row["per_seed_confirmation_ce"]]
            + [f"{row['mean_paired_confirmation_delta']:+.4f}", f"{row['paired_confirmation_delta_sd']:.4f}"])
    for row in continuation:
        expected_rows.append([LABELS[ARMS.index(row["arm"])], f"{row['tokens']:,}", f"{row['additional_tokens']:,}",
            f"{row['total_training_seconds']:.2f}", f"{row['development_ce']:.4f}", f"{row['confirmation_ce']:.4f}"])
    same(report_rows, expected_rows, "published result tables")
    extension_by_arm = {row["arm"]: row for row in continuation}
    return dict(status="PASS_STORED_RECEIPT_CONSISTENCY", screen=recomputed, extensions=continuation,
        selected_recurrent_variant=winner, total_target_exposures=exposures, training_seconds=seconds,
        trajectory_rows=len(bundle["steps"]), per_run_receipts=len(expected_receipts),
        development_window_cells=21 * 128, confirmation_window_cells=21 * 256,
        saved_qualification_rows=24, saved_absolute_error_bounds=24 * 6, printed_result_rows=len(expected_rows),
        transformer_to_lean_extension_throughput_ratio=extension_by_arm["transformer_lean"]["tokens_per_second"] / extension_by_arm[winner]["tokens_per_second"],
        checked_model_or_corpus_bodies=False, new_model_execution=False,
        limitations=["Hashes and saved scalars do not replay logits, gradients, tokenizer execution or checkpoints.",
            "Qualification maxima are saved receipts, with FP64 recurrence in both parameter dtypes.",
            "Continuation is one seed, equal added time, unequal lifetime time and target exposures.",
            "Development/confirmation regions were reused; duplicates are not independent seeds.",
            "No fresh total-memory, latency, reasoning, long-context, generalization or SOTA claim."])


def main():
    result = audit(load_bundle())
    inventory = (HERE / "INPUTS.json").read_bytes()
    result.update(input_files=len(json.loads(inventory)["files"]), inputs_sha256=sha(inventory),
                  audit_source_sha256=sha(Path(__file__).read_bytes()), protocol_sha256=sha((HERE / "PROTOCOL.json").read_bytes()))
    print(json.dumps(result, indent=2, allow_nan=False) + "\n", end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, ZeroDivisionError, OverflowError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
