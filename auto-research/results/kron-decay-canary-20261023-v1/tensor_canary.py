"""Authorized bounded synthetic mechanics canary; no corpus.

The standard-library prefix sets resource limits before importing tensor code.
All archived modules are hash-checked and used without changing their files.
"""
import copy
import functools
import hashlib
import importlib
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1] / "results" / "minkron-fast-20261021-v1"


def run():
    protocol = json.loads((HERE / "CANARY_PROTOCOL.json").read_text())
    if not protocol["authorization"]["approved"]:
        raise RuntimeError("DRAFT_NOT_AUTHORIZED: exact donor approval has not been recorded")
    cpu_limit = int(os.environ.get("KRON_CANARY_CPU_SECONDS", "600"))
    if not 1 <= cpu_limit <= 600:
        raise RuntimeError("Invalid remaining cumulative CPU allowance")
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
    for name, expected in json.loads((HERE / "SOURCE_INPUTS.json").read_text())["files"].items():
        actual = hashlib.sha256((PARENT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Source input changed: " + name)
    started, cpu_started = time.monotonic(), time.process_time()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(PARENT), str(HERE)]
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_device("cpu")
    if str(torch.__version__) != "2.9.1+cpu" or torch.version.cuda is not None:
        raise RuntimeError("Approved CPU-only torch build is required")
    source = importlib.import_module("model")
    common = importlib.import_module("common")
    candidate = importlib.import_module("candidate_core")
    variants = {
        "source_direct": source.kron_chunk,
        "source_factored": source.kron_chunk_factored,
        "prefix_scaled": functools.partial(candidate.chunk, scaled=True),
        "interval_direct": functools.partial(candidate.chunk, interval=True),
        "interval_scaled": functools.partial(candidate.chunk, interval=True, scaled=True),
    }
    tables, tiny_metadata, tiny_outputs, model_rows = [], [], [], []

    def tick(label):
        if time.monotonic() - started > 900:
            raise RuntimeError("Fixed walltime cap reached at " + label)
        print(json.dumps(dict(progress=label, cpu_seconds=time.process_time() - cpu_started)),
              file=sys.stderr, flush=True)

    def flat(value, prefix=""):
        if isinstance(value, torch.Tensor):
            return [(prefix, value)]
        if isinstance(value, dict):
            return [item for key, child in value.items() for item in flat(child, prefix + "/" + str(key))]
        return [item for i, child in enumerate(value) for item in flat(child, prefix + "/" + str(i))]

    def compare(left, right, tolerance):
        if len(left) != len(right):
            return dict(passed=False, finite=False, max_abs="length mismatch", failed=["length"])
        maximum, normalized, failed, finite = 0.0, 0.0, [], True
        for (name, a), (_, b) in zip(left, right):
            if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
                failed.append(name + ":missing/shape/dtype")
                finite = False
                continue
            ok = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            finite = finite and ok
            if not ok:
                failed.append(name + ":nonfinite")
                continue
            difference = (a.detach().double() - b.detach().double()).abs()
            maximum = max(maximum, float(difference.max()))
            normalized = max(normalized, float((difference /
                (tolerance["atol"] + tolerance["rtol"] * b.detach().double().abs())).max()))
            if not torch.allclose(a, b, **tolerance):
                failed.append(name)
        return dict(passed=not failed, finite=finite, max_abs=maximum if finite else "nonfinite",
                    max_normalized_error=normalized if finite else "nonfinite", failed=failed)

    def core_case(label, inputs, reset, dtype):
        tolerance = protocol["tolerances"][str(dtype).split(".")[-1]]
        for name, function in variants.items():
            tick(label + "/" + name)
            left = [item.detach().clone().requires_grad_(True) for item in inputs]
            right = [item.detach().clone().requires_grad_(True) for item in inputs]
            actual, state = function(*left[:6], left[6], reset)
            expected, reference_state = source.kron_recurrent(*right[:6], right[6], reset)
            forward = compare([("out", actual), ("state", state)],
                              [("out", expected), ("state", reference_state)], tolerance)
            gradient = dict(passed=False, finite=False, max_abs="not run", failed=["nonfinite forward"])
            if forward["finite"]:
                (actual.square().mean() + .17 * state.square().mean()).backward()
                (expected.square().mean() + .17 * reference_state.square().mean()).backward()
                gradient = compare([(str(i), value.grad) for i, value in enumerate(left)],
                                   [(str(i), value.grad) for i, value in enumerate(right)], tolerance)
            invariants = {}
            with torch.no_grad():
                cut = max(1, inputs[0].shape[1] // 3)
                first, carried = function(*[value[:, :cut] for value in inputs[:6]], inputs[6], reset[:, :cut])
                rest, final = function(*[value[:, cut:] for value in inputs[:6]], carried, reset[:, cut:])
                invariants["streaming"] = compare([("out", actual.detach()), ("state", state.detach())],
                    [("out", torch.cat((first, rest), 1)), ("state", final)], tolerance)
                future_inputs = [value.clone() for value in inputs]
                for value in future_inputs[:3]:
                    value[:, cut:] *= -1
                future_inputs[3][:, cut:] = -.25
                changed, _ = function(*future_inputs[:6], future_inputs[6], reset)
                invariants["causality"] = compare([("prefix", actual[:, :cut].detach())],
                    [("prefix", changed[:, :cut])], tolerance)
                positions = torch.nonzero(reset[0], as_tuple=False).flatten().tolist()
                boundary = next((position for position in positions if position > 0), None)
                if boundary is None:
                    invariants["reset"] = dict(passed=True, finite=True, max_abs=0.0, failed=[])
                else:
                    reset_inputs = [value.clone() for value in inputs]
                    for value in reset_inputs[:3]:
                        value[0, :boundary] *= -1
                    reset_inputs[3][0, :boundary] = -.125
                    changed, _ = function(*reset_inputs[:6], reset_inputs[6], reset)
                    invariants["reset"] = compare([("after reset", actual[0, boundary:].detach())],
                        [("after reset", changed[0, boundary:])], tolerance)
            row = [label, name, forward["passed"], forward["max_abs"], gradient["passed"],
                   gradient["max_abs"], invariants["streaming"]["passed"], invariants["streaming"]["max_abs"],
                   invariants["reset"]["passed"], invariants["reset"]["max_abs"],
                   invariants["causality"]["passed"], invariants["causality"]["max_abs"],
                   forward["finite"] and gradient["finite"] and all(value["finite"] for value in invariants.values()),
                   str(state.dtype), gradient["failed"]]
            if label.startswith("tiny/"):
                tiny_outputs.append(dict(case=label, variant=name,
                    outputs=actual.detach().flatten().tolist(), state=state.detach().flatten().tolist(),
                    reference_outputs=expected.detach().flatten().tolist(),
                    reference_state=reference_state.detach().flatten().tolist(),
                    forward=forward, gradients=gradient))
            tables.append(row)
            print(json.dumps(dict(core_result=row)), file=sys.stderr, flush=True)
            if name.startswith("interval") and not row[12]:
                raise RuntimeError("Nonfinite candidate; stopped this line at " + label + "/" + name)

    for dtype in (torch.float32, torch.float64):
        for gates in ((-1, -1), (-(2**27), -1), (-(2**56), -1), (-1, -(2**27), -1)):
            length = len(gates)
            for initial in (0, 1):
                for first_reset in (False, True):
                    q = torch.ones((1, length, 1, 1), dtype=dtype)
                    g = torch.tensor(gates, dtype=dtype).reshape(1, length, 1, 1)
                    beta = torch.full((1, length, 1), .5, dtype=dtype)
                    state = torch.full((1, 1, 1, 1), float(initial), dtype=dtype)
                    reset = torch.zeros((1, length), dtype=torch.bool)
                    reset[:, 0] = first_reset
                    label = f"tiny/{dtype}/{gates}/S{initial}/R{int(first_reset)}"
                    cumulative = g.cumsum(1)
                    center = (cumulative[:, :1] + cumulative[:, -1:]) * .5
                    tiny_metadata.append(dict(case=label, cumulative=cumulative.flatten().tolist(),
                                              centered_range=float((cumulative - center).abs().max())))
                    core_case(label, [q, q, q, g, beta, beta, state], reset, dtype)

    for dtype in (torch.float32, torch.float64):
        for seed in (73, 191, 307):
            for batch, length, heads, keys, values in ((1, 2, 1, 1, 1), (2, 7, 2, 4, 3), (2, 19, 1, 4, 3)):
                torch.manual_seed(seed)
                q, k, v = [functional.normalize(torch.randn((batch, length, heads, width), dtype=dtype), dim=-1)
                           for width in (keys, keys, values)]
                v[:, 0] = 0
                if length > 2:
                    v[:, 1] *= 1e-10
                g = -functional.softplus(torch.randn_like(q))
                beta, write = [torch.randn((batch, length, heads), dtype=dtype).sigmoid() for _ in range(2)]
                state = torch.randn((batch, heads, keys, values), dtype=dtype) * .2
                reset = torch.zeros((batch, length), dtype=torch.bool)
                reset[:, 0] = seed != 191
                reset[0, length // 2] = True
                label = f"random/{dtype}/{seed}/{batch},{length},{heads},{keys},{values}"
                core_case(label, [q, k, v, g, beta, write, state], reset, dtype)

    for name in (*variants, "source_factored_fp64_core"):
        tick("model/k16/25101/" + name)
        torch.manual_seed(25101)
        model = common.make_model("k16", 25101).to(torch.float32)
        if sum(p.numel() for p in model.parameters()) != 1970370:
            raise RuntimeError("Unexpected model parameter count")
        mixed = name == "source_factored_fp64_core"
        for block in model.blocks:
            block.mixer.recurrence_fp64 = mixed
        reference = copy.deepcopy(model)
        for block in reference.blocks:
            block.mixer.backend = "recurrent"
        for block in model.blocks:
            block.mixer.chunk_core = source.kron_chunk_factored if mixed else variants[name]
        if not all(torch.equal(a, b) for a, b in zip(model.parameters(), reference.parameters())):
            raise RuntimeError("Paired initialization mismatch")
        x = torch.randint(0, 4095, (2, 19))
        reset = torch.zeros_like(x, dtype=torch.bool)
        reset[:, 0] = True
        reset[0, 7] = True
        reset[1, 13] = True
        tolerance = protocol["tolerances"]["float32"]
        actual, state = model(x, reset=reset)
        expected, expected_state = reference(x, reset=reset)
        record = dict(variant=name, recurrence_fp64=mixed,
                      forward_state=compare([("logits", actual)] + flat(state),
                                            [("logits", expected)] + flat(expected_state), tolerance))
        if not record["forward_state"]["finite"]:
            model_rows.append(record)
            continue
        targets = torch.roll(x, -1, 1)
        loss = functional.cross_entropy(actual.reshape(-1, 4096), targets.flatten())
        expected_loss = functional.cross_entropy(expected.reshape(-1, 4096), targets.flatten())
        record["loss"] = compare([("loss", loss)], [("loss", expected_loss)], tolerance)
        loss.backward()
        expected_loss.backward()
        record["gradients"] = compare([(n, p.grad) for n, p in model.named_parameters()],
                                      [(n, p.grad) for n, p in reference.named_parameters()], tolerance)
        if not record["gradients"]["finite"]:
            model_rows.append(record)
            print(json.dumps(dict(model_result=record)), file=sys.stderr, flush=True)
            continue
        optimizer, expected_optimizer = source.optimizer(model, .001), source.optimizer(reference, .001)
        optimizer.step()
        expected_optimizer.step()
        record["optimizer_step"] = compare(list(model.named_parameters()), list(reference.named_parameters()), tolerance)
        record["optimizer_state"] = compare(flat(list(optimizer.state.values())), flat(list(expected_optimizer.state.values())), tolerance)
        with torch.no_grad():
            full, full_state = model(x, reset=reset)
            left, left_state = model(x[:, :6], reset=reset[:, :6])
            right, right_state = model(x[:, 6:], left_state, reset[:, 6:])
            record["streaming"] = compare([("logits", full)] + flat(full_state),
                                          [("logits", torch.cat((left, right), 1))] + flat(right_state), tolerance)
            altered = x.clone()
            altered[0, :7] = (altered[0, :7] + 77) % 4095
            other, _ = model(altered, reset=reset)
            record["reset"] = compare([("reset", full[0, 7:])], [("reset", other[0, 7:])], tolerance)
            changed = x.clone()
            changed[:, 12:] = (changed[:, 12:] + 101) % 4095
            future, _ = model(changed, reset=reset)
            record["causality"] = compare([("prefix", full[:, :12])], [("prefix", future[:, :12])], tolerance)
        model_rows.append(record)
        print(json.dumps(dict(model_result=record)), file=sys.stderr, flush=True)
    primary = [row for row in tables if row[1] == "interval_direct"]
    return dict(status="COMPLETED_SYNTHETIC_MECHANICS_CANARY", torch_version=torch.__version__,
                torch_git_version=torch.version.git_version, python=sys.version,
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                cpu_limit_seconds=cpu_limit, cuda_build=torch.version.cuda,
                torch_config=torch.__config__.show(), core_columns=["case", "variant", "forward_pass", "max_abs_forward",
                    "gradient_pass", "max_abs_gradient", "streaming_pass", "max_abs_streaming", "reset_pass",
                    "max_abs_reset", "causality_pass", "max_abs_causality", "all_finite", "state_dtype", "failed_gradient_inputs"],
                core_rows=tables, cumulative_metadata=tiny_metadata, tiny_outputs=tiny_outputs, model_rows=model_rows,
                primary_core_gate_passed=len(primary) == 50 and all(row[2] and row[4] and row[6] and row[8] and row[10] and row[12] for row in primary),
                model_gate_passed_by_variant={row["variant"]: len([value for value in row.values() if isinstance(value, dict)]) == 8
                    and all(value["passed"] for value in row.values() if isinstance(value, dict)) for row in model_rows},
                cpu_seconds=time.process_time() - cpu_started, wall_seconds=time.monotonic() - started,
                corpus_training=False, official_test_access=False, checkpoints_saved=False)


if __name__ == "__main__":
    try:
        result = run()
        print(json.dumps(result, allow_nan=False))
        sys.exit(0 if result["primary_core_gate_passed"] else 1)
    except Exception as error:
        print(json.dumps(dict(status="FAILED_OR_INCOMPLETE", error=repr(error))), file=sys.stderr)
        sys.exit(1)
