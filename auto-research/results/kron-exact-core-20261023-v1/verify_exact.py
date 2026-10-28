"""Run the preregistered finite exact domain, with a 600-second wall cap."""
import copy
import hashlib
import json
import sys
import time
from fractions import Fraction as F
from pathlib import Path

import exact_core as core
import fixtures

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "minkron-fast-20261021-v1"
NUMERIC_FIELDS = ("q", "k", "v", "decay", "beta", "write", "state")
PROTOCOL_SHA256 = "4dbb2d89cbedea4be9e1358d25363b487d8e882e825e0234ea6e9be6b8c1596d"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    if isinstance(value, core.Dual):
        return dict(value=str(value.value), derivative=str(value.derivative))
    if isinstance(value, F):
        return str(value)
    if isinstance(value, list):
        return [encoded(item) for item in value]
    if isinstance(value, dict):
        return {key: encoded(item) for key, item in value.items()}
    return value


def directions(case):
    for field in NUMERIC_FIELDS:
        for i, entry in enumerate(case[field]):
            if isinstance(entry, list):
                for j in range(len(entry)):
                    yield field, i, j
            else:
                yield field, i


def directional_case(case, active):
    result = copy.deepcopy(case)
    for path in directions(case):
        target = result[path[0]]
        for index in path[1:-1]:
            target = target[index]
        target[path[-1]] = core.Dual(target[path[-1]], F(int(path == active)))
    return result


def squared(matrix):
    return sum((entry * entry for row in matrix for entry in row), F(0))


def compare(reference, actual, identity, mode, case):
    if not core.core_equal(reference, actual):
        raise ValueError(json.dumps(dict(case=identity, mode=mode, input=encoded(case),
                                        reference=encoded(reference), actual=encoded(actual))))


def conditioning_controls():
    result = []
    for epsilon in (F(1, 10), F(1, 10**6), F(1, 10**10)):
        for family, values in (("small_current", (F(1), epsilon)),
                               ("small_previous", (epsilon, F(1))), ("unit_control", (F(1), F(1)))):
            case = dict(q=[[F(1)]] * 2, k=[[F(1)]] * 2, v=[[value] for value in values],
                        decay=[[F(1)]] * 2, beta=[F(1)] * 2, write=[F(1)] * 2,
                        reset=[False] * 2, state=[[F(0)]])
            direct, scaled = core.block(case, "direct"), core.block(case, "scaled")
            require(core.core_equal(direct, scaled), "conditioning control changed outputs")
            before, after = core.condition_infinity(direct["matrix"]), core.condition_infinity(scaled["matrix"])
            result.append(dict(family=family, epsilon=str(epsilon), original_condition_inf=str(before),
                               scaled_condition_inf=str(after),
                               comparison="better" if after < before else "worse" if after > before else "equal"))
    return result


def verify():
    started, cpu_started = time.monotonic(), time.process_time()
    require(hashlib.sha256((HERE / "PROTOCOL.json").read_bytes()).hexdigest() == PROTOCOL_SHA256,
            "frozen protocol changed")
    inputs = json.loads((HERE / "INPUTS.json").read_text())
    for name, digest in inputs["files"].items():
        require(hashlib.sha256((PARENT / name).read_bytes()).hexdigest() == digest, "parent input hash: " + name)
    digest = hashlib.sha256()
    counts = dict(forward_cases=0, whole_block_comparisons=0, chunk_comparisons=0,
                  state_contraction_checks=0, scaled_matrix_bound_checks=0,
                  exact_derivative_cases=0, exact_derivative_directions=0)
    worst_condition, worst_case = F(0), None
    maximum_derivative = F(0)
    maximum_derivative_location = None
    last_progress = started

    def tick():
        nonlocal last_progress
        elapsed = time.monotonic() - started
        require(elapsed <= 600, "INCOMPLETE: fixed 600-second cap; counts=" + json.dumps(counts))
        if time.monotonic() - last_progress >= 20:
            print(json.dumps(dict(progress=counts, wall_seconds=round(elapsed, 2))), file=sys.stderr, flush=True)
            last_progress = time.monotonic()

    for identity, case in fixtures.forward_cases():
        tick()
        reference = core.sequential(case)
        blocks = {mode: core.block(case, mode) for mode in ("direct", "factored", "scaled")}
        for mode, result in blocks.items():
            compare(reference, result, identity, mode, case)
            counts["whole_block_comparisons"] += 1
        for pieces in fixtures.compositions(len(case["q"])):
            for mode in ("sequential", "direct", "factored", "scaled"):
                result = core.run_chunks(case, pieces, mode)
                compare(reference, result, identity, f"{mode}/chunks={pieces}", case)
                counts["chunk_comparisons"] += 1
        perturbation = [[F((i + 2 * j) % 3 - 1, 5) for j in range(len(row))]
                        for i, row in enumerate(case["state"])]
        changed = copy.deepcopy(case)
        changed["state"] = [[entry + perturbation[i][j] for j, entry in enumerate(row)]
                            for i, row in enumerate(case["state"])]
        changed_result = core.sequential(changed)
        difference = [[a - b for a, b in zip(left, right)]
                      for left, right in zip(changed_result["state"], reference["state"])]
        require(squared(difference) <= squared(perturbation), "state contraction: " + identity)
        counts["state_contraction_checks"] += 1
        matrix = blocks["scaled"]["matrix"]
        inverse = core.inverse_lower(matrix)
        length = len(matrix)
        require(all(abs(entry) <= 1 for row in matrix + inverse for entry in row), "scaled entry bound: " + identity)
        condition = core.infinity_norm(matrix) * core.infinity_norm(inverse)
        require(condition <= length * length, "scaled condition bound: " + identity)
        if condition > worst_condition:
            worst_condition, worst_case = condition, identity
        counts["scaled_matrix_bound_checks"] += 1
        digest.update(json.dumps(dict(case=identity, result=encoded(reference)), sort_keys=True).encode())
        counts["forward_cases"] += 1

    for identity, case in fixtures.derivative_cases():
        for active in directions(case):
            tick()
            seeded = directional_case(case, active)
            reference = core.sequential(seeded)
            for mode in ("direct", "factored", "scaled"):
                compare(reference, core.block(seeded, mode), identity, f"{mode}/partial={active}", seeded)
            for field in ("outputs", "state"):
                for i, row in enumerate(reference[field]):
                    for j, entry in enumerate(row):
                        derivative = core.dual(entry).derivative
                        if abs(derivative) > maximum_derivative:
                            maximum_derivative = abs(derivative)
                            maximum_derivative_location = dict(case=identity, input=list(active),
                                output=[field, i, j], signed_derivative=str(derivative))
            digest.update(json.dumps(dict(case=identity, direction=active, result=encoded(reference)), sort_keys=True).encode())
            counts["exact_derivative_directions"] += 1
        counts["exact_derivative_cases"] += 1
    require(counts["forward_cases"] == 3840 and counts["exact_derivative_cases"] == 64
            and counts["exact_derivative_directions"] == 1536, "declared domain coverage")
    return dict(status="PASS_FINITE_EXACT_DOMAIN", **counts, exact_discrepancies=0,
                domain_result_sha256=digest.hexdigest(), worst_scaled_condition_inf=str(worst_condition),
                worst_scaled_condition_case=worst_case, maximum_absolute_core_partial=str(maximum_derivative),
                maximum_partial_location=maximum_derivative_location, conditioning_controls=conditioning_controls(),
                wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started,
                real_model_or_tensor_library_execution=False,
                limits=["Finite checks support the separately written algebraic derivation; they do not prove all inputs.",
                        "Rational fixtures have rational norm roots and omit the common 1/sqrt(K) output factor.",
                        "Exact core partials exclude the norm-floor kink and do not include upstream normalization or model gradients.",
                        "The factored oracle tests an arbitrary detached rational center, not floating exp/log numerics.",
                        "No FP32 accuracy, throughput, model quality, deployment or scaling improvement was measured."])


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        print(json.dumps(dict(status="FAILED_OR_INCOMPLETE", error=str(error))), file=sys.stderr)
        sys.exit(1)
