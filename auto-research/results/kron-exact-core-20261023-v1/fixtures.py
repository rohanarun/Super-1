"""The finite rational domain frozen in PROTOCOL.json; no external data."""
import random
from fractions import Fraction as F

SEEDS = tuple(int(word, 16) for word in ("dcbb91d0", "7e51cf13", "ee6fda1f", "efc14ca2"))
DIMENSIONS = ((1, 1), (2, 1), (2, 2), (4, 3))
REGIMES = ("unit", "zero", "near_floor", "mixed")


def unit_vector(width, rng):
    if width == 1:
        return [F(rng.choice((-1, 1)))]
    parameters = [F(rng.randint(-2, 2), 2) for _ in range(width - 1)]
    square = sum((value * value for value in parameters), F(0))
    return [2 * value / (1 + square) for value in parameters] + [(1 - square) / (1 + square)]


def case(length, keys, values, seed, regime, nonzero, reset_mask, derivative_domain=False):
    rng = random.Random(seed ^ (length << 24) ^ (keys << 16) ^ (values << 8))
    q, k, v, decay, beta, write = [], [], [], [], [], []
    for t in range(length):
        query, key, value = unit_vector(keys, rng), unit_vector(keys, rng), unit_vector(values, rng)
        if regime == "zero":
            query, key, value = [F(0)] * keys, [F(0)] * keys, [F(0)] * values
        elif regime == "near_floor":
            scales = (F(1, 10**10), F(1)) if derivative_domain else (F(1, 10**12), F(1, 10**10), F(1, 10**9), F(1))
            value = [entry * scales[t % len(scales)] for entry in value]
        elif regime == "mixed":
            scale = (F(0), F(1, 10**10), F(1))[t % 3]
            value = [entry * scale for entry in value]
            if t % 4 == 0:
                key = [F(0)] * keys
            if t % 5 == 0:
                query = [F(0)] * keys
        q.append(query)
        k.append(key)
        v.append(value)
        decay.append([rng.choice((F(1, 8), F(1, 2), F(3, 4), F(1))) for _ in range(keys)])
        beta.append(rng.choice((F(0), F(1, 2), F(1))))
        write.append(rng.choice((F(0), F(1, 3), F(1))))
    state = [[F(rng.randint(-2, 2), 3) if nonzero else F(0) for _ in range(values)] for _ in range(keys)]
    return dict(q=q, k=k, v=v, decay=decay, beta=beta, write=write,
                reset=[bool(reset_mask & (1 << t)) for t in range(length)], state=state)


def reset_masks(length):
    return range(1 << length) if length <= 4 else (0, 255, 1, 128, 85, 170, 24, 16)


def compositions(length):
    if length == 8:
        return ((8,), (1, 7), (4, 4), (3, 2, 3), (1,) * 8)
    result = []
    for mask in range(1 << (length - 1)):
        pieces, start = [], 0
        for t in range(1, length):
            if mask & (1 << (t - 1)):
                pieces.append(t - start)
                start = t
        pieces.append(length - start)
        result.append(tuple(pieces))
    return result


def forward_cases():
    for length in (1, 2, 4, 8):
        for keys, values in DIMENSIONS:
            for seed in SEEDS:
                for regime in REGIMES:
                    for nonzero in (False, True):
                        for mask in reset_masks(length):
                            identity = f"T{length}-K{keys}-V{values}-{seed:08x}-{regime}-S{int(nonzero)}-R{mask}"
                            yield identity, case(length, keys, values, seed, regime, nonzero, mask)


def derivative_cases():
    for seed in SEEDS:
        for regime in ("unit", "near_floor"):
            for nonzero in (False, True):
                for mask in range(4):
                    identity = f"J-{seed:08x}-{regime}-S{int(nonzero)}-R{mask}"
                    yield identity, case(2, 2, 2, seed, regime, nonzero, mask, derivative_domain=True)
