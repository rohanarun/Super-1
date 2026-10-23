"""Exact core algebra, not a trained model. Outputs omit the common 1/sqrt(K)."""
from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt

FLOOR = F(1, 10**18)


@dataclass(frozen=True)
class Dual:
    value: F
    derivative: F = F(0)

    def __add__(self, other):
        other = dual(other)
        return Dual(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + -dual(other)

    def __rsub__(self, other):
        return dual(other) + -self

    def __mul__(self, other):
        other = dual(other)
        return Dual(self.value * other.value,
                    self.derivative * other.value + self.value * other.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = dual(other)
        return Dual(self.value / other.value,
                    (self.derivative * other.value - self.value * other.derivative) / (other.value * other.value))

    def __rtruediv__(self, other):
        return dual(other) / self


def dual(value):
    return value if isinstance(value, Dual) else Dual(F(value))


def scalar(value):
    return value.value if isinstance(value, Dual) else value


def rational_sqrt(value):
    raw = scalar(value)
    numerator, denominator = isqrt(raw.numerator), isqrt(raw.denominator)
    if numerator * numerator != raw.numerator or denominator * denominator != raw.denominator:
        raise ValueError("This exact fixture requires a rational square root")
    root = F(numerator, denominator)
    return Dual(root, value.derivative / (2 * root)) if isinstance(value, Dual) else root


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("Vector dimensions differ")
    return sum((a * b for a, b in zip(left, right)), F(0))


def norm_floor(vector):
    norm = dot(vector, vector)
    return norm, norm if scalar(norm) >= FLOOR else FLOOR


def dimensions(case):
    length, keys = len(case["q"]), len(case["state"])
    values = len(case["state"][0]) if keys else 0
    valid = length > 0 and keys > 0 and values > 0
    valid = valid and all(len(case[name]) == length for name in ("k", "v", "decay", "beta", "write", "reset"))
    valid = valid and all(len(row) == width for name, width in (("q", keys), ("k", keys), ("decay", keys), ("v", values), ("state", values)) for row in case[name])
    if not valid:
        raise ValueError("Inconsistent core dimensions")
    return length, keys, values


def sequential(case):
    dimensions(case)
    state = [row[:] for row in case["state"]]
    outputs = []
    for q, k, v, decay, beta, write, reset in zip(
            case["q"], case["k"], case["v"], case["decay"], case["beta"], case["write"], case["reset"]):
        state = [[(F(0) if reset else entry) * decay[i] for entry in row]
                 for i, row in enumerate(state)]
        norm, denominator = norm_floor(v)
        score = sum((k[i] * state[i][j] * v[j] for i in range(len(k)) for j in range(len(v))), F(0))
        update = write * norm / denominator - beta * score / denominator
        state = [[entry + update * k[i] * v[j] for j, entry in enumerate(row)]
                 for i, row in enumerate(state)]
        outputs.append([sum((q[i] * state[i][j] for i in range(len(k))), F(0)) for j in range(len(v))])
    return dict(outputs=outputs, state=state)


def forward_solve(matrix, rhs):
    result = []
    for i, row in enumerate(matrix):
        result.append((rhs[i] - dot(row[:i], result)) / row[i])
    return result


def block(case, mode="direct"):
    if mode not in ("direct", "factored", "scaled"):
        raise ValueError("Unknown exact representation")
    length, keys, values = dimensions(case)
    prefix, docs = [], []
    running, document = [F(1)] * keys, 0
    for decay, reset in zip(case["decay"], case["reset"]):
        running = [a * b for a, b in zip(running, decay)]
        prefix.append(running)
        document += int(reset)
        docs.append(document)
    origin = [[entry if docs[t] == 0 else F(0) for entry in row] for t, row in enumerate(prefix)]
    # Any positive, detached scale cancels exactly; rational fixtures avoid exp/log.
    centers = [scalar(entry) for entry in prefix[length // 2]]

    def pair(query, t, s):
        if s > t or docs[t] != docs[s]:
            return F(0)
        if mode == "factored":
            return sum((query[i] * prefix[t][i] / centers[i]
                        * case["k"][s][i] * centers[i] / prefix[s][i] for i in range(keys)), F(0))
        products = [F(1)] * keys
        for u in range(s + 1, t + 1):
            products = [a * b for a, b in zip(products, case["decay"][u])]
        return sum((query[i] * case["k"][s][i] * products[i] for i in range(keys)), F(0))

    norms, denominators = zip(*(norm_floor(vector) for vector in case["v"]))
    roots = [rational_sqrt(value) for value in denominators] if mode == "scaled" else None
    vectors = [[entry / roots[t] for entry in row] for t, row in enumerate(case["v"])] if roots else case["v"]
    factors = case["beta"] if roots else [beta / denominator for beta, denominator in zip(case["beta"], denominators)]
    matrix, rhs = [], []
    for t in range(length):
        matrix.append([F(1) if s == t else
                       factors[t] * pair(case["k"][t], t, s) * dot(vectors[t], vectors[s]) if s < t else F(0)
                       for s in range(length)])
        score = sum((case["k"][t][i] * origin[t][i] * case["state"][i][j] * vectors[t][j]
                     for i in range(keys) for j in range(values)), F(0))
        divisor = roots[t] if roots else denominators[t]
        rhs.append(case["write"][t] * norms[t] / divisor - factors[t] * score)
    updates = forward_solve(matrix, rhs)
    outputs = [[sum((case["q"][t][i] * origin[t][i] * case["state"][i][j] for i in range(keys)), F(0))
                + sum((pair(case["q"][t], t, s) * updates[s] * vectors[s][j] for s in range(t + 1)), F(0))
                for j in range(values)] for t in range(length)]
    final = []
    for i in range(keys):
        tails = []
        for s in range(length):
            product = F(1)
            for u in range(s + 1, length):
                product *= case["decay"][u][i]
            tails.append(case["k"][s][i] * product if docs[s] == docs[-1] else F(0))
        final.append([origin[-1][i] * case["state"][i][j]
                      + sum((tails[s] * updates[s] * vectors[s][j] for s in range(length)), F(0))
                      for j in range(values)])
    return dict(outputs=outputs, state=final, matrix=matrix, updates=updates)


def inverse_lower(matrix):
    length = len(matrix)
    columns = [forward_solve(matrix, [F(int(i == j)) for i in range(length)]) for j in range(length)]
    return [[columns[j][i] for j in range(length)] for i in range(length)]


def infinity_norm(matrix):
    return max(sum((abs(scalar(entry)) for entry in row), F(0)) for row in matrix)


def condition_infinity(matrix):
    return infinity_norm(matrix) * infinity_norm(inverse_lower(matrix))


def run_chunks(case, lengths, mode):
    state = case["state"]
    outputs, start = [], 0
    for length in lengths:
        end = start + length
        piece = {key: value[start:end] for key, value in case.items() if key != "state"}
        piece["state"] = state
        result = sequential(piece) if mode == "sequential" else block(piece, mode)
        outputs.extend(result["outputs"])
        state, start = result["state"], end
    if start != len(case["q"]):
        raise ValueError("Chunk lengths do not cover this case")
    return dict(outputs=outputs, state=state)


def equivalent(left, right):
    if isinstance(left, list) or isinstance(right, list):
        return isinstance(left, list) and isinstance(right, list) and len(left) == len(right) and all(
            equivalent(a, b) for a, b in zip(left, right))
    return dual(left) == dual(right)


def core_equal(left, right):
    return equivalent(left["outputs"], right["outputs"]) and equivalent(left["state"], right["state"])
