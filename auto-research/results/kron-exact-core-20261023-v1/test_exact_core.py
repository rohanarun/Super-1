"""Hand-derived controls and bounded exact checks; no tensor library or model."""
import copy
import unittest
from fractions import Fraction as F

import exact_core as core
import fixtures


def one_step(value=F(1), reset=False):
    return dict(q=[[F(1)]], k=[[F(1)]], v=[[value]], decay=[[F(1, 2)]],
                beta=[F(1, 4)], write=[F(3, 4)], reset=[reset], state=[[F(2)]])


def assert_all(test, case, expected):
    for mode in ("sequential", "direct", "factored", "scaled"):
        with test.subTest(mode=mode):
            result = core.sequential(case) if mode == "sequential" else core.block(case, mode)
            test.assertTrue(core.equivalent(result["state"], [[expected]]))
            test.assertTrue(core.equivalent(result["outputs"], [[expected]]))


class ExactCoreTests(unittest.TestCase):
    def test_hand_computed_nonzero_initial_state(self):
        assert_all(self, one_step(), F(3, 2))

    def test_reset_happens_before_the_current_token(self):
        assert_all(self, one_step(reset=True), F(3, 4))

    def test_zero_value_does_not_erase_or_write_state(self):
        assert_all(self, one_step(F(0)), F(1))

    def test_floor_keeps_the_original_write_ratio(self):
        assert_all(self, one_step(F(1, 10**10)), F(399, 400) + F(3, 4 * 10**12))

    def test_floor_boundary_forward_values(self):
        for value in (F(1, 10**12), F(1, 10**10), F(1, 10**9), F(1, 10**8)):
            case = one_step(value)
            reference = core.sequential(case)
            for mode in ("direct", "factored", "scaled"):
                self.assertTrue(core.core_equal(reference, core.block(case, mode)))

    def test_exact_derivative_is_large_despite_bounded_state(self):
        case = one_step(core.Dual(F(1, 10**10), F(1)))
        expected = -F(50_000_000) + F(9, 400)
        for mode in ("sequential", "direct", "factored", "scaled"):
            result = core.sequential(case) if mode == "sequential" else core.block(case, mode)
            self.assertEqual(result["state"][0][0].derivative, expected)

    def test_dual_product_quotient_and_square_root(self):
        x = core.Dual(F(9, 4), F(3))
        root = core.rational_sqrt(x)
        self.assertEqual(root, core.Dual(F(3, 2), F(1)))
        self.assertEqual(x / x, core.Dual(F(1), F(0)))
        self.assertEqual((x * x).derivative, F(27, 2))

    def test_generator_has_the_declared_cardinality_and_domains(self):
        self.assertEqual(sum(1 for _ in fixtures.forward_cases()), 3840)
        self.assertEqual(sum(1 for _ in fixtures.derivative_cases()), 64)
        for width in (1, 2, 3, 4):
            import random
            vector = fixtures.unit_vector(width, random.Random(17))
            self.assertEqual(core.dot(vector, vector), F(1))

    def test_nontrivial_reset_chunks_match_all_representations(self):
        case = fixtures.case(4, 4, 3, fixtures.SEEDS[0], "near_floor", True, 0b0110)
        reference = core.sequential(case)
        for pieces in fixtures.compositions(4):
            for mode in ("sequential", "direct", "factored", "scaled"):
                with self.subTest(chunks=pieces, mode=mode):
                    self.assertTrue(core.core_equal(reference, core.run_chunks(case, pieces, mode)))

    def test_diagonal_similarity_and_solution_transform(self):
        case = fixtures.case(4, 2, 2, fixtures.SEEDS[1], "near_floor", True, 0)
        original, scaled = core.block(case, "direct"), core.block(case, "scaled")
        roots = [core.rational_sqrt(core.norm_floor(vector)[1]) for vector in case["v"]]
        for i in range(4):
            self.assertEqual(scaled["updates"][i], roots[i] * original["updates"][i])
            for j in range(4):
                self.assertEqual(scaled["matrix"][i][j], roots[i] * original["matrix"][i][j] / roots[j])

    def test_reverse_scaling_is_not_a_universal_condition_improvement(self):
        def condition(values):
            case = dict(q=[[F(1)]] * 2, k=[[F(1)]] * 2, v=[[value] for value in values],
                        decay=[[F(1)]] * 2, beta=[F(1)] * 2, write=[F(1)] * 2,
                        reset=[False] * 2, state=[[F(0)]])
            return tuple(core.condition_infinity(core.block(case, mode)["matrix"]) for mode in ("direct", "scaled"))
        original, scaled = condition((F(1), F(1, 10**6)))
        self.assertGreater(original, scaled)
        reverse_original, reverse_scaled = condition((F(1, 10**6), F(1)))
        self.assertLess(reverse_original, reverse_scaled)
        self.assertEqual(condition((F(1), F(1))), (F(4), F(4)))

    def test_scaled_matrix_and_inverse_have_bounded_entries(self):
        case = fixtures.case(8, 4, 3, fixtures.SEEDS[2], "mixed", True, 24)
        matrix = core.block(case, "scaled")["matrix"]
        inverse = core.inverse_lower(matrix)
        self.assertTrue(all(abs(entry) <= 1 for row in matrix + inverse for entry in row))
        self.assertLessEqual(core.condition_infinity(matrix), F(8 * 8))

    def test_malformed_shapes_are_rejected(self):
        case = one_step()
        case["reset"] = []
        for mode in ("sequential", "direct", "factored", "scaled"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "dimensions"):
                core.sequential(case) if mode == "sequential" else core.block(case, mode)

    def test_wrong_reset_is_a_decisive_control(self):
        case = one_step(reset=True)
        wrong = copy.deepcopy(case)
        wrong["reset"][0] = False
        self.assertFalse(core.core_equal(core.sequential(case), core.block(wrong, "scaled")))


if __name__ == "__main__":
    unittest.main()
