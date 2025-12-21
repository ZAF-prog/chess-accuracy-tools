import unittest
import math
from src.Regan_IPR_Workflow import compute_sae_delta, solve_p0_equation3, calculate_move_probabilities, centipawns_to_winprob

class TestReganMath(unittest.TestCase):
    def test_sae_delta_same_sign(self):
        # v0=1.00 (100cp), vi=0.50 (50cp). Both positive.
        # Delta = |log(1+|1.0|) - log(1+|0.5|)| = |log(2) - log(1.5)| = log(2/1.5) = log(1.333...)
        # log(1.333) ~= 0.28768
        delta = compute_sae_delta(100, 50)
        expected = math.log(1 + 1.0) - math.log(1 + 0.5)
        self.assertAlmostEqual(delta, expected, places=5)

    def test_sae_delta_cross_zero(self):
        # v0=0.50 (50cp), vi=-0.50 (-50cp).
        # Delta = log(1+|0.5|) + log(1+|-0.5|) = 2 * log(1.5)
        # 2 * 0.4054 = 0.8109
        delta = compute_sae_delta(50, -50)
        expected = math.log(1.5) + math.log(1.5)
        self.assertAlmostEqual(delta, expected, places=5)

    def test_p0_solver_normalization(self):
        # Test if probabilities sum to 1.0
        # Case: Best move + 2 others
        values = [100.0, 50.0, 0.0] 
        s, c = 0.5, 1.0
        
        probs = calculate_move_probabilities(values, s, c)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        
        # Best move probability p0 should be highest
        self.assertTrue(probs[0] > probs[1])
        self.assertTrue(probs[1] > probs[2])
        
        # Check Equation 3 relationship: p_i = p_0 ^ exp( (delta/s)^c )
        # log(pi) = exp(...) * log(p0)
        p0 = probs[0]
        pi = probs[1]
        delta_i = compute_sae_delta(100, 50)
        exponent = math.exp((delta_i/s)**c)
        
        # floating point drift might be slight, but p0^exponent should be approx pi
        self.assertAlmostEqual(pi, math.pow(p0, exponent), places=4)

    def test_degenerate_s(self):
        # s very small -> best move gets all prob
        values = [100, 50]
        probs = calculate_move_probabilities(values, 0.000001, 1.0)
        self.assertAlmostEqual(probs[0], 1.0)
        self.assertAlmostEqual(probs[1], 0.0)

if __name__ == '__main__':
    unittest.main()
