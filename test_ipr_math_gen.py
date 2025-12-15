
import unittest
import math
import sys
import os

# Functions under test (Copied from V4 to verify logic independent of file structure)
def centipawns_to_winprob(cp):
    try:
        if cp > 20000: return 1.0
        if cp < -20000: return 0.0
        return 1.0 / (1.0 + math.pow(10, -cp / 400.0))
    except (OverflowError, ValueError):
        return 0.0 if cp < 0 else 1.0

def calculate_move_probabilities(values, s, c):
    if not values: return []
    win_probs = [centipawns_to_winprob(v * 100.0) for v in values]
    best_wp = max(win_probs)
    deltas = [max(0.0, best_wp - wp) for wp in win_probs]
    weights = []
    for d in deltas:
        if s <= 1e-9: 
            weights.append(1.0 if d == 0 else 0.0)
            continue
        try:
            term = d / s
            if term > 1000: 
                w = 0.0
            else:
                exponent = math.pow(term, c)
                if exponent > 700:
                    w = 0.0
                else:
                    w = math.exp(-exponent)
        except (OverflowError, ValueError):
            w = 0.0
        weights.append(w)
    total_w = sum(weights)
    if total_w == 0:
        return [1.0] + [0.0]*(len(values)-1)
    return [w / total_w for w in weights]

class TestIPRMath(unittest.TestCase):
    def test_win_prob_conversion(self):
        self.assertAlmostEqual(centipawns_to_winprob(0), 0.5)
        self.assertAlmostEqual(centipawns_to_winprob(400), 1/1.1)

    def test_extreme_values_relaxed(self):
        # Mate (100.0) vs +5.00 (5.0)
        # In WinProb: Mate=1.0, +5.00 ~= 0.95
        # Delta ~= 0.05.
        # With s=0.1, delta/s = 0.5. exp(-0.5) is significant (~0.6).
        # So "Mate" is NOT infinitely better than +5.00.
        values = [100.0, 5.0, 0.0] 
        s, c = 0.1, 1.0
        probs = calculate_move_probabilities(values, s, c)
        
        # Prob[0] should be higher than Prob[1], but not 1.0
        self.assertTrue(probs[0] > probs[1])
        self.assertTrue(probs[1] > probs[2])
        # Prob[2] (0.00 equal position vs Mate) should be near 0
        # Delta(1.0 - 0.5) = 0.5. 0.5/0.1 = 5. exp(-5) = 0.006. Small but non-zero.
        self.assertTrue(probs[2] < 0.05)

if __name__ == '__main__':
    unittest.main()
