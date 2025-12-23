
import numpy as np
import math
import time
from scipy.optimize import brentq
from typing import List

# --- ORIGINAL SCALAR LOGIC ---
def compute_delta_scalar(v0_cp: float, vi_cp: float) -> float:
    z0 = v0_cp / 100.0
    zi = vi_cp / 100.0
    if zi > z0: return 0.0
    if z0 * zi >= 0:
        return abs(math.log(1 + abs(z0)) - math.log(1 + abs(zi)))
    else:
        return math.log(1 + abs(z0)) + math.log(1 + abs(zi))

def solve_p0_equation3_scalar(deltas: List[float], s: float, c: float) -> float:
    if s <= 1e-9: return 1.0
    alphas = []
    for d in deltas:
        try:
            term = d / s
            val = math.exp(math.pow(term, c))
            alphas.append(val if val < 100 else 100.0)
        except:
            alphas.append(100.0)
    def f(p0):
        if p0 <= 0: return -1.0
        if p0 >= 1: return sum(1 for a in alphas if a == 1.0) - 1.0
        return sum(math.pow(p0, a) for a in alphas) - 1.0
    try:
        return brentq(f, 1e-12, 1 - 1e-12)
    except:
        return 1.0 / len(deltas)

# --- OPTIMIZED VECTORIZED LOGIC (from the script) ---
def compute_delta_vec(v0: np.ndarray, vi: np.ndarray) -> np.ndarray:
    def log_scale(z_cp):
        z = z_cp / 100.0
        return np.sign(z) * np.log1p(np.abs(z))
    delta = log_scale(v0) - log_scale(vi)
    return np.maximum(0.0, delta)

def solve_p0_equation3_vec(deltas_list: List[np.ndarray], s: float, c: float) -> np.ndarray:
    if s <= 1e-9: return np.ones(len(deltas_list))
    alphas_list = []
    max_alpha = 100.0
    for d in deltas_list:
        with np.errstate(over='ignore', invalid='ignore'):
            term = np.power(d / s, c)
            a = np.exp(term)
            a = np.where(np.isnan(a) | np.isinf(a) | (a > max_alpha), max_alpha, a)
        alphas_list.append(a)
    p0_results = np.zeros(len(deltas_list))
    for idx, alphas in enumerate(alphas_list):
        def f(p):
            if p <= 0: return -1.0
            if p >= 1: return np.sum(alphas == 1.0) - 1.0
            return np.sum(np.power(p, alphas)) - 1.0
        try:
            p0_results[idx] = brentq(f, 1e-12, 1 - 1e-12)
        except:
            p0_results[idx] = 1.0 / len(alphas)
    return p0_results

def test_correctness():
    print("Testing correctness...")
    np.random.seed(42)
    s, c = 0.15, 0.8
    
    for _ in range(5):
        # Generate 20 random moves in CP
        moves = np.random.randint(-200, 200, 20).astype(float)
        moves[0] = np.max(moves) # Ensure first is best
        
        # Scalar check
        deltas_sc = [compute_delta_scalar(moves[0], m) for m in moves]
        p0_sc = solve_p0_equation3_scalar(deltas_sc, s, c)
        
        # Vec check
        deltas_v = compute_delta_vec(moves[0], moves)
        p0_v = solve_p0_equation3_vec([deltas_v], s, c)[0]
        
        np.testing.assert_allclose(deltas_sc, deltas_v, rtol=1e-5, err_msg="Delta mismatch")
        np.testing.assert_allclose(p0_sc, p0_v, rtol=1e-5, err_msg="p0 mismatch")
        
    print("Correctness check PASSED.")

def run_benchmark():
    print("\nRunning benchmark...")
    # 10,000 positions with 20 moves each
    n_pos = 10000
    n_moves = 20
    data = np.random.randint(-300, 300, (n_pos, n_moves)).astype(float)
    for i in range(n_pos):
        data[i, 0] = np.max(data[i])
        
    s, c = 0.12, 0.75
    
    # Scalar Timing
    start = time.time()
    for i in range(n_pos):
        dists = [compute_delta_scalar(data[i, 0], m) for m in data[i]]
        _ = solve_p0_equation3_scalar(dists, s, c)
    scalar_time = time.time() - start
    print(f"Scalar time: {scalar_time:.4f}s")
    
    # Vectorized Timing
    start = time.time()
    # compute_delta_vec is vectorized over moves but we still loop over positions for deltas_list
    # Note: solve_p0_equation3_vec takes a list of arrays
    all_deltas = [compute_delta_vec(data[i, 0], data[i]) for i in range(n_pos)]
    _ = solve_p0_equation3_vec(all_deltas, s, c)
    vec_time = time.time() - start
    print(f"Vectorized time: {vec_time:.4f}s")
    
    print(f"Speedup: {scalar_time / vec_time:.2f}x")

if __name__ == "__main__":
    test_correctness()
    run_benchmark()
