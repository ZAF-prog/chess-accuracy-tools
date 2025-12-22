#!/usr/bin/env python
"""
Regan IPR Workflow v2.0 - Benchmark Projection
==============================================
Implements the "Strict" Regan methodology where all results are projected 
onto a standardized benchmark set (the "Solitaire Set") to eliminate 
game difficulty bias.

Workflow:
1.  Phase A (Calibration): 
    - Fits (s, c) for representative Elo buckets.
    - Projects each bucket's (s, c) onto a standardized BENCHMARK PGN.
    - Calculates Expected Average Error (AEe) on that benchmark.
    - Performs regression: Elo ~ AEe(Benchmark).
2.  Phase B (Player Analysis):
    - Fits (s, c) for a player on their own games (Test Set T).
    - Projects the player's (s, c) onto the BENCHMARK PGN.
    - Calculates the player's AEe(Benchmark).
    - Uses the Phase A regression to determine the final IPR.

Usage:
    python Regan_IPR_Workflow_V2.py <buckets_list_file> <training_pgn> --benchmark <benchmark_pgn>
"""

import multiprocessing
import chess
import chess.engine
import chess.pgn
import numpy as np
import math
import csv
import sys
import argparse
import pickle
import time
import os
import platform
import logging
import warnings
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize, brentq
from typing import List, Tuple
try:
    import psutil
except ImportError:
    psutil = None
import atexit

# --- CONFIGURATION ---
BOOK_MOVES = 8
CAP_EVAL = 300
MULTI_PV = 5
MATE_SCORE = 10000

# =============================================================================
# MATH MODULE: Regan Equation 3 & SAE
# =============================================================================

def compute_sae_delta(v0_cp: float, vi_cp: float) -> float:
    z0 = v0_cp / 100.0
    zi = vi_cp / 100.0
    if zi > z0: return 0.0
    if z0 * zi >= 0:
        return abs(math.log(1 + abs(z0)) - math.log(1 + abs(zi)))
    else:
        return math.log(1 + abs(z0)) + math.log(1 + abs(zi))

def solve_p0_equation3(deltas: List[float], s: float, c: float) -> float:
    if s <= 1e-9: return 1.0
    exponents = []
    for d in deltas:
        if d == 0: exponents.append(1.0)
        else:
            try:
                term = (d / s)
                E = math.exp(math.pow(term, c)) if term <= 20 else float('inf')
            except OverflowError: E = float('inf')
            exponents.append(E)

    def f(p0):
        sum_p = 0.0
        for E in exponents:
            if E == float('inf'): continue
            try: sum_p += math.pow(p0, E)
            except: pass
        return sum_p - 1.0

    try: return brentq(f, 0.0 + 1e-9, 1.0 - 1e-9)
    except: return 0.5

def calculate_move_probabilities(values_cp: List[float], s: float, c: float) -> List[float]:
    if not values_cp: return []
    best_cp = values_cp[0]
    deltas = [compute_sae_delta(best_cp, v) for v in values_cp]
    p0 = solve_p0_equation3(deltas, s, c)
    probs = []
    for d in deltas:
        if s <= 1e-9:
            probs.append(1.0 if d==0 else 0.0)
            continue
        try:
            term = (d / s)
            E = math.exp(math.pow(term, c)) if term <= 100 else float('inf')
            pi = 0.0 if E == float('inf') else math.pow(p0, E)
        except: pi = 0.0
        probs.append(pi)
    total = sum(probs)
    return [p/total for p in probs] if total > 0 else [1.0] + [0.0]*(len(values_cp)-1)

# =============================================================================
# WORKER / ANALYSIS INFRASTRUCTURE
# =============================================================================

def get_default_engine_path():
    system = platform.system()
    if system == "Windows":
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    return Path("stockfish")

def create_player_data():
    return {'test_set': [], 'elos': [], 'games': 0}

worker_engine = None
worker_config = {}

def init_worker(engine_path: Path, hash_mb: int):
    global worker_engine, worker_config
    worker_config = {'path': engine_path, 'hash': hash_mb}
    try:
        worker_engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        worker_engine.configure({"Hash": int(hash_mb), "Threads": 1})
        atexit.register(worker_engine.quit)
    except: worker_engine = None

def restart_worker_engine():
    global worker_engine, worker_config
    if worker_engine:
        try: worker_engine.quit()
        except: pass
    try:
        worker_engine = chess.engine.SimpleEngine.popen_uci(str(worker_config['path']))
        worker_engine.configure({"Hash": int(worker_config['hash']), "Threads": 1})
    except: worker_engine = None

def worker_analyze_chunk(args):
    chunk_id, offsets, pgn_path, depth, time_limit, multipv, cache_dir, verbose = args
    result_file = cache_dir / f"chunk_{chunk_id}.pkl"
    if result_file.exists(): return f"Chunk {chunk_id} skipped"
    global worker_engine
    if not worker_engine: return f"Chunk {chunk_id} No Engine"
    chunk_data = defaultdict(create_player_data)
    try:
        with open(pgn_path, 'r', encoding='utf-8') as f:
            for offset in offsets:
                f.seek(offset)
                game = chess.pgn.read_game(f)
                if not game: continue
                analyze_single_game(game, worker_engine, chunk_data, depth, time_limit, multipv, verbose)
    except Exception as e: return f"Chunk {chunk_id} Error: {e}"
    with open(result_file, 'wb') as f: pickle.dump(chunk_data, f)
    return f"Chunk {chunk_id} Done"

def analyze_single_game(game, engine, data_store, depth, time_limit, multipv, verbose=False):
    white, black = game.headers.get("White", "Unknown"), game.headers.get("Black", "Unknown")
    try: w_elo = int(game.headers.get("WhiteElo", 0))
    except: w_elo = 0
    try: b_elo = int(game.headers.get("BlackElo", 0))
    except: b_elo = 0
    for p, elo in [(white, w_elo), (black, b_elo)]:
        data_store[p]['games'] += 1
        if elo > 0: data_store[p]['elos'].append(elo)
    board, node, ply = game.board(), game, 0
    while node.variations:
        next_node = node.variation(0)
        move, ply = next_node.move, ply + 1
        if ply <= BOOK_MOVES * 2:
            board.push(move)
            node = next_node
            continue
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth, time=time_limit), multipv=multipv)
            res = []
            for pv in info:
                if 'pv' not in pv: continue
                sc = pv['score'].white()
                cp = MATE_SCORE if sc.is_mate() and sc.mate() > 0 else (-MATE_SCORE if sc.is_mate() else sc.score())
                res.append((pv['pv'][0], cp))
            if not res: raise Exception("No pv")
            if abs(res[0][1]) > CAP_EVAL:
                board.push(move)
                node = next_node
                continue
            raw_values_cp = [x[1] for x in res]
            if board.turn == chess.BLACK: raw_values_cp = [-x for x in raw_values_cp]
            actual_idx = -1
            for i, (m, _) in enumerate(res):
                if m == move:
                    actual_idx = i
                    break
            if actual_idx != -1:
                player = white if board.turn == chess.WHITE else black
                data_store[player]['test_set'].append((raw_values_cp, actual_idx))
        except:
            restart_worker_engine()
            global worker_engine
            engine = worker_engine
        board.push(move)
        node = next_node

def run_analysis_parallel(pgn_path, engine_path, depth, time_limit, multipv, cores, verbose=False):
    offsets = []
    with open(pgn_path, 'rb') as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line: break
            if line.startswith(b'[Event '): offsets.append(off)
    if not offsets: return {}
    cache_dir = pgn_path.parent / ".ipr_cache" / pgn_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    use_cores = cores if cores else max(1, int(multiprocessing.cpu_count() * 0.75))
    chunk_size = min(50, max(1, int(len(offsets) / (use_cores * 4))))
    pool_args = [(i, offsets[i:i+chunk_size], pgn_path, depth, time_limit, multipv, cache_dir, verbose) 
                 for i in range(0, len(offsets), chunk_size)]
    with multiprocessing.Pool(use_cores, initializer=init_worker, initargs=(engine_path, 128)) as pool:
        for _ in pool.imap_unordered(worker_analyze_chunk, pool_args): pass
    master_data = defaultdict(create_player_data)
    for pkl in sorted(cache_dir.glob("chunk_*.pkl")):
        with open(pkl, 'rb') as f:
            chunk_data = pickle.load(f)
            for pl, val in chunk_data.items():
                master_data[pl]['test_set'].extend(val['test_set'])
                master_data[pl]['elos'].extend(val['elos'])
                master_data[pl]['games'] += val['games']
        pkl.unlink()
    return master_data

# =============================================================================
# FITTING & PROJECTION LOGIC
# =============================================================================

def percentile_score_func(params, dataset):
    s, c = params
    if s <= 0.001 or c <= 0.001: return 1e9
    qs = np.arange(0.05, 1.0, 0.05)
    counts, valid = np.zeros(len(qs)), 0
    for vals_cp, idx in dataset:
        if idx >= len(vals_cp): continue
        probs = calculate_move_probabilities(vals_cp, s, c)
        p, pm = probs[idx], sum(probs[:idx])
        pp = pm + p
        counts += (qs >= pp).astype(float)
        if p > 1e-9:
            mask = (qs > pm) & (qs < pp)
            if np.any(mask): counts[mask] += (qs[mask] - pm) / p
        valid += 1
    return np.sum((counts/valid - qs)**2) if valid > 0 else 1e9

def fit_sc_global(dataset):
    if len(dataset) < 30: return None
    res = minimize(percentile_score_func, [0.09, 0.5], args=(dataset,),
                   bounds=[(0.01, 1.0), (0.1, 2.0)], method='L-BFGS-B')
    return res.x if res.success else None

def calculate_expected_ae(dataset, s, c):
    """PROJECTS (s, c) onto a dataset to get mathematical Expected AE."""
    total, count = 0, 0
    for vals_cp, _ in dataset:
        probs = calculate_move_probabilities(vals_cp, s, c)
        best_v = vals_cp[0]
        deltas = [compute_sae_delta(best_v, v) for v in vals_cp]
        total += sum(p*d for p, d in zip(probs, deltas))
        count += 1
    return total / count if count > 0 else 0

def weighted_regression(x, y, weights):
    w = np.array(weights)
    X = np.vstack([np.ones(len(x)), x]).T
    W = np.diag(w)
    try:
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        i, s = beta[0], beta[1]
        yp = i + s * np.array(x)
        rss = np.sum(w * (y - yp)**2)
        tss = np.sum(w * (y - np.average(y, weights=w))**2)
        r2 = 1 - (rss/tss) if tss > 0 else 0
        return i, s, r2, np.average(np.abs(y - yp), weights=w)
    except: return 0, 0, 0, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("buckets_list", type=Path)
    parser.add_argument("training_pgn", type=Path)
    parser.add_argument("--benchmark", type=Path, help="Standard Benchmark PGN (Solitaire Set)")
    parser.add_argument("--engine", type=Path, default=get_default_engine_path())
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--multipv", type=int, default=20)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    start_time = time.time()
    
    # 1. Benchmark Data (Load once)
    benchmark_path = args.benchmark if args.benchmark else args.training_pgn
    print(f"--- Benchmark Preparation: {benchmark_path.name} ---")
    bench_data_raw = run_analysis_parallel(benchmark_path, args.engine, args.depth, args.time, args.multipv, args.cores, args.verbose)
    benchmark_set = []
    for p in bench_data_raw.values(): benchmark_set.extend(p['test_set'])
    print(f"  Benchmark contains {len(benchmark_set)} positions.")

    # 2. Phase A: Calibration
    print("\n--- PHASE A: Calibration ---")
    bucket_files = []
    if args.buckets_list.exists():
        with open(args.buckets_list, 'r') as f:
            for l in f:
                l = l.strip()
                if not l: continue
                p = Path(l) if Path(l).exists() else args.buckets_list.parent / l
                if p.exists(): bucket_files.append(p)
    
    output_path_a = args.buckets_list.parent / f"{Path(sys.argv[0]).stem}_s,c-fit.csv"
    fieldnames_a = ['Bucket','AvgElo','s','c','N','AEe_Benchmark'] + ['INT','SLOPE','R2']
    with open(output_path_a, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames_a).writeheader()
    
    calib_results = []
    for bf in bucket_files:
        print(f"Analyzing Bucket: {bf.name}")
        data = run_analysis_parallel(bf, args.engine, args.depth, args.time, args.multipv, args.cores, args.verbose)
        all_m, all_e = [], []
        for p in data.values():
            all_m.extend(p['test_set'])
            all_e.extend(p['elos'])
        if not all_m: continue
        avg_e, fit = np.mean(all_e), fit_sc_global(all_m)
        if fit is not None:
            s, c = fit
            aee = calculate_expected_ae(benchmark_set, s, c)
            res = {'Bucket': bf.name, 'AvgElo': avg_e, 's': s, 'c': c, 'N': len(all_m), 'AEe_Benchmark': aee}
            calib_results.append(res)
            print(f"  Fit: Elo={avg_e:.0f}, s={s:.4f}, c={c:.4f} -> AEe(Bench)={aee:.6f}")
    
    # Regression Elo ~ AEe
    if calib_results:
        X = [r['AEe_Benchmark'] for r in calib_results]
        Y = [r['AvgElo'] for r in calib_results]
        W = [math.sqrt(r['N']) for r in calib_results]
        reg_int, reg_slope, reg_r2, _ = weighted_regression(X, Y, W)
        print(f"Linear Fit: IPR = {reg_int:.2f} + ({reg_slope:.2f}) * AEe")
        with open(output_path_a, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames_a)
            w.writeheader()
            for r in calib_results:
                r.update({'INT': reg_int, 'SLOPE': reg_slope, 'R2': reg_r2})
                w.writerow(r)
    
    # 3. Phase B: Player Analysis
    print("\n--- PHASE B: Player Analysis ---")
    train_data = run_analysis_parallel(args.training_pgn, args.engine, args.depth, args.time, args.multipv, args.cores, args.verbose)
    output_path_b = args.training_pgn.parent / f"{args.training_pgn.stem}_AE-fit_V2.csv"
    fieldnames_b = ['Player','Elo','s','c','AEe_Benchmark','IPR','Moves']
    
    player_results = []
    for player, pdata in train_data.items():
        if len(pdata['test_set']) < 20: continue
        p_elo = np.mean(pdata['elos']) if pdata['elos'] else 2000
        fit = fit_sc_global(pdata['test_set'])
        if fit is not None:
            s, c = fit
            aee = calculate_expected_ae(benchmark_set, s, c)
            ipr = reg_int + reg_slope * aee if calib_results else 0
            
            player_results.append({
                'Player': player, 'Elo': p_elo, 's': s, 'c': c, 
                'AEe_Benchmark': aee, 'IPR': ipr, 'Moves': len(pdata['test_set'])
            })
            print(f"Player {player:20} -> AEe={aee:.6f} -> IPR={ipr:.0f}")

    # Write results and Summary Row
    if player_results:
        with open(output_path_b, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_b)
            writer.writeheader()
            for r in player_results:
                writer.writerow(r)
            
            # Add summary row
            avg_ipr = np.mean([r['IPR'] for r in player_results])
            avg_elo = np.mean([r['Elo'] for r in player_results])
            avg_aee = np.mean([r['AEe_Benchmark'] for r in player_results])
            total_moves = sum([r['Moves'] for r in player_results])
            
            writer.writerow({
                'Player': 'AVERAGE',
                'Elo': avg_elo,
                's': '',
                'c': '',
                'AEe_Benchmark': avg_aee,
                'IPR': avg_ipr,
                'Moves': total_moves
            })
        print(f"Player results saved to {output_path_b.name}")
    else:
        print("No players with sufficient data found.")

    print(f"\nTotal Runtime: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
