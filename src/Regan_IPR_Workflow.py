#!/usr/bin/env python
"""
Regan IPR Workflow v1.0
=======================
A comprehensive implementation of the Regan-Haworth (2012) methodology for 
Chess Intrinsic Performance Ratings.

Features:
1. "Strict" Regan Model (Equation 3) with Double Exponential probability distribution.
2. Correct Scaled Average Error (SAE) integration.
3. Two-phase workflow:
   - Phase A: Global (s, c) fitting on Elo buckets -> Weighted Regression.
   - Phase B: Player-specific IPR estimation using regression-derived consistency (c).

Usage:
    python Regan_IPR_Workflow.py <buckets_list_file> <training_pgn>
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
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize, brentq
from typing import List, Tuple
import logging
import warnings
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
CHUNK_SIZE = 50

# --- GLOBAL WORKER VARIABLE ---
worker_engine = None

# =============================================================================
# MATH MODULE: Regan Equation 3 & SAE
# =============================================================================

def centipawns_to_winprob(cp: float) -> float:
    """Logistic conversion from centipawns to win probability."""
    if cp > 20000: return 1.0
    if cp < -20000: return 0.0
    try:
        return 1.0 / (1.0 + math.pow(10, -cp / 400.0))
    except (OverflowError, ValueError):
        return 0.0 if cp < 0 else 1.0

def compute_sae_delta(v0_cp: float, vi_cp: float) -> float:
    """
    Computes Scaled Difference (Delta) between best move (v0) and move i (vi).
    Uses the integral of 1 / (1 + |z|) dz.
    """
    # Convert to pawns
    z0 = v0_cp / 100.0
    zi = vi_cp / 100.0

    # Ensure v0 >= vi (best move vs other move)
    # If engine returns higher score for other move (rare noise), delta is 0
    if zi > z0:
        return 0.0

    # If same sign (both >= 0 or both <= 0), simple log difference
    if z0 * zi >= 0:
        return abs(math.log(1 + abs(z0)) - math.log(1 + abs(zi)))
    else:
        # Crosses zero: Integral split into [zi, 0] and [0, z0]
        # Int(1/(1+|z|)) from 0 to A is log(1+A) for A>0
        # Int(1/(1+|z|)) from B to 0 is log(1+|B|) for B<0
        return math.log(1 + abs(z0)) + math.log(1 + abs(zi))

def solve_p0_equation3(deltas: List[float], s: float, c: float) -> float:
    """
    Solves for p0 in Sum( pi ) = 1, where pi = p0 ^ exp( (delta/s)^c ).
    
    Let E_i = exp( (delta_i / s)^c ). Note E_0 = 1 (since delta_0 = 0).
     Equation: Sum( p0 ^ E_i ) - 1 = 0.
    
    This function finds root for p0 in [0, 1].
    """
    # Precompute exponents E_i
    # Optimization: If (delta/s)^c is large, E_i is huge, p0^E_i -> 0 quickly.
    
    if s <= 1e-9:
        return 1.0 # Degenerate case, best move prob -> 1.0

    exponents = []
    for d in deltas:
        if d == 0:
            exponents.append(1.0)
        else:
            try:
                term = (d / s)
                if term > 20: # Safety cap, exp(20^c) is huge
                    # if c >= 1, this is massive.
                    E = math.exp(math.pow(term, c))
                else:
                    E = math.exp(math.pow(term, c))
            except OverflowError:
                E = float('inf')
            exponents.append(E)

    def f(p0):
        # f(p0) = Sum( p0^E_i ) - 1
        # if p0=0, sum=0 (assuming exponents > 0) -> -1
        # if p0=1, sum=N -> N-1 (>0 if N>1)
        sum_p = 0.0
        for E in exponents:
            if E == float('inf'):
                continue # Term is 0
            # If p0 is tiny and E is huge, pow is 0
            try:
                # Optimized pow or standard
                sum_p += math.pow(p0, E)
            except:
                pass
        return sum_p - 1.0

    try:
        # brentq usually robust for monotonic functions
        root = brentq(f, 0.0 + 1e-9, 1.0 - 1e-9)
        return root
    except Exception:
        # Fallback if solver fails (e.g., flat regions)
        return 0.5

def calculate_move_probabilities(values_cp: List[float], s: float, c: float) -> List[float]:
    """
    Calculates move probabilities using Regan's Equation 3.
    values_cp: List of evaluations in centipawns (best first).
    """
    if not values_cp: return []
    
    best_cp = values_cp[0]
    deltas = [compute_sae_delta(best_cp, v) for v in values_cp]
    
    # Solve for p0
    p0 = solve_p0_equation3(deltas, s, c)
    
    # Calculate all pi
    probs = []
    for d in deltas:
        if s <= 1e-9:
            probs.append(1.0 if d==0 else 0.0)
            continue
            
        try:
            term = (d / s)
            # Heavy clamping to avoid overflow in exp( (d/s)^c )
            if term > 100: 
                E = float('inf')
            else:
                E = math.exp(math.pow(term, c))
            
            if E == float('inf'):
                pi = 0.0
            else:
                pi = math.pow(p0, E)
        except:
            pi = 0.0
        probs.append(pi)
        
    # Re-normalize slightly to handle float precision drift (sum should be ~1.0)
    total = sum(probs)
    if total > 0:
        return [p/total for p in probs]
    else:
        return [1.0] + [0.0]*(len(values_cp)-1)


# =============================================================================
# WORKER / ANALYSIS INFRASTRUCTURE
# =============================================================================

def get_default_engine_path():
    system = platform.system()
    if system == "Windows":
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    else:
        return Path("stockfish")

def create_player_data():
    return {
        'test_set': [],  # [(raw_values_cp, actual_idx), ...] 
        'elos': [],
        'games': 0
    }

# --- GLOBAL WORKER VARIABLE ---
worker_engine = None
worker_config = {}

def init_worker(engine_path: Path, hash_mb: int):
    global worker_engine
    global worker_config
    worker_config = {'path': engine_path, 'hash': hash_mb}
    
    print(f"Worker {os.getpid()}: Initializing Stockfish...", flush=True)
    try:
        worker_engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        worker_engine.configure({"Hash": int(hash_mb), "Threads": 1})
        atexit.register(worker_engine.quit)
    except Exception as e:
        print(f"Init failed: {e}")
        worker_engine = None

def restart_worker_engine():
    global worker_engine
    global worker_config
    
    # print(f"Worker {os.getpid()}: Restarting engine...", flush=True)
    if worker_engine:
        try:
            worker_engine.quit()
        except:
            pass
        worker_engine = None
        
    try:
        worker_engine = chess.engine.SimpleEngine.popen_uci(str(worker_config['path']))
        worker_engine.configure({"Hash": int(worker_config['hash']), "Threads": 1})
    except Exception as e:
        print(f"Worker {os.getpid()} Restart Failed: {e}", flush=True)
        worker_engine = None

def worker_analyze_chunk(args):
    chunk_id, offsets, pgn_path, depth, time_limit, multipv, cache_dir, verbose = args
    result_file = cache_dir / f"chunk_{chunk_id}.pkl"
    if result_file.exists():
        return f"Chunk {chunk_id} (Skipped/Exists)"

    global worker_engine
    if not worker_engine:
        return f"Chunk {chunk_id} Failed: No Engine"

    chunk_data = defaultdict(create_player_data)
    
    try:
        with open(pgn_path, 'r', encoding='utf-8') as f:
            for offset in offsets:
                f.seek(offset)
                game = chess.pgn.read_game(f)
                if not game: continue
                analyze_single_game(game, worker_engine, chunk_data, depth, time_limit, multipv, verbose)
    except Exception as e:
        return f"Chunk {chunk_id} Error: {e}"

    with open(result_file, 'wb') as f:
        pickle.dump(chunk_data, f)
    
    return f"Chunk {chunk_id} Done ({len(offsets)})"

def analyze_single_game(game, engine, data_store, depth, time_limit, multipv, verbose=False):
    white = game.headers.get("White", "Unknown")
    black = game.headers.get("Black", "Unknown")
    
    # Elos
    try: w_elo = int(game.headers.get("WhiteElo", 0)) 
    except: w_elo = 0
    try: b_elo = int(game.headers.get("BlackElo", 0)) 
    except: b_elo = 0
    
    for p, elo in [(white, w_elo), (black, b_elo)]:
        data_store[p]['games'] += 1
        if elo > 0: data_store[p]['elos'].append(elo)

    board = game.board()
    node = game
    ply = 0
    
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        ply += 1
        
        if verbose and ply % 10 == 0:
            print(f"    Worker {os.getpid()} | {white} vs {black} | ...analyzing move {ply}...", flush=True)

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
                if sc.is_mate():
                    cp = MATE_SCORE if sc.mate() > 0 else -MATE_SCORE
                else:
                    cp = sc.score()
                res.append((pv['pv'][0], cp))
            
            if not res: raise Exception("No pv")
            
            # Filter garbage time
            best_cp = res[0][1]
            if abs(best_cp) > CAP_EVAL:
                board.push(move)
                node = next_node
                continue
                
            # Standardize Perspective: Always from side to move
            # Engine scores are white-centric. If Black to move, flip.
            raw_values_cp = [x[1] for x in res]
            if board.turn == chess.BLACK:
                raw_values_cp = [-x for x in raw_values_cp]
            
            # Find actual move index
            actual_idx = -1
            for i, (m, _) in enumerate(res):
                if m == move:
                    actual_idx = i
                    break
            
            if actual_idx != -1:
                player = white if board.turn == chess.WHITE else black
                data_store[player]['test_set'].append( (raw_values_cp, actual_idx) )
                
        except Exception as e:
            if verbose:
                print(f"    Worker {os.getpid()} Error: {e}. Restarting engine...", flush=True)
            restart_worker_engine()
            global worker_engine
            engine = worker_engine # Refresh local reference
            pass
            
        board.push(move)
        node = next_node

def index_pgn(path):
    offsets = []
    with open(path, 'rb') as f:
        while True:
            off = f.tell()
            line = f.readline()
            if not line: break
            if line.startswith(b'[Event '):
                offsets.append(off)
    return offsets

def run_analysis_parallel(pgn_path, engine_path, depth, time_limit, multipv, cores, verbose=False):
    """Orchestrates parallel analysis and returns aggregated master_data."""
    print(f"Processing {pgn_path}...")
    offsets = index_pgn(pgn_path)
    if not offsets: return {}
    
    cache_dir = pgn_path.parent / ".ipr_cache" / pgn_path.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    use_cores = cores if cores else max(1, int(multiprocessing.cpu_count() * 0.75))
    # Use fixed 128MB hash per worker (sufficient for single-position analysis)
    hash_per_worker = 128

    # Dynamic chunk sizing to ensure utilization
    total_games = len(offsets)
    # Target approx 4 chunks per core to balance load, but cap at 50 games/chunk
    chunk_size = max(1, int(total_games / (use_cores * 4)))
    chunk_size = min(50, chunk_size)
    
    pool_args = []
    chunks = [offsets[i:i+chunk_size] for i in range(0, len(offsets), chunk_size)]
    for i, c in enumerate(chunks):
        pool_args.append( (i, c, pgn_path, depth, time_limit, multipv, cache_dir, verbose) )
        
    print(f"  Launching {use_cores} workers for {len(chunks)} chunks (size ~{chunk_size})...")
    with multiprocessing.Pool(use_cores, initializer=init_worker, initargs=(engine_path, hash_per_worker)) as pool:
        for _ in pool.imap_unordered(worker_analyze_chunk, pool_args):
            pass # Progress could be printed here
            
    # Aggregate with progressive cleanup to reduce memory and disk usage
    master_data = defaultdict(create_player_data)
    for pkl in sorted(cache_dir.glob("chunk_*.pkl")):
        try:
            with open(pkl, 'rb') as f:
                chunk_data = pickle.load(f)
                for pl, val in chunk_data.items():
                    master_data[pl]['test_set'].extend(val['test_set'])
                    master_data[pl]['elos'].extend(val['elos'])
                    master_data[pl]['games'] += val['games']
                del chunk_data  # Explicit cleanup
        except:
            pass
        finally:
            # Delete cache file immediately after reading to reduce disk usage
            try:
                pkl.unlink()
            except:
                pass
    return master_data

# =============================================================================
# FITTING LOGIC
# =============================================================================

def percentile_score_func(params, dataset):
    s, c = params
    if s <= 0.001 or c <= 0.001: return 1e9
    
    qs = np.arange(0.05, 1.0, 0.05)
    counts = np.zeros(len(qs))
    valid = 0
    
    for vals_cp, idx in dataset:
        if idx >= len(vals_cp): continue
        probs = calculate_move_probabilities(vals_cp, s, c)
        
        p = probs[idx]
        p_minus = sum(probs[:idx])
        p_plus = p_minus + p
        
        counts += (qs >= p_plus).astype(float)
        if p > 1e-9:
            mask = (qs > p_minus) & (qs < p_plus)
            if np.any(mask):
                counts[mask] += (qs[mask] - p_minus) / p
        valid += 1
        
    if valid == 0: return 1e9
    obs = counts / valid
    return np.sum((obs - qs)**2)

def fit_sc_global(dataset):
    """Fits s and c for a dataset."""
    if len(dataset) < 30: return None
    try:
        res = minimize(percentile_score_func, [0.09, 0.5], args=(dataset,),
                       bounds=[(0.01, 1.0), (0.1, 2.0)], method='L-BFGS-B')
        return res.x if res.success else None
    except:
        return None

def fit_s_fixed_c(dataset, fixed_c):
    """Fits s only, given a fixed c."""
    if len(dataset) < 10: return None
    
    def score_s(x):
        return percentile_score_func([x[0], fixed_c], dataset)
    
    try:
        res = minimize(score_s, [0.1], bounds=[(0.01, 1.0)], method='L-BFGS-B')
        return res.x[0] if res.success else None
    except:
        return None

def calculate_ae(dataset, s, c):
    """Calculates Average SAE (Equation 3 Expected vs Actual optional)."""
    # The prompt asks for "Calculate the respective AE".
    # We will return the EXPECTED SAE from the model parameters (AE_model)
    # as this is usually the benchmark.
    total_ae = 0
    count = 0
    for vals_cp, idx in dataset:
        probs = calculate_move_probabilities(vals_cp, s, c)
        best_v = vals_cp[0]
        deltas = [compute_sae_delta(best_v, v) for v in vals_cp]
        
        expected_delta = sum(p*d for p, d in zip(probs, deltas))
        total_ae += expected_delta
        count += 1
    return total_ae / count if count > 0 else 0


# =============================================================================
# MAIN PHASE CONTROL
# =============================================================================

def weighted_regression(x, y, weights):
    """Weighted Linear Regression: y = alpha + beta * x"""
    # W = weights matrix
    # Standard formula
    w = np.array(weights)
    X = np.vstack([np.ones(len(x)), x]).T
    W = np.diag(w)
    
    # (X^T W X)^-1 X^T W y
    try:
        beta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        intercept = beta[0]
        slope = beta[1]
        
        # R2
        y_pred = intercept + slope * np.array(x)
        res_ss = np.sum(w * (y - y_pred)**2)
        tot_ss = np.sum(w * (y - np.average(y, weights=w))**2)
        r2 = 1 - (res_ss/tot_ss) if tot_ss > 0 else 0
        
        # Avg Y Error
        avg_err = np.average(np.abs(y - y_pred), weights=w)
        
        return intercept, slope, r2, avg_err
    except:
        return 0, 0, 0, 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("buckets_list", type=Path)
    parser.add_argument("training_pgn", type=Path)
    parser.add_argument("--engine", type=Path, default=get_default_engine_path())
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--time", type=float, default=1.0, help="Time limit per move in seconds")
    parser.add_argument("--multipv", type=int, default=20)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--verbose", action="store_true", help="Print heartbeat every 10 moves")
    args = parser.parse_args()
    
    # Suppress asyncio error tracebacks (errors are still caught and handled)
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Start timing
    start_time = time.time()
    
    # --- PHASE A: BUCKETS ---
    print("--- PHASE A: Bucket Analysis ---")
    phase_a_start = time.time()
    bucket_files = []
    if args.buckets_list.exists():
        with open(args.buckets_list, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                p = Path(line)
                if p.exists():
                    bucket_files.append(p)
                else:
                    # Try relative to list file
                    p_rel = args.buckets_list.parent / line
                    if p_rel.exists():
                        bucket_files.append(p_rel)
                    else:
                        print(f"Warning: Bucket file '{line}' not found (checked '{p}' and '{p_rel}')")
    
    # Prepare Phase A output file with header
    prog_name = Path(sys.argv[0]).stem
    output_path_a = args.buckets_list.parent / f"{prog_name}_s,c-fit.csv"
    fieldnames_a = ['Bucket','MinElo','MaxElo','AvgElo','s','c','N'] + ['INT','SLOPE','R2','AVG_ERR']
    
    # Write header immediately
    with open(output_path_a, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_a)
        writer.writeheader()
    
    bucket_results = []
    for bf in bucket_files:
        print(f"Analyzing Bucket: {bf.name}")
        data = run_analysis_parallel(bf, args.engine, args.depth, args.time, args.multipv, args.cores, args.verbose)
        
        # Merge all data for bucket-level fit
        all_moves = []
        all_elos = []
        for p in data.values():
            all_moves.extend(p['test_set'])
            all_elos.extend(p['elos'])
        
        if not all_moves: continue
        
        avg_elo = np.mean(all_elos) if all_elos else 0
        min_elo = np.min(all_elos) if all_elos else 0
        max_elo = np.max(all_elos) if all_elos else 0
        fit = fit_sc_global(all_moves)
        
        if fit is not None:
            s, c = fit
            n_moves = len(all_moves)
            print(f"  Fit: Elo={avg_elo:.0f} ({min_elo}-{max_elo}), s={s:.4f}, c={c:.4f}, N={n_moves}")
            result = {
                'Bucket': bf.name, 'MinElo': min_elo, 'MaxElo': max_elo, 'AvgElo': avg_elo, 
                's': s, 'c': c, 'N': n_moves
            }
            bucket_results.append(result)
            
            # Write this bucket's result immediately (without regression stats yet)
            with open(output_path_a, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames_a)
                writer.writerow(result)
    
    # Regression c vs Elo
    if bucket_results:
        X = [r['AvgElo'] for r in bucket_results]
        Y = [r['c'] for r in bucket_results]
        W = [math.sqrt(r['N']) for r in bucket_results] # Weight = Sqrt(N)
        
        c_int, c_slope, c_r2, c_err = weighted_regression(X, Y, W)
        print(f"Regression c vs Elo: c = {c_int:.4f} + {c_slope:.6f}*Elo (R2={c_r2:.3f})")
        
        # Re-write entire CSV with regression stats now available
        with open(output_path_a, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_a)
            writer.writeheader()
            for i, r in enumerate(bucket_results):
                row = r.copy()
                # Per-row error: difference between actual c and regression-predicted c
                predicted_c = c_int + c_slope * r['AvgElo']
                row_err = abs(r['c'] - predicted_c)
                row.update({'INT': c_int, 'SLOPE': c_slope, 'R2': c_r2, 'AVG_ERR': row_err})
                writer.writerow(row)
    else:
        print("No valid bucket results. Utilizing defaults.")
        c_int, c_slope = 0.5, 0.0 # Fallback
        
    phase_a_time = time.time() - phase_a_start
    
    # --- PHASE B: PLAYER TRAINING SET ---
    print("\n--- PHASE B: Player Analysis ---")
    phase_b_start = time.time()
    train_data = run_analysis_parallel(args.training_pgn, args.engine, args.depth, args.time, args.multipv, args.cores, args.verbose)
    
    player_results = []
    
    # Aggregate data for regression
    reg_elo = []
    reg_ae = []
    reg_weights = []
    
    # Prepare Phase B output file with header
    fname = f"{args.training_pgn.stem}_AE-fit.csv"
    output_path_b = args.training_pgn.parent / fname
    fieldnames_b = ['Player','Elo','s','c_fixed','AE','Moves','IPR'] + ['INT','SLOPE','R2','AVG_ERR']
    
    with open(output_path_b, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_b)
        writer.writeheader()

    for player, pdata in train_data.items():
        if len(pdata['test_set']) < 20: continue
        
        p_elo = np.mean(pdata['elos']) if pdata['elos'] else 2000 # Default if unknown
        
        # Calculate consistency c from Phase A regression
        c_fixed = c_int + c_slope * p_elo
        
        # Fit s with fixed c
        s_fit = fit_s_fixed_c(pdata['test_set'], c_fixed)
        
        if s_fit:
            # Calculate AE
            ae = calculate_ae(pdata['test_set'], s_fit, c_fixed)
            
            player_results.append({
                'Player': player,
                'Elo': p_elo,
                's': s_fit,
                'c_fixed': c_fixed,
                'AE': ae,
                'Moves': len(pdata['test_set'])
            })
            
            reg_elo.append(p_elo)
            reg_ae.append(ae)
            reg_weights.append(math.sqrt(len(pdata['test_set'])))
            
            # Write this player's result immediately (without IPR and regression stats yet)
            with open(output_path_b, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames_b)
                writer.writerow({
                    'Player': player,
                    'Elo': p_elo,
                    's': s_fit,
                    'c_fixed': c_fixed,
                    'AE': ae,
                    'Moves': len(pdata['test_set'])
                })
            
    # Regression Elo vs AE
    if player_results:
        ae_int, ae_slope, ae_r2, ae_err = weighted_regression(reg_elo, reg_ae, reg_weights)
        print(f"Regression AE vs Elo: AE = {ae_int:.4f} + {ae_slope:.6f}*Elo (R2={ae_r2:.3f})")
        
        # Calculate IPR for each player
        for r in player_results:
            if abs(ae_slope) > 1e-9:
                # AE = Int + Slope * IPR  =>  IPR = (AE - Int) / Slope
                r['IPR'] = (r['AE'] - ae_int) / ae_slope
            else:
                r['IPR'] = 0.0

        # Output AE Fit CSV
        fname = f"{args.training_pgn.stem}_AE-fit.csv"
        output_path = args.training_pgn.parent / fname
        with open(output_path, 'w', newline='', encoding='utf-8') as f: # Encoding for names
            writer = csv.DictWriter(f, fieldnames=['Player','Elo','s','c_fixed','AE','Moves','IPR'] + 
                                               ['INT','SLOPE','R2','AVG_ERR'])
            writer.writeheader()
            for i, r in enumerate(player_results):
                row = r.copy()
                # Per-row error: difference between actual AE and regression-predicted AE
                predicted_ae = ae_int + ae_slope * r['Elo']
                row_err = abs(r['AE'] - predicted_ae)
                row.update({'INT': ae_int, 'SLOPE': ae_slope, 'R2': ae_r2, 'AVG_ERR': row_err})
                writer.writerow(row)
            
            # Add summary row with average IPR
            avg_ipr = np.mean([r['IPR'] for r in player_results])
            avg_elo = np.mean([r['Elo'] for r in player_results])
            avg_ae = np.mean([r['AE'] for r in player_results])
            total_moves = sum([r['Moves'] for r in player_results])
            
            # Calculate RMSE for summary
            squared_errors = [(r['AE'] - (ae_int + ae_slope * r['Elo']))**2 for r in player_results]
            rmse = np.sqrt(np.mean(squared_errors))
            
            summary_row = {
                'Player': 'AVERAGE',
                'Elo': avg_elo,
                's': '',
                'c_fixed': '',
                'AE': avg_ae,
                'Moves': total_moves,
                'IPR': avg_ipr,
                'INT': ae_int,
                'SLOPE': ae_slope,
                'R2': ae_r2,
                'AVG_ERR': rmse
            }
            writer.writerow(summary_row)
        print(f"Player results saved to {fname}")
    else:
        print("No players with sufficient data found.")
    
    phase_b_time = time.time() - phase_b_start
    total_time = time.time() - start_time
    
    # Print timing summary
    print("\n" + "="*50)
    print("EXECUTION TIME SUMMARY")
    print("="*50)
    print(f"Phase A (Bucket Analysis):  {phase_a_time:.2f}s")
    print(f"Phase B (Player Analysis):  {phase_b_time:.2f}s")
    print(f"Total Runtime:              {total_time:.2f}s")
    print("="*50)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
