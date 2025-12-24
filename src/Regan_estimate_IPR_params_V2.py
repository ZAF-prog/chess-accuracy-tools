#!/usr/bin/env python3
"""
Regan IPR Parameter Estimation Stage 1 (Optimized V2)
=====================================================
Calculates (s, c) parameter pairs for various Elo buckets and 
establishes a linearly consistent dependence of s & c on Elo.

Methodology:
- Engine: Stockfish Multi-PV (up to 20-50 moves).
- Scaling: Logarithmic centipawn scaling (Equation 3).
- Fitting: Percentiling method to minimize deviation from uniform distribution.
- Consistency: Iterative linear regression (IRWLS) with feedback.
- Optimization: Persistent engine pool and efficient parallelization.

Usage:
    python Regan_estimate_IPR_params_V2.py buckets.txt --iterations 2 --depth 13
"""

import os
import sys
import re
import math
import time
import argparse
import platform
import logging
import csv
import pickle
import multiprocessing
import numpy as np
import chess
import chess.engine
import chess.pgn
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize, brentq
from sklearn.linear_model import LinearRegression

# --- CONFIGURATION ---
DEFAULT_BOOK_MOVES = 8
DEFAULT_CAP_EVAL = 300
DEFAULT_MULTI_PV = 20
MATE_SCORE = 10000

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ENGINE MANAGEMENT ---
worker_engine = None

def init_worker(engine_path, hash_mb=64):
    """Initialize worker process with a persistent engine instance."""
    global worker_engine
    try:
        import atexit
        worker_engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        worker_engine.configure({"Hash": hash_mb, "Threads": 1})
        atexit.register(lambda: worker_engine.quit() if worker_engine else None)
    except Exception as e:
        logger.error(f"Failed to initialize worker engine: {e}")

def get_default_engine_path():
    system = platform.system()
    if system == "Windows":
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    return Path("stockfish")

# =============================================================================
# MATH MODULE: Regan Equation & Delta
# =============================================================================

def compute_delta(v0_cp: float, vi_cp: float) -> float:
    """
    Computes the scaled difference delta using logarithmic scaling.
    delta = integral from vi to v0 of (1/(1 + abs(z))) dz
    """
    z0 = v0_cp / 100.0
    zi = vi_cp / 100.0
    if zi > z0: return 0.0
    
    # Same sign case
    if z0 * zi >= 0:
        return abs(math.log(1 + abs(z0)) - math.log(1 + abs(zi)))
    else:
        # Opposite signs
        return math.log(1 + abs(z0)) + math.log(1 + abs(zi))

def solve_p0_equation3(deltas: List[float], s: float, c: float) -> float:
    """
    Solves sum(p0^alpha_i) = 1 where alpha_i = exp((delta_i/s)^c).
    This corresponds to Equation 3 in Regan's work (corrected for sign).
    """
    if s <= 1e-9: return 1.0
    
    alphas = []
    for d in deltas:
        if d == 0:
            alphas.append(1.0)
        else:
            try:
                term = d / s
                val = math.exp(math.pow(term, c))
                alphas.append(val if val < 200 else 200.0) # Cap for stability
            except OverflowError:
                alphas.append(200.0)

    def f(p0):
        if p0 <= 0: return -1.0
        if p0 >= 1: return sum(1 for a in alphas if a == 1.0) - 1.0
        try:
            return sum(math.pow(p0, a) for a in alphas) - 1.0
        except:
            return 1e9

    try:
        # Use brentq to find root in (0, 1)
        return brentq(f, 1e-12, 1 - 1e-12)
    except:
        # Fallback if no root in (0, 1), though sum(p0^1) usually crosses 1
        return 1.0 / len(deltas)

def calculate_move_probabilities(values_cp: List[float], s: float, c: float) -> List[float]:
    if not values_cp: return []
    v0 = values_cp[0]
    deltas = [compute_delta(v0, v) for v in values_cp]
    p0 = solve_p0_equation3(deltas, s, c)
    
    probs = []
    for d in deltas:
        try:
            alpha = math.exp(math.pow(d / s, c)) if d > 0 else 1.0
            pi = math.pow(p0, alpha) if alpha < 200 else 0.0
            probs.append(pi)
        except:
            probs.append(0.0)
    
    total = sum(probs)
    if total <= 0:
        return [1.0] + [0.0] * (len(values_cp) - 1)
    return [p / total for p in probs]

# =============================================================================
# ANALYSIS MODULE: PGN/Engine Processing
# =============================================================================

def analyze_position(engine, board, depth, multipv):
    """Analyze a single position and return UCI moves and CP scores."""
    try:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        results = []
        for pv in info:
            if 'pv' not in pv: continue
            score = pv['score'].white()
            if score.is_mate():
                cp = MATE_SCORE - abs(score.mate()) if score.mate() > 0 else -MATE_SCORE + abs(score.mate())
            else:
                cp = score.score()
            results.append((pv['pv'][0].uci(), cp))
        return results
    except Exception as e:
        logger.error(f"Engine analysis failed: {e}")
        return None

def process_pgn_chunk(args):
    """Worker function for parallel PGN processing."""
    chunk_id, offsets, pgn_path, depth, multipv, book_moves, cap_eval, verbose, cache_dir = args
    global worker_engine
    
    # Check cache first
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = Path(cache_dir) / f"chunk_{chunk_id}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}. Rerunning...")

    if not worker_engine:
        logger.error(f"Worker {chunk_id} has no engine instance.")
        return []

    results = []
    moves_processed = 0
    engine = worker_engine
    
    try:
        with open(pgn_path, 'r', encoding='utf-8') as f:
                for offset in offsets:
                    f.seek(offset)
                    game = chess.pgn.read_game(f)
                    if not game: continue
                    
                    # Track player Elos
                    w_elo = int(game.headers.get("WhiteElo", 0))
                    b_elo = int(game.headers.get("BlackElo", 0))
                    
                    board = game.board()
                    node = game
                    ply = 0
                    history = set()
                    
                    while node.variations:
                        next_node = node.variation(0)
                        move = next_node.move
                        ply += 1
                        
                        # Filter criteria
                        if ply <= book_moves * 2:
                            board.push(move)
                            node = next_node
                            continue
                            
                        # Repetition check (simple FEN prefix)
                        fen_pos = board.fen().split(' ')[0]
                        if fen_pos in history:
                            board.push(move)
                            node = next_node
                            continue
                        history.add(fen_pos)
                        
                        # Analyze
                        analysis = analyze_position(engine, board, depth, multipv)
                        if not analysis:
                            board.push(move)
                            node = next_node
                            continue
                            
                        # Best move score from current player's perspective
                        best_move_uci, best_cp = analysis[0]
                        turn_multiplier = 1 if board.turn == chess.WHITE else -1
                        v0_cp = best_cp * turn_multiplier
                        
                        # Skip extreme positions
                        if abs(v0_cp) > cap_eval:
                            board.push(move)
                            node = next_node
                            continue
                        
                        # Find played move CP
                        played_move_uci = move.uci()
                        played_idx = -1
                        values_cp = []
                        
                        for i, (m_uci, cp) in enumerate(analysis):
                            v_i = cp * turn_multiplier
                            values_cp.append(v_i)
                            if m_uci == played_move_uci:
                                played_idx = i
                        
                        # Only keep if played move was in the top N analysis
                        if played_idx != -1:
                            # Store (values_cp, played_idx, current_player_elo)
                            p_elo = w_elo if board.turn == chess.WHITE else b_elo
                            results.append((values_cp, played_idx, p_elo))
                            moves_processed += 1
                            if verbose > 0 and moves_processed % verbose == 0:
                                logger.info(f"Chunk {chunk_id}: Processed {moves_processed} moves...")
                            
                        board.push(move)
                        node = next_node
    except Exception as e:
        logger.error(f"Chunk {chunk_id} failed: {e}")
        return []
        
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(results, f)
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_file}: {e}")
        
    return results

# =============================================================================
# FITTING MODULE: Percentiling & IRWLS
# =============================================================================

def percentile_score_func(params, datasets):
    """
    Computes S_{s,c} = sum over q of (R_{q,s,c} - q)^2.
    """
    s, c = params
    if s <= 0.001 or c <= 0.01: return 1e9
    
    qs = np.linspace(0.05, 0.95, 19)
    counts = np.zeros(len(qs))
    total_valid = 0
    
    for values_cp, played_idx, _ in datasets:
        probs = calculate_move_probabilities(values_cp, s, c)
        if not probs: continue
        
        # p_minus: sum of probs of moves better than the played one
        p_minus = sum(probs[:played_idx])
        p_move = probs[played_idx]
        p_plus = p_minus + p_move
        
        # Determine R_q contribution
        # If q < p_minus: played move is "above" q
        # If q > p_plus: played move is "below" q
        # If p_minus < q < p_plus: fractional contribution
        for i, q in enumerate(qs):
            if q >= p_plus:
                counts[i] += 1.0
            elif q > p_minus:
                # Fractional contribution: (q - p_minus) / p_move
                counts[i] += (q - p_minus) / p_move
        
        total_valid += 1
        
    if total_valid == 0: return 1e9
    R_q = counts / total_valid
    return np.sum((R_q - qs)**2)

def fit_bucket_params(datasets, initial_guess=(0.1, 0.5)):
    """Finds (s, c) that minimizes the percentile score."""
    if len(datasets) < 50:
        logger.warning(f"Insufficient data for fitting ({len(datasets)} moves).")
        return None
    
    res = minimize(percentile_score_func, initial_guess, args=(datasets,),
                   bounds=[(0.01, 0.5), (0.1, 5.0)], method='L-BFGS-B')
    
    if res.success:
        return res.x
    return None

def weighted_linear_regression(x, y, w):
    """Performs weighted linear regression: y = m*x + b"""
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    w = np.array(w)
    
    model = LinearRegression()
    model.fit(x, y, sample_weight=w)
    
    r2 = model.score(x, y, sample_weight=w)
    m = model.coef_[0]
    b = model.intercept_
    
    return m, b, r2

# =============================================================================
# MAIN CLI APPARATUS
# =============================================================================

def extract_elo_from_filename(filename: str) -> Optional[int]:
    """Tries to find ELO~2400 or similar in the filename."""
    match = re.search(r'ELO~([0-9]+)', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="IPR Parameter Estimation Stage 1")
    parser.add_argument("buckets_list", type=Path, help="Text file with PGN bucket paths")
    parser.add_argument("--iterations", type=int, default=2, help="Number of fitting/regression cycles")
    parser.add_argument("--depth", type=int, default=13, help="Search depth (default 13)")
    parser.add_argument("--multipv", type=int, default=20, help="Multi-PV count (default 20)")
    parser.add_argument("--engine", type=Path, default=get_default_engine_path(), help="Path to Stockfish")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count(), help="Parallel cores")
    parser.add_argument("--output", type=str, help="Output CSV path (default: {input_base}_IPR-fit.csv)")
    parser.add_argument("--verbose", type=int, default=10, help="Print heartbeat messages after every N moves (0 to disable)")
    
    args = parser.parse_args()
    
    # Set default output name based on input list if not specified
    if not args.output:
        suffix = f"_iterations={args.iterations}_depth={args.depth}_multipv={args.multipv}"
        args.output = args.buckets_list.parent / f"{args.buckets_list.stem}_IPR-fit{suffix}.csv"
    else:
        args.output = Path(args.output)
    
    if not args.buckets_list.exists():
        logger.error(f"Buckets list file not found: {args.buckets_list}")
        return

    # 1. Read bucket files
    bucket_files = []
    with open(args.buckets_list, 'r') as f:
        for line in f:
            p = line.strip()
            if not p: continue
            path = Path(p)
            if not path.is_absolute():
                path = args.buckets_list.parent / path
            if path.exists():
                bucket_files.append(path)
            else:
                logger.warning(f"PGN file not found: {path}")

    if not bucket_files:
        logger.error("No PGN files to process.")
        return

    # 2. Extract positions and analyze (Cache results)
    logger.info(f"Starting analysis of {len(bucket_files)} buckets...")
    bucket_data_map = {} # path -> list of move analysis
    
    # Use a persistent pool for all buckets to avoid engine churn
    with multiprocessing.Pool(processes=args.cores, initializer=init_worker, initargs=(args.engine,)) as pool:
        for bf in bucket_files:
            logger.info(f"Processing bucket: {bf.name}")
            
            # Define cache directory for this bucket and configuration
            cache_root = bf.parent / ".ipr_cache"
            cache_subdir = f"{bf.stem}_d{args.depth}_mpv{args.multipv}_b{DEFAULT_BOOK_MOVES}_c{DEFAULT_CAP_EVAL}"
            bucket_cache_dir = cache_root / cache_subdir
            
            # Get offsets for parallelization
            offsets = []
            with open(bf, 'rb') as f:
                while True:
                    off = f.tell()
                    line = f.readline()
                    if not line: break
                    if line.startswith(b'[Event '): offsets.append(off)
            
            if not offsets:
                logger.warning(f"No games found in {bf.name}")
                continue
                
            # Larger chunks for better efficiency
            num_tasks = args.cores * 4
            chunk_size = max(1, len(offsets) // num_tasks)
            pool_args = []
            for i, start_idx in enumerate(range(0, len(offsets), chunk_size)):
                chunk = offsets[start_idx : start_idx + chunk_size]
                pool_args.append((i, chunk, bf, args.depth, args.multipv, DEFAULT_BOOK_MOVES, DEFAULT_CAP_EVAL, args.verbose, bucket_cache_dir))
                
            bucket_results = []
            for chunk_res in pool.imap_unordered(process_pgn_chunk, pool_args):
                bucket_results.extend(chunk_res)
            
            bucket_data_map[bf] = bucket_results
            logger.info(f"  Analyzed {len(bucket_results)} eligible moves from {bf.name}")

    # 3. Iterative Fitting and Regression
    # Store per-bucket current guesses and results
    bucket_stats = []
    for bf, moves in bucket_data_map.items():
        if not moves: continue
        
        # Calculate actual Elo stats from moves
        player_elos = [m[2] for m in moves if m[2] > 0]
        if player_elos:
            min_elo = int(np.min(player_elos))
            max_elo = int(np.max(player_elos))
            avg_elo = int(np.mean(player_elos))
        else:
            # Fallback to filename or default
            file_elo = extract_elo_from_filename(bf.name)
            avg_elo = file_elo if file_elo else 0
            min_elo = avg_elo
            max_elo = avg_elo

        bucket_stats.append({
            'path': bf,
            'name': bf.name,
            'avg_elo': avg_elo,
            'min_elo': min_elo,
            'max_elo': max_elo,
            'moves': moves,
            's': 0.1, # Initial guess
            'c': 0.5, # Initial guess
            'n': len(moves)
        })

    reg_s = (0, 0.1, 0) # slope, intercept, r2
    reg_c = (0, 0.5, 0)
    
    for iteration in range(args.iterations):
        logger.info(f"--- Iteration {iteration+1}/{args.iterations} ---")
        
        # Step A: Independent Fit for each bucket
        current_fits = []
        for b in bucket_stats:
            logger.info(f"Fitting bucket: {b['name']} at AvgElo {b['avg_elo']}")
            initial = (b['s'], b['c'])
            res = fit_bucket_params(b['moves'], initial_guess=initial)
            if res is not None:
                b['s'], b['c'] = res
                current_fits.append(b)
                logger.info(f"  Result: s={b['s']:.4f}, c={b['c']:.4f}")
            else:
                logger.warning(f"  Fitting failed for {b['name']}")
        
        if not current_fits:
            logger.error("No successful fits in this iteration.")
            break
            
        # Step B: Linear Regression
        elos = [b['avg_elo'] for b in current_fits]
        s_vals = [b['s'] for b in current_fits]
        c_vals = [b['c'] for b in current_fits]
        weights = [math.sqrt(b['n']) for b in current_fits]
        
        ms, bs, r2s = weighted_linear_regression(elos, s_vals, weights)
        mc, bc, r2c = weighted_linear_regression(elos, c_vals, weights)
        
        reg_s = (ms, bs, r2s)
        reg_c = (mc, bc, r2c)
        
        logger.info(f"Regression Results:")
        logger.info(f"  s = {ms:.6f} * Elo + {bs:.4f} (R2={r2s:.4f})")
        logger.info(f"  c = {mc:.6f} * Elo + {bc:.4f} (R2={r2c:.4f})")
        
        # Step C: Update guesses if R2 quality is met (or just proceed to next iter)
        # Consistent with prompt: "If one or both regressions got R2>0.8, plug the fitted values as initial guesses"
        if r2s > 0.8 or r2c > 0.8:
            for b in bucket_stats:
                b['s'] = ms * b['avg_elo'] + bs
                b['c'] = mc * b['avg_elo'] + bc
        else:
            logger.info("  R2 below 0.8 threshold, keeping current fits as guesses.")

    # 4. Save Final Results
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['Bucket', 'MinElo', 'MaxElo', 'AvgElo', 's', 'c', 'n_moves', 's_slope', 's_int', 's_r2', 'c_slope', 'c_int', 'c_r2']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for b in bucket_stats:
            row = {
                'Bucket': b['name'],
                'MinElo': b['min_elo'],
                'MaxElo': b['max_elo'],
                'AvgElo': b['avg_elo'],
                's': b['s'],
                'c': b['c'],
                'n_moves': b['n'],
                's_slope': reg_s[0],
                's_int': reg_s[1],
                's_r2': reg_s[2],
                'c_slope': reg_c[0],
                'c_int': reg_c[1],
                'c_r2': reg_c[2]
            }
            writer.writerow(row)

    logger.info(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    # Ensure correct start method for Windows
    if platform.system() == "Windows":
        multiprocessing.set_start_method('spawn', force=True)
    main()
