#!/usr/bin/env python3
"""
Regan IPR Parameter Estimation Stage 1
=======================================
Calculates (s, c) parameter pairs for various Elo buckets and 
establishes a linearly consistent dependence of s & c on Elo.

Methodology:
- Engine: Stockfish Multi-PV (up to 20-50 moves).
- Scaling: Logarithmic centipawn scaling (Equation 3).
- Fitting: Percentiling method to minimize deviation from uniform distribution.
- Consistency: Iterative linear regression (IRWLS) with feedback.

Usage:
    python Regan_estimate_IPR_params_V1.py buckets.txt --iterations 2 --depth 13
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
import gc
import multiprocessing
import traceback
import psutil
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
MATE_SCORE = 10000
TOTAL_HASH_BUDGET = 6144  # 6GB in MB
MIN_HASH_LIMIT = 64
ESTIMATED_PROCESS_OVERHEAD_MB = 100 # Approx RAM for Python + Engine binary

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_default_engine_path():
    system = platform.system()
    if system == "Windows":
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    return Path("stockfish")

# =============================================================================
# MATH MODULE: Regan Equation & Delta
# =============================================================================

def compute_delta_vec(v0: np.ndarray, vi: np.ndarray) -> np.ndarray:
    """Vectorized calculation of Regan deltas."""
    z0 = np.abs(v0) / 100.0
    zi = np.abs(vi) / 100.0
    
    # Logic: if same sign, diff of log(1+abs); if diff sign, sum of log(1+abs)
    # This simplifies to: log(1+abs(z0)) - (sign(z0)*sign(zi)) * log(1+abs(zi)) 
    # BUT Regan's formula in Equation 3 implies delta is the integral of 1/(1+|z|).
    # Since we defined v0 as the best move and turn-neutralized:
    # delta(v0, vi) = L(v0) - L(vi) where L(z) = sign(z)*log(1+|z|)
    
    def log_scale(z_cp):
        z = z_cp / 100.0
        return np.sign(z) * np.log1p(np.abs(z))

    delta = log_scale(v0) - log_scale(vi)
    # Ensure no negative deltas due to precision (v0 is always best)
    return np.maximum(0.0, delta)

def solve_p0_equation3_vec(deltas_list: List[np.ndarray], s: float, c: float) -> np.ndarray:
    """
    Vectorized solver for p0 for multiple positions.
    Each element in deltas_list is an array of deltas for one position.
    """
    if s <= 1e-9: return np.ones(len(deltas_list))
    
    # We want to solve f(p0) = sum(p0^alpha_i) - 1 = 0
    # where alpha_i = exp((delta_i/s)^c)
    
    alphas_list = []
    max_alpha = 100.0 # Stability cap
    
    for d in deltas_list:
        # Avoid Power/Exp overflow
        # term = (d/s)^c
        # alpha = exp(term)
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

def calculate_move_probabilities_vec(datasets: List[Tuple[np.ndarray, int]], s: float, c: float) -> List[np.ndarray]:
    """Calculates probabilities for a whole dataset at once."""
    if not datasets: return []
    
    all_deltas = []
    for values_cp, played_idx, p_elo in datasets:
        v0 = values_cp[0]
        # values_cp is already turn-neutralized
        d = compute_delta_vec(v0, values_cp)
        all_deltas.append(d)
    
    p0_vals = solve_p0_equation3_vec(all_deltas, s, c)
    
    results = []
    for i, d in enumerate(all_deltas):
        p0 = p0_vals[i]
        with np.errstate(over='ignore', invalid='ignore'):
            alpha = np.exp(np.power(d / s, c))
            # pi = p0 ^ alpha
            pi = np.power(p0, alpha)
        
        total = np.sum(pi)
        if total <= 0:
            res = np.zeros_like(pi)
            res[0] = 1.0
            results.append(res)
        else:
            results.append(pi / total)
            
    return results

# =============================================================================
# ANALYSIS MODULE: PGN/Engine Processing
# =============================================================================

# Global shared state for worker processes
_worker_engine = None
_current_worker_hash = None
_shared_hash_val = None # Global synchronization object

def init_worker(engine_path, shared_hash_val):
    global _worker_engine, _current_worker_hash, _shared_hash_val
    try:
        _worker_engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        _shared_hash_val = shared_hash_val
        _current_worker_hash = _shared_hash_val.value
        # Explicitly set 1 thread and the initial hash
        _worker_engine.configure({"Threads": 1, "Hash": _current_worker_hash})
    except Exception as e:
        logger.error(f"Failed to initialize engine in worker: {e}")

def analyze_position(board, depth, multipv):
    """Analyze a single position using the global worker engine, adjusting hash if needed."""
    global _worker_engine, _current_worker_hash, _shared_hash_val
    if _worker_engine is None or _shared_hash_val is None:
        return None
    
    # Check for dynamic hash update
    try:
        target_hash = _shared_hash_val.value
        if _current_worker_hash != target_hash:
            logger.info(f"Worker process updating Stockfish hash: {_current_worker_hash}MB -> {target_hash}MB")
            _worker_engine.configure({"Hash": target_hash})
            _current_worker_hash = target_hash
    except Exception as e:
        logger.warning(f"Failed to update engine hash: {e}")

    try:
        info = _worker_engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
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
    

    try:
        os.makedirs(cache_dir, exist_ok=True)
        start_offset = offsets[0] if offsets else 0
        cache_file = Path(cache_dir) / f"chunk_start_{start_offset}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return chunk_id, pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}. Rerunning...")

        results = []
        moves_processed = 0
        
        with open(pgn_path, 'r', encoding='utf-8') as f:
            for offset in offsets:
                f.seek(offset)
                game = chess.pgn.read_game(f)
                if not game: continue
                
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
                    if ply <= book_moves * 2:
                        board.push(move)
                        node = next_node
                        continue
                    fen_pos = board.fen().split(' ')[0]
                    if fen_pos in history:
                        board.push(move)
                        node = next_node
                        continue
                    history.add(fen_pos)
                    analysis = analyze_position(board, depth, multipv)
                    if not analysis:
                        board.push(move)
                        node = next_node
                        continue
                    best_move_uci, best_cp = analysis[0]
                    turn_multiplier = 1 if board.turn == chess.WHITE else -1
                    v0_cp = best_cp * turn_multiplier
                    if abs(v0_cp) > cap_eval:
                        board.push(move)
                        node = next_node
                        continue
                    played_move_uci = move.uci()
                    played_idx = -1
                    values_cp = []
                    for i, (m_uci, cp) in enumerate(analysis):
                        v_i = cp * turn_multiplier
                        values_cp.append(v_i)
                        if m_uci == played_move_uci:
                            played_idx = i
                    if played_idx != -1:
                        p_elo = w_elo if board.turn == chess.WHITE else b_elo
                        results.append((np.array(values_cp), played_idx, p_elo))
                        moves_processed += 1
                        if verbose > 0 and moves_processed % verbose == 0:
                            logger.info(f"Chunk {chunk_id}: Processed {moves_processed} moves...")
                    board.push(move)
                    node = next_node
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_file}: {e}")
        return chunk_id, results

    except Exception:
        with open(f"error_chunk_{chunk_id}.log", "w") as ef:
            ef.write(traceback.format_exc())
        return chunk_id, []

# =============================================================================
# FITTING MODULE: Percentiling & IRWLS
# =============================================================================

def percentile_score_func(params, datasets):
    """
    Computes S_{s,c} = sum over q of (R_{q,s,c} - q)^2.
    Vectorized over positions.
    """
    s, c = params
    if s <= 0.001 or c <= 0.01: return 1e9
    
    qs = np.linspace(0.05, 0.95, 19)
    counts = np.zeros(len(qs))
    
    # Process all positions in the dataset at once
    # datasets is a list of (values_cp_array, played_idx, elo)
    all_probs = calculate_move_probabilities_vec(datasets, s, c)
    if not all_probs: return 1e9
    
    total_valid = 0
    for i, probs in enumerate(all_probs):
        played_idx = datasets[i][1]
        
        # p_minus: sum of probs of moves better than the played one
        p_minus = np.sum(probs[:played_idx])
        p_move = probs[played_idx]
        p_plus = p_minus + p_move
        
        # Determine R_q contribution
        # Logic: count is 1 if q >= p_plus, 0 if q <= p_minus, else fractional
        # We can vectorize this inner loop over qs
        counts += np.where(qs >= p_plus, 1.0, 
                  np.where(qs > p_minus, (qs - p_minus) / p_move, 0.0))
        
        total_valid += 1
        
    if total_valid == 0: return 1e9
    R_q = counts / total_valid
    return np.sum((R_q - qs)**2)

def wrap_fit_parallel(args):
    """Wrapper for fitting in a pool."""
    datasets, initial_guess, bucket_name, elo = args
    logger.info(f"Fitting bucket: {bucket_name} at AvgElo {elo}")
    res = fit_bucket_params(datasets, initial_guess)
    return res

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
    parser.add_argument("--cores", type=int, default=max(1, int(multiprocessing.cpu_count() * 0.8)), help="Parallel cores (default: 80%% of total)")
    parser.add_argument("--memory-limit", type=int, default=75, help="Max system memory usage percent (default: 75%%)")
    parser.add_argument("--hash", type=int, help=f"Initial Stockfish hash size in MB (default: capped 6GB pool / cores, min {MIN_HASH_LIMIT})")
    parser.add_argument("--output", type=str, help="Output CSV path (default: {input_base}_IPR-fit.csv)")
    parser.add_argument("--verbose", type=int, default=10, help="Print heartbeat messages after every N moves (0 to disable)")
    
    args = parser.parse_args()
    
    # Memory Safety Guard: Calculate available budget for 75% target
    mem = psutil.virtual_memory()
    total_ram_mb = mem.total // (1024 * 1024)
    used_ram_mb = mem.used // (1024 * 1024)
    # Target 75% of total RAM as maximum "safe" ceiling
    safe_ceiling_mb = int(total_ram_mb * 0.75)
    # Account for process overhead (Python objects + Engine binaries)
    total_overhead_mb = args.cores * ESTIMATED_PROCESS_OVERHEAD_MB
    # Budget remaining for Hash before hitting the ceiling
    safety_budget_mb = max(0, safe_ceiling_mb - used_ram_mb - total_overhead_mb)
    
    effective_budget = TOTAL_HASH_BUDGET
    if effective_budget > safety_budget_mb:
        effective_budget = safety_budget_mb
        logger.warning(f"Memory Safety Guard: Capping total hash budget to {effective_budget}MB to maintain <75% system usage (Safety Budget: {safety_budget_mb}MB, Overhead: {total_overhead_mb}MB).")

    # Calculate initial hash if not explicitly provided
    if args.hash is None:
        args.hash = max(MIN_HASH_LIMIT, effective_budget // args.cores)
    
    logger.info(f"Memory Management: Using {args.cores} workers, {args.hash}MB hash each (Total hash pool: {args.cores * args.hash}MB, Floor: {MIN_HASH_LIMIT}MB)")
    
    # Set default output name based on input list if not specified
    if not args.output:
        args.output = args.buckets_list.parent / f"{args.buckets_list.stem}_IPR-fit--multipv={args.multipv}--depth={args.depth}--iterations={args.iterations}.csv"
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

    # 2. Extract positions and analyze (Global Chunk Pool)
    logger.info(f"Starting analysis of {len(bucket_files)} buckets...")
    
    all_pool_args = []
    bucket_chunk_map = {} # path -> list of chunk indices in all_pool_args
    bucket_stats = []
    
    # Shared value for dynamic hash control
    shared_hash_val = multiprocessing.Value('i', args.hash)
    
    for bf in bucket_files:
        # Initialize bucket stats entry
        bucket_stats.append({
            'name': bf.name,
            'path': bf,
            'start_time': time.time()
        })
        
        # Define cache directory for this bucket and configuration
        cache_root = bf.parent / ".ipr_cache"
        cache_subdir = f"{bf.stem}_d{args.depth}_mpv{args.multipv}_b{DEFAULT_BOOK_MOVES}_c{DEFAULT_CAP_EVAL}"
        bucket_cache_dir = cache_root / cache_subdir
        
        # Get offsets for parallelization (Optimized buffered scanning)
        offsets = []
        logger.info(f"Indexing bucket: {bf.name}...")
        with open(bf, 'rb') as f:
            chunk_size = 1024 * 1024  # 1MB buffer
            overlap = 1024  # Ensure we don't miss tags at buffer boundaries
            pos = 0
            
            # Check if very first line is an Event tag
            first_line = f.readline()
            if first_line.startswith(b'[Event '):
                offsets.append(0)
            f.seek(0)

            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                
                # Search for [Event in the buffer
                idx = -1
                while True:
                    # Look for [Event at the start of a line. 
                    # We check for \n[Event or if it's the very beginning of the file.
                    idx = data.find(b'[Event ', idx + 1)
                    if idx == -1:
                        break
                    
                    # Ensure it's preceded by a newline or start of file
                    if idx == 0 or data[idx-1] == ord('\n') or data[idx-1] == ord('\r'):
                        offsets.append(pos + idx)
                
                pos += len(data)
                # Backup slightly to handle tags split across chunks
                if len(data) == chunk_size:
                    pos -= overlap
                    f.seek(pos)
                else:
                    break
        
        # Remove duplicates in case backup caused overlaps
        offsets = sorted(list(set(offsets)))
        
        logger.info(f"  Found {len(offsets)} games.")
        
        if not offsets:
            logger.warning(f"No games found in {bf.name}")
            continue
            
        GAMES_PER_CHUNK = 20
        bucket_chunk_indices = []
        for i in range(0, len(offsets), GAMES_PER_CHUNK):
            chunk = offsets[i : i + GAMES_PER_CHUNK]
            
            # Use deterministic chunk identifier based on bucket name and start index
            chunk_global_id = len(all_pool_args)
            # shared_hash_val is passed via initializer/global to avoid pickling issues on Windows
            all_pool_args.append((chunk_global_id, chunk, bf, args.depth, args.multipv, DEFAULT_BOOK_MOVES, DEFAULT_CAP_EVAL, args.verbose, bucket_cache_dir))
            bucket_chunk_indices.append(chunk_global_id)
        
        bucket_chunk_map[bf] = bucket_chunk_indices

    # Run all chunks in one large pool to saturate CPU
    all_results = [[] for _ in range(len(all_pool_args))]
    logger.info(f"Processing {len(all_pool_args)} total chunks across all buckets using {args.cores} cores (Target memory: <{args.memory_limit}%%).")
    
    def check_memory():
        """Aggressively manage memory by reducing worker hash size if limit exceeded."""
        gc.collect() # Proactive cleanup
        current = psutil.virtual_memory().percent
        if current > args.memory_limit:
            # Try reducing hash before waiting
            with shared_hash_val.get_lock():
                if shared_hash_val.value > MIN_HASH_LIMIT:
                    old_h = shared_hash_val.value
                    new_h = max(MIN_HASH_LIMIT, old_h // 2)
                    shared_hash_val.value = new_h
                    logger.warning(f"Memory high ({current}%). Proactively reducing worker hash {old_h}MB -> {new_h}MB to continue...")
                else:
                    logger.warning(f"Memory high ({current}%) and hash is already at floor ({MIN_HASH_LIMIT}MB). Waiting for pool to free memory...")
            
            while psutil.virtual_memory().percent > args.memory_limit:
                time.sleep(5)
            logger.info(f"Memory cleared ({psutil.virtual_memory().percent}%). Resuming...")
            
    with multiprocessing.Pool(processes=args.cores, initializer=init_worker, initargs=(args.engine, shared_hash_val)) as pool:
        # imap_unordered with index tracking to place results correctly
        it = pool.imap_unordered(process_pgn_chunk, all_pool_args)
        while True:
            try:
                check_memory()
                res = next(it)
                i, chunk_res = res
                all_results[i] = chunk_res
            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error during parallel analysis: {e}")
                # Clean up if engine was started
                if _worker_engine:
                    _worker_engine.quit()
                raise

    # Assemble bucket data map
    bucket_data_map = {}
    for b_idx, (bf, indices) in enumerate(bucket_chunk_map.items()):
        bucket_results = []
        for idx in indices:
            bucket_results.extend(all_results[idx])
        bucket_data_map[bf] = bucket_results
        
        # Update bucket stats with correct Elo and move count
        moves = bucket_results
        player_elos = [m[2] for m in moves if m[2] > 0]
        if player_elos:
            min_elo = int(np.min(player_elos))
            max_elo = int(np.max(player_elos))
            avg_elo = int(np.mean(player_elos))
        else:
            file_elo = extract_elo_from_filename(bf.name)
            avg_elo = file_elo if file_elo else 0
            min_elo = avg_elo
            max_elo = avg_elo
            
        # bucket_stats was populated in the same order as bucket_files loop
        bucket_stats[b_idx].update({
            'avg_elo': avg_elo,
            'min_elo': min_elo,
            'max_elo': max_elo,
            'n': len(moves),
            'moves': moves,
            's': 0.1, # Initial guess defaults
            'c': 0.5
        })
        logger.info(f"Bucket {bf.name}: Analyzed {len(bucket_results)} eligible moves")

    # 3. Iterative Fitting and Regression
    # We already initialized bucket_stats and updated it with results.

    # 3. Iterative Fitting and Outlier Cleaning
    logger.info("--- Fitting and Outlier Cleaning Phase ---")
    
    while True:
        # Step A-C: IRWLS Iterations
        reg_s = (0, 0.1, 0)
        reg_c = (0, 0.5, 0)
        
        for iteration in range(args.iterations):
            logger.info(f"  --- IRWLS Iteration {iteration+1}/{args.iterations} ---")
            
            # Step A: Parallel Fit
            fit_pool_args = []
            for b in bucket_stats:
                initial = (b['s'], b['c'])
                fit_pool_args.append((b['moves'], initial, b['name'], b['avg_elo']))
            
            with multiprocessing.Pool(processes=min(args.cores, len(fit_pool_args) if fit_pool_args else 1)) as pool:
                results = pool.map(wrap_fit_parallel, fit_pool_args)
            
            current_fits = []
            for i, res in enumerate(results):
                b = bucket_stats[i]
                if res is not None:
                    b['s'], b['c'] = res
                    current_fits.append(b)
                else:
                    logger.warning(f"    Fitting failed for {b['name']}")
            
            if not current_fits:
                break
                
            # Step B: Linear Regression for 'c' only
            elos = [b['avg_elo'] for b in current_fits]
            c_vals = [b['c'] for b in current_fits]
            weights = [math.sqrt(b['n']) for b in current_fits]
            
            mc, bc, r2c = weighted_linear_regression(elos, c_vals, weights)
            reg_c = (mc, bc, r2c)
            
            # Logging 's' regression
            s_vals = [b['s'] for b in current_fits]
            ms, bs, r2s = weighted_linear_regression(elos, s_vals, weights)
            reg_s = (ms, bs, r2s)
            
            logger.info(f"    Regression: c = {mc:.6f}*Elo + {bc:.4f} (R2={r2c:.4f}), s_slope={ms:.6f}")
            
            # Step C: Update 'c' guesses
            if r2c > 0.8:
                for b in bucket_stats:
                    b['c'] = mc * b['avg_elo'] + bc
            
            gc.collect() # Cleanup after each IRWLS iteration

        # Step D: Outlier Inspection
        current_fits = [b for b in bucket_stats if 's' in b and 'c' in b]
        if not current_fits: break
        
        elos = np.array([b['avg_elo'] for b in current_fits])
        c_vals = np.array([b['c'] for b in current_fits])
        weights = np.array([math.sqrt(b['n']) for b in current_fits])
        
        mc, bc, r2c = weighted_linear_regression(elos, c_vals, weights)
        preds = mc * elos + bc
        residuals = c_vals - preds
        
        min_elo, max_elo = np.min(elos), np.max(elos)
        range_elo = max_elo - min_elo
        low_bound = min_elo + 0.3 * range_elo
        high_bound = max_elo - 0.3 * range_elo
        
        extreme_indices = np.where((elos <= low_bound) | (elos >= high_bound))[0]
        if len(extreme_indices) < 2:
            break
            
        sigma = np.std(residuals[extreme_indices])
        if sigma == 0: break
        
        outliers = []
        for idx in extreme_indices:
            if abs(residuals[idx]) > 4 * sigma:
                outliers.append(idx)
        
        if not outliers:
            logger.info("  No outliers detected at extremes. Cleaning complete.")
            break
            
        logger.info(f"  Detected {len(outliers)} outliers at extremes. Dropping and redoing IRWLS...")
        outlier_names = [current_fits[i]['name'] for i in outliers]
        bucket_stats = [b for b in bucket_stats if b['name'] not in outlier_names]

    # 5. Save Final Results
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['Bucket', 'MinElo', 'MaxElo', 'AvgElo', 's', 'c', 'n_moves', 's_slope', 's_int', 's_r2', 'c_slope', 'c_int', 'c_r2', 'runtime_sec']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for b in bucket_stats:
            b['end_time'] = time.time()
            row = {
                'Bucket': b['name'],
                'MinElo': b['min_elo'],
                'MaxElo': b['max_elo'],
                'AvgElo': b['avg_elo'],
                's': b['s'] if 's' in b else 0,
                'c': b['c'] if 'c' in b else 0,
                'n_moves': b['n'],
                's_slope': reg_s[0],
                's_int': reg_s[1],
                's_r2': reg_s[2],
                'c_slope': reg_c[0],
                'c_int': reg_c[1],
                'c_r2': reg_c[2],
                'runtime_sec': b['end_time'] - b['start_time']
            }
            writer.writerow(row)

    logger.info(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    # Ensure correct start method for Windows
    if platform.system() == "Windows":
        multiprocessing.set_start_method('spawn', force=True)
    main()
