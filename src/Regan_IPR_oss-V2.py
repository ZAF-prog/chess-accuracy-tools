#!/usr/bin/env python
"""
Regan IPR Calculator V2 (Optimized & Parallelized)
==================================================

Improvements over V1:
1. Parallel processing using 80% of available CPU cores.
2. Chunk-based processing with automatic restart capability (checkpoints).
3. Reusable 'S' set (export/import keys).
4. Improved robustness and error handling.

Usage:
    python Regan_IPR_oss-V2.py <pgn_file> [options]

    Resume a run:
    Just run the same command again. It will skip already processed chunks.

    Export S-set:
    python Regan_IPR_oss-V2.py games.pgn --export-s solitaire_data.pkl

    Calculate with fixed S-set:
    python Regan_IPR_oss-V2.py games.pgn --use-s solitaire_data.pkl
"""

import multiprocessing
import chess
import chess.engine
import chess.pgn
import numpy as np
import math
import sys
import argparse
import pickle
import time
import os
import platform
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize, brentq
from typing import List, Tuple, Dict, Any

# --- CONFIGURATION ---
BOOK_MOVES = 8          # Skip first 8 moves (16 ply)
CAP_EVAL = 300          # Garbage time filter (centipawns)
MULTI_PV = 5            # Number of principal variations
IPR_INTERCEPT = 3571
IPR_SLOPE = -15413
MATE_SCORE = 10000
CHUNK_SIZE = 50         # Games per chunk

def get_default_engine_path():
    """Determines default Stockfish path based on OS."""
    system = platform.system()
    if system == "Windows":
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    elif system in ("Linux", "Darwin"):
        return Path("binaries/stockfish-ubuntu-x86-64-avx512")
    return Path("stockfish")

# --- CORE MATH FUNCTIONS ---

def calculate_delta(v0: float, vi: float) -> float:
    """Calculate scaled value difference (logarithmic scaling)."""
    try:
        if v0 <= -1 or vi <= -1: 
            return float('inf')
        return abs(math.log(1 + v0) - math.log(1 + vi))
    except (ValueError, OverflowError):
        return float('inf')

def solve_p0(alphas: List[float]) -> float:
    """Solve for p0 such that sum(p0^alpha_i) = 1."""
    def func(p0):
        # Avoid p0=0 error, though bound is 1e-9
        return sum(p0**a for a in alphas) - 1.0
    
    try:
        return brentq(func, 1e-9, 1.0 - 1e-9)
    except (ValueError, RuntimeError):
        return 1.0 / len(alphas) if alphas else 1.0

def calculate_move_probabilities(values: List[float], s: float, c: float) -> List[float]:
    """
    Calculate move probabilities pi = p0^alpha_i.
    Where alpha_i = exp((delta_i / s)^c). [Positive exponent for power law]
    """
    if not values: return []
    v0 = values[0]
    alphas = []
    
    for vi in values:
        delta = calculate_delta(v0, vi)
        if delta == 0:
            alpha = 1.0
        elif delta == float('inf'):
            alpha = float('inf') 
        else:
            try:
                # IMPORTANT IMPROVEMENT V2:
                # V1 used alpha = exp( - (delta/s)^c ), yielding alpha < 1.
                # If alpha < 1, then p0^alpha > p0 (for p0 < 1).
                # This implied bad moves were MORE probable than best move.
                # V2 assumes power law implies probabilities drop off, so we need p0^alpha < p0 => alpha > 1.
                # Thus we use POSITIVE exponent: alpha = exp( (delta/s)^c )
                exponent = (delta / s) ** c
                if exponent > 700: # Avoid overflow
                    alpha = float('inf')
                else:
                    alpha = math.exp(exponent)
            except (OverflowError, ValueError, ZeroDivisionError):
                alpha = float('inf')
        alphas.append(alpha)
        
    p0 = solve_p0(alphas)
    
    probs = []
    for a in alphas:
        if a == float('inf'):
            probs.append(0.0)
        else:
            probs.append(p0**a)
            
    # Normalize to be safe
    total = sum(probs)
    if total > 0:
        return [p/total for p in probs]
    return probs

def create_player_data():
    """Factory for player data structure (picklable)."""
    return {'test_set': [], 'solitaire_set': [], 'games': 0, 'elos': []}

# --- WORKER FUNCTIONS ---

def worker_analyze_chunk(args):
    """
    Worker process to analyze a chunk of games.
    Args: (chunk_id, offsets, pgn_path, engine_path, depth, multipv, cache_dir)
    """
    chunk_id, offsets, pgn_path, engine_path, depth, multipv, cache_dir = args
    
    # Checkpoint result file
    result_file = cache_dir / f"chunk_{chunk_id}.pkl"
    if result_file.exists():
        return f"Chunk {chunk_id} (Skipped)"

    # Setup Engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    except Exception as e:
        return f"Chunk {chunk_id} Failed: Engine error {e}"

    chunk_data = defaultdict(create_player_data)

    try:
        with open(pgn_path, 'r', encoding='utf-8') as f:
            for offset in offsets:
                f.seek(offset)
                game = chess.pgn.read_game(f)
                if not game: continue
                
                analyze_single_game(game, engine, chunk_data, depth, multipv)
                
    except Exception as e:
        engine.quit()
        return f"Chunk {chunk_id} Failed: {e}"
        
    engine.quit()
    
    # Save checkpoint
    with open(result_file, 'wb') as f:
        pickle.dump(chunk_data, f)
        
    return f"Chunk {chunk_id} Done ({len(offsets)} games)"

def analyze_single_game(game, engine, data_store, depth, multipv):
    white = game.headers.get("White", "Unknown")
    black = game.headers.get("Black", "Unknown")
    
    # Elos
    try: data_store[white]['elos'].append(int(game.headers.get("WhiteElo", 0)))
    except: pass
    try: data_store[black]['elos'].append(int(game.headers.get("BlackElo", 0)))
    except: pass
    
    data_store[white]['games'] += 1
    data_store[black]['games'] += 1
    
    board = game.board()
    node = game
    ply = 0
    
    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        ply += 1
        
        # Filters
        if ply <= BOOK_MOVES * 2:
            board.push(move)
            node = next_node
            continue
            
        # Analysis
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            res = []
            for pv in info:
                if 'pv' not in pv: continue
                sc = pv['score'].white()
                cp = MATE_SCORE if sc.is_mate() else sc.score()
                res.append((pv['pv'][0], cp))
                
            if not res: raise Exception("No result")
            
            best_cp = res[0][1]
            if abs(best_cp) > CAP_EVAL: # Garbage time
                board.push(move)
                node = next_node
                continue
                
            # Formatting
            raw_values = [x[1]/100.0 for x in res]
            if board.turn == chess.BLACK:
                raw_values = [-x for x in raw_values]
            
            # Identify Played Move
            actual_idx = -1
            for i, (m, _) in enumerate(res):
                if m == move:
                    actual_idx = i
                    break
            
            if actual_idx != -1:
                # Add to data
                player = white if board.turn == chess.WHITE else black
                
                # Solitaire Deltas
                v0 = raw_values[0]
                deltas = [calculate_delta(v0, v) for v in raw_values[1:]]
                
                data_store[player]['test_set'].append( (raw_values, actual_idx) )
                data_store[player]['solitaire_set'].append( (raw_values, deltas) )
                
        except Exception:
            pass # Skip position on error
            
        board.push(move)
        node = next_node

# --- MAIN DRIVER ---

def index_pgn_games(path):
    offsets = []
    with open(path, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line: break
            if line.startswith(b'[Event '):
                offsets.append(offset)
    return offsets

def main():
    parser = argparse.ArgumentParser(description="Multi-Core IPR Calculator V2")
    parser.add_argument("pgn_file", type=Path, help="Input PGN")
    parser.add_argument("--engine", type=Path, default=get_default_engine_path(), help="Stockfish path")
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--cores", type=int, help="Force CPU cores (default: 80% of available)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--export-s", type=Path, help="Save S-set data to file")
    parser.add_argument("--use-s", type=Path, help="Load S-set data from file")
    
    args = parser.parse_args()
    
    if not args.pgn_file.exists():
        print("PGN not found.")
        sys.exit(1)
        
    start_time = time.time()
    
    # 1. Indexing
    print("Indexing PGN...")
    offsets = index_pgn_games(args.pgn_file)
    print(f"Found {len(offsets)} games.")
    
    # 2. Setup Checkpoint Dir
    cache_dir = args.pgn_file.parent / ".ipr_cache" / args.pgn_file.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Validation
    # Simple check if engine runs
    try:
        eng = chess.engine.SimpleEngine.popen_uci(str(args.engine))
        eng.quit()
    except Exception as e:
        print(f"Error starting engine at {args.engine}: {e}")
        sys.exit(1)

    # 4. Multiprocessing Pool
    total_cores = multiprocessing.cpu_count()
    use_cores = args.cores if args.cores else max(1, int(total_cores * 0.8))
    
    print(f"Starting analysis with {use_cores} cores ({len(offsets)} games)...")
    
    chunks = [offsets[i:i + args.chunk_size] for i in range(0, len(offsets), args.chunk_size)]
    pool_args = []
    for i, chunk in enumerate(chunks):
        pool_args.append((i, chunk, args.pgn_file, args.engine, args.depth, MULTI_PV, cache_dir))
        
    # Run Pool
    with multiprocessing.Pool(processes=use_cores) as pool:
        for res in pool.imap_unordered(worker_analyze_chunk, pool_args):
            print(res)
            
    print(f"Analysis complete. Time: {time.time() - start_time:.1f}s")
    
    # 5. Aggregate Results
    print("Aggregating results...")
    master_data = defaultdict(create_player_data)
    
    # S-set accumulation (global) if exporting
    global_s_set = [] 
    
    for pkl in cache_dir.glob("chunk_*.pkl"):
        try:
            with open(pkl, 'rb') as f:
                chunk_data = pickle.load(f)
                for player, pdata in chunk_data.items():
                    master_data[player]['test_set'].extend(pdata['test_set'])
                    master_data[player]['solitaire_set'].extend(pdata['solitaire_set'])
                    master_data[player]['games'] += pdata['games']
                    master_data[player]['elos'].extend(pdata['elos'])
                    
                    if args.export_s:
                        global_s_set.extend(pdata['solitaire_set'])
        except Exception as e:
            print(f"Error reading checkpoint {pkl}: {e}")

    # Export S-Set if requested
    if args.export_s:
        with open(args.export_s, 'wb') as f:
            pickle.dump(global_s_set, f)
        print(f"Exported {len(global_s_set)} positions to {args.export_s}")
        
    # Import S-Set if requested
    fixed_s_set = None
    if args.use_s:
        if args.use_s.exists():
            with open(args.use_s, 'rb') as f:
                fixed_s_set = pickle.load(f)
            print(f"Loaded fixed S-set with {len(fixed_s_set)} positions.")
        else:
            print("Warning: specified S-set file not found.")

    # 6. Calculate IPRs
    print("Calculating IPRs...")
    final_output = []
    
    for player, data in master_data.items():
        if not data['test_set']: continue
        
        # Fit parameters
        # Optimization: s in [0.01, 1.0], c in [0.1, 2.0]
        # Log likelihood maximization
        def neg_log_lik(params):
            s, c = params
            if s<=0 or c<=0: return 1e9
            ll = 0
            for vals, played_idx in data['test_set']:
                if played_idx >= len(vals): continue
                probs = calculate_move_probabilities(vals, s, c)
                p = probs[played_idx] if played_idx < len(probs) else 0
                ll += math.log(p + 1e-12)
            return -ll
            
        res = minimize(neg_log_lik, [0.09, 0.5], bounds=[(0.01, 1.0), (0.1, 2.0)], method='L-BFGS-B')
        s_fit, c_fit = res.x
        
        # Calculate AE_e
        # If fixed_s_set is provided, use that. Otherwise use player's own solitaire_set (from their games)
        target_s_set = fixed_s_set if fixed_s_set else data['solitaire_set']
        
        total_ae = 0
        count = 0
        for vals, deltas in target_s_set:
            probs = calculate_move_probabilities(vals, s_fit, c_fit)
            # AE = sum over moves (pi * delta_i)
            # deltas are for moves[1:]
            # probs[0] is best move (delta=0)
            if len(probs) > 1 and len(deltas) == len(probs)-1:
                step_ae = sum(p * d for p, d in zip(probs[1:], deltas))
                total_ae += step_ae
                count += 1
                
        AE_e = total_ae / count if count > 0 else 0
        IPR = IPR_INTERCEPT + IPR_SLOPE * AE_e
        
        if math.isfinite(IPR):
            avg_elo = int(sum(data['elos'])/len(data['elos'])) if data['elos'] else 0
            final_output.append({
                'Player': player,
                'Games': data['games'],
                'Elo': avg_elo,
                's': round(s_fit, 4),
                'c': round(c_fit, 4),
                'IPR': int(IPR)
            })
            print(f"Player: {player:20} IPR: {int(IPR)} (s={s_fit:.3f}, c={c_fit:.3f})")

    # 7. Write CSV
    if final_output:
        csv_path = args.pgn_file.parent / (args.pgn_file.stem + "_IPR_V2.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Player', 'Games', 'Elo', 's', 'c', 'IPR'])
            writer.writeheader()
            writer.writerows(sorted(final_output, key=lambda x: x['Player']))
        print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
