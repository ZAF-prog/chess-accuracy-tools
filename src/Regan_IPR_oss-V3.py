#!/usr/bin/env python
"""
Regan IPR Calculator V3 (Persistent Engine Optimization)
========================================================

Improvements:
1. Persistent Engine: Stockfish is initialized once per worker process, 
   saving significant time on startup/shutdown overhead.
2. Parallel processing using ~80% of available CPU cores.
3. Chunk-based processing with automatic restart capability (checkpoints).
4. Reusable 'S' set (export/import keys).

Usage:
    python Regan_IPR_oss-V3.py <pgn_file> [options]

    Example:
    python Regan_IPR_oss-V3.py games.pgn --cores 4 --depth 16
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
from typing import List, Tuple, Dict, Any

# --- CONFIGURATION ---
BOOK_MOVES = 8          # Skip first 8 moves (16 ply)
CAP_EVAL = 300          # Garbage time filter (centipawns)
MULTI_PV = 5            # Number of principal variations
IPR_INTERCEPT = 3571
IPR_SLOPE = -15413
MATE_SCORE = 10000
CHUNK_SIZE = 50         # Games per chunk

# --- GLOBAL WORKER VARIABLE ---
# This variable holds the engine instance within each worker process.
worker_engine = None

def get_default_engine_path():
    """Determines default Stockfish path based on OS."""
    system = platform.system()
    if system == "Windows":
        # Adjust this path to your actual Stockfish location
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    elif system in ("Linux", "Darwin"):
        return Path("stockfish") # Assumes 'stockfish' is in PATH
    return Path("stockfish")

def get_system_memory_mb():
    """Returns total system memory in MB."""
    # Method 1: psutil (if installed)
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 * 1024)
    except ImportError:
        pass
        
    # Method 2: Windows ctypes
    if platform.system() == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 * 1024)
        except Exception:
            pass
            
    # Fallback: Assume 8GB
    return 8192

# --- CORE MATH FUNCTIONS ---

def calculate_delta(v0: float, vi: float) -> float:
    """
    Calculate scaled value difference (logarithmic scaling).
    Uses log(1 + exp(v)) form to handle negative evaluations gracefully.
    """
    try:
        # For standard chess values (v < 100 pawns), exp(v) is safe from overflow.
        # If values were > 700, we would need branching, but CAP_EVAL prevents that.
        term0 = math.log(1 + math.exp(v0))
        termi = math.log(1 + math.exp(vi))
        return abs(term0 - termi)
    except (ValueError, OverflowError):
        return float('inf')

def solve_p0(alphas: List[float]) -> float:
    """Solve for p0 such that sum(p0^alpha_i) = 1."""
    def func(p0):
        return sum(p0**a for a in alphas) - 1.0
    
    try:
        # Search for root between 0 and 1
        return brentq(func, 1e-9, 1.0 - 1e-9)
    except (ValueError, RuntimeError):
        # Fallback if solver fails or no solution found
        return 1.0 / len(alphas) if alphas else 1.0

def calculate_move_probabilities(values: List[float], s: float, c: float) -> List[float]:
    """
    Calculate move probabilities pi = p0^alpha_i.
    Where alpha_i = exp((delta_i / s)^c). [Positive exponent for power law behavior]
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
                # We use a positive exponent so that alpha > 1 for inferior moves.
                # Since p0 < 1, p0^alpha will be smaller than p0, reducing probability of bad moves.
                exponent = (delta / s) ** c
                if exponent > 700: # Avoid overflow (exp(700) is huge)
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
            try:
                probs.append(p0**a)
            except (OverflowError, ValueError):
                probs.append(0.0)
            
    # Normalize to ensure sum is exactly 1.0
    total = sum(probs)
    if total > 0:
        return [p/total for p in probs]
    return probs

def create_player_data():
    """Factory for player data structure (picklable)."""
    return {'test_set': [], 'solitaire_set': [], 'games': 0, 'elos': []}

# --- WORKER FUNCTIONS ---

def init_worker(engine_path: Path, hash_size: int):
    """
    Initializer function run once per worker process.
    Initializes the engine with specific Hash size.
    """
    global worker_engine
    try:
        # Initialize the engine once per process
        worker_engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        # Use calculated Hash size
        worker_engine.configure({"Hash": hash_size})
    except Exception as e:
        print(f"Worker initialization failed for engine at {engine_path}: {e}")
        worker_engine = None

def worker_analyze_chunk(args):
    """
    Worker process to analyze a chunk of games.
    Uses the persistent global 'worker_engine'.
    
    Args: (chunk_id, offsets, pgn_path, depth, multipv, eval_scale, cache_dir)
    Note: engine_path is NOT passed here; it's handled by init_worker.
    """
    chunk_id, offsets, pgn_path, depth, multipv, eval_scale, cache_dir = args
    
    # Checkpoint result file
    result_file = cache_dir / f"chunk_{chunk_id}.pkl"
    if result_file.exists():
        return f"Chunk {chunk_id} (Skipped)"

    # Access the global persistent engine
    global worker_engine
    if worker_engine is None:
        return f"Chunk {chunk_id} Failed: Engine not initialized properly."

    chunk_data = defaultdict(create_player_data)

    try:
        with open(pgn_path, 'r', encoding='utf-8') as f:
            for offset in offsets:
                f.seek(offset)
                game = chess.pgn.read_game(f)
                if not game: continue
                
                # Analyze using the persistent engine
                analyze_single_game(game, worker_engine, chunk_data, depth, multipv, eval_scale)
                
    except Exception as e:
        # Do not quit the engine here; it must survive for the next chunk!
        return f"Chunk {chunk_id} Failed: {e}"
        
    # Save checkpoint
    try:
        with open(result_file, 'wb') as f:
            pickle.dump(chunk_data, f)
    except Exception as e:
        return f"Chunk {chunk_id} Failed to save results: {e}"
        
    return f"Chunk {chunk_id} Done ({len(offsets)} games)"

def analyze_single_game(game, engine, data_store, depth, multipv, eval_scale=1.0):
    white = game.headers.get("White", "Unknown")
    black = game.headers.get("Black", "Unknown")
    
    # Store Elo if available
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
        
        # 1. Skip Book Moves
        if ply <= BOOK_MOVES * 2:
            board.push(move)
            node = next_node
            continue
            
        # 2. Engine Analysis
        try:
            # Using the persistent engine instance passed in
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            res = []
            for pv in info:
                if 'pv' not in pv: continue
                sc = pv['score'].white()
                
                # Handle Mate Scores
                if sc.is_mate():
                    cp = MATE_SCORE if sc.mate() > 0 else -MATE_SCORE
                else:
                    cp = sc.score()
                
                # Store the primary move of this variation and its score
                res.append((pv['pv'][0], cp))
                
            if not res: raise Exception("No result from engine")
            
            # 3. Garbage Time Filter
            best_cp = res[0][1]
            if abs(best_cp) > CAP_EVAL: 
                board.push(move)
                node = next_node
                continue
                
            # 4. Format Scores (Centipawns to Pawns) and Scale
            # Adjust perspective so positive is always "good for the player to move"
            # Scale down modern engines (e.g. 0.75 or 0.8) to match 2011 calibration
            raw_values = [(x[1]/100.0) * eval_scale for x in res]
            if board.turn == chess.BLACK:
                raw_values = [-x for x in raw_values]
            
            # 5. Identify Actual Move Played
            actual_idx = -1
            for i, (m, _) in enumerate(res):
                if m == move:
                    actual_idx = i
                    break
            
            if actual_idx != -1:
                player = white if board.turn == chess.WHITE else black
                
                # 6. Calculate Deltas for Solitaire Set
                v0 = raw_values[0]
                deltas = [calculate_delta(v0, v) for v in raw_values[1:]]
                
                # Store Data
                # test_set: used for fitting (requires index of move played)
                data_store[player]['test_set'].append( (raw_values, actual_idx) )
                # solitaire_set: used for projecting AE (requires deltas)
                data_store[player]['solitaire_set'].append( (raw_values, deltas) )
                
        except Exception:
            pass # Skip position on analysis error
            
        board.push(move)
        node = next_node

# --- MAIN DRIVER ---

def index_pgn_games(path):
    """Scans PGN file and returns file offsets for each game."""
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
    parser = argparse.ArgumentParser(description="Regan IPR Calculator V3 (Persistent Engine)")
    parser.add_argument("pgn_file", type=Path, help="Input PGN")
    parser.add_argument("--engine", type=Path, default=get_default_engine_path(), help="Stockfish path")
    parser.add_argument("--depth", type=int, default=14, help="Analysis depth (default: 14)")
    parser.add_argument("--cores", type=int, help="Force CPU cores (default: 80% of available)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Games per processing chunk")
    parser.add_argument("--eval-scale", type=float, default=1.0, help="Scale engine evals (e.g. 0.8 for modern Stockfish)")
    parser.add_argument("--export-s", type=Path, help="Save aggregated S-set data to file")
    parser.add_argument("--use-s", type=Path, help="Load fixed S-set data from file")
    
    args = parser.parse_args()
    
    if not args.pgn_file.exists():
        print(f"Error: PGN file not found at {args.pgn_file}")
        sys.exit(1)
        
    start_time = time.time()
    
    # 1. Indexing
    print("Indexing PGN...")
    offsets = index_pgn_games(args.pgn_file)
    print(f"Found {len(offsets)} games.")
    
    if not offsets:
        print("No games found in PGN.")
        sys.exit(0)

    # 2. Setup Checkpoint Dir
    cache_dir = args.pgn_file.parent / ".ipr_cache" / args.pgn_file.stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Engine Validation
    try:
        eng = chess.engine.SimpleEngine.popen_uci(str(args.engine))
        eng.quit()
        print(f"Engine validated: {args.engine}")
    except Exception as e:
        print(f"Error starting engine at {args.engine}: {e}")
        sys.exit(1)

    # 4. Multiprocessing Pool Setup
    total_cores = multiprocessing.cpu_count()
    use_cores = args.cores if args.cores else max(1, int(total_cores * 0.8))
    
    print(f"Starting analysis with {use_cores} cores.")
    print(f"Processing in chunks of {args.chunk_size} games...")

    # Memory calculation
    total_ram = get_system_memory_mb()
    target_ram = total_ram * 0.8
    hash_per_worker = int(target_ram / use_cores)
    # Safety caps
    hash_per_worker = max(16, hash_per_worker)
    # Stockfish often hashes in powers of 2, but accepts any.
    # We should cap it reasonably (e.g. not more than 8GB per core generally supported?)
    # Stockfish max is usually huge.
    print(f"System Memory: {int(total_ram)} MB. Allocating 80% ({int(target_ram)} MB) to Hash.")
    print(f"Hash per worker: {hash_per_worker} MB")
    
    chunks = [offsets[i:i + args.chunk_size] for i in range(0, len(offsets), args.chunk_size)]
    
    # Prepare arguments for tasks
    # Note: engine path is removed from here and passed to initializer
    pool_args = []
    for i, chunk in enumerate(chunks):
        pool_args.append((i, chunk, args.pgn_file, args.depth, MULTI_PV, args.eval_scale, cache_dir))
        
    # Run Pool with Initializer
    # initializer=init_worker ensures the engine starts once per process
    with multiprocessing.Pool(processes=use_cores, 
                              initializer=init_worker, 
                              initargs=(args.engine, hash_per_worker)) as pool:
        
        # Use imap_unordered for responsive output during processing
        for res in pool.imap_unordered(worker_analyze_chunk, pool_args):
            print(res)
            
    print(f"Analysis complete. Time: {time.time() - start_time:.1f}s")
    
    # 5. Aggregate Results
    print("Aggregating results from checkpoints...")
    master_data = defaultdict(create_player_data)
    global_s_set = [] 
    
    pickle_files = list(cache_dir.glob("chunk_*.pkl"))
    
    for pkl in pickle_files:
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
            
    total_moves = sum(len(p['test_set']) for p in master_data.values())
    print(f"Total aggregated data: {len(master_data)} players, {total_moves} moves.")
    
    if total_moves == 0:
        print("WARNING: No moves were analyzed. Check engine config or game data.")


        
    # Import S-Set
    # Import S-Set
    fixed_s_set = None
    calibration = None
    
    if args.use_s:
        if args.use_s.exists():
            with open(args.use_s, 'rb') as f:
                loaded_data = pickle.load(f)
                # Handle old format (list) vs new format (dict)
                if isinstance(loaded_data, dict):
                    fixed_s_set = loaded_data.get('positions')
                    calibration = loaded_data.get('calibration')
                else:
                    fixed_s_set = loaded_data
            
            print(f"Loaded fixed S-set with {len(fixed_s_set) if fixed_s_set else 0} positions.")
            if calibration:
                print(f"Loaded Calibration: IPR = {calibration[0]:.2f} + {calibration[1]:.2f} * AE")
        else:
            print("Warning: specified S-set file not found.")

    # 6. Calculate IPRs and Calibrate if needed
    print("\n--- Calibration & Calculation ---")
    
    # We first calculate AE for ALL players to enable regression
    player_results = []
    
    for player, data in master_data.items():
        if not data['test_set'] or len(data['test_set']) < 50: 
            continue
            
        # Fit s, c
        def neg_log_lik(params):
            s, c = params
            if s <= 0.001 or c <= 0.001: return 1e9
            ll = 0
            for vals, played_idx in data['test_set']:
                if played_idx >= len(vals): continue
                probs = calculate_move_probabilities(vals, s, c)
                p = probs[played_idx] if played_idx < len(probs) else 0
                ll += math.log(p + 1e-12)
            return -ll

        try:
            res = minimize(neg_log_lik, [0.09, 0.5], bounds=[(0.01, 1.0), (0.1, 2.0)], method='L-BFGS-B')
            s_fit, c_fit = res.x
        except Exception:
            continue
            
        # Calculate AE using Target Set
        target_s_set = fixed_s_set if fixed_s_set else data['solitaire_set']
        total_ae = 0.0
        count = 0 
        
        for vals, deltas in target_s_set:
            probs = calculate_move_probabilities(vals, s_fit, c_fit)
            if len(probs) > 1 and len(deltas) == len(probs)-1:
                step_ae = 0.0
                valid_step = True
                for p, d in zip(probs[1:], deltas):
                    if p > 1e-15:
                        if d == float('inf'):
                            valid_step = False; break
                        step_ae += p * d
                if valid_step and math.isfinite(step_ae):
                    total_ae += step_ae
                    count += 1
                    
        if count > 0:
            AE_e = total_ae / count
            valid_elos = [e for e in data['elos'] if e > 0]
            avg_elo = int(sum(valid_elos)/len(valid_elos)) if valid_elos else 0
            
            player_results.append({
                'Player': player,
                'Games': data['games'],
                'Elo': avg_elo,
                's': s_fit, 'c': c_fit,
                'AE': AE_e
            })
            print(f"Analyzed {player}: AE={AE_e:.6f}, Elo={avg_elo}")

    # Perform Regression or Use Loaded/Default
    final_intercept = IPR_INTERCEPT
    final_slope = IPR_SLOPE
    
    if calibration:
        final_intercept, final_slope = calibration
        print("Using Loaded Calibration from S-set.")
    else:
        # Try to self-calibrate
        valid_points = [(r['AE'], r['Elo']) for r in player_results if r['Elo'] > 0]
        if len(valid_points) >= 3:
            aes = [p[0] for p in valid_points]
            elos = [p[1] for p in valid_points]
            slope, intercept = np.polyfit(aes, elos, 1)
            final_intercept = intercept
            final_slope = slope
            print(f"Self-Calibration Successful ({len(valid_points)} players).")
            print(f"New Formula: IPR = {final_intercept:.2f} {final_slope:+.2f} * AE")
        else:
            print(f"Insufficient rated players ({len(valid_points)}) for calibration. Using Rybka defaults.")

    # Calculate final IPRs
    final_output = []
    for res in player_results:
        IPR = final_intercept + final_slope * res['AE']
        res['IPR'] = int(IPR)
        res['s'] = round(res['s'], 4)
        res['c'] = round(res['c'], 4)
        print(f"  {res['Player']:20} AE={res['AE']:.6f} -> IPR={int(IPR)}")
        final_output.append(res)
        
    # Export S-Set (with Calibration)
    if args.export_s:
        with open(args.export_s, 'wb') as f:
            export_data = {
                'positions': global_s_set,
                'calibration': (final_intercept, final_slope)
            }
            pickle.dump(export_data, f)
        print(f"Exported S-set and Calibration to {args.export_s}")

    # 7. Write CSV Output
    if final_output:
        csv_path = args.pgn_file.parent / (args.pgn_file.stem + "_IPR_V3.csv")
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fields = ['Player', 'Games', 'Elo', 's', 'c', 'IPR']
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                # Sort by IPR descending
                writer.writerows(sorted(final_output, key=lambda x: x['IPR'], reverse=True))
            print(f"\nResults successfully saved to: {csv_path.resolve()}")
        except Exception as e:
            print(f"Error writing CSV: {e}")
    else:
        print("\nNo results generated.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
