#!/usr/bin/env python
"""
Dr. Kenneth Regan's Intrinsic Performance Rating (IPR) algorithm
with Multi-PV analysis using Stockfish (replacing Rybka).

This implementation follows Regan's methodology:
 Fit skill parameters (s, c) using maximum likelihood on actual moves
 This should take input from a PGN file
  with narrow range of Elo ratings for all players!

Regan IPR Parameter Fitter (s, c) V7.1
=========================================

Purpose:
1. Analyze PGN games using a persistent Stockfish engine (optimized).
2. Collect move-by-move data for parameter fitting.
3. Perform Maximum Likelihood Estimation (MLE) to fit the player parameters (s, c).
4. Extract Elo, First Year, and Last Year played.
5. APPEND results to a specified cumulative CSV file.
6. Does calculate Average Error (AE).

Usage:
    python regan_fit_sc.py <pgn_file> --output-csv results.csv
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
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Any
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
import atexit

# --- CONFIGURATION ---
BOOK_MOVES = 8          # Skip first 8 moves (16 ply)
CAP_EVAL = 300          # Garbage time filter (centipawns)
MULTI_PV = 5            # Default number of principal variations (can be overridden via CLI)
MATE_SCORE = 10000      # Value used to cap mate scores
CHUNK_SIZE = 50         # Games per chunk

# --- GLOBAL WORKER VARIABLE ---
worker_engine = None

def get_default_engine_path():
    """Determines default Stockfish path based on OS."""
    system = platform.system()
    if system == "Windows":
        # Adjust this path to your actual Stockfish location
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    elif system in ("Linux", "Darwin"):
        return Path("stockfish")  # Assumes 'stockfish' is in PATH
    return Path("stockfish")

def centipawns_to_winprob(cp: float) -> float:
    """Converts centipawns to win probability using a logistic formula."""
    try:
        if cp > 20000:
            return 1.0
        if cp < -20000:
            return 0.0
        return 1.0 / (1.0 + math.pow(10, -cp / 400.0))
    except (OverflowError, ValueError):
        return 0.0 if cp < 0 else 1.0

def calculate_move_probabilities(values: List[float], s: float, c: float) -> List[float]:
    """Calculate move probabilities using inverse-exponential over win-prob deltas.

    values: engine scores in pawns for each candidate move.
    """
    if not values:
        return []

    # 1. Convert to centipawns then win probability
    win_probs = [centipawns_to_winprob(v * 100.0) for v in values]

    # Use the maximum win prob as best reference
    best_wp = max(win_probs)

    # 2. Deltas in win-prob space
    deltas = [max(0.0, best_wp - wp) for wp in win_probs]

    # 3. Weights = exp(-(delta/s)^c)
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
        return [1.0] + [0.0] * (len(values) - 1)

    return [w / total_w for w in weights]

def create_player_data():
    """Factory for player data structure (picklable)."""
    return {
        'test_set': [],  # [(raw_values, actual_idx), ...]
        'elos': [],
        'games': 0,
        'first_year': float('inf'),
        'last_year': float('-inf'),
    }

# --- WORKER FUNCTIONS (Parallel Analysis) ---

def init_worker(engine_path: Path, hash_mb: int):
    """Initializer run once per worker process to start the engine."""
    global worker_engine
    try:
        worker_engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
        worker_engine.configure({"Hash": int(hash_mb), "Threads": 1})
        atexit.register(worker_engine.quit)
    except Exception as e:
        print(f"Worker initialization failed for engine at {engine_path}: {e}")
        worker_engine = None

def worker_analyze_chunk(args):
    """Worker process to analyze a chunk of games using the persistent engine."""
    chunk_id, offsets, pgn_path, depth, multipv, cache_dir = args

    result_file = cache_dir / f"chunk_{chunk_id}.pkl"
    if result_file.exists():
        return f"Chunk {chunk_id} (Skipped)"

    global worker_engine
    if worker_engine is None:
        return f"Chunk {chunk_id} Failed: Engine not initialized properly."

    chunk_data = defaultdict(create_player_data)

    try:
        with open(pgn_path, 'r', encoding='utf-8') as f:
            for offset in offsets:
                f.seek(offset)
                game = chess.pgn.read_game(f)
                if not game:
                    continue
                analyze_single_game(game, worker_engine, chunk_data, depth, multipv)
    except Exception as e:
        return f"Chunk {chunk_id} Failed: {e}"

    try:
        with open(result_file, 'wb') as f:
            pickle.dump(chunk_data, f)
    except Exception as e:
        return f"Chunk {chunk_id} Failed to save results: {e}"

    return f"Chunk {chunk_id} Done ({len(offsets)} games)"

def analyze_single_game(game, engine, data_store, depth, multipv):
    white = game.headers.get("White", "Unknown")
    black = game.headers.get("Black", "Unknown")
    date_str = game.headers.get("Date", "????" + ".??.??")

    try:
        year = int(date_str.split('.')[0].strip('?'))
    except (ValueError, IndexError):
        year = None

    for player in [white, black]:
        pdata = data_store[player]
        pdata['games'] += 1
        try:
            elo = int(game.headers.get("WhiteElo" if player == white else "BlackElo", 0))
            if elo > 0:
                pdata['elos'].append(elo)
        except Exception:
            pass
        if year is not None:
            pdata['first_year'] = min(pdata['first_year'], year)
            pdata['last_year'] = max(pdata['last_year'], year)

    board = game.board()
    node = game
    ply = 0

    while node.variations:
        next_node = node.variation(0)
        move = next_node.move
        ply += 1

        if ply <= BOOK_MOVES * 2:
            board.push(move)
            node = next_node
            continue

        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            res = []
            for pv in info:
                if 'pv' not in pv:
                    continue
                sc = pv['score'].white()
                if sc.is_mate():
                    cp = MATE_SCORE if sc.mate() > 0 else -MATE_SCORE
                else:
                    cp = sc.score()
                res.append((pv['pv'][0], cp))

            if not res:
                raise Exception("No result from engine")

            best_cp = res[0][1]
            if abs(best_cp) > CAP_EVAL:
                board.push(move)
                node = next_node
                continue

            raw_values = [x[1] / 100.0 for x in res]
            if board.turn == chess.BLACK:
                raw_values = [-x for x in raw_values]

            actual_idx = -1
            for i, (m, _) in enumerate(res):
                if m == move:
                    actual_idx = i
                    break

            if actual_idx != -1:
                player = white if board.turn == chess.WHITE else black
                data_store[player]['test_set'].append((raw_values, actual_idx))
        except Exception:
            pass

        board.push(move)
        node = next_node

# --- MAIN DRIVER (Aggregation and Fitting) ---

def index_pgn_games(path):
    """Scans PGN file and returns file offsets for each game."""
    offsets = []
    with open(path, 'rb') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.startswith(b'[Event '):
                offsets.append(offset)
    return offsets

def main():
    parser = argparse.ArgumentParser(description="Regan IPR Parameter Fitter (s, c)")
    parser.add_argument("pgn_file", type=Path, help="Input PGN")
    parser.add_argument("--engine", type=Path, default=get_default_engine_path(), help="Stockfish path")
    parser.add_argument("--output-csv", type=Path, required=True, help="Cumulative CSV file to append results to (e.g., results.csv)")
    parser.add_argument("--depth", type=int, default=14, help="Analysis depth (default: 14)")
    parser.add_argument("--cores", type=int, help="Force CPU cores (default: 80% of available)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Games per processing chunk")
    parser.add_argument("--multipv", type=int, default=MULTI_PV, help="Number of principal variations (MultiPV), default: 5")

    args = parser.parse_args()

    if not args.pgn_file.exists():
        print(f"Error: PGN file not found at {args.pgn_file}")
        sys.exit(1)

    start_time = time.time()

    print("Indexing PGN...")
    offsets = index_pgn_games(args.pgn_file)
    print(f"Found {len(offsets)} games.")
    if not offsets:
        print("No games found in PGN.")
        sys.exit(0)

    cache_dir = args.pgn_file.parent / ".ipr_fit_cache" / args.pgn_file.stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        eng = chess.engine.SimpleEngine.popen_uci(str(args.engine))
        eng.quit()
        print(f"Engine validated: {args.engine}")
    except Exception as e:
        print(f"Error starting engine at {args.engine}: {e}")
        sys.exit(1)

    total_cores = multiprocessing.cpu_count()
    use_cores = args.cores if args.cores else max(1, int(total_cores * 0.8))

    total_ram = psutil.virtual_memory().total
    total_hash_mb = (total_ram * 0.6) / (1024 * 1024)
    hash_per_worker = max(16, int(total_hash_mb / use_cores))

    print(f"Starting parallel analysis with {use_cores} cores...")
    print(f"System RAM: {total_ram / (1024**3):.1f} GB. Allocating {hash_per_worker} MB Hash per worker (Total Hash: {int(hash_per_worker * use_cores / 1024)} GB)")

    chunks = [offsets[i:i + args.chunk_size] for i in range(0, len(offsets), args.chunk_size)]
    pool_args = []
    for i, chunk in enumerate(chunks):
        pool_args.append((i, chunk, args.pgn_file, args.depth, args.multipv, cache_dir))

    with multiprocessing.Pool(
        processes=use_cores,
        initializer=init_worker,
        initargs=(args.engine, hash_per_worker),
    ) as pool:
        for res in pool.imap_unordered(worker_analyze_chunk, pool_args):
            print(res)

    print(f"\nAnalysis complete. Time: {time.time() - start_time:.1f}s")

    print("Aggregating results from checkpoints...")
    master_data = defaultdict(create_player_data)
    pickle_files = list(cache_dir.glob("chunk_*.pkl"))

    for pkl in pickle_files:
        try:
            with open(pkl, 'rb') as f:
                chunk_data = pickle.load(f)
                for player, pdata in chunk_data.items():
                    master_data[player]['test_set'].extend(pdata['test_set'])
                    master_data[player]['elos'].extend(pdata['elos'])
                    master_data[player]['games'] += pdata['games']
                    master_data[player]['first_year'] = min(master_data[player]['first_year'], pdata['first_year'])
                    master_data[player]['last_year'] = max(master_data[player]['last_year'], pdata['last_year'])
        except Exception as e:
            print(f"Error reading checkpoint {pkl}: {e}")

    total_moves = sum(len(p['test_set']) for p in master_data.values())
    print(f"Total aggregated data: {len(master_data)} players, {total_moves} moves analyzed.")

    print("\n--- Performing Global Parameter Fitting (s, c) ---")
    final_output = []
    MIN_MOVES = 30

    global_test_set = []
    all_elos = []
    global_first_year = float('inf')
    global_last_year = float('-inf')

    for player, data in master_data.items():
        if len(data['test_set']) > 0:
            global_test_set.extend(data['test_set'])
            all_elos.extend(data['elos'])
            global_first_year = min(global_first_year, data['first_year'])
            global_last_year = max(global_last_year, data['last_year'])

    total_moves_fit = len(global_test_set)
    print(f"Total moves for fitting: {total_moves_fit}")

    if total_moves_fit < MIN_MOVES:
        print(f"Insufficient total moves ({total_moves_fit} < {MIN_MOVES}). Fitting skipped.")
    else:
        def percentile_score(params):
            s, c = params
            if s <= 0.001 or c <= 0.001:
                return 1e9

            qs = np.arange(0.05, 1.0, 0.05)
            observation_counts = np.zeros(len(qs))
            total_valid_moves = 0

            for vals, played_idx in global_test_set:
                if played_idx >= len(vals):
                    continue

                probs = calculate_move_probabilities(vals, s, c)
                if not probs:
                    continue

                p_played = probs[played_idx]
                p_minus = sum(probs[:played_idx])
                p_plus = p_minus + p_played

                observation_counts += (qs >= p_plus).astype(float)

                if p_played > 1e-9:
                    mask = (qs > p_minus) & (qs < p_plus)
                    if np.any(mask):
                        fractions = (qs[mask] - p_minus) / p_played
                        observation_counts[mask] += fractions

                total_valid_moves += 1

            if total_valid_moves == 0:
                return 1e9

            observed_R = observation_counts / total_valid_moves
            sq_errors = (observed_R - qs) ** 2
            return np.sum(sq_errors)

        print(f"Fitting global parameters using Percentile Method on {total_moves_fit} moves...")
        try:
            res = minimize(
                percentile_score,
                [0.09, 0.5],
                bounds=[(0.01, 1.0), (0.1, 2.0)],
                method='L-BFGS-B',
            )
            s_fit, c_fit = res.x
            success = res.success
            print(f"  Final Score (Sq,c): {res.fun:.6f}")
        except Exception as e:
            print(f"  Fit failed: {e}")
            success = False
            s_fit, c_fit = 0, 0

        total_expected_error = 0
        total_actual_error = 0
        total_expected_match = 0
        total_actual_match = 0
        total_positions = 0

        for vals, played_idx in global_test_set:
            if played_idx >= len(vals):
                continue

            probs = calculate_move_probabilities(vals, s_fit, c_fit)
            if not probs:
                continue

            wps = [centipawns_to_winprob(v * 100) for v in vals]
            best_wp = wps[0]
            deltas = []
            for wp in wps:
                d = best_wp - wp
                if d < 0:
                    d = 0
                deltas.append(d)

            expected_pos_error = sum(p * d for p, d in zip(probs, deltas))
            expected_match = probs[0]

            actual_pos_error = deltas[played_idx] if played_idx < len(deltas) else 0.0
            actual_match = 1.0 if played_idx == 0 else 0.0

            total_expected_error += expected_pos_error
            total_actual_error += actual_pos_error
            total_expected_match += expected_match
            total_actual_match += actual_match
            total_positions += 1

        if total_positions > 0:
            ae_e = total_expected_error / total_positions
            ae_a = total_actual_error / total_positions
            match_e = (total_expected_match / total_positions) * 100
            match_a = (total_actual_match / total_positions) * 100
        else:
            ae_e = 0
            ae_a = 0
            match_e = 0
            match_a = 0

        qfit = math.sqrt(res.fun / 19.0) if success else 0.0

        if success:
            filtered_elos = [e for e in all_elos if e > 0]
            if filtered_elos:
                avg_elo = int(sum(filtered_elos) / len(filtered_elos))
                min_elo = min(filtered_elos)
                max_elo = max(filtered_elos)
            else:
                avg_elo = 0
                min_elo = 0
                max_elo = 0

            first_year = int(global_first_year) if global_first_year != float('inf') else 'N/A'
            last_year = int(global_last_year) if global_last_year != float('-inf') else 'N/A'

            print(f"  Result: s={s_fit:.4f}, c={c_fit:.4f}")
            print(
                f"  Metrics: AE(Pred/Act)={ae_e:.4f}/{ae_a:.4f}, "
                f"Match(Pred/Act)={match_e:.1f}%/{match_a:.1f}%, Qfit={qfit:.4f}"
            )

            final_output.append(
                {
                    'filename': args.pgn_file.name,
                    'Games': len(offsets),
                    'Moves': total_moves_fit,
                    'AvgElo': avg_elo,
                    'MinElo': min_elo,
                    'MaxElo': max_elo,
                    's': round(s_fit, 4),
                    'c': round(c_fit, 4),
                    'AE_e': round(ae_e, 6),
                    'AE_a': round(ae_a, 6),
                    'Match_e': round(match_e, 2),
                    'Match_a': round(match_a, 2),
                    'Qfit': round(qfit, 5),
                    'FirstYear': first_year,
                    'LastYear': last_year,
                }
            )

    if final_output:
        csv_path = args.output_csv
        fields = [
            'filename',
            'Games',
            'Moves',
            'AvgElo',
            'MinElo',
            'MaxElo',
            's',
            'c',
            'AE_e',
            'AE_a',
            'Match_e',
            'Match_a',
            'Qfit',
            'FirstYear',
            'LastYear',
        ]

        file_exists = csv_path.exists()

        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if not file_exists or os.path.getsize(csv_path) == 0:
                    writer.writeheader()
                writer.writerows(final_output)
            print(f"\nSuccessfully appended results to: {csv_path.resolve()}")
        except Exception as e:
            print(f"Error writing/appending to CSV: {e}")
    else:
        print("\nNo parameters were successfully fitted.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
