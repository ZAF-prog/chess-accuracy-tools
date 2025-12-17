#!/usr/bin/env python
import argparse
import os
import pickle
import platform
import csv
from pathlib import Path
import chess
import chess.pgn
import chess.engine
import numpy as np
from scipy.integrate import quad
from math import exp, log

# =============================================================================
# ENGINE SETUP
# =============================================================================
def get_default_engine_path():
    """Determines default Stockfish path based on OS."""
    system = platform.system()
    if system == "Windows":
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    elif system in ("Linux", "Darwin"):
        return Path("stockfish")
    return Path("stockfish")

# =============================================================================
# PHASE 1: DATA COLLECTION AND PREPROCESSING
# =============================================================================
def analyze_game_with_engine(game, engine, depth, multipv, move_start=8, pawn_cutoff=300):
    positions = []
    board = game.board()
    
    for move_number, move in enumerate(game.mainline_moves()):
        if move_number < move_start:
            board.push(move)
            continue

        try:
            analysis = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
            best_eval_cp = analysis[0]['score'].pov(board.turn).score(mate_score=10000)
            if abs(best_eval_cp) > pawn_cutoff or board.is_repetition(2):
                board.push(move)
                continue

            positions.append({
                'move_evals': {info['pv'][0]: info['score'].pov(board.turn).score(mate_score=10000) for info in analysis},
                'move_played': move,
            })
        except (chess.engine.EngineError, IndexError) as e:
            print(f"Skipping position due to engine error: {e}")
        
        board.push(move)
        
    return positions

def _integrand(z):
    return 1.0 / (1.0 + abs(z))

def compute_delta(v0, vi):
    v0_pawns, vi_pawns = v0 / 100.0, vi / 100.0
    if v0_pawns * vi_pawns >= 0:
        return abs(log(1 + abs(v0_pawns)) - log(1 + abs(vi_pawns)))
    else:
        delta, _ = quad(_integrand, vi_pawns, v0_pawns)
        return delta

def create_spread_vector(position_data, max_moves):
    move_evals = position_data['move_evals']
    sorted_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_moves: return None
    best_move, v0 = sorted_moves[0]
    
    spread, move_to_index = [], {}
    for idx, (move, vi) in enumerate(sorted_moves[:max_moves]):
        spread.append(compute_delta(v0, vi))
        move_to_index[move] = idx
    
    while len(spread) < max_moves: spread.append(float('inf'))

    played_index = move_to_index.get(position_data['move_played'], -1)
    return {'spread': spread, 'played_index': played_index} if played_index != -1 else None

def build_training_dataset(pgn_path, engine, elo_min, elo_max, max_games, depth, multipv):
    training_data, stats = [], {"num_games": 0, "num_moves": 0, "elos": [], "years": []}
    
    with open(pgn_path) as f:
        while max_games is None or stats["num_games"] < max_games:
            game = chess.pgn.read_game(f)
            if game is None: break
            
            white_elo, black_elo = int(game.headers.get("WhiteElo", "0")), int(game.headers.get("BlackElo", "0"))
            year = int(game.headers.get("Date", "1970").split('.')[0])

            if not (elo_min <= white_elo <= elo_max and elo_min <= black_elo <= elo_max): continue
            
            print(f"Processing game {stats['num_games'] + 1}...")
            positions = analyze_game_with_engine(game, engine, depth, multipv)
            
            if positions:
                stats["num_games"] += 1
                stats["elos"].extend([white_elo, black_elo])
                stats["years"].append(year)
                for pos_data in positions:
                    spread_data = create_spread_vector(pos_data, multipv)
                    if spread_data: training_data.append(spread_data)
    
    stats["num_moves"] = len(training_data)
    print(f"Created {stats['num_moves']} training positions from {stats['num_games']} games.")
    return training_data, stats

# =============================================================================
# PHASE 2: PARAMETER FITTING VIA PERCENTILING
# =============================================================================
def compute_move_probabilities(spread, s, c):
    shares = [exp(-((delta / s) ** c)) if delta != float('inf') else 0.0 for delta in spread]
    total_shares = sum(shares)
    return [share / total_shares for share in shares] if total_shares > 0 else [1.0/len(spread)]*len(spread)

def classify_position_for_percentile(spread, played_index, s, c, q):
    probs = compute_move_probabilities(spread, s, c)
    p_minus = sum(probs[:played_index])
    p_played = probs[played_index]
    p_plus = p_minus + p_played

    if p_plus <= q: return 1.0
    if p_minus >= q: return 0.0
    return abs(q - p_minus) / p_played

def compute_R_qsc(training_data, s, c, q):
    up_sum = sum(classify_position_for_percentile(pos['spread'], pos['played_index'], s, c, q) for pos in training_data)
    return up_sum / len(training_data) if training_data else 0.0

def compute_fit_score(training_data, s, c, percentiles):
    return sum((compute_R_qsc(training_data, s, c, q) - q) ** 2 for q in percentiles)

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# =============================================================================
# PHASE 2: PARAMETER FITTING VIA PERCENTILING (Optimized)
# =============================================================================
def compute_move_probabilities(spread, s, c):
    # This function remains the same
    shares = [exp(-((delta / s) ** c)) if delta != float('inf') else 0.0 for delta in spread]
    total_shares = sum(shares)
    if total_shares == 0:
        num_valid_moves = sum(1 for d in spread if d != float('inf'))
        return [1.0/num_valid_moves if d != float('inf') else 0.0 for d in spread]
    return [share / total_shares for share in shares]

def calculate_score_for_sc_pair(s, c, training_data, percentiles):
    """Calculates the fit score for a single (s, c) pair. Suitable for parallel execution."""
    if s == 0: return float('inf')
    
    # Pre-calculate probabilities for all positions for this (s, c) pair
    all_probs = [compute_move_probabilities(pos['spread'], s, c) for pos in training_data]
    
    total_score = 0
    for q in percentiles:
        up_sum = 0
        for i, pos in enumerate(training_data):
            probs = all_probs[i]
            played_index = pos['played_index']
            
            p_minus = sum(probs[:played_index])
            p_played = probs[played_index]
            p_plus = p_minus + p_played

            if p_plus <= q:
                up_sum += 1.0
            elif p_minus < q < p_plus:
                up_sum += abs(q - p_minus) / p_played
        
        R_qsc = up_sum / len(training_data) if training_data else 0.0
        total_score += (R_qsc - q) ** 2
        
    return total_score

from scipy.optimize import minimize
from functools import partial
import multiprocessing

def fit_parameters_hybrid(training_data):
    """
    More efficient hybrid optimization: a coarse grid search followed by local refinement.
    """
    percentiles = [q/100.0 for q in range(0, 101, 5)]

    # 1. Coarse Grid Search (to find a good starting point)
    print("Starting coarse grid search...")
    coarse_s_values = [s/10.0 for s in range(1, 8)]  # 0.1, 0.2, ..., 0.7
    coarse_c_values = [c/10.0 for c in range(10, 51, 5)] # 1.0, 1.5, ..., 5.0
    
    coarse_best_score = float('inf')
    s_initial_guess, c_initial_guess = None, None

    sc_pairs = [(s, c) for s in coarse_s_values for c in coarse_c_values]

    # Parallelize the coarse search
    with multiprocessing.Pool() as pool:
        func = partial(calculate_score_for_sc_pair, training_data=training_data, percentiles=percentiles)
        results = pool.starmap(func, sc_pairs)

    for (s, c), score in zip(sc_pairs, results):
        if score < coarse_best_score:
            coarse_best_score = score
            s_initial_guess, c_initial_guess = s, c
            
    print(f"Coarse search best guess: s={s_initial_guess:.3f}, c={c_initial_guess:.2f}, score={coarse_best_score:.6f}")

    # 2. Local Refinement using Nelder-Mead
    print("Starting local refinement with Nelder-Mead...")
    
    def objective_function(params):
        s, c = params
        if s <= 0 or c <= 0:  # Constraints
            return float('inf')
        return calculate_score_for_sc_pair(s, c, training_data, percentiles)

    initial_guess = [s_initial_guess, c_initial_guess]
    
    result = minimize(
        objective_function,
        initial_guess,
        method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 1e-4, 'disp': True},
        bounds=[(0.01, 1.0), (0.1, 10.0)]
    )
    
    refined_s, refined_c = result.x
    final_score = result.fun
    
    print(f"Refinement complete. Final best: s={refined_s:.4f}, c={refined_c:.3f}, score={final_score:.6f}")
    
    return refined_s, refined_c


# =============================================================================
# MAIN EXECUTION
# =============================================================================
# (save_summary_csv remains the same)
def save_summary_csv(summary_data, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary_data.keys())
        writer.writerow(summary_data.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit s and c parameters from a PGN training set.")
    parser.add_argument("pgn_path", type=str, help="Path to the PGN training file.")
    # ... (rest of arguments are the same)
    parser.add_argument("--engine", type=str, default=get_default_engine_path(), help="Path to UCI engine.")
    parser.add_argument("--depth", type=int, default=15, help="Engine analysis depth.")
    parser.add_argument("--multipv", type=int, default=20, help="Engine Multi-PV setting.")
    parser.add_argument("--elo_min", type=int, default=2590, help="Minimum Elo for games.")
    parser.add_argument("--elo_max", type=int, default=2610, help="Maximum Elo for games.")
    parser.add_argument("--max_games", type=int, default=None, help="Max games to process.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--use_cache", action="store_true", help="Use cached pre-processed data if available.")
    args = parser.parse_args()

    input_path = Path(args.pgn_path)
    if args.output_dir:
        output_basename = Path(args.output_dir) / f"{input_path.stem}_sc_fit"
    else:
        output_basename = input_path.parent / f"{input_path.stem}_sc_fit"

    preprocessed_data_path = output_basename.with_suffix(".pkl")
    
    # --- PHASE 1 (with Caching) ---
    if args.use_cache and preprocessed_data_path.exists():
        print(f"Loading cached pre-processed data from {preprocessed_data_path}...")
        with open(preprocessed_data_path, "rb") as f:
            cached_data = pickle.load(f)
        training_data = cached_data['training_data']
        stats = cached_data['stats']
    else:
        print("Starting Phase 1: Data Collection and Preprocessing...")
        engine_path = args.engine
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        training_data, stats = build_training_dataset(args.pgn_path, engine, args.elo_min, args.elo_max, args.max_games, args.depth, args.multipv)
        engine.quit()
        
        print(f"Caching pre-processed data to {preprocessed_data_path}...")
        with open(preprocessed_data_path, "wb") as f:
            pickle.dump({'training_data': training_data, 'stats': stats}, f)

    # --- PHASE 2 ---
    if not training_data:
        print("No training data was generated. Cannot fit parameters.")
        s_fit, c_fit = "N/A", "N/A"
    else:
        print("Starting Phase 2: Parameter Fitting...")
        s_fit, c_fit = fit_parameters_hybrid(training_data)
    
    # --- OUTPUT ---
    summary_data = {
        "filename": input_path.name,
        "MULTIPV": args.multipv,
        "Number_Games": stats["num_games"],
        "Number_Moves": stats["num_moves"],
        "MinElo": min(stats["elos"]) if stats["elos"] else 0,
        "MaxElo": max(stats["elos"]) if stats["elos"] else 0,
        "AvgElo": np.mean(stats["elos"]) if stats["elos"] else 0,
        "s": s_fit,
        "c": c_fit,
        "AE_e": "N/A",
        "FirstYear": min(stats["years"]) if stats["years"] else 0,
        "LastYear": max(stats["years"]) if stats["years"] else 0
    }
    
    csv_output_path = output_basename.with_suffix(".csv")
    save_summary_csv(summary_data, csv_output_path)
    print(f"Summary with fitted s/c values saved to {csv_output_path}")
