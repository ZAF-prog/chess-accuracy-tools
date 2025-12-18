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
def analyze_game_with_engine(game, engine, depth, multipv, timeout, verbose, move_start=8, pawn_cutoff=300):
    positions = []
    board = game.board()
    
    for move_number, move in enumerate(game.mainline_moves()):
        if move_number < move_start:
            board.push(move)
            continue
        
        if verbose and move_number > move_start and move_number % 5 == 0:
            print(f"    ... analyzing move {move_number}")

        try:
            # CRITICAL: Added timeout to the analysis limit
            limit = chess.engine.Limit(depth=depth, time=timeout)
            analysis = engine.analyse(board, limit, multipv=multipv)
            
            if not analysis:
                print(f"Skipping position at move {move_number}: No analysis returned.")
                board.push(move)
                continue

            best_eval_cp = analysis[0]['score'].pov(board.turn).score(mate_score=10000)
            if abs(best_eval_cp) > pawn_cutoff or board.is_repetition(2):
                board.push(move)
                continue

            positions.append({
                'move_evals': {info['pv'][0]: info['score'].pov(board.turn).score(mate_score=10000) for info in analysis if 'pv' in info and info['pv']},
                'move_played': move,
            })
        # CRITICAL: Catch TimeoutError to prevent hangs
        except (chess.engine.EngineError, chess.engine.TimeoutError, IndexError) as e:
            print(f"Skipping position at move {move_number} due to error: {e}")
        
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

def build_training_dataset(pgn_path, engine, max_games, depth, multipv, timeout, verbose):
    """
    Analyzes all valid games from a PGN file without any ELO filtering.
    """
    print("Processing all valid games from PGN file (no ELO filter applied).")
    training_data, stats = [], {"num_games": 0, "num_moves": 0, "elos": [], "years": []}
    
    with open(pgn_path) as f:
        while max_games is None or stats["num_games"] < max_games:
            game = chess.pgn.read_game(f)
            if game is None: break
            
            white_elo = int(game.headers.get("WhiteElo", "0"))
            black_elo = int(game.headers.get("BlackElo", "0"))
            year_str = game.headers.get("Date", "1970.01.01")
            year = int(year_str.split('.')[0])

            if white_elo == 0 or black_elo == 0:
                continue
            
            print(f"Processing game {stats['num_games'] + 1} ({game.headers.get('White', '?')} vs {game.headers.get('Black', '?')})...")
            positions = analyze_game_with_engine(game, engine, depth, multipv, timeout, verbose)
            
            if positions:
                stats["num_games"] += 1
                stats["elos"].extend([white_elo, black_elo])
                stats["years"].append(year)
                for pos_data in positions:
                    spread_data = create_spread_vector(pos_data, multipv)
                    if spread_data:
                        training_data.append(spread_data)
    
    stats["num_moves"] = len(training_data)
    print(f"Created {stats['num_moves']} training positions from {stats['num_games']} games.")
    return training_data, stats

# =============================================================================
# PHASE 2: PARAMETER FITTING VIA PERCENTILING (Optimized)
# =============================================================================
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize
from functools import partial
import multiprocessing

def calculate_score_for_sc_pair(s, c, training_data, percentiles):
    """Calculates the fit score for a single (s, c) pair. Suitable for parallel execution."""
    if s <= 0 or c <= 0: return float('inf')
    
    total_score = 0
    for q in percentiles:
        up_sum = 0
        for pos in training_data:
            probs = compute_move_probabilities(pos['spread'], s, c)
            played_index = pos['played_index']
            
            p_minus = sum(probs[:played_index])
            p_played = probs[played_index]
            p_plus = p_minus + p_played

            if p_plus <= q:
                up_sum += 1.0
            elif p_minus < q < p_plus and p_played > 0:
                up_sum += abs(q - p_minus) / p_played
        
        R_qsc = up_sum / len(training_data) if training_data else 0.0
        total_score += (R_qsc - q) ** 2
        
    return total_score

def fit_parameters_hybrid(training_data):
    """
    More efficient hybrid optimization: a coarse grid search followed by local refinement.
    """
    percentiles = [q/100.0 for q in range(0, 101, 5)]

    print("Starting coarse grid search...")
    coarse_s_values = [s/10.0 for s in range(1, 8)]
    coarse_c_values = [c/10.0 for c in range(10, 51, 5)]
    
    s_initial_guess, c_initial_guess = 0.1, 1.0 # Fallback values
    coarse_best_score = float('inf')
    sc_pairs = [(s, c) for s in coarse_s_values for c in coarse_c_values]

    with ProcessPoolExecutor() as executor:
        func = partial(calculate_score_for_sc_pair, training_data=training_data, percentiles=percentiles)
        results = {sc: executor.submit(func, sc[0], sc[1]) for sc in sc_pairs}
        for sc, future in results.items():
            score = future.result()
            if score < coarse_best_score:
                coarse_best_score = score
                s_initial_guess, c_initial_guess = sc
            
    print(f"Coarse search best guess: s={s_initial_guess:.3f}, c={c_initial_guess:.2f}, score={coarse_best_score:.6f}")

    print("Starting local refinement with Nelder-Mead...")
    
    def objective_function(params):
        return calculate_score_for_sc_pair(params[0], params[1], training_data, percentiles)

    result = minimize(
        objective_function,
        [s_initial_guess, c_initial_guess],
        method='Nelder-Mead',
        options={'xatol': 1e-4, 'fatol': 1e-4, 'disp': True},
        bounds=[(0.01, 1.0), (0.1, 10.0)]
    )
    
    refined_s, refined_c = result.x
    print(f"Refinement complete. Final best: s={refined_s:.4f}, c={refined_c:.3f}, score={result.fun:.6f}")
    
    return refined_s, refined_c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit s and c parameters from a PGN training set.")
    parser.add_argument("pgn_path", type=str, help="Path to the PGN training file.")
    parser.add_argument("--engine", type=str, default=get_default_engine_path(), help="Path to UCI engine.")
    parser.add_argument("--depth", type=int, default=15, help="Engine analysis depth.")
    parser.add_argument("--multipv", type=int, default=20, help="Engine Multi-PV setting.")
    parser.add_argument("--move_timeout", type=float, default=20.0, help="Timeout in seconds for a single move analysis.")
    parser.add_argument("--max_games", type=int, default=None, help="Max games to process.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--use_cache", action="store_true", help="Use cached pre-processed data if available.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for move-by-move progress.")
    args = parser.parse_args()

    input_path = Path(args.pgn_path)
    output_basename = (Path(args.output_dir) if args.output_dir else input_path.parent) / f"{input_path.stem}_IPR_s,c-fit"
    preprocessed_data_path = output_basename.with_suffix(".pkl")
    
    if args.use_cache and preprocessed_data_path.exists():
        print(f"Loading cached data from {preprocessed_data_path}...")
        with open(preprocessed_data_path, "rb") as f:
            cache = pickle.load(f)
        training_data, stats = cache['training_data'], cache['stats']
    else:
        print("Starting Phase 1: Data Collection...")
        engine = chess.engine.SimpleEngine.popen_uci(args.engine)
        training_data, stats = build_training_dataset(
            pgn_path=args.pgn_path, 
            engine=engine, 
            max_games=args.max_games, 
            depth=args.depth, 
            multipv=args.multipv,
            timeout=args.move_timeout,
            verbose=args.verbose
        )
        engine.quit()
        
        print(f"Caching pre-processed data to {preprocessed_data_path}...")
        with open(preprocessed_data_path, "wb") as f:
            pickle.dump({'training_data': training_data, 'stats': stats}, f)

    if not training_data:
        print("No training data generated. Cannot fit parameters.")
        s_fit, c_fit = "N/A", "N/A"
    else:
        print("Starting Phase 2: Parameter Fitting...")
        s_fit, c_fit = fit_parameters_hybrid(training_data)
    
    summary_data = {
        "filename": input_path.name, "MULTIPV": args.multipv, "Number_Games": stats["num_games"],
        "Number_Moves": stats["num_moves"], "MinElo": min(stats["elos"]) if stats["elos"] else 0,
        "MaxElo": max(stats["elos"]) if stats["elos"] else 0, "AvgElo": np.mean(stats["elos"]) if stats["elos"] else 0,
        "s": s_fit, "c": c_fit, "AE_e": "N/A",
        "FirstYear": min(stats["years"]) if stats["years"] else 0, "LastYear": max(stats["years"]) if stats["years"] else 0
    }
    
    csv_path = output_basename.with_suffix(".csv")
    save_summary_csv(summary_data, csv_path)
    print(f"Summary saved to {csv_path}")
