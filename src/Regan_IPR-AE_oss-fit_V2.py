#!/usr/bin/env python
import argparse
import pickle
import platform
import os
import sys
from pathlib import Path
from math import exp, log
from functools import partial
from io import StringIO

import chess
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
from scipy.integrate import quad
from concurrent.futures import ProcessPoolExecutor

try:
    import psutil
except ImportError:
    psutil = None

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

# IPR conversion formula from the paper: IPR = 3571 - 15413 * AE_e
# These constants define the linear relationship found by Regan between AEe and Elo.
IPR_INTERCEPT = 3571
IPR_SLOPE = -15413

# Engine analysis parameters.
MATE_SCORE = 10000
MOVE_START_DEFAULT = 8
PAWN_CUTOFF_DEFAULT = 300


# =============================================================================
# ENGINE SETUP
# =============================================================================

def get_default_engine_path() -> Path:
    """Determines the default Stockfish engine path based on the operating system."""
    system = platform.system()
    if system == "Windows":
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    elif system in ("Linux", "Darwin"):
        return Path("stockfish")
    return Path("stockfish")


# =============================================================================
# PARALLEL DATA COLLECTION & PROCESSING (from Reference PGN)
# =============================================================================

def analyze_game_worker(
    game_text: str, engine_path: str, depth: int, multipv: int,
    timeout: float, hash_mb: int
) -> list[list[float]]:
    """
    A single worker's task: analyze one game and return a list of spread vectors.
    
    This function is designed to be executed in a separate process. It initializes
    its own persistent, single-threaded engine instance to avoid the overhead of
    restarting the engine for every game and to prevent thread contention within
    a single engine when running multiple processes.
    """
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 1, "Hash": hash_mb})
    
    game = chess.pgn.read_game(StringIO(game_text))
    spread_vectors = []
    
    if game:
        board = game.board()
        for move_number, move in enumerate(game.mainline_moves()):
            if move_number < MOVE_START_DEFAULT:
                board.push(move)
                continue

            try:
                limit = chess.engine.Limit(depth=depth, time=timeout)
                analysis = engine.analyse(board, limit, multipv=multipv)
                if not analysis:
                    board.push(move)
                    continue

                if isinstance(analysis, dict):
                    analysis = [analysis]

                best_eval_cp = analysis[0]["score"].pov(board.turn).score(mate_score=MATE_SCORE)
                if best_eval_cp is None or abs(best_eval_cp) > PAWN_CUTOFF_DEFAULT or board.is_repetition(2):
                    board.push(move)
                    continue

                move_evals = {
                    info["pv"][0]: info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
                    for info in analysis if "pv" in info and info["pv"] and info["score"].pov(board.turn).score(mate_score=MATE_SCORE) is not None
                }
                
                if move_evals:
                    sorted_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)
                    if not sorted_moves:
                        board.push(move)
                        continue
                    
                    v0 = sorted_moves[0][1]
                    spread = [compute_delta(v0, vi) for _, vi in sorted_moves]
                    spread_vectors.append(spread)

            except (chess.engine.EngineError, chess.engine.TimeoutError, IndexError):
                pass  # Ignore errors in worker, just move to the next position
            
            board.push(move)
            
    engine.quit()
    return spread_vectors

def compute_delta(v0: float, vi: float) -> float:
    """Computes the scaled difference (delta) between two centipawn evaluations."""
    v0_pawns, vi_pawns = v0 / 100.0, vi / 100.0
    if v0_pawns * vi_pawns >= 0:
        return abs(log(1 + abs(v0_pawns)) - log(1 + abs(vi_pawns)))
    delta, _ = quad(lambda z: 1.0 / (1.0 + abs(z)), vi_pawns, v0_pawns)
    return delta

def build_reference_dataset(
    pgn_path: str, engine_path: str, depth: int,
    multipv: int, timeout: float, verbose: bool
) -> list[list[float]]:
    """
    Analyzes the reference PGN in parallel to create a dataset of spread vectors.
    
    This is the main data collection function. It orchestrates the performance-
    intensive task of analyzing every game in the reference PGN. It works by:
    1. Calculating available system resources (CPU cores, RAM).
    2. Splitting the PGN file into individual game strings.
    3. Creating a pool of worker processes (ProcessPoolExecutor).
    4. Distributing the game analysis tasks to the workers.
    5. Collecting the results (lists of spread vectors) from each worker.
    """
    print("Processing reference PGN to generate spread vectors...")
    
    # --- Resource Calculation ---
    # Automatically configure parallel workers and memory based on system specs.
    if psutil:
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        num_cpus = os.cpu_count() or 1
        # Use 80% of available cores for worker processes.
        num_workers = max(1, int(num_cpus * 0.8))
        # Allocate 60% of total system memory for engine hash, divided among workers.
        total_hash_mb = int((total_mem_gb * 0.6) * 1024)
        hash_per_worker = max(16, total_hash_mb // num_workers)
        
        if verbose:
            print(f"System Info: {num_cpus} CPUs, {total_mem_gb:.1f} GB RAM")
            print(f"Using {num_workers} worker processes with {hash_per_worker} MB hash each.")
    else:
        print("Warning: psutil not found. Using default 4 workers and 256MB hash.")
        num_workers = 4
        hash_per_worker = 256

    # --- PGN Splitting for Workers ---
    # The PGN file is split into a list of strings, where each string is a full
    # game. This allows each worker to receive a game's text and process it
    # independently without file I/O conflicts.
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        full_pgn_text = f.read()
    game_texts = [f"[Event {game_text}" for game_text in full_pgn_text.split("\n[Event ") if game_text.strip()]
    if game_texts and not full_pgn_text.startswith("[Event"):
        game_texts[0] = game_texts[0][7:]

    print(f"Found {len(game_texts)} games to analyze.")
    reference_spreads = []

    # --- Parallel Processing ---
    # A ProcessPoolExecutor manages the pool of worker processes.
    # The 'partial' function creates a callable with fixed arguments for the
    # engine configuration, which is then passed to each worker.
    worker_func = partial(
        analyze_game_worker,
        engine_path=engine_path, depth=depth, multipv=multipv,
        timeout=timeout, hash_mb=hash_per_worker
    )
    
    # The executor's 'map' function efficiently distributes the game_texts
    # to the worker pool and collects the results as they are completed.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(worker_func, game_texts)
        for i, game_spreads in enumerate(results):
            if verbose and (i + 1) % 20 == 0:
                print(f"  ... processed {i + 1}/{len(game_texts)} games.")
            if game_spreads:
                reference_spreads.extend(game_spreads)

    print(f"Created {len(reference_spreads)} reference positions from {len(game_texts)} games.")
    return reference_spreads


# =============================================================================
# AEe and IPR CALCULATION
# =============================================================================

def compute_move_probabilities(spread: list[float], s: float, c: float) -> list[float]:
    """Computes move probabilities given a spread vector, s, and c."""
    if s <= 0 or c <= 0: return [0.0] * len(spread)
    shares = [exp(-((delta / s) ** c)) if delta != float("inf") else 0.0 for delta in spread]
    total_shares = sum(shares)
    if total_shares == 0:
        num_valid = sum(1 for d in spread if d != float("inf"))
        return [1.0 / num_valid if d != float("inf") else 0.0 for d in spread] if num_valid > 0 else [0.0] * len(spread)
    return [share / total_shares for share in shares]

def calculate_aee(ref_spreads: list[list[float]], s: float, c: float) -> float:
    """
    Calculates the Average Expected Error (AEe) for a given s and c
    over the reference dataset of spread vectors.
    
    AEe is the model's prediction for the average error (delta) a player with
    skill (s, c) would make on the positions in the reference set.
    Formula: AEe = (1/T) * Σ_t Σ_{i≥1} [ p_{i,t} * δ_{i,t} ]
    where T is the total number of positions.
    """
    # For each position (spread) in the reference set, we calculate the
    # expected error for that position. The expected error is the sum of
    # (probability * error) for every possible non-best move.
    total_expected_error = sum(
        p * d
        for spread in ref_spreads
        # We compute probabilities for all moves, then zip them with their deltas.
        # We skip the first element ([1:]) because the best move has a delta of 0.
        for p, d in zip(compute_move_probabilities(spread, s, c)[1:], spread[1:])
        if d != float("inf")
    )
    # The final AEe is the average of these expected errors over all positions.
    return total_expected_error / len(ref_spreads) if ref_spreads else 0.0

def calculate_ipr(aee: float) -> float:
    """
    Converts an AEe value to its corresponding Intrinsic Performance Rating (IPR)
    using the linear formula established in Regan's research.
    """
    return IPR_INTERCEPT + IPR_SLOPE * aee


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calculate AEe and IPR on a reference PGN set using pre-fitted s and c parameters."
    )
    parser.add_argument("sc_results_csv", type=str, help="Path to the CSV file from Step 1 (s, c fits).")
    parser.add_argument("reference_pgn", type=str, help="Path to the reference (Solitaire) PGN file.")
    parser.add_argument("--engine_path", type=str, default=str(get_default_engine_path()), help="Path to Stockfish.")
    parser.add_argument("--depth", type=int, default=12, help="Engine search depth.")
    parser.add_argument("--multipv", type=int, default=20, help="Engine MultiPV setting.")
    parser.add_argument("--timeout", type=float, default=0.5, help="Engine time limit per move.")
    parser.add_argument("--use_cache", action="store_true", help="Use/create pickle cache for reference PGN analysis.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    # --- Load pre-fitted s,c data ---
    # The script begins by loading the results from the Step 1 script.
    # This file contains the empirically fitted (s, c) pairs for various
    # Elo levels or player populations.
    try:
        sc_df = pd.read_csv(args.sc_results_csv)
    except FileNotFoundError:
        sys.exit(f"Error: Input CSV file not found at {args.sc_results_csv}")

    # --- Build or Load Reference Dataset from PGN ---
    # This is the most computationally expensive step. The script analyzes the
    # reference PGN (e.g., world championship games) to create a standardized
    # dataset of "spread vectors". This dataset represents the "test" that
    # each (s, c) pair will be measured against.
    # Caching this dataset to a pickle file is crucial for performance,
    # allowing subsequent runs to skip the analysis entirely.
    cache_path = Path(args.reference_pgn).with_suffix(".reference_spreads_cache.pkl")
    if args.use_cache and cache_path.exists():
        if args.verbose: print(f"Loading cached reference data from {cache_path}...")
        with open(cache_path, "rb") as f: reference_spreads = pickle.load(f)
    else:
        reference_spreads = build_reference_dataset(
            args.reference_pgn, args.engine_path, args.depth, args.multipv, args.timeout, args.verbose
        )
        if args.use_cache:
            if args.verbose: print(f"Saving reference data cache to {cache_path}...")
            with open(cache_path, "wb") as f: pickle.dump(reference_spreads, f)
    
    if not reference_spreads:
        sys.exit("No reference positions could be analyzed from the PGN. Exiting.")

    # --- Calculate AEe and IPR for each (s, c) pair ---
    # This is the core calculation loop. For each row (representing a player
    # or Elo level) from the input CSV, the script takes the (s, c) parameters
    # and uses them to calculate two key metrics against the reference dataset:
    # 1. AEe: The theoretical average error for that skill level.
    # 2. IPR: The direct conversion of that AEe into an Elo-like rating.
    print("\nCalculating AEe and IPR for each s,c pair...")
    aee_results = []
    ipr_results = []
    for index, row in sc_df.iterrows():
        try:
            s = float(row["s"])
            c = float(row["c"])
            if s <= 0 or c <= 0:
                raise ValueError("s and c must be positive")
        except (ValueError, KeyError):
            print(f"Warning: Skipping row {index+2} due to invalid or missing s/c values.")
            aee_results.append(np.nan)
            ipr_results.append(np.nan)
            continue
        
        aee = calculate_aee(reference_spreads, s, c)
        ipr = calculate_ipr(aee)
        aee_results.append(aee)
        ipr_results.append(ipr)
        if args.verbose:
            elo_label = f" (for Elo ~{row['AvgElo']:.0f})" if 'AvgElo' in row else ""
            print(f"  s={s:.4f}, c={c:.4f}{elo_label} -> AEe={aee:.6f} -> IPR={ipr:.1f}")

    sc_df["AEe"] = aee_results
    sc_df["IPR"] = ipr_results

    # --- Write Final Output CSV ---
    # The final step is to save the results. The output CSV contains all the
    # original data from the input file, now augmented with the calculated
    # AEe and IPR values, providing a complete picture of the analysis.
    output_filename = f"{Path(args.sc_results_csv).stem}_with_IPR.csv"
    output_path = Path("data") / output_filename
    output_path.parent.mkdir(exist_ok=True)
    
    sc_df.to_csv(output_path, index=False, float_format="%.8f")
    print(f"\nSuccessfully calculated and saved results to {output_path}")

if __name__ == "__main__":
    main()
