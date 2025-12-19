#!/usr/bin/env python
import argparse
import pickle
import platform
import csv
from pathlib import Path
import sys
from math import exp, log
from functools import partial

import chess
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
import psutil
from concurrent.futures import ProcessPoolExecutor
from sklearn.linear_model import LinearRegression

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

MATE_SCORE = 10000
MOVE_START_DEFAULT = 8
PAWN_CUTOFF_DEFAULT = 300
PERCENTILES = [q / 100.0 for q in range(0, 101, 5)]


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
# PARALLEL WORKER SETUP
# =============================================================================

def analyze_game_worker(
    game_text: str, engine_path: str, depth: int, multipv: int,
    timeout: float, hash_mb: int
) -> list[dict]:
    """
    A single worker's task: analyze one game from its PGN text.
    Initializes its own persistent, single-threaded engine instance.
    """
    # Each worker gets its own engine instance.
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    # Configure the engine for this specific worker.
    engine.configure({"Threads": 1, "Hash": hash_mb})
    
    # The python-chess library needs an IO object to read a game.
    from io import StringIO
    game = chess.pgn.read_game(StringIO(game_text))
    
    results = []
    if game:
        # We pass verbose=False because parallel output would be scrambled.
        positions = analyze_game_with_engine(game, engine, depth, multipv, timeout, verbose=False)
        for pos_data in positions:
            spread_data = create_spread_data_for_position(pos_data, multipv)
            if spread_data:
                results.append(spread_data)
    
    engine.quit()
    return results


# =============================================================================
# DATA COLLECTION & PROCESSING (from Reference PGN)
# =============================================================================

def analyze_game_with_engine(
    game: chess.pgn.Game, engine: chess.engine.SimpleEngine, depth: int,
    multipv: int, timeout: float, verbose: bool
) -> list[dict]:
    """
    Analyzes a single game, returning data needed for optimization and AEe calculation.

    
    This is a critical data collection step. For each valid position, it gathers:
    1.  'move_evals': A dictionary of engine evaluations for all top moves.
    2.  'move_played': The actual move played in the game. This is essential for
        the Stage 2 re-optimization of 's', which fits the model to the moves
        observed in this reference set.
    """
    positions = []
    board = game.board()

    for move_number, move in enumerate(game.mainline_moves()):
        if move_number < MOVE_START_DEFAULT:
            board.push(move)
            continue

        if verbose and move_number > MOVE_START_DEFAULT and move_number % 10 == 0:
            print(f"    ... analyzing move {move_number}")

        try:
            limit = chess.engine.Limit(depth=depth, time=timeout)
            analysis = engine.analyse(board, limit, multipv=multipv)

            if not analysis or isinstance(analysis, dict):
                analysis = [analysis] if analysis else []

            best_eval_cp = analysis[0]["score"].pov(board.turn).score(mate_score=MATE_SCORE) if analysis else None
            if best_eval_cp is None or abs(best_eval_cp) > PAWN_CUTOFF_DEFAULT or board.is_repetition(2):
                board.push(move)
                continue

            move_evals = {
                info["pv"][0]: info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
                for info in analysis if "pv" in info and info["pv"] and info["score"].pov(board.turn).score(mate_score=MATE_SCORE) is not None
            }

            if move_evals:
                positions.append({"move_evals": move_evals, "move_played": move})

        except (chess.engine.EngineError, chess.engine.TimeoutError, IndexError) as e:
            if verbose:
                print(f"Skipping position at move {move_number} due to error: {e}")

        board.push(move)

    return positions

def compute_delta(v0: float, vi: float) -> float:
    v0_pawns, vi_pawns = v0 / 100.0, vi / 100.0
    if v0_pawns * vi_pawns >= 0:
        return abs(log(1 + abs(v0_pawns)) - log(1 + abs(vi_pawns)))
    delta, _ = quad(lambda z: 1.0 / (1.0 + abs(z)), vi_pawns, v0_pawns)
    return delta

def create_spread_data_for_position(position_data: dict, max_moves: int) -> dict | None:

    """
    Creates a spread vector and finds the index of the move that was played.
    
    The 'spread' is the vector of delta values (δ₀, δ₁, ...), representing the
    error of each move compared to the best move. The 'played_index' tells us
    which of these moves was actually chosen in the game, which is the ground
    truth for the percentiling optimization.
    """
    move_evals = position_data["move_evals"]
    sorted_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)

    if not sorted_moves:
        return None

    best_move, v0 = sorted_moves[0]
    spread = [compute_delta(v0, vi) for move, vi in sorted_moves[:max_moves]]
    
    move_to_index = {move: idx for idx, (move, _) in enumerate(sorted_moves[:max_moves])}
    played_index = move_to_index.get(position_data["move_played"], -1)

    if played_index == -1:
        return None

    return {"spread": spread, "played_index": played_index}

def build_reference_dataset(
    pgn_path: str, engine_path: str, depth: int,
    multipv: int, timeout: float, verbose: bool
) -> list[dict]:
    """
    Analyzes reference PGN in parallel to create a dataset for optimizing 's'.
    
    This function sets up a pool of worker processes, each running a dedicated,
    single-threaded Stockfish instance. It splits the input PGN file by games
    and distributes the analysis tasks across the workers.
    """
    print("Processing reference PGN for re-optimization dataset...")
    
    # --- Resource Calculation ---
    try:
        total_mem_gb = psutil.virtual_memory().total / (1024**3)
        num_cpus = os.cpu_count() or 1
        
        # Use 80% of available cores for workers.
        num_workers = max(1, int(num_cpus * 0.8))
        
        # Use 60% of total system memory for hash, divided among workers.
        total_hash_mb = int((total_mem_gb * 0.6) * 1024)
        hash_per_worker = max(16, total_hash_mb // num_workers)
        
        if verbose:
            print(f"System Info: {num_cpus} CPUs, {total_mem_gb:.1f} GB RAM")
            print(f"Using {num_workers} worker processes with {hash_per_worker} MB hash each.")
            
    except (ImportError, ModuleNotFoundError):
        print("Warning: psutil not found. Using default 4 workers and 256MB hash.")
        num_workers = 4
        hash_per_worker = 256

    # --- PGN Splitting ---
    # Read the entire PGN and split it into text blocks for each game.
    # This is more robust for multiprocessing than passing parsed game objects.
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        full_pgn_text = f.read()
    
    # A simple split based on a common PGN game header. Assumes PGNs are well-formed.
    game_texts = [game_text for game_text in full_pgn_text.split("\n[Event ") if game_text.strip()]
    # The first game won't have the split string, so we prepend it.
    if len(game_texts) > 0 and not game_texts[0].startswith("[Event "):
        game_texts = [f"[Event {text}" for text in game_texts]

    print(f"Found {len(game_texts)} games to analyze.")
    reference_data = []

    # --- Parallel Processing ---
    # Create a partial function to pass fixed arguments to the worker.
    worker_func = partial(
        analyze_game_worker,
        engine_path=engine_path,
        depth=depth,
        multipv=multipv,
        timeout=timeout,
        hash_mb=hash_per_worker
    )
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # map() distributes the game_texts to the workers and collects results.
        results = executor.map(worker_func, game_texts)
        
        for i, game_results in enumerate(results):
            if verbose and (i + 1) % 10 == 0:
                print(f"  ... processed {i + 1}/{len(game_texts)} games.")
            if game_results:
                reference_data.extend(game_results)

    print(f"Created {len(reference_data)} reference positions from {len(game_texts)} games.")
    return reference_data


# =============================================================================
# PROBABILITY, OPTIMIZATION & AEe
# =============================================================================

def compute_move_probabilities(spread: list[float], s: float, c: float) -> list[float]:
    """Computes move probabilities given spread, s, and c."""
    if s <= 0 or c <= 0: return [0.0] * len(spread)
    shares = [exp(-((delta / s) ** c)) if delta != float("inf") else 0.0 for delta in spread]
    total_shares = sum(shares)
    if total_shares == 0:
        num_valid = sum(1 for d in spread if d != float("inf"))
        return [1.0 / num_valid if d != float("inf") else 0.0 for d in spread] if num_valid > 0 else [0.0] * len(spread)
    return [share / total_shares for share in shares]

def calculate_percentile_score(s: float, c: float, ref_data: list[dict]) -> float:

    """
    Calculates the fit score for a given 's' and a fixed 'c' using the percentiling method.

    The goal is to find parameters (s, c) that make the model's predicted move
    probabilities best match the moves actually played in the reference data.
    This score is the sum of squared deviations between the model's predicted
    cumulative distribution and the ideal uniform distribution (R_qsc ≈ q).
    A lower score means a better fit.
    """
    if s <= 0: return float("inf")
    total_score = 0.0
    
    # Iterate over a range of percentiles (q = 0.00, 0.05, ..., 1.00)
    for q in PERCENTILES:
        up_sum = 0.0
        for pos in ref_data:
            probs = compute_move_probabilities(pos["spread"], s, c)


            played_index = pos["played_index"]

            # Calculate the cumulative probability mass *before* the played move
            p_minus = sum(probs[:played_index])
            p_played = probs[played_index]
            # Calculate the cumulative probability mass *including* the played move
            p_plus = p_minus + p_played

            # This is the core of the "percentiling" logic. It checks where the
            # played move falls relative to the q-th percentile of the
            # predicted probability distribution.
            if p_plus <= q:
                # The entire probability of the played move is below the percentile q.
                up_sum += 1.0
            elif p_minus < q < p_plus and p_played > 0:
                # The played move's probability "straddles" the percentile q.
                # We use fractional counting to get a smooth objective function.
                # This measures what fraction of the move's probability mass is below q.
                up_sum += abs(q - p_minus) / p_played
        
        # R_qsc is the observed fraction of moves that fall below percentile q.
        # If the model is a perfect fit, R_qsc should be equal to q.
        R_qsc = up_sum / len(ref_data) if ref_data else 0.0
        
        # The score is the sum of squared errors. We want to minimize this.
        total_score += (R_qsc - q) ** 2
        
    return total_score

def optimize_s_for_fixed_c(c_fixed: float, ref_data: list[dict], verbose: bool) -> float:
    """Finds the best 's' for a fixed 'c' using the reference dataset."""
    if verbose:
        print(f"  Optimizing 's' for fixed c = {c_fixed:.4f}...")
    
    objective = partial(calculate_percentile_score, c=c_fixed, ref_data=ref_data)
    
    # Use a bounded 1D minimizer
    result = minimize(
        objective,
        x0=[0.1],  # Initial guess for s
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "disp": False},
    )
    
    s_optimized = result.x[0]
    if verbose:
        print(f"  -> Found best s = {s_optimized:.6f} (score: {result.fun:.6f})")
    return s_optimized

def calculate_aee(ref_spreads_only: list[list[float]], s: float, c: float) -> float:








    """
    Calculates the Average Expected Error (AEe) for a given s and c.
    
    AEe is the model's prediction for the average error (delta) a player with
    skill (s, c) would make on the positions in the reference set.
    Formula: AEe = (1/T) * Σ_t Σ_{i≥1} [ p_{i,t} * δ_{i,t} ]
    """
    total_expected_error = 0.0
    num_positions = len(ref_spreads_only)

    if num_positions == 0:
        return 0.0

    # For each position in the reference set...
    for spread in ref_spreads_only:
        probabilities = compute_move_probabilities(spread, s, c)
        
        # Calculate the expected error for this single position.
        # This is the sum of (probability * error) for all non-best moves.
        # We skip the best move (i=0) because its delta is always 0.
        expected_error_for_position = sum(
            p * d for p, d in zip(probabilities[1:], spread[1:]) if d != float("inf")
        )
        total_expected_error += expected_error_for_position

    # Average the expected error over all positions.
    return total_expected_error / num_positions


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Calibrate IPR vs. AEe using the 'central artery' method.")
    parser.add_argument("sc_results_csv", type=str, help="Path to the CSV from Step 1 (s, c fits).")
    parser.add_argument("reference_pgn", type=str, help="Path to the reference (Solitaire) PGN file.")
    parser.add_argument("--engine_path", type=str, default=str(get_default_engine_path()), help="Path to Stockfish.")
    parser.add_argument("--depth", type=int, default=12, help="Engine search depth.")
    parser.add_argument("--multipv", type=int, default=20, help="Engine MultiPV setting.")
    parser.add_argument("--timeout", type=float, default=0.5, help="Engine time limit per move.")
    parser.add_argument("--use_cache", action="store_true", help="Use/create pickle cache for reference PGN analysis.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    # --- Load Step 1 Data ---
    try:
        sc_df = pd.read_csv(args.sc_results_csv)
        sc_df["AvgElo"] = pd.to_numeric(sc_df["AvgElo"])
        sc_df["c"] = pd.to_numeric(sc_df["c"])
    except (FileNotFoundError, KeyError, ValueError) as e:
        sys.exit(f"Error loading or parsing CSV file: {e}")

    # --- STAGE 1: Fit 'c' vs. Elo ("Central Artery") ---
    # The first step is to establish a general relationship between player skill (Elo)
    # and the 'c' (consistency) parameter. We use the results from the Step 1 script
    # to fit a linear model: c = m * Elo + b. This trend line is called the
    # "central artery" and provides a smoothed, generalized value for 'c' at any
    # given Elo, removing noise from individual training sets.
    X_elo = sc_df[["AvgElo"]]
    y_c = sc_df["c"]
    c_fitter = LinearRegression().fit(X_elo, y_c)
    c_slope, c_intercept = c_fitter.coef_[0], c_fitter.intercept_
    print("="*60, "\nSTAGE 1: 'Central Artery' Fit for c vs. Elo", "\n" + "="*60)
    print(f"Fit successful: c = {c_slope:.6f} * Elo + {c_intercept:.6f}")
    sc_df["c_smoothed"] = c_fitter.predict(X_elo)

    # --- Build or Load Reference Dataset ---
    # Now, we process the high-quality "Solitaire" or "Reference" PGN file.
    # This dataset serves two purposes:
    # 1. It provides the ground truth (played moves) for re-optimizing 's' in Stage 2.
    # 2. It provides the set of positions (spreads) for calculating AEe in Stage 3.
    cache_path = Path(args.reference_pgn).with_suffix(".reference_optimization_cache.pkl")
    if args.use_cache and cache_path.exists():
        if args.verbose: print(f"\nLoading cached reference data from {cache_path}...")
        with open(cache_path, "rb") as f: ref_data_for_s_opt = pickle.load(f)
    else:
        # The new build_reference_dataset manages its own engines and parallel processing.
        # We no longer create a single engine in the main thread.
        ref_data_for_s_opt = build_reference_dataset(
            args.reference_pgn, args.engine_path, args.depth, args.multipv, args.timeout, args.verbose
        )
        if args.use_cache:
            if args.verbose: print(f"Saving reference data cache to {cache_path}...")
            with open(cache_path, "wb") as f: pickle.dump(ref_data_for_s_opt, f)
    
    if not ref_data_for_s_opt:
        sys.exit("No reference positions could be analyzed. Exiting.")

    # --- STAGE 2: Re-Optimize 's' for each Elo level ---
    # With a smoothed 'c' value for each Elo level from the "central artery",
    # we now find the best 's' (sensitivity) parameter that explains the move
    # choices made in the reference PGN. This is a crucial step: instead of
    # using the 's' from the original noisy training sets, we are tuning 's'
    # against a single, high-quality standard (the reference games).
    print("\n" + "="*60, "\nSTAGE 2: Re-optimizing 's' using Reference Set", "\n" + "="*60)
    s_optimized_values = [
        optimize_s_for_fixed_c(row["c_smoothed"], ref_data_for_s_opt, args.verbose)
        for _, row in sc_df.iterrows()
    ]
    sc_df["s_optimized"] = s_optimized_values

    # --- STAGE 3: Calculate AEe with new (s, c) pairs ---
    # Now that we have a refined (s_optimized, c_smoothed) pair for each Elo
    # level, we can calculate the final Average Expected Error (AEe). This AEe
    # value represents the theoretical average error a player of that skill
    # level would produce on the reference set of positions.
    print("\n" + "="*60, "\nSTAGE 3: Calculating AEe", "\n" + "="*60)
    ref_spreads_only = [pos["spread"] for pos in ref_data_for_s_opt]
    aee_values = [
        calculate_aee(ref_spreads_only, row["s_optimized"], row["c_smoothed"])
        for _, row in sc_df.iterrows()
    ]
    sc_df["AEe_calculated"] = aee_values
    if args.verbose:
        print("Calculated AEe values:")
        print(sc_df[["AvgElo", "s_optimized", "c_smoothed", "AEe_calculated"]])

    # --- STAGE 4: Final IPR Calibration ---
    # The final step is to create a mapping from the abstract AEe metric to the
    # familiar Elo rating scale. We perform another linear regression, this time
    # fitting Elo against AEe. The resulting formula, IPR = intercept - slope * AEe,
    # is the main output of this entire calibration process. It can be used to
    # convert an AEe score from any player on any set of games into an Intrinsic
    # Performance Rating.
    X_aee = sc_df[["AEe_calculated"]]
    y_elo = sc_df["AvgElo"]
    ipr_fitter = LinearRegression().fit(X_aee, y_elo)
    ipr_slope, ipr_intercept = ipr_fitter.coef_[0], ipr_fitter.intercept_
    print("\n" + "="*60, "\nSTAGE 4: Final IPR vs. AEe Calibration", "\n" + "="*60)
    print("Fit successful. The final IPR formula is:")
    print(f"IPR = {ipr_intercept:.4f} + ({ipr_slope:.4f}) * AEe\n")

    # --- Write Final CSV ---
    # The output CSV saves all the intermediate and final results for analysis.
    output_path = Path(f"{Path(args.sc_results_csv).stem}_IPR-AE_oss-fit.csv")
    output_df = sc_df.rename(columns={"c": "c_original_fit", "s": "s_original_fit"})
    output_df.to_csv(output_path, index=False, float_format="%.8f")
    print(f"Successfully saved detailed results to {output_path}")

if __name__ == "__main__":
    main()
