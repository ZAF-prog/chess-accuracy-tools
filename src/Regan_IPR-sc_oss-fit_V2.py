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

from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
from functools import partial
import multiprocessing

# =============================================================================
# CONSTANTS
# =============================================================================

MATE_SCORE = 10000
MOVE_START_DEFAULT = 8
PAWN_CUTOFF_DEFAULT = 300
DEFAULT_YEAR = 1970
MIN_ELO_DEFAULT = 0
COARSE_S_RANGE = [s / 10.0 for s in range(1, 8)]  # 0.1 to 0.7
COARSE_C_RANGE = [c / 10.0 for c in range(10, 51, 5)]  # 1.0 to 5.0
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
# PHASE 1: DATA COLLECTION AND PREPROCESSING
# =============================================================================

def analyze_game_with_engine(
    game: chess.pgn.Game,
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
    timeout: float,
    verbose: bool,
    move_start: int = MOVE_START_DEFAULT,
    pawn_cutoff: int = PAWN_CUTOFF_DEFAULT,
) -> list[dict]:
    """
    Analyzes a single game with Stockfish and returns a list of dictionaries,
    one for each analyzed position.

    Args:
        game: The chess game to analyze.
        engine: The chess engine to use for analysis.
        depth: The search depth for the engine.
        multipv: The number of principal variations to consider.
        timeout: The time limit for the engine per move (in seconds).
        verbose: If True, print progress messages.
        move_start: The move number to start analysis from.
        pawn_cutoff: Threshold for centipawn evaluation to skip tactical positions.

    Returns:
        A list of dictionaries, where each dictionary represents an analyzed position
        and contains 'move_evals' and 'move_played'.
    """
    positions = []
    board = game.board()

    for move_number, move in enumerate(game.mainline_moves()):
        if move_number < move_start:
            board.push(move)
            continue

        if verbose and move_number > move_start and move_number % 5 == 0:
            print(f"    ... analyzing move {move_number}")

        try:
            # Limit: depth plus a time cap (timeout is in seconds)
            limit = chess.engine.Limit(depth=depth, time=timeout)
            analysis = engine.analyse(board, limit, multipv=multipv)

            # python-chess returns a list (length multipv) when multipv > 1,
            # or a single dict when multipv == 1. Normalize to list.
            if not analysis:
                if verbose:
                    print(f"Skipping position at move {move_number}: No analysis returned.")
                board.push(move)
                continue

            if isinstance(analysis, dict):
                analysis = [analysis]

            best_eval_cp = analysis[0]["score"].pov(board.turn).score(mate_score=MATE_SCORE)
            if best_eval_cp is None:
                board.push(move)
                continue

            # Skip tactically wild positions or trivial repetitions
            if abs(best_eval_cp) > pawn_cutoff or board.is_repetition(2):
                board.push(move)
                continue

            move_evals = {}
            for info in analysis:
                if "pv" not in info or not info["pv"]:
                    continue
                this_move = info["pv"][0]
                score_cp = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
                if score_cp is not None:
                    move_evals[this_move] = score_cp

            if not move_evals:
                board.push(move)
                continue

            positions.append({
                "move_evals": move_evals,
                "move_played": move,
            })

        except (chess.engine.EngineError, chess.engine.TimeoutError, IndexError) as e:
            if verbose:
                print(f"Skipping position at move {move_number} due to error: {e}")

        board.push(move)

    return positions


# =============================================================================
# SPREAD COMPUTATION
# =============================================================================


def _integrand(z):
    return 1.0 / (1.0 + abs(z))


def compute_delta(v0: float, vi: float) -> float:
    """
    Computes the spread delta between two evaluations, v0 and vi, in centipawns.

    Args:
        v0: The evaluation of the best move.
        vi: The evaluation of the move to compare against.

    Returns:
        The computed spread delta.
    """
    v0_pawns, vi_pawns = v0 / 100.0, vi / 100.0
    # Same sign or zero: log-based distance
    if v0_pawns * vi_pawns >= 0:
        return abs(log(1 + abs(v0_pawns)) - log(1 + abs(vi_pawns)))
    # Opposite sign: integrate 1/(1+|z|) from vi to v0
    else:
        delta, _ = quad(_integrand, vi_pawns, v0_pawns)
        return delta


def create_spread_vector(position_data: dict, max_moves: int) -> dict | None:
    """
    Creates a spread vector and identifies the index of the move played for a given position.

    Args:
        position_data: A dictionary containing 'move_evals' and 'move_played'.
        max_moves: The maximum number of moves to include in the spread vector.

    Returns:
        A dictionary with 'spread' and 'played_index', or None if the position
        is not usable.
    """
    move_evals = position_data["move_evals"]
    sorted_moves = sorted(move_evals.items(), key=lambda x: x[1], reverse=True)

    if not sorted_moves:
        return None

    best_move, v0 = sorted_moves[0]

    spread = []
    move_to_index = {}

    for idx, (move, vi) in enumerate(sorted_moves[:max_moves]):
        spread.append(compute_delta(v0, vi))
        move_to_index[move] = idx

    # Pad with +inf to reach max_moves length
    while len(spread) < max_moves:
        spread.append(float("inf"))

    played_index = move_to_index.get(position_data["move_played"], -1)
    if played_index == -1:
        return None

    return {"spread": spread, "played_index": played_index}


# =============================================================================
# DATASET BUILDING
# =============================================================================


def build_training_dataset(
    pgn_path: str,
    engine: chess.engine.SimpleEngine,
    max_games: int | None,
    depth: int,
    multipv: int,
    timeout: float,
    verbose: bool,
) -> tuple[list[dict], dict]:
    """
    Analyzes games from a PGN file to build a training dataset without ELO filtering.

    Args:
        pgn_path: The path to the PGN file.
        engine: The chess engine to use for analysis.
        max_games: The maximum number of games to process.
        depth: The search depth for the engine.
        multipv: The number of principal variations to consider.
        timeout: The time limit for the engine per move (in seconds).
        verbose: If True, print progress messages.

    Returns:
        A tuple containing the training data and statistics about the processed games.
    """
    print("Processing all valid games from PGN file (no ELO filter applied).")

    training_data = []
    stats = {"num_games": 0, "num_moves": 0, "elos": [], "years": []}

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while max_games is None or stats["num_games"] < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
            except (ValueError, AttributeError):
                continue

            date_str = game.headers.get("Date", f"{DEFAULT_YEAR}.01.01")
            try:
                year = int(date_str.split(".")[0])
            except (ValueError, IndexError):
                year = DEFAULT_YEAR

            if white_elo == MIN_ELO_DEFAULT or black_elo == MIN_ELO_DEFAULT:
                continue

            if verbose:
                print(
                    f"Processing game {stats['num_games'] + 1} "
                    f"({game.headers.get('White', '?')} vs {game.headers.get('Black', '?')})..."
                )

            positions = analyze_game_with_engine(
                game=game,
                engine=engine,
                depth=depth,
                multipv=multipv,
                timeout=timeout,
                verbose=verbose,
            )

            if positions:
                stats["num_games"] += 1
                stats["elos"].extend([white_elo, black_elo])
                stats["years"].append(year)

                for pos_data in positions:
                    spread_data = create_spread_vector(pos_data, multipv)
                    if spread_data is not None:
                        training_data.append(spread_data)

    stats["num_moves"] = len(training_data)
    print(
        f"Created {stats['num_moves']} training positions from "
        f"{stats['num_games']} games."
    )
    return training_data, stats


# =============================================================================
# PROBABILITY MODEL AND LOSS
# =============================================================================


def compute_move_probabilities(spread: list[float], s: float, c: float) -> list[float]:
    """
    Computes the probability for each move given a spread vector and parameters s and c.

    Args:
        spread: A list of spread deltas for each move.
        s: The scaling parameter.
        c: The shape parameter.

    Returns:
        A list of probabilities for each move.
    """
    shares = [
        exp(-((delta / s) ** c)) if delta != float("inf") else 0.0
        for delta in spread
    ]
    total_shares = sum(shares)

    if total_shares == 0:
        # If all zero, distribute uniformly over finite entries
        num_valid_moves = sum(1 for d in spread if d != float("inf"))
        if num_valid_moves == 0:
            return [0.0] * len(spread)
        return [1.0 / num_valid_moves if d != float("inf") else 0.0 for d in spread]

    return [share / total_shares for share in shares]


# =============================================================================
# SCORING / OPTIMIZATION
# =============================================================================


def calculate_score_for_sc_pair(
    s: float, c: float, training_data: list[dict], percentiles: list[float]
) -> float:
    """
    Calculates the fit score for a single pair of (s, c) parameters.

    Args:
        s: The scaling parameter.
        c: The shape parameter.
        training_data: The dataset of analyzed positions.
        percentiles: A list of percentiles to use for scoring.

    Returns:
        The calculated fit score.
    """
    if s <= 0 or c <= 0:
        return float("inf")

    total_score = 0.0

    for q in percentiles:
        up_sum = 0.0

        for pos in training_data:
            probs = compute_move_probabilities(pos["spread"], s, c)
            played_index = pos["played_index"]

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


def fit_parameters_hybrid(
    training_data: list[dict], verbose: bool = True
) -> tuple[float, float]:
    """
    Finds the best-fit (s, c) parameters using a hybrid optimization approach.

    This method combines a coarse grid search with a local refinement (Nelder-Mead)
    to efficiently find the optimal parameters.

    Args:
        training_data: The dataset of analyzed positions.
        verbose: If True, print progress and results of the optimization.

    Returns:
        A tuple containing the best-fit s and c parameters.
    """
    if verbose:
        print("Starting coarse grid search...")

    s_initial_guess, c_initial_guess = 0.1, 1.0  # Fallback values
    coarse_best_score = float("inf")
    sc_pairs = [(s, c) for s in COARSE_S_RANGE for c in COARSE_C_RANGE]

    func = partial(
        calculate_score_for_sc_pair,
        training_data=training_data,
        percentiles=PERCENTILES,
    )

    # Use all available CPU cores (you may adjust this policy if desired)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(func, sc[0], sc[1]): sc for sc in sc_pairs}
        for future in futures:
            sc = futures[future]
            score = future.result()
            if score < coarse_best_score:
                coarse_best_score = score
                s_initial_guess, c_initial_guess = sc

    if verbose:
        print(
            f"Coarse search best guess: s={s_initial_guess:.3f}, "
            f"c={c_initial_guess:.2f}, score={coarse_best_score:.6f}"
        )
        print("Starting local refinement with Nelder-Mead...")

    def objective_function(params):
        return calculate_score_for_sc_pair(
            params[0], params[1], training_data, PERCENTILES
        )

    result = minimize(
        objective_function,
        [s_initial_guess, c_initial_guess],
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "disp": verbose},
    )

    refined_s, refined_c = result.x

    if verbose:
        print(
            f"Refinement complete. Final best: s={refined_s:.4f}, "
            f"c={refined_c:.3f}, score={result.fun:.6f}"
        )

    return refined_s, refined_c


# =============================================================================
# CSV SUMMARY OUTPUT
# =============================================================================


def save_summary_csv(data: dict, path: Path):
    """
    Saves summary data to a CSV file, appending if the file already exists.

    Args:
        data: A dictionary containing the summary data.
        path: The path to the output CSV file.
    """
    file_exists = path.is_file()

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data.keys()))
        if not file_exists:
            writer.writeheader()  # Write header only if file is new
        writer.writerow(data)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fit s and c parameters from a PGN training set."
    )
    parser.add_argument("pgn_path", type=str, help="Path to the PGN training file.")
    parser.add_argument(
        "--engine_path",
        type=str,
        default=str(get_default_engine_path()),
        help="Path to the Stockfish executable.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=12,
        help="Search depth for engine analysis.",
    )
    parser.add_argument(
        "--multipv",
        type=int,
        default=20,
        help="Number of moves to consider (MultiPV). Also used as max_moves.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.5,
        help="Per-position time limit (seconds) for engine analysis.",
    )
    parser.add_argument(
        "--max_games",
        type=int,
        default=None,
        help="Maximum number of games to process (None = all).",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use / create a pickle cache for training data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Custom name for the output CSV file (without extension).",
    )

    args = parser.parse_args()

    pgn_path = Path(args.pgn_path)
    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    # Cache path
    cache_path = pgn_path.with_suffix(".training_cache.pkl")

    # -----------------------------------------------------------------
    # Load / compute training data
    # -----------------------------------------------------------------
    if args.use_cache and cache_path.exists():
        if args.verbose:
            print(f"Loading cached training data from {cache_path}...")
        with open(cache_path, "rb") as f:
            training_data, stats = pickle.load(f)
    else:
        if args.verbose:
            print(f"Starting engine {args.engine_path}...")
        engine = chess.engine.SimpleEngine.popen_uci([args.engine_path])

        try:
            training_data, stats = build_training_dataset(
                pgn_path=str(pgn_path),
                engine=engine,
                max_games=args.max_games,
                depth=args.depth,
                multipv=args.multipv,
                timeout=args.timeout,
                verbose=args.verbose,
            )
        finally:
            engine.quit()

        if args.use_cache:
            if args.verbose:
                print(f"Saving training data cache to {cache_path}...")
            with open(cache_path, "wb") as f:
                pickle.dump((training_data, stats), f)

    if not training_data:
        print("No training data collected. Exiting.")
        return

    # -----------------------------------------------------------------
    # Fit s and c
    # -----------------------------------------------------------------
    s_fit, c_fit = fit_parameters_hybrid(training_data, verbose=args.verbose)

    # -----------------------------------------------------------------
    # Prepare summary statistics
    # -----------------------------------------------------------------
    num_games = stats.get("num_games", 0)
    num_moves = stats.get("num_moves", 0)
    elos = stats.get("elos", [])
    years = stats.get("years", [])

    min_elo = min(elos) if elos else 0
    max_elo = max(elos) if elos else 0
    avg_elo = float(np.mean(elos)) if elos else 0.0

    first_year = min(years) if years else 0
    last_year = max(years) if years else 0

    # AE_e is not computed in this script; placeholder "N/A" kept.
    summary_data = {
        "filename": pgn_path.name,
        "MULTI_PV": args.multipv,
        "Number_Games": num_games,
        "Number_Moves": num_moves,
        "MinElo": min_elo,
        "MaxElo": max_elo,
        "AvgElo": avg_elo,
        "s": s_fit,
        "c": c_fit,
        "AE_e": "N/A",
        "FirstYear": first_year,
        "LastYear": last_year,
    }

    # -----------------------------------------------------------------
    # Write CSV summary
    # -----------------------------------------------------------------
    # Ensure the output directory exists
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if args.output_name:
        output_basename = Path(args.output_name)
    else:
        output_basename = Path(f"{pgn_path.stem}_IPR_s,c-fit")
    
    csv_path = output_dir / output_basename.with_suffix(".csv")

    save_summary_csv(summary_data, csv_path)
    print(f"Summary saved to {csv_path}")


if __name__ == "__main__":
    main()
