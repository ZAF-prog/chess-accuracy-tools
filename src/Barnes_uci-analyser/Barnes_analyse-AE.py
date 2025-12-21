#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script provides a functional implementation for detecting potential cheating in chess
by analysing a game record (PGN) with a UCI-compliant chess engine. It calculates
several key metrics as described by David J. Barnes and Julio Hernandez-Castro.

This implementation is based on the work of David J. Barnes and Julio Hernandez-Castro,
as described in their paper: "Detecting computer-assisted cheating in chess"
<http://dx.doi.org/10.1016/j.cose.2014.10.002>

The original C++ source code from the authors can be found at:
<http://www.cs.kent.ac.uk/~djb/chessplag/>

This Python script was significantly reworked to be a complete, functional tool
using the 'python-chess' library for engine communication and game processing.
"""

import chess
import chess.pgn
import chess.engine
import platform
from pathlib import Path
from typing import List, Tuple, Optional

# =============================================================================
# ENGINE SETUP
# =============================================================================

def get_default_engine_path() -> Path:
    """
    Determines the default Stockfish engine path based on the operating system.
    Users should modify this function to point to their specific engine location.
    """
    system = platform.system()
    if system == "Windows":
        # On Windows, you might need to provide a full path to the .exe
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    elif system in ("Linux", "Darwin"):
        # On Linux and macOS, if 'stockfish' is in the system's PATH, this is sufficient.
        # Otherwise, provide a full path, e.g., Path("/usr/games/stockfish").
        return Path("stockfish")
    # Fallback for other systems
    return Path("stockfish")

# =============================================================================
# PLAYER NAME STANDARDIZATION
# =============================================================================

def standardize_player_names(game: chess.pgn.Game, name_map: dict) -> Tuple[str, str]:
    """
    Standardizes player names to handle variations (e.g., "Carlsen" vs. "Carlsen, Magnus").

    It maintains a map of known names. When a new name is encountered, it checks if it's
    a shorter version of an existing name or vice-versa. The longest variant is
    kept as the canonical name.

    Args:
        game: The game object with "White" and "Black" headers.
        name_map: A dictionary mapping known name variants to a canonical name.

    Returns:
        A tuple containing the standardized names for White and Black.
    """
    white_name = game.headers.get("White", "Unknown")
    black_name = game.headers.get("Black", "Unknown")
    
    standardized_white = white_name
    standardized_black = black_name

    # Helper to update the map
    def update_map(name):
        canonical_name = name
        found = False
        # Check if the new name is a substring of an existing key or vice-versa
        for existing_name in list(name_map.keys()):
            if name in existing_name:
                # The new name is shorter, use the existing canonical name
                canonical_name = name_map[existing_name]
                found = True
                break
            elif existing_name in name:
                # The new name is longer, it becomes the new canonical name
                canonical_name = name
                # Update all previous entries that pointed to the old name
                old_canonical = name_map[existing_name]
                for k, v in name_map.items():
                    if v == old_canonical:
                        name_map[k] = canonical_name
                found = True
                break
        
        if not found:
            name_map[name] = canonical_name
        return canonical_name

    standardized_white = update_map(white_name)
    standardized_black = update_map(black_name)

    return standardized_white, standardized_black


# =============================================================================
# CORE ANALYSIS LOGIC
# =============================================================================

def analyse_game(game: chess.pgn.Game, engine: chess.engine.SimpleEngine, analysis_depth: int, book_moves: int) -> Tuple[List[float], List[float], List[str], List[str]]:
    """
    Analyzes a single chess game move by move.

    For each position past the opening book, it retrieves the engine's evaluation
    for the best move and for the move the player actually made.

    Args:
        game: The python-chess game object to analyze.
        engine: The UCI engine instance to perform the analysis.
        analysis_depth: The search depth for the engine analysis.
        book_moves: The number of opening moves (plies) to skip.

    Returns:
        A tuple containing four lists:
        - engine_evals_cp: Centipawn evaluations of the engine's best move.
        - player_evals_cp: Centipawn evaluations of the player's chosen move.
        - engine_moves_uci: The engine's best move in UCI format.
        - player_moves_uci: The player's chosen move in UCI format.
    """
    engine_evals_cp = []
    player_evals_cp = []
    engine_moves_uci = []
    player_moves_uci = []
    
    board = game.board()
    
    for i, move in enumerate(game.mainline_moves()):
        # Skip the defined number of opening "book" moves
        if i < book_moves:
            board.push(move)
            continue

                                                                # Analyze the position *before* the player's move is made
        try:
            # We ask for the top 5 moves to see if the player's move was one of them
            info = engine.analyse(board, chess.engine.Limit(depth=analysis_depth), multipv=5)
        except chess.engine.EngineTerminatedError:
            print("Engine terminated unexpectedly. Aborting analysis.")
            break

        # The engine's best move and its evaluation
        best_engine_move = info[0]['pv'][0]
        engine_score = info[0]['score'].white()

        # Find the evaluation for the move the player actually made
        player_score = None
        for variation in info:
            if variation['pv'][0] == move:
                player_score = variation['score'].white()
                break
        
                # If the player's move was not in the top 5, we must analyze it specifically
        if player_score is None:
            try:
                # To get the score for a specific move, we use the `root_moves` parameter
                specific_info = engine.analyse(board, chess.engine.Limit(depth=analysis_depth), root_moves=[move])
                if not specific_info:
                    print(f"Could not get analysis for specific move: {move.uci()}. Skipping.")
                    board.push(move)
                    continue
                player_score = specific_info[0]['score'].white()
            except (chess.engine.EngineTerminatedError, IndexError):
                print(f"Could not analyze player's move: {move.uci()}. Skipping.")
                board.push(move)
                continue
            except KeyError:
                print(f"Could not get score for move: {move.uci()}. Skipping.")
                board.push(move)
                continue

        # The paper's method for Average Error excludes positions with mate scores.
        # We will only include centipawn evaluations for the AE and CV calculations.
        if not engine_score.is_mate() and not player_score.is_mate():
            engine_evals_cp.append(engine_score.score())
            player_evals_cp.append(player_score.score())

        # For Move Matching, we store the actual moves
        engine_moves_uci.append(best_engine_move.uci())
        player_moves_uci.append(move.uci())

        # Apply the move to the board to proceed to the next position
        board.push(move)
        
    return engine_evals_cp, player_evals_cp, engine_moves_uci, player_moves_uci

# =============================================================================
# METRIC CALCULATIONS
# =============================================================================

def calculate_coincidence_value(engine_evals: List[float], player_evals: List[float]) -> float:
    """
    Calculates the Coincidence Value (CV).
    This is the proportion of non-book moves where the player's move had the
    same evaluation as the engine's preferred move.
    """
    total_non_book_moves = len(player_evals)
    if total_non_book_moves == 0:
        return 0.0

    # A small tolerance can be added for floating point comparisons, but for
    # engine analysis at the same depth, scores should be identical.
    same_evaluation_count = sum(1 for eng_eval, ply_eval in zip(engine_evals, player_evals) if eng_eval == ply_eval)
    
    cv = (same_evaluation_count / total_non_book_moves)
    return cv

def calculate_move_matching_percentage(engine_moves: List[str], player_moves: List[str]) -> float:
    """
    Calculates the Move Matching Percentage (MM).
    This is the percentage of moves where the player's move was identical
    to the engine's top choice.
    """
    total_non_book_moves = len(player_moves)
    if total_non_book_moves == 0:
        return 0.0

    exact_match_count = sum(1 for engine_move, player_move in zip(engine_moves, player_moves) if engine_move == player_move)
    
    mm_percentage = (exact_match_count / total_non_book_moves) * 100
    return mm_percentage

def calculate_average_error(engine_evals: List[float], player_evals: List[float]) -> float:
    """
    Calculates the Average Error (AE) in centipawns.
    This is the mean difference between the evaluation of the best move and the
    evaluation of the player's move. The value should be non-positive.
    """
    total_non_book_moves = len(player_evals)
    if total_non_book_moves == 0:
        return 0.0

    # Formula: sum(best_move_eval - played_move_eval).
    # Since we are using White's perspective, a drop for Black is a positive number,
    # so we must account for this. However, `python-chess` `.white()` score perspective
    # handles this automatically. A score of +100 is good for white, +50 is less good.
    # The difference is 100 - 50 = 50. This represents the error.
    error_sum = sum((engine_eval - player_eval) for engine_eval, player_eval in zip(engine_evals, player_evals))
    
    ae_centipawns = error_sum / total_non_book_moves
    return ae_centipawns

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the analysis.
    """
    # --- Configuration ---
    # Path to the PGN file you want to analyze.
    pgn_file_path = Path(r"C:\Users\Public\Github\chess-accuracy-tools\src\Barnes_uci-analyser\example.pgn")
    # Path to your UCI chess engine executable.
    engine_path = get_default_engine_path()
    # Analysis depth in plies. Higher values are more accurate but much slower.
    ANALYSIS_DEPTH = 15
    # Number of moves (plies) to skip at the start of the game (opening book).
    BOOK_MOVES_TO_SKIP = 8 # Corresponds to 4 full moves

    print("--- Chess Accuracy Analysis ---")
    print(f"PGN File: {pgn_file_path}")
    print(f"Engine: {engine_path}")
    print(f"Analysis Depth: {ANALYSIS_DEPTH}")
    print(f"Skipping first {BOOK_MOVES_TO_SKIP} plies (book moves).")
    print("-" * 30)

    if not pgn_file_path.exists():
        print(f"Error: PGN file not found at '{pgn_file_path}'")
        return

    if not engine_path.exists():
        print(f"Error: Engine not found at '{engine_path}'")
        print("Please check the path in the get_default_engine_path() function.")
        return

    player_name_map = {}
    game_count = 0

    try:
        # Start the chess engine process once
        with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
            print("Engine started successfully.")
            
            with open(pgn_file_path) as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    
                    game_count += 1
                    white_name, black_name = standardize_player_names(game, player_name_map)
                    print(f"\n--- Analyzing Game {game_count}: {white_name} vs {black_name} ---")

                    # --- Run Analysis ---
                    engine_evals, player_evals, engine_moves, player_moves = analyse_game(
                        game, engine, ANALYSIS_DEPTH, BOOK_MOVES_TO_SKIP
                    )

                    if not player_moves:
                        print("No non-book moves were analyzed. Cannot calculate metrics.")
                        continue

                    # --- Calculate Metrics ---
                    cv = calculate_coincidence_value(engine_evals, player_evals)
                    mm_percentage = calculate_move_matching_percentage(engine_moves, player_moves)
                    ae_centipawns = calculate_average_error(engine_evals, player_evals)

                    # --- Print Results ---
                    print(f"Total non-book moves analyzed: {len(player_moves)}")
                    print(f"Coincidence Value (CV): {cv:.3f}")
                    print(f"Move Matching (MM): {mm_percentage:.2f}%")
                    print(f"Average Error (AE): {ae_centipawns:.2f} centipawns")

            print("\n--- Standardization Map ---")
            for name, canonical in player_name_map.items():
                if name != canonical:
                    print(f"'{name}' -> '{canonical}'")
            print("-" * 30)


    except chess.engine.EngineTerminatedError:
        print("Engine process terminated unexpectedly.")
    except PermissionError:
        print(f"Error: Insufficient permissions to execute the engine at '{engine_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

