#!/usr/bin/env python
# written with Gemini 3.0 Thinking, then fixed
"""
 Dr. Kenneth Regan's Intrinsic Performance Rating (IPR) algorithm,
 approximated with OSS code 
"""
import chess
import chess.engine
import chess.pgn
import math
import sys

# --- Configuration ---
# --- CROSS-PLATFORM ENGINE PATH LOGIC ---
def get_engine_path():
    """
    Determines the correct Stockfish binary path based on the operating system.
    Paths are set according to user's specified configuration.
    """
    os_name = platform.system()
    
    if os_name == "Windows":
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    elif os_name in ("Linux", "Darwin"):
        return Path("binaries/stockfish-ubuntu-x86-64-avx512")
    else:
        print(f"Error: Unsupported operating system detected: {os_name}")
        sys.exit(1)

# STOCKFISH_PATH: Path to the executable. 
# You need to download Stockfish separately and point this to it.
STOCKFISH_PATH = get_engine_path()
print(f"Using Stockfish engine path: {ENGINE_PATH}")

# ANALYSIS_TIME: Dr. Regan uses fixed depth (e.g., depth=13) rather than time 
# for consistency, but time is easier for casual testing. 
# 0.1s is fast; for real accuracy, use at least 0.5s or depth=15.
ANALYSIS_TIME = 0.1  

# BOOK_MOVES: We skip the first 8 moves (16 ply) because playing known theory 
# is memorization, not calculation skill.
BOOK_MOVES = 8       

# CAP_EVAL: "Garbage Time" filter.
# If a player is winning by > 3 pawns (300 cp), the engine might find a "mate in 10" 
# while the human plays a safe "mate in 20". This huge numeric difference 
# isn't a "blunder" in human terms, so Regan excludes these positions entirely.
CAP_EVAL = 300       

def cp_to_win_probability(cp):
    """
    Converts Centipawn (cp) evaluation to Win Probability (0.0 to 1.0).
    
    Why? 
    A 50 centipawn loss in an equal position (0.00 -> -0.50) is a disaster.
    A 50 centipawn loss in a winning position (+5.00 -> +4.50) is irrelevant.
    
    Using Win Probability normalizes this. Losing 50cp at +5.00 changes 
    Win% from 99.9% to 99.8% (tiny error), whereas at 0.00 it drops 
    to 35% (huge error).
    """
    if cp is None: return 0.5 
    
    # The Logistic Function (Sigmoid curve).
    # 400 is a standard scaling factor used in Elo calculations.
    # Some models use different constants, but the shape is what matters.
    return 1 / (1 + 10 ** (-cp / 400.0))

def calculate_ipr(pgn_file_path):
    print(f"Analyzing {pgn_file_path}...")
    
    # Initialize Engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except FileNotFoundError:
        print(f"Error: Stockfish not found at {STOCKFISH_PATH}")
        return

    with open(pgn_file_path) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None: break
            
            board = game.board()
            errors = []
            
            print(f"\nGame: {game.headers.get('White', '?')} vs {game.headers.get('Black', '?')}")
            
            move_count = 0
            for move in game.mainline_moves():
                move_count += 1
                
                # --- FILTER 1: Skip Opening Theory ---
                if move_count <= BOOK_MOVES * 2:
                    board.push(move)
                    continue

                # --- STEP 1: Establish the "Truth" (Best Move) ---
                # We ask the engine: "What is the absolute best evaluation possible here?"
                info_best = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
                
                # We use a cap for Mate scores (+/- 10000) so they don't break the math.
                score_best = info_best["score"].white().score(mate_score=10000)
                
                # --- FILTER 2: Cap Evaluation (Garbage Time) ---
                # If one side is already crushing the other (>3.00 pawns), 
                # we stop tracking errors because "good enough" moves are acceptable.
                if abs(score_best) > CAP_EVAL:
                    board.push(move)
                    continue

                # --- STEP 2: Evaluate the Human's Actual Move ---
                # We assume the move is played, then ask the engine "How good is this new position?"
                board.push(move)
                info_played = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
                
                # Important: Engine gives score for side-to-move. 
                # If White moved, now it's Black's turn, so the engine gives Black's advantage.
                # We negate it to get the score relative to White.
                score_played = -info_played["score"].white().score(mate_score=10000)

                # --- STEP 3: Normalize to Win Probability ---
                # We need to see the move from the perspective of the player who just moved.
                if board.turn == chess.BLACK: # White just moved
                    wp_best = cp_to_win_probability(score_best)
                    wp_played = cp_to_win_probability(score_played)
                else: # Black just moved
                    # Flip scores because -1.00 is good for Black.
                    wp_best = cp_to_win_probability(-score_best)
                    wp_played = cp_to_win_probability(-score_played)
                
                # --- STEP 4: Calculate the "Cost" of the move ---
                # The error is simply: (Win% chance if I played perfectly) - (Win% chance after my actual move)
                # Ideally, this is 0.0. 
                delta = max(0.0, wp_best - wp_played)
                errors.append(delta)

            # --- STEP 5: Aggregate and Convert to IPR ---
            if not errors:
                print("Not enough analyzable moves (game too short or too unbalanced).")
                continue

            # Average Error (AE): The average win probability lost per move.
            average_error = sum(errors) / len(errors)
            
            # --- THE MAGIC FORMULA ---
            # Derived from linear regression on thousands of games in Regan's dataset.
            # 3571 represents the theoretical rating of a player with 0.0 error (Engine).
            # 15413 is the slope: for every 1% (0.01) of win probability you lose, 
            # your rating drops by ~154 points.
            ipr = 3571 - (15413 * average_error)
            
            print(f"Analyzed Moves: {len(errors)}")
            print(f"Average Error (Win%): {average_error:.4f}")
            print(f"Estimated IPR: {int(ipr)}")

    engine.quit()
