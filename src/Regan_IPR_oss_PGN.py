#!/usr/bin/env python
"""
Regan_IPR_oss_PGN.py
--------------------
Calculates Dr. Kenneth Regan's Intrinsic Performance Rating (IPR) 
using existing [%eval] annotations in a PGN file (instead of running an engine).
"""

import chess.pgn
import math
import sys
import argparse
import re
import csv
import random
import statistics
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
BOOK_MOVES = 8       # Skip first 8 moves (16 ply)
CAP_EVAL = 300       # Garbage time filter (censored if > 3.00 pawns)
IPR_INTERCEPT = 3571
IPR_SLOPE = 15413
MATE_SCORE = 10000

def cp_to_win_probability(cp):
    """Converts Centipawn (cp) evaluation to Win Probability (0.0 to 1.0)."""
    if cp is None: return 0.5 
    return 1 / (1 + 10 ** (-cp / 400.0))

def parse_eval(comment):
    """
    Extracts evaluation from a PGN comment string.
    Expected format: "[%eval 0.35]" or "[%eval #+3]"
    Returns: float (cp score, white centric) or None if not found.
    Handles #mate by converting to +/- MATE_SCORE.
    """
    if not comment:
        return None
        
    match = re.search(r'\[%eval\s+([-+]?\d*\.?\d+|#[-+]?\d+)\]', comment)
    if not match:
        return None
        
    val_str = match.group(1)
    
    if '#' in val_str:
        # Mate
        if '-' in val_str:
            return -MATE_SCORE
        else:
            return MATE_SCORE
    else:
        # Centipawns (usually formatted as 0.35 -> 35 cp? No, typically 0.35 means pawns)
        # CHECK: pgn_evaluator.py writes "f'{cp/100:.2f}'". So 0.35 means 35 cp.
        # But Regan logic often works on Centipawns (integers).
        # cp_to_win_probability expects CP (e.g. 50 for 0.5 pawns).
        # If val_str is "0.35", float(val_str) is 0.35. We need to multiply by 100.
        return float(val_str) * 100.0

def bootstrap_ipr(errors, num_samples=1000):
    """
    Bootstraps the IPR calculation to estimate Standard Deviation.
    """
    if len(errors) < 2:
        return 0.0
        
    ipr_samples = []
    
    for _ in range(num_samples):
        # Sample with replacement
        sample_errors = random.choices(errors, k=len(errors))
        avg_error = sum(sample_errors) / len(sample_errors)
        ipr = IPR_INTERCEPT - (IPR_SLOPE * avg_error)
        ipr_samples.append(ipr)
        
    return statistics.stdev(ipr_samples)

def ipr_from_errors(errors):
    if not errors:
        return 0
    avg_error = sum(errors) / len(errors)
    return IPR_INTERCEPT - (IPR_SLOPE * avg_error)

class PlayerStats:
    def __init__(self):
        self.errors = []
        self.elos = []
        self.games = 0
        self.years = []

def process_pgn(pgn_file):
    print(f"Processing {pgn_file}...")
    
    # Sanity Check: Read first bit of file to check for %eval
    with open(pgn_file, 'r', encoding='utf-8') as f:
        content_head = f.read(10000)
        if "[%eval" not in content_head:
             # It might be further down if games are long/headers long, but usually it appears early.
             # We will do a stronger check: parse the first game fully.
             pass

    players = defaultdict(PlayerStats)
    
    with open(pgn_file, 'r', encoding='utf-8') as pgn:
        # Check first game for evals as sanity check
        first_game = True
        
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None: break
            
            white = game.headers.get("White", "Unknown")
            black = game.headers.get("Black", "Unknown")
            date = game.headers.get("Date", "????")
            year = date.split('.')[0] if '.' in date else date
            
            # ELOS
            try:
                w_elo = int(game.headers.get("WhiteElo", "0"))
            except ValueError:
                w_elo = 0
            try:
                b_elo = int(game.headers.get("BlackElo", "0"))
            except ValueError:
                b_elo = 0
                
            # We need to traverse the game node by node
            node = game
            board = game.board()
            
            # Tracking
            ply_count = 0 
            curr_eval = None
            
            # To calculate error for a move, we need:
            # Eval BEFORE move (which is Eval AFTER previous move)
            # Eval AFTER move
            
            # Initialize "Previous Eval". 
            # For the very start of game, we don't usually have an eval unless root comment.
            # We assume None until we see one.
            prev_eval = None
            
            # Root comment?
            if node.comment:
                prev_eval = parse_eval(node.comment)
                
            has_evals = False
            
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                ply_count += 1
                
                # Get eval after this move
                raw_eval_score = parse_eval(next_node.comment)
                
                if raw_eval_score is not None:
                    has_evals = True
                    curr_eval = raw_eval_score
                else:
                    curr_eval = None
                
                # LOGIC:
                # We need prev_eval (Best) and curr_eval (Played).
                # Only calculate if we have BOTH and we are past BOOK_MOVES.
                
                if (ply_count > BOOK_MOVES * 2) and (prev_eval is not None) and (curr_eval is not None):
                    
                    # Filter: Garbage Time (Cap Eval) based on PREVIOUS position (Best)
                    # If the position was already won/lost > CAP_EVAL, skip.
                    if abs(prev_eval) <= CAP_EVAL:
                        
                        # Calculate Win Probs
                        # Note: prev_eval and curr_eval are WHITE-CENTRIC scores (CP).
                        
                        # If WHITE just moved (board.turn == WHITE before push, BLACK after push)
                        # Wait, board is updated at bottom of loop? No, usually we push after logic.
                        # Let's check state.
                        
                        # board.turn is the side whose move `move` IS.
                        # e.g. Start: White to move. `move` is e4.
                        
                        if board.turn == chess.WHITE:
                             # White Moved.
                             # Goal: Maximize Eval.
                             # Prev (Best) was say +0.50.
                             # Curr (Actual) is +0.40.
                             # Diff = Best - Actual.
                             
                             wp_best = cp_to_win_probability(prev_eval)
                             wp_played = cp_to_win_probability(curr_eval)
                             
                             delta = max(0.0, wp_best - wp_played)
                             players[white].errors.append(delta)
                             
                        else:
                             # Black Moved.
                             # Goal: Minimize Eval (Make more negative).
                             # Prev (Best) was say +0.50.
                             # Curr (Actual) is +0.60 (Worse for black).
                             # Black wants -Eval to be high.
                             
                             # Flip to Black perspective
                             wp_best = cp_to_win_probability(-prev_eval)
                             wp_played = cp_to_win_probability(-curr_eval)
                             
                             delta = max(0.0, wp_best - wp_played)
                             players[black].errors.append(delta)

                # UPDATE STATE
                if curr_eval is not None:
                    prev_eval = curr_eval
                
                board.push(move)
                node = next_node
                
            # End of game stats
            if has_evals:
                if w_elo > 0: players[white].elos.append(w_elo)
                if b_elo > 0: players[black].elos.append(b_elo)
                players[white].games += 1
                players[black].games += 1
                if year.isdigit():
                    players[white].years.append(int(year))
                    players[black].years.append(int(year))
            
            if first_game:
                if not has_evals:
                    print("WARNING: First game has no [%eval] annotations! Please check input.")
                    # We continue, assuming maybe some games have it? But warn user.
                first_game = False

    return players

def main():
    parser = argparse.ArgumentParser(description="Calculate IPR from PGN with [%eval] tags.")
    parser.add_argument("pgn_file", help="Path to input PGN file")
    args = parser.parse_args()
    
    input_path = Path(args.pgn_file)
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        sys.exit(1)
        
    players = process_pgn(input_path)
    
    if not players:
        print("No players found or processed.")
        return

    output_csv = input_path.parent / (input_path.stem + "_IPR.csv")
    
    print(f"Writing results to {output_csv}...")
    
    all_errors = []
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Player', 'Elo', 'Number_Games', 'Avg_IPR', 'Std_IPR', 'FirstYear', 'LastYear']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for player_name in sorted(players.keys()):
            stats = players[player_name]
            if not stats.errors:
                continue
                
            avg_ipr = ipr_from_errors(stats.errors)
            std_ipr = bootstrap_ipr(stats.errors)
            
            avg_elo = sum(stats.elos) / len(stats.elos) if stats.elos else 0
            
            first_year = min(stats.years) if stats.years else "N/A"
            last_year = max(stats.years) if stats.years else "N/A"
            
            writer.writerow({
                'Player': player_name,
                'Elo': f"{int(avg_elo)}" if avg_elo > 0 else "N/A",
                'Number_Games': stats.games,
                'Avg_IPR': f"{int(avg_ipr)}",
                'Std_IPR': f"{int(std_ipr)}",
                'FirstYear': first_year,
                'LastYear': last_year
            })
            
            all_errors.extend(stats.errors)
            
        # Overall Summary
        if all_errors:
            avg_ipr_total = ipr_from_errors(all_errors)
            std_ipr_total = bootstrap_ipr(all_errors)
            
            writer.writerow({
                'Player': 'Overall Summary',
                'Elo': 'N/A',
                'Number_Games': sum(p.games for p in players.values()) // 2, # Approx games
                'Avg_IPR': f"{int(avg_ipr_total)}",
                'Std_IPR': f"{int(std_ipr_total)}",
                'FirstYear': 'N/A',
                'LastYear': 'N/A'
            })

    print("Done.")

if __name__ == "__main__":
    main()
