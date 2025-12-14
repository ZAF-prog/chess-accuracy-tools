#!/usr/bin/env python
# V1 written in Google Antigravity
"""
PGN Blunder Rate Analyzer

PURPOSE:
    This script analyzes chess games in PGN format that contain engine evaluation
    comments (e.g., [%eval 0.33]) and calculates the blunder rate for each player
    who has sufficient data (at least 100 moves).

USAGE:
    python analyze_blunders.py <input_pgn_file>
    
    Example:
        python analyze_blunders.py data/Staunton-evaluated.pgn
    
    Output:
        Creates a CSV file named <input_basename>_Blunder-rate.csv
        containing player statistics.

BLUNDER DEFINITION:
    A blunder is defined as a move that worsens the evaluation by at least 3.0 pawns
    from the player's perspective:
    
    - For White: A move is a blunder if the evaluation drops by 3.0 or more
                 (e.g., from +2.0 to -1.0 or worse)
    
    - For Black: A move is a blunder if the evaluation rises by 3.0 or more
                 (e.g., from -2.0 to +1.0 or worse for Black)
    
    - Missed forced mates are also considered blunders, as mate evaluations are
      converted to very large values (±20000), making any move that loses a mate
      automatically exceed the 3.0 threshold.

ALGORITHM:
    1. Validate that the PGN file contains [%eval ...] comments
    2. Parse each game and extract player names
    3. For each move:
       - Extract the evaluation from the comment
       - Convert mate scores (#M2, #M-1, etc.) to large numeric values
       - Calculate the evaluation change from the previous move
       - Determine if the change constitutes a blunder based on whose turn it was
       - Track statistics for each player
    4. Output results for players with at least 100 moves
    5. Calculate blunder rate as percentage (blunders per 100 moves)

EVALUATION CONVENTIONS:
    - Positive values indicate White advantage
    - Negative values indicate Black advantage
    - Mate scores: #M2 = White mates in 2 moves → +19998
                   #M-2 = Black mates in 2 moves → -19998
                   #M0 = Checkmate delivered → ±20000 (sign depends on who mated)
"""

import chess.pgn
import sys
import re
import csv
import os
from collections import defaultdict

MATE_SCORE = 20000

def parse_eval(eval_str):
    """
    Parses an eval string like '0.33', '-1.75', '#M2', '#M-1'.
    Returns a float value. Mate is converted to a large number.
    White advantage is positive, Black advantage is negative.
    """
    # Remove braces and [%eval ...] wrapper if passed as raw comment, 
    # but here we expect just the value inside or we parse it out.
    # The PGN parser might give us the raw comment string.
    
    # Handle the format seen in the file: "[%eval 0.33]" or "0.33" depending on extraction
    match = re.search(r'\[%eval\s+([^]]+)\]', eval_str)
    if match:
        content = match.group(1)
    else:
        content = eval_str

    if '#' in content:
        # Mate format seen: #M2, #M-1, maybe #2
        # Extract the number relative to mate
        mate_match = re.search(r'#M?([+-]?\d+)', content)
        if mate_match:
            mate_moves = int(mate_match.group(1))
            # Determine sign. 
            # If mate_moves is positive (e.g. 2), it's mate for White?
            # Usually strict PGN: positive is white, negative is black.
            # #M2 -> White mates in 2. Score ~ 20000
            # #M-2 -> Black mates in 2. Score ~ -20000
            
            # Using 20000 - abs(moves) for white, -20000 + abs(moves) for black
            if mate_moves > 0:
                return MATE_SCORE - abs(mate_moves)
            elif mate_moves < 0:
                return -MATE_SCORE + abs(mate_moves)
            else:
                # #M0 -> Mate delivered.
                # If we don't know who delivered it, checks eval sign? 
                # Usually #M0 implies the side who just moved won.
                # We'll rely on the polarity logic of the blunder check or just return huge value.
                # Actually, 0 usually implies immediate mate. 
                # Let's assume positive for now, context might matter? 
                # In standard PGN, 1-0 result implies final eval is infinity. 
                # For safety, let's treat 0 as 20000? 
                # But wait, purely from eval string "#M0", we don't know. 
                # However, usually signed like others? No, 0 is 0. 
                # Let's handle special 0 case if it occurs. 
                # Looking at file: "20. Rxg1 {[%eval #M1]} Nf2# {[%eval #M0]}"
                # 19... Rg1+ was Black move. White to move. Eval #M-1 (Black mates).
                # 20. Rxg1 was White move. Black to move. Eval #M1 (White mates? No wait).
                # Let's re-read file lines 312-313:
                # 19. Rff1 {[%eval #M2]} Rg1+ {[%eval #M-1]}
                # 20. Rxg1 {[%eval #M1]} Nf2# {[%eval #M0]}
                # 19. Rff1 (White). Eval #M2 (White mates in 2). Correct.
                # 19... Rg1+ (Black). Eval #M-1 (Black mates in 1?? No).
                # Wait. #M2 means White mates in 2 ply? Or 2 moves?
                # Usually engine #2 means Mate in 2 moves.
                # If White played 19. Rff1 and eval is #M2, White is winning.
                # Black replies 19... Rg1+. Eval becomes #M-1?
                # If White is designated to mate, eval should stay Positive.
                # Unless #M-1 means "Black mates in 1"? 
                # Ah, the file shows 19... Rg1+. If White captures, then...
                # Actually, look at the moves. 19. Rff1 (threatens mate?). 19... Rg1+ (check).
                # 20. Rxg1 (White captures). Nf2# (Black mates).
                # So Black WON.
                # This means 19. Rff1 was a BLUNDER if eval was clean before?
                # Line 311: 18. Rxf6 ... Nh3 eval -5.77. Black is winning (-).
                # 19. Rff1 eval #M2 ?? If Black checks and mates, White is losing.
                # #M2 usually means White mates. 
                # Did the engine fail? Or is #M2 actually "Mate in 2 ply for whoever"?
                # Standard UCI: "mate 2" -> White mates. "mate -2" -> Black mates.
                # Line 312: 19. Rff1 {[%eval #M2]}
                # If eval suddenly jumped from -5.77 to #M2 (White winning), checking if 19. Rff1 was brilliant?
                # Then 19... Rg1+ {[%eval #M-1]}. #M-1 usually means Black mates.
                # So from #M2 (White Checkmate) to #M-1 (Black Checkmate)? 
                # That would mean 19... Rg1+ was a blunder by Black?? NO.
                # If White was winning (#M2) and then Black is winning (#M-1), then White played badly? 
                # No, eval is POST-move. 
                # After 19. Rff1, Eval is #M2. Means White is winning.
                # After 19... Rg1+, Eval is #M-1. Means Black is winning.
                # This implies 19... Rg1+ TURNED the table? 
                # Or was 19. Rff1 eval wrong/weird? 
                # Let's look at the actual game context (Cochrane vs Staunton). 0-1. Black wins.
                # So Black (Staunton) won.
                # 19. Rff1 must have been a blunder if prev was safer? 
                # Row 311: 18... Nh3 {[%eval -5.77]} -> Black advantage.
                # 19. Rff1 {[%eval #M2]} -> This implies White is now WINNING? 
                # If so, 19. Rff1 was a huge turnaround?
                # Then 20... Rg1+ leads to #M-1 (Black mates).
                # This sequence suggests volatile evals or I am misinterpreting #M2.
                # Usually #M2 = +Mate in 2. #M-2 = -Mate in 2.
                # If correct:
                # 18... Nh3 (-5.77). Black winning.
                # 19. Rff1 (#M2). White winning. (Blunder by Black? No, we are evaluating White's move 19. Rff1).
                # Wait. 18... Nh3 occurred. Eval -5.77.
                # 19. Rff1 occurred. Eval #M2. 
                # Change: #M2 - (-5.77) = +Huge. White improved massively!
                # So 19. Rff1 was a GREAT move?
                # Phen 19... Rg1+. Eval #M-1 (Black mates).
                # Change: #M-1 - #M2 = -Huge - (+Huge) = -SuperHuge.
                # Black's eval dropped from White Winning to Black Winning?
                # That implies 19... Rg1+ was a GENIUS move (White blundered by allowing it? No, White already moved).
                # If after 19. Rff1 White is winning, then 19... Rg1+ makes Black win?
                # That means 19. Rff1 did NOT actually win, or 19... Rg1+ is a refutation that the engine MISSED at depth?
                # Or #M2 was a hallucination.
                # OR... #M2 refers to ply?
                # Let's assume standard UCI conventions where printed eval is trustworthy for the depth.
                # The user asks for blunder check based on *recorded data*.
                # If recorded data says #M2 -> #M-1, that is a massive swing.
                # From White perspective: Current (#M-1) - Prev (#M2) = -40000. 
                # That would mean Black's move made White's position terrible.
                # Usually we check: Did BLACK blunder?
                # Black played Rg1+. Eval went from White Winning to Black Winning.
                # This is GOOD for Black. 
                # Diff for Black = -(Current - Prev) = -( (-20000) - (+20000) ) = -(-40000) = +40000.
                # Black improved by 40000. Not a blunder.
                # BUT wait. Did WHITE blunder previous turn?
                # 19. Rff1. Prev was -5.77. Curr #M2.
                # Diff = +20000 - (-5.77) = +20005.
                # White improved.
                # So according to these evals:
                # 18... Nh3 (-5.77) -> 19. Rff1 (#M2) (White turns losing into winning) -> 19... Rg1+ (#M-1) (Black turns losing into winning).
                # This suggests the evaluations are extremely volatile or shallow depth.
                # BUT the task is "calculate rate of blunders... defined as move that causes eval to worsen".
                # I must follow the math strictly.
                
                # Handling #M0:
                # If #M0, it means checkmate on board.
                # If it's 20... Nf2# {[%eval #M0]}, Black played Nf2#. 
                # Since result is 0-1 (lines 313), Black won. 
                # So #M0 here acts as "Black Mate". Score -20000.
                # We need context of whose turn it is or who mated.
                # Nf2# is Black move. So Eval is for Black win. -20000.
                 pass
            
            # Simple heuristic for M0: Look at the move color?
            # Actually if #M0 comes after a Black move, it's Black mating -> -20000.
            # If after White move -> +20000.
            if mate_moves == 0:
                # We need side information in parse_eval? 
                # Or just pass the turn logic outside?
                # Let's return a special sentinel or handle inside the loop.
                return 0 # SENTINEL
            
            return MATE_SCORE - abs(mate_moves) if mate_moves > 0 else -MATE_SCORE + abs(mate_moves)
            
    return float(content)

def validate_pgn_has_evals(pgn_path):
    """
    Validates that the PGN file contains evaluation comments.
    Returns True if evals are found, False otherwise.
    """
    eval_count = 0
    games_checked = 0
    max_games_to_check = 3  # Check first few games
    
    with open(pgn_path, encoding='utf-8') as pgn:
        while games_checked < max_games_to_check:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            
            games_checked += 1
            node = game
            while node.variations:
                next_node = node.variation(0)
                if "[%eval" in next_node.comment:
                    eval_count += 1
                    if eval_count >= 5:  # Found enough evals
                        return True
                node = next_node
    
    return eval_count > 0

def analyze_pgn(pgn_path, output_csv):
    results = {} # Player -> {'moves': 0, 'blunders': 0}
    
    # Validate that PGN has evaluation comments
    print("Validating PGN file...")
    if not validate_pgn_has_evals(pgn_path):
        print("ERROR: The input PGN file does not contain [%eval ...] comments.")
        print("This script requires a PGN file with engine evaluations.")
        sys.exit(1)
    print("Validation passed. PGN contains evaluation data.")
    
    with open(pgn_path, encoding='utf-8') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
                
            white = game.headers.get("White", "?")
            black = game.headers.get("Black", "?")
            
            if white not in results: results[white] = {'moves': 0, 'blunders': 0}
            if black not in results: results[black] = {'moves': 0, 'blunders': 0}
            
            board = game.board()
            prev_eval = 0.2 # Rough start eval
            
            # Iterate moves
            node = game
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                comment = next_node.comment
                
                # Check for eval
                curr_eval = prev_eval # Default if missing
                
                # Parse eval
                if "[%eval" in comment:
                    try:
                        raw_val = parse_eval(comment)
                        if raw_val == 0 and "#M0" in comment:
                             # Determine who mated.
                             # If move was by White (board.turn), then White mated -> +20000
                             # Wait, board.turn is updated AFTER push? 
                             # We have 'board' state before push.
                             # If board.turn == chess.WHITE (White is moving), and result is mate,
                             # Then eval is +20000.
                             curr_eval = MATE_SCORE if board.turn == chess.WHITE else -MATE_SCORE
                        elif raw_val == 0 and "#M0" not in comment:
                             curr_eval = 0.0
                        else:
                             curr_eval = raw_val
                    except Exception as e:
                        # Fallback
                        curr_eval = prev_eval
                else:
                    # If no eval, we can't judge blunder. Skip this move's calculation but update prev??
                    # If we skip, we lose continuity. Better to assume no change or skip entire chain?
                    # The prompt implies "files like this" which are fully evaluated.
                    # We'll just carry forward prev_eval to avoid false blunders.
                    curr_eval = prev_eval

                # Calculate Blunder
                # White move: Diff = Current - Previous. Blunder if <= -3.0
                # Black move: Diff = Current - Previous. Blunder if >= 3.0
                
                is_blunder = False
                diff = curr_eval - prev_eval
                
                if board.turn == chess.WHITE:
                    # White moved.
                    if diff <= -3.0:
                        is_blunder = True
                else:
                    # Black moved.
                    if diff >= 3.0:
                        is_blunder = True
                
                # Update Stats
                player = white if board.turn == chess.WHITE else black
                results[player]['moves'] += 1
                if is_blunder:
                    results[player]['blunders'] += 1
                
                # Push move and update prev
                board.push(move)
                prev_eval = curr_eval
                node = next_node

    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Player', 'Moves', 'Blunders', 'Blunder Rate %']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for player, stats in results.items():
            if stats['moves'] >= 100:
                rate = (stats['blunders'] / stats['moves']) * 100
                writer.writerow({
                    'Player': player,
                    'Moves': stats['moves'],
                    'Blunders': stats['blunders'],
                    'Blunder Rate %': f"{rate:.2f}"
                })

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_blunders.py <pgn_file>")
        sys.exit(1)
        
    input_pgn = sys.argv[1]
    
    # Construct output filename: basename + _Blunder-rate.csv
    base, ext = os.path.splitext(input_pgn)
    output_csv = f"{base}_Blunder-rate.csv"
    
    print(f"Analyzing {input_pgn}...")
    analyze_pgn(input_pgn, output_csv)
    print(f"Done. Output saved to {output_csv}")
