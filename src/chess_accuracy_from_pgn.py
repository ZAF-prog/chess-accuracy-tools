import io
import math
import sys
import traceback
import re
import csv
import os
from pathlib import Path
from collections import defaultdict
import chess.pgn

# Constants
MATE_SCORE = 1000
START_EVAL = 17  # Approximate start position eval in centipawns

def get_eval_from_comment(comment):
    """
    Extracts evaluation from comment.
    Format: [%eval 0.33] or [%eval #M6]
    Returns: (score_type, value)
        score_type: 'cp' or 'mate'
        value: float (for cp) or int (for mate)
    """
    match = re.search(r'\[%eval\s+([^\]]+)\]', comment)
    if not match:
        return None
    
    eval_str = match.group(1)
    if eval_str.startswith('#M'):
        return 'mate', int(eval_str[2:])
    else:
        return 'cp', float(eval_str)

def move_accuracy_percent(before, after):
    if after >= before:
        return 100.0
    else:
        win_diff = before - after
        raw = 103.1668100711649 * math.exp(-0.04354415386753951 * win_diff) + -3.166924740191411
        return max(min(raw + 1, 100), 0)

def winning_chances_percent(cp):
    multiplier = -0.00368208
    chances = 2 / (1 + math.exp(multiplier * cp)) - 1
    return 50 + 50 * max(min(chances, 1), -1)

def harmonic_mean(values):
    n = len(values)
    if n == 0:
        return 0
    reciprocal_sum = sum(1 / x for x in values if x)
    return n / reciprocal_sum if reciprocal_sum else 0

def std_dev(seq):
    if len(seq) == 0:
        return 0.0
    mean = sum(seq) / len(seq)
    variance = sum((x - mean) ** 2 for x in seq) / len(seq)
    return math.sqrt(variance)

def median(values):
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
    else:
        return sorted_vals[n//2]

def median_absolute_deviation(values):
    if not values:
        return 0.0
    med = median(values)
    deviations = [abs(v - med) for v in values]
    return median(deviations)

def volatility_weighted_mean(accuracies, win_chances, is_white):
    weights = []
    for i in range(len(accuracies)):
        base_index = i * 2 + 1 if is_white else i * 2 + 2
        start_idx = max(base_index - 2, 0)
        end_idx = min(base_index + 2, len(win_chances) - 1)

        sub_seq = win_chances[start_idx:end_idx]
        weight = max(min(std_dev(sub_seq), 12), 0.5)
        weights.append(weight)

    weighted_sum = sum(accuracies[i] * weights[i] for i in range(len(accuracies)))
    total_weight = sum(weights)
    weighted_mean = weighted_sum / total_weight if total_weight else 0

    return weighted_mean

def process_game(game, is_verbose):
    accuracies_white, accuracies_black, win_chances = [], [], []
    total_cp_loss_white, total_cp_loss_black = 0, 0
    prev_evaluation = START_EVAL
    move_number = 1
    
    board = game.board()
    win_chances.append(winning_chances_percent(prev_evaluation))
    
    node = game
    while not node.is_end():
        if node.variations:
            next_node = node.variations[0]
            move = next_node.move
            comment = next_node.comment
            
            eval_data = get_eval_from_comment(comment)
            
            if eval_data:
                score_type, value = eval_data
                
                if score_type == 'cp':
                    score = value * 100
                else: # score_type == 'mate'
                    # Temporarily push move to determine side_to_move for mate score interpretation
                    board.push(move)
                    
                    # board.turn is now the side to move in the evaluated position
                    if board.turn == chess.WHITE:
                        white_mate_score = value
                    else:
                        white_mate_score = -value
                    
                    # Pop the move back to correctly apply the single push later
                    board.pop()
                        
                    if white_mate_score > 0:
                        score = MATE_SCORE
                    else:
                        score = -MATE_SCORE
                
                # Now push the move for real
                board.push(move)
                
                win_before_white = winning_chances_percent(prev_evaluation)
                win_after_white = winning_chances_percent(score)
                win_chances.append(win_after_white)

                # Determine who made the move
                # If board.turn is now BLACK, it means WHITE just moved.
                mover = not board.turn 
                
                if mover == chess.WHITE:
                    win_before = 100 - win_before_white
                    win_after = 100 - win_after_white
                else:
                    win_before = win_before_white
                    win_after = win_after_white

                accuracy = move_accuracy_percent(win_before, win_after)

                if mover == chess.WHITE:
                    # White moved. Did score drop?
                    # White wants Max score.
                    # Loss = Prev - Current (if Prev > Current)
                    cp_loss = max(0, prev_evaluation - score)
                    total_cp_loss_white += cp_loss
                    accuracies_white.append(accuracy)
                else:
                    # Black moved. Did score rise?
                    # Black wants Min score.
                    # Loss = Current - Prev (if Current > Prev)
                    cp_loss = max(0, score - prev_evaluation)
                    total_cp_loss_black += cp_loss
                    accuracies_black.append(accuracy)

                if is_verbose:
                    board.pop() # Pop to get SAN from previous board state
                    san_move = board.san(move)
                    board.push(move) # Push back
                    
                    move_number_str = f'{move_number:3}.' if mover == chess.WHITE else "    "
                    eval_display = f"{value:.2f}" if score_type == 'cp' else f"#{value}"
                    print(
                        f'{move_number_str} {san_move:5}: Eval: {eval_display:5}, '
                        f'Centipawn Loss: {cp_loss:3.0f}, Accuracy %: {accuracy:3.0f}, Win %: {win_after_white:2.0f}')
                
                prev_evaluation = score
                if mover == chess.WHITE:
                    move_number += 1
            else:
                # No eval found, just push move and continue
                board.push(move)
                if board.turn == chess.BLACK: # White just moved
                    move_number += 1
            
            node = next_node
        else:
            break
            
    return accuracies_white, accuracies_black, total_cp_loss_white, total_cp_loss_black, win_chances

def analyze_pgn_file(input_file, is_verbose):
    print(f"Processing {input_file.name}...")
    
    # Dictionary to store player data: player_name -> list of accuracy scores
    player_data = defaultdict(lambda: {'elo': None, 'accuracies': []})
    
    games_count = 0
    
    while True:
        game = chess.pgn.read_game(input_file)
        if game is None:
            break
        
        games_count += 1
        white_name = game.headers.get("White", "Unknown")
        black_name = game.headers.get("Black", "Unknown")
        white_elo = game.headers.get("WhiteElo", "")
        black_elo = game.headers.get("BlackElo", "")
        
        if is_verbose:
            print(f"\nGame {games_count}: {white_name} vs {black_name}")
            
        (acc_white, acc_black, cp_loss_white, cp_loss_black, win_chances) = process_game(game, is_verbose)
        
        if acc_white and acc_black:
            move_count_white = len(acc_white)
            move_count_black = len(acc_black)
            
            avg_cp_white = cp_loss_white / move_count_white if move_count_white else 0
            avg_cp_black = cp_loss_black / move_count_black if move_count_black else 0
            
            harm_white = harmonic_mean(acc_white)
            harm_black = harmonic_mean(acc_black)
            
            weight_white = volatility_weighted_mean(acc_white, win_chances, True)
            weight_black = volatility_weighted_mean(acc_black, win_chances, False)
            
            final_white = (harm_white + weight_white) / 2
            final_black = (harm_black + weight_black) / 2
            
            # Store player data
            if white_elo and player_data[white_name]['elo'] is None:
                player_data[white_name]['elo'] = white_elo
            player_data[white_name]['accuracies'].append(final_white)
            
            if black_elo and player_data[black_name]['elo'] is None:
                player_data[black_name]['elo'] = black_elo
            player_data[black_name]['accuracies'].append(final_black)
            
            print(f"Game {games_count}: {white_name} ({final_white:.1f}%) vs {black_name} ({final_black:.1f}%) | CP Loss: {avg_cp_white:.1f} / {avg_cp_black:.1f}")
        else:
            print(f"Game {games_count}: Insufficient data")
    
    return player_data

def write_csv_output(player_data, output_file):
    """Write player statistics to CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Player', 'Elo', 'Games', 'Avg_Accuracy', 'Std_Accuracy', 'Median_Accuracy', 'MAD_Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Sort players by Elo (descending), then by name
        sorted_players = sorted(
            player_data.items(),
            key=lambda x: (int(x[1]['elo']) if x[1]['elo'] and x[1]['elo'].isdigit() else 0, x[0]),
            reverse=True
        )
        
        for player_name, data in sorted_players:
            accuracies = data['accuracies']
            if accuracies:
                writer.writerow({
                    'Player': player_name,
                    'Elo': data['elo'] if data['elo'] else 'N/A',
                    'Games': len(accuracies),
                    'Avg_Accuracy': f"{sum(accuracies) / len(accuracies):.2f}",
                    'Std_Accuracy': f"{std_dev(accuracies):.2f}",
                    'Median_Accuracy': f"{median(accuracies):.2f}",
                    'MAD_Accuracy': f"{median_absolute_deviation(accuracies):.2f}"
                })

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate chess accuracy from PGN evaluations')
    parser.add_argument('input_file', help='Input PGN file with [%eval] comments')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show move-by-move details')
    parser.add_argument('-o', '--output', help='Output CSV file (default: same basename as input)')
    
    args = parser.parse_args()
    
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            player_data = analyze_pgn_file(f, args.verbose)
        
        # Determine output file name
        if args.output:
            output_file = args.output
        else:
            input_path = Path(args.input_file)
            output_file = input_path.with_suffix('.csv')
        
        write_csv_output(player_data, output_file)
        print(f"\nStatistics written to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}")
    except Exception as e:
        traceback.print_exc()
