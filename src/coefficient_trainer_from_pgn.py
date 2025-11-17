#!/usr/bin/env python
"""
Program 2: coefficient_trainer_from_pgn.py
---------------------------------
Trains the centipawn-to-win-probability coefficient from pre-annotated PGN
containing {[%eval]} comments. Does not require Stockfish.

Note:
    based on
https://github.com/lichess-org/lila/pull/11148
    modified for retraining from './annotated-gamedata.pgn'
"""
import chess.pgn
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from tqdm import tqdm
import sys
import re

# --- CONFIGURATION ---
ANNOTATED_PGN = Path("./annotated-gamedata.pgn")
CP_BIN_SIZE = 50
MAX_ABS_CP = 3000

def logistic_model(cp, k):
    """
    Sigmoid curve mapping centipawn score to win probability.
    """
    return 0.5 + 0.5 * (2 / (1 + np.exp(-k * cp)) - 1)

def parse_eval_comment(comment):
    """
    Extracts evaluation from comment like '{[%eval 1.23]}' or '{[%eval #M5]}'.
    Returns centipawn value or None if parsing fails.
    """
    if not comment:
        return None
    
    # Match {[%eval X.XX]} or {[%eval #MX]}
    match = re.search(r'\{\[%eval ([^}]+)\]\}', comment)
    if not match:
        return None
    
    eval_str = match.group(1).strip()
    
    # Handle mate scores
    if eval_str.startswith('#M'):
        try:
            mate_in = int(eval_str[2:])
            return MAX_ABS_CP if mate_in > 0 else -MAX_ABS_CP
        except ValueError:
            return None
    
    # Handle centipawn scores
    try:
        return int(float(eval_str) * 100)
    except ValueError:
        return None

def collect_training_data():
    """
    Reads annotated PGN and extracts (centipawn, game_result) pairs.
    """
    print(f"Loading annotated games from {ANNOTATED_PGN}...")
    
    raw_data = []
    
    with open(ANNOTATED_PGN, encoding="utf-8") as pgn_file:
        game_count = 0
        
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
                
            game_count += 1
            result = game.headers.get("Result")
            final_score = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}.get(result, 0.5)
            
            node = game
            position_count = 0
            
            while node.variations:
                node = node.variation(0)
                
                cp = parse_eval_comment(node.comment)
                
                if cp is not None and abs(cp) <= MAX_ABS_CP:
                    raw_data.append((cp, final_score))
                    position_count += 1
            
            if game_count % 100 == 0:
                print(f"Processed {game_count} games, {len(raw_data)} positions...")
    
    print(f"\n✅ Collected {len(raw_data)} data points from {game_count} games.")
    return raw_data

def bin_and_aggregate(raw_data, bin_size):
    """
    Groups centipawn scores into bins and calculates average win rate per bin.
    """
    binned_results = {}
    
    for cp, p_obs in raw_data:
        bin_center = int(round(cp / bin_size) * bin_size)
        
        if bin_center not in binned_results:
            binned_results[bin_center] = []
        
        binned_results[bin_center].append(p_obs)
    
    cp_bins = []
    p_observed = []
    
    for cp_bin in sorted(binned_results.keys()):
        results = binned_results[cp_bin]
        
        if len(results) >= 50:
            cp_bins.append(cp_bin)
            p_observed.append(np.mean(results))
    
    print(f"\n✅ Created {len(cp_bins)} bins with sufficient data.")
    return np.array(cp_bins), np.array(p_observed)

def main():
    if not ANNOTATED_PGN.exists():
        print(f"Error: Annotated PGN not found at {ANNOTATED_PGN}")
        print("Run pgn_evaluator.py first to generate annotated data.")
        sys.exit(1)
    
    # Step 1: Collect data from annotated PGN
    raw_data = collect_training_data()
    
    if not raw_data:
        print("No valid data points found. Check annotated PGN format.")
        sys.exit(1)
    
    # Step 2: Aggregate and bin
    cp_bins, p_observed = bin_and_aggregate(raw_data, CP_BIN_SIZE)
    
    if len(cp_bins) < 5:
        print("Not enough bins for curve fitting.")
        sys.exit(1)
    
    # Step 3: Fit curve
    print("\nFitting logistic curve to find optimal 'k'...")
    
    initial_guess_k = 0.00368208
    
    try:
        params, covariance = curve_fit(
            logistic_model,
            cp_bins,
            p_observed,
            p0=[initial_guess_k]
        )
        
        k_fit = params[0]
        
        print("\n" + "="*50)
        print("✅ NEW CALIBRATION COEFFICIENT FOUND")
        print("="*50)
        print(f"Original Lichess k: 0.00368208")
        print(f"Recalculated k_fit: {k_fit:.8f}")
        print(f"Difference: {((k_fit/0.00368208 - 1)*100):+.2f}%")
        print("="*50)
        
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
