#!/usr/bin/env python
"""
Dr. Kenneth Regan's Intrinsic Performance Rating (IPR) algorithm
with Multi-PV analysis using Stockfish (replacing Rybka).

This implementation follows Regan's methodology:
1. Fit skill parameters (s, c) using maximum likelihood on actual moves
2. Project expected error (AE_e) on solitaire set using fitted parameters
3. Calculate IPR from AE_e using linear regression formula
"""

import chess
import chess.engine
import chess.pgn
import numpy as np
import math
import sys
import argparse
import csv
import platform
from pathlib import Path
from collections import defaultdict
from scipy.optimize import minimize, brentq
from typing import List, Tuple, Dict, Optional

# --- CONFIGURATION ---
BOOK_MOVES = 8          # Skip first 8 moves (16 ply)
CAP_EVAL = 300          # Garbage time filter (centipawns)
MULTI_PV = 5            # Number of principal variations to analyze
IPR_INTERCEPT = 3571    # IPR formula constant
IPR_SLOPE = -15413      # IPR formula slope
MATE_SCORE = 10000      # Mate score cap for calculations

def get_engine_path():
    """Determines the correct Stockfish binary path based on the operating system."""
    os_name = platform.system()
    
    if os_name == "Windows":
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    elif os_name in ("Linux", "Darwin"):
        return Path("binaries/stockfish-ubuntu-x86-64-avx512")
    else:
        print(f"Error: Unsupported operating system detected: {os_name}")
        sys.exit(1)

STOCKFISH_PATH = get_engine_path()

class MultiPVIPRCalculator:
    """
    Implements Regan's IPR algorithm with Multi-PV analysis.
    """
    
    def __init__(self, engine_path: Path, depth: int = 15, multi_pv: int = MULTI_PV):
        """Initialize with engine configuration."""
        self.engine_path = engine_path
        self.depth = depth
        self.multi_pv = multi_pv
        self.engine = None
        
        # Fitted parameters (will be set during analysis)
        self.s = 1.0
        self.c = 1.0
        
    def start_engine(self):
        """Start the Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))
            print(f"Stockfish engine started with MultiPV={self.multi_pv}, depth={self.depth}")
        except FileNotFoundError:
            print(f"Error: Stockfish not found at {self.engine_path}")
            sys.exit(1)
            
    def stop_engine(self):
        """Stop the Stockfish engine."""
        if self.engine:
            self.engine.quit()
            
    def analyze_position_multipv(self, board: chess.Board) -> List[Tuple[chess.Move, float]]:
        """
        Analyze position with Multi-PV to get top N moves with evaluations.
        
        Returns:
            List of (move, eval_cp) tuples, sorted by evaluation (best first)
        """
        if not self.engine:
            raise RuntimeError("Engine not started")
            
        # Get Multi-PV analysis
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=self.multi_pv)
        
        results = []
        for pv_info in info:
            if "pv" not in pv_info or len(pv_info["pv"]) == 0:
                continue
                
            move = pv_info["pv"][0]
            score = pv_info["score"].white()
            
            # Convert score to centipawns (white perspective)
            if score.is_mate():
                mate_in = score.mate()
                eval_cp = MATE_SCORE if mate_in > 0 else -MATE_SCORE
            else:
                eval_cp = score.score()
                
            results.append((move, eval_cp))
            
        return results
    
    def calculate_delta(self, v0: float, vi: float) -> float:
        """
        Calculate scaled value difference (delta_i) between best move and alternative.
        Uses logarithmic scaling: delta = |log(1+v0) - log(1+vi)|
        
        Args:
            v0, vi: Evaluations in pawn units (e.g., 0.20 for 20 centipawns)
        """
        try:
            # Ensure values are not too negative for log domain
            if v0 <= -1 or vi <= -1:
                return float('inf')
            delta = abs(math.log(1 + v0) - math.log(1 + vi))
            return delta
        except (ValueError, OverflowError):
            return float('inf')
    
    def calculate_move_probabilities(self, values: List[float]) -> List[float]:
        """
        Calculate probabilities p_i for all moves using current (s, c) parameters.
        
        Args:
            values: List of evaluations (pawn units), best move first
            
        Returns:
            List of probabilities summing to 1.0
        """
        if not values:
            return []
            
        v0 = values[0]
        alphas = []
        
        for vi in values:
            delta_i = self.calculate_delta(v0, vi)
            
            if delta_i == float('inf'):
                alpha = 0.0
            elif delta_i == 0:
                alpha = 1.0
            else:
                try:
                    # alpha = e^(-(delta/s)^c)
                    alpha = math.exp(-((delta_i / self.s) ** self.c))
                except (OverflowError, ValueError):
                    alpha = 0.0
                    
            alphas.append(alpha)
        
        # Solve for p0 such that sum(p0^alpha_i) = 1
        def func_to_minimize(p0):
            return sum(p0**alpha for alpha in alphas) - 1.0
        
        try:
            p0_solution = brentq(func_to_minimize, 1e-12, 1.0)
        except (ValueError, RuntimeError):
            # Fallback: uniform distribution
            p0_solution = 1.0 / len(alphas) if len(alphas) > 0 else 1.0
        
        probabilities = [p0_solution**alpha for alpha in alphas]
        
        # Normalize
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
            
        return probabilities
    
    def fit_parameters_sc(self, test_set_data: List[Tuple[List[float], int]]) -> Tuple[float, float]:
        """
        Fit (s, c) parameters using maximum likelihood estimation.
        
        Args:
            test_set_data: List of (move_values, actual_move_index) tuples
            
        Returns:
            (s_fit, c_fit) parameters
        """
        if not test_set_data:
            print("Warning: No test set data for fitting. Using default parameters.")
            return 0.089, 0.506  # Default for ~2600 Elo
        
        def negative_log_likelihood(params):
            """Objective function: negative log-likelihood of observed moves."""
            s_trial, c_trial = params
            
            if s_trial <= 0 or c_trial <= 0:
                return 1e10  # Invalid parameters
            
            # Temporarily set parameters
            old_s, old_c = self.s, self.c
            self.s, self.c = s_trial, c_trial
            
            log_likelihood = 0.0
            
            for move_values, actual_idx in test_set_data:
                if actual_idx >= len(move_values):
                    continue
                    
                probs = self.calculate_move_probabilities(move_values)
                
                if actual_idx < len(probs) and probs[actual_idx] > 0:
                    log_likelihood += math.log(probs[actual_idx])
                else:
                    log_likelihood += math.log(1e-10)  # Avoid log(0)
            
            # Restore parameters
            self.s, self.c = old_s, old_c
            
            return -log_likelihood
        
        # Initial guess based on typical values
        initial_guess = [0.089, 0.506]
        
        # Bounds: s in (0.01, 1.0), c in (0.1, 2.0)
        bounds = [(0.01, 1.0), (0.1, 2.0)]
        
        print("Fitting (s, c) parameters using maximum likelihood...")
        result = minimize(negative_log_likelihood, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            s_fit, c_fit = result.x
            # Check if parameters hit boundaries (indicates poor fit)
            if s_fit >= 0.99 or c_fit >= 1.99 or s_fit <= 0.011 or c_fit <= 0.11:
                print(f"Warning: Fitted parameters at boundary: s={s_fit:.4f}, c={c_fit:.4f}")
                print("Using default parameters instead.")
                return 0.089, 0.506
            print(f"Fitted parameters: s={s_fit:.4f}, c={c_fit:.4f}")
            return s_fit, c_fit
        else:
            print(f"Warning: Optimization did not converge. Using default parameters.")
            return 0.089, 0.506
    
    def calculate_projected_ae_e(self, solitaire_set_data: List[Tuple[List[float], List[float]]]) -> float:
        """
        Calculate Expected Average Error (AE_e) on solitaire set.
        
        Args:
            solitaire_set_data: List of (move_values, deltas) tuples
            
        Returns:
            AE_e value
        """
        if not solitaire_set_data:
            return 0.0
            
        total_expected_error = 0.0
        
        for move_values, deltas in solitaire_set_data:
            probabilities = self.calculate_move_probabilities(move_values)
            
            # Sum p_i * delta_i for i >= 1 (non-best moves)
            expected_error_t = 0.0
            if len(probabilities) > 1 and len(deltas) == len(probabilities) - 1:
                for p_i, delta_i in zip(probabilities[1:], deltas):
                    expected_error_t += p_i * delta_i
                    
            total_expected_error += expected_error_t
        
        AE_e = total_expected_error / len(solitaire_set_data)
        return AE_e
    
    def calculate_ipr(self, AE_e: float) -> float:
        """Calculate IPR from Expected Average Error."""
        return IPR_INTERCEPT + IPR_SLOPE * AE_e
    
    def process_pgn(self, pgn_path: Path) -> Dict:
        """
        Process PGN file and calculate IPR for each player.
        
        Returns:
            Dictionary with player statistics
        """
        print(f"Processing {pgn_path}...")
        
        if not pgn_path.exists():
            print(f"Error: PGN file not found at {pgn_path}")
            sys.exit(1)
        
        self.start_engine()
        
        player_data = defaultdict(lambda: {
            'test_set': [],      # (move_values, actual_move_idx)
            'solitaire_set': [], # (move_values, deltas)
            'games': 0,
            'elos': []
        })
        
        try:
            with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
                game_count = 0
                
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    game_count += 1
                    print(f"Processing game {game_count}...", end='\r')
                    
                    white = game.headers.get("White", "Unknown")
                    black = game.headers.get("Black", "Unknown")
                    
                    # Get Elos
                    try:
                        w_elo = int(game.headers.get("WhiteElo", "0"))
                    except ValueError:
                        w_elo = 0
                    try:
                        b_elo = int(game.headers.get("BlackElo", "0"))
                    except ValueError:
                        b_elo = 0
                    
                    if w_elo > 0:
                        player_data[white]['elos'].append(w_elo)
                    if b_elo > 0:
                        player_data[black]['elos'].append(b_elo)
                    
                    player_data[white]['games'] += 1
                    player_data[black]['games'] += 1
                    
                    # Process moves
                    board = game.board()
                    node = game
                    ply_count = 0
                    
                    while node.variations:
                        next_node = node.variation(0)
                        actual_move = next_node.move
                        ply_count += 1
                        
                        # Skip book moves
                        if ply_count <= BOOK_MOVES * 2:
                            board.push(actual_move)
                            node = next_node
                            continue
                        
                        # Get Multi-PV analysis
                        try:
                            multipv_results = self.analyze_position_multipv(board)
                        except Exception as e:
                            print(f"\nWarning: Analysis failed for position: {e}")
                            board.push(actual_move)
                            node = next_node
                            continue
                        
                        if not multipv_results:
                            board.push(actual_move)
                            node = next_node
                            continue
                        
                        # Best move evaluation
                        best_eval_cp = multipv_results[0][1]
                        
                        # Apply garbage time filter
                        if abs(best_eval_cp) > CAP_EVAL:
                            board.push(actual_move)
                            node = next_node
                            continue
                        
                        # Convert to pawn units
                        move_values = [eval_cp / 100.0 for _, eval_cp in multipv_results]
                        
                        # Adjust for player perspective
                        current_player = white if board.turn == chess.WHITE else black
                        
                        if board.turn == chess.BLACK:
                            # Flip evaluations for Black
                            move_values = [-v for v in move_values]
                        
                        # Find actual move index
                        actual_move_idx = -1
                        for idx, (move, _) in enumerate(multipv_results):
                            if move == actual_move:
                                actual_move_idx = idx
                                break
                        
                        if actual_move_idx == -1:
                            # Actual move not in top N
                            board.push(actual_move)
                            node = next_node
                            continue
                        
                        # Calculate deltas for solitaire set
                        v0 = move_values[0]
                        deltas = [self.calculate_delta(v0, vi) for vi in move_values[1:]]
                        
                        # Add to player's data
                        player_data[current_player]['test_set'].append((move_values, actual_move_idx))
                        player_data[current_player]['solitaire_set'].append((move_values, deltas))
                        
                        board.push(actual_move)
                        node = next_node
                
                print(f"\nProcessed {game_count} games.")
                
        finally:
            self.stop_engine()
        
        return player_data
    
    def calculate_player_iprs(self, player_data: Dict) -> List[Dict]:
        """
        Calculate IPR for each player.
        
        Returns:
            List of player statistics dictionaries
        """
        results = []
        
        for player_name, data in player_data.items():
            if not data['test_set'] or not data['solitaire_set']:
                continue
            
            print(f"\nCalculating IPR for {player_name}...")
            
            # Step 1: Fit parameters
            s_fit, c_fit = self.fit_parameters_sc(data['test_set'])
            self.s, self.c = s_fit, c_fit
            
            # Step 2: Calculate AE_e
            AE_e = self.calculate_projected_ae_e(data['solitaire_set'])
            
            # Step 3: Calculate IPR
            ipr = self.calculate_ipr(AE_e)
            
            # Validate IPR value
            if not math.isfinite(ipr):
                print(f"  Warning: Invalid IPR value (infinity or NaN). Skipping player.")
                continue
            
            avg_elo = sum(data['elos']) / len(data['elos']) if data['elos'] else 0
            
            results.append({
                'Player': player_name,
                'Elo': int(avg_elo) if avg_elo > 0 else 'N/A',
                'Games': data['games'],
                's_fit': f"{s_fit:.4f}",
                'c_fit': f"{c_fit:.4f}",
                'AE_e': f"{AE_e:.4f}",
                'IPR': int(ipr)
            })
            
            print(f"  s={s_fit:.4f}, c={c_fit:.4f}, AE_e={AE_e:.4f}, IPR={int(ipr)}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Calculate IPR using Multi-PV analysis with Stockfish.")
    parser.add_argument("pgn_file", help="Path to input PGN file")
    parser.add_argument("--depth", type=int, default=15, help="Analysis depth (default: 15)")
    parser.add_argument("--multipv", type=int, default=MULTI_PV, help=f"Number of PV lines (default: {MULTI_PV})")
    
    args = parser.parse_args()
    
    input_path = Path(args.pgn_file)
    if not input_path.exists():
        print(f"Error: File {input_path} not found.")
        sys.exit(1)
    
    # Initialize calculator
    calculator = MultiPVIPRCalculator(STOCKFISH_PATH, depth=args.depth, multi_pv=args.multipv)
    
    # Process PGN
    player_data = calculator.process_pgn(input_path)
    
    # Calculate IPRs
    results = calculator.calculate_player_iprs(player_data)
    
    if not results:
        print("No results to output.")
        return
    
    # Write CSV
    output_csv = input_path.parent / (input_path.stem + "_MultiPV_IPR.csv")
    
    print(f"\nWriting results to {output_csv}...")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Player', 'Elo', 'Games', 's_fit', 'c_fit', 'AE_e', 'IPR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in sorted(results, key=lambda x: x['Player']):
            writer.writerow(row)
    
    print("Done.")

if __name__ == "__main__":
    main()
