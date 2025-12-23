#!/usr/bin/env python3
"""
IPR Parameter Estimation from PGN Databases

Estimates (s, c) parameter pairs for Intrinsic Performance Ratings
from chess games organized by Elo rating buckets.

Usage:
    python estimate_ipr_params.py buckets.txt --engine stockfish --depth 13
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import chess
import chess.pgn
import chess.engine
from scipy.optimize import minimize, differential_evolution
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MoveAnalysis:
    """Stores analysis results for a single move."""
    position_fen: str
    move_played: str
    move_index: int
    best_move: str
    best_score: float  # in centipawns
    move_score: float  # in centipawns
    alternatives: List[Tuple[str, float]]  # [(move, score), ...]
    game_phase: str  # 'opening', 'middlegame', 'endgame'


@dataclass
class GameAnalysis:
    """Stores analysis for an entire game."""
    white_elo: int
    black_elo: int
    moves: List[MoveAnalysis]
    result: str


@dataclass
class BucketConfig:
    """Configuration for an Elo bucket."""
    name: str
    elo_min: int
    elo_max: int
    pgn_files: List[str]


class EngineAnalyzer:
    """Handles chess engine analysis."""
    
    def __init__(self, engine_path: str, depth: int = 13, multipv: int = 5):
        self.engine_path = engine_path
        self.depth = depth
        self.multipv = multipv
        self.engine = None
    
    def __enter__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.engine:
            self.engine.quit()
    
    def analyze_position(self, board: chess.Board) -> Optional[List[Tuple[str, float]]]:
        """
        Analyze position and return top moves with scores.
        
        Returns:
            List of (move_uci, score_cp) tuples, or None if analysis fails
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Use context manager.")
        
        try:
            info = self.engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth),
                multipv=self.multipv
            )
            
            results = []
            for pv_info in info:
                if 'pv' in pv_info and len(pv_info['pv']) > 0:
                    move = pv_info['pv'][0]
                    score = pv_info['score'].relative
                    
                    # Convert score to centipawns
                    if score.is_mate():
                        # Assign large values for mate scores
                        mate_in = score.mate()
                        cp_score = 10000 - abs(mate_in) * 100
                        if mate_in < 0:
                            cp_score = -cp_score
                    else:
                        cp_score = score.score()
                    
                    results.append((move.uci(), cp_score))
            
            return results if results else None
            
        except Exception as e:
            logger.warning(f"Analysis failed: {e}")
            return None


class IPRCalculator:
    """Calculates IPR-related metrics."""
    
    @staticmethod
    def scaled_difference(v0: float, vi: float) -> float:
        """
        Calculate scaled difference δ using logarithmic scaling.
        
        δ = ∫[vi to v0] 1/(1+|z|) dz
        
        When v0 and vi have same sign, simplifies to:
        |log(1+v0) - log(1+vi)|
        """
        # Convert centipawns to pawn units
        v0_pawns = v0 / 100.0
        vi_pawns = vi / 100.0
        
        # Handle same sign case (most common)
        if (v0_pawns >= 0 and vi_pawns >= 0) or (v0_pawns <= 0 and vi_pawns <= 0):
            delta = abs(math.log(1 + abs(v0_pawns)) - math.log(1 + abs(vi_pawns)))
        else:
            # Different signs - need numerical integration
            # For simplicity, use approximation
            delta = abs(v0_pawns - vi_pawns) / (1 + (abs(v0_pawns) + abs(vi_pawns)) / 2)
        
        return delta
    
    @staticmethod
    def move_probability(delta: float, s: float, c: float, p0: float) -> float:
        """
        Calculate probability of move given parameters.
        
        p_i = p_0^α where α = e^(-(δ/s)^c)
        """
        if delta == 0:
            return p0
        
        alpha = math.exp(-((delta / s) ** c))
        return p0 ** alpha
    
    @staticmethod
    def normalize_probabilities(deltas: List[float], s: float, c: float) -> List[float]:
        """
        Calculate normalized probabilities for all moves.
        
        Returns list of probabilities that sum to 1.
        """
        # Binary search for p0 such that sum of probabilities = 1
        def sum_probs(p0):
            total = p0  # Best move
            for delta in deltas[1:]:  # Other moves
                alpha = math.exp(-((delta / s) ** c))
                total += p0 ** alpha
            return total
        
        # Find p0 using binary search
        p0_low, p0_high = 0.01, 0.99
        for _ in range(50):  # Iterations
            p0_mid = (p0_low + p0_high) / 2
            total = sum_probs(p0_mid)
            
            if abs(total - 1.0) < 1e-6:
                break
            elif total > 1.0:
                p0_high = p0_mid
            else:
                p0_low = p0_mid
        
        p0 = p0_mid
        
        # Calculate all probabilities
        probs = [p0]
        for delta in deltas[1:]:
            alpha = math.exp(-((delta / s) ** c))
            probs.append(p0 ** alpha)
        
        return probs


class GameProcessor:
    """Processes PGN games and extracts move data."""
    
    def __init__(self, engine_path: str, depth: int = 13, multipv: int = 5):
        self.engine_path = engine_path
        self.depth = depth
        self.multipv = multipv
    
    def should_analyze_position(self, board: chess.Board, move_number: int, 
                                position_history: Dict) -> bool:
        """
        Determine if position should be analyzed based on IPR criteria.
        
        Skip:
        - Moves 1-8 (opening book)
        - Repeated positions
        - Positions with >300 cp advantage
        """
        # Skip early moves
        if move_number <= 8:
            return False
        
        # Check for repetition
        fen = board.fen().split(' ')[0]  # Position only, ignore move counters
        if fen in position_history:
            return False
        
        return True
    
    def analyze_game(self, game: chess.pgn.Game) -> Optional[GameAnalysis]:
        """Analyze a single game and extract move data."""
        # Extract Elo ratings
        try:
            white_elo = int(game.headers.get('WhiteElo', 0))
            black_elo = int(game.headers.get('BlackElo', 0))
            
            if white_elo == 0 or black_elo == 0:
                return None
        except (ValueError, TypeError):
            return None
        
        moves_analysis = []
        board = game.board()
        position_history = {}
        move_number = 0
        
        with EngineAnalyzer(self.engine_path, self.depth, self.multipv) as analyzer:
            for node in game.mainline():
                move_number += 1
                
                if not self.should_analyze_position(board, move_number, position_history):
                    board.push(node.move)
                    continue
                
                # Analyze position before move
                analysis = analyzer.analyze_position(board)
                
                if analysis is None or len(analysis) < 2:
                    board.push(node.move)
                    continue
                
                # Get best move and score
                best_move, best_score = analysis[0]
                
                # Skip if position is too one-sided
                if abs(best_score) > 300:
                    board.push(node.move)
                    continue
                
                # Find played move in analysis
                played_move = node.move.uci()
                move_score = None
                
                for move_uci, score in analysis:
                    if move_uci == played_move:
                        move_score = score
                        break
                
                # If move not in top N, analyze it separately
                if move_score is None:
                    board.push(node.move)
                    board.pop()
                    # Would need separate analysis here
                    # For now, skip
                    board.push(node.move)
                    continue
                
                # Store analysis
                move_analysis = MoveAnalysis(
                    position_fen=board.fen(),
                    move_played=played_move,
                    move_index=move_number,
                    best_move=best_move,
                    best_score=best_score,
                    move_score=move_score,
                    alternatives=analysis,
                    game_phase=self._determine_phase(board)
                )
                
                moves_analysis.append(move_analysis)
                
                # Update position history
                fen = board.fen().split(' ')[0]
                position_history[fen] = True
                
                board.push(node.move)
        
        if not moves_analysis:
            return None
        
        return GameAnalysis(
            white_elo=white_elo,
            black_elo=black_elo,
            moves=moves_analysis,
            result=game.headers.get('Result', '*')
        )
    
    def _determine_phase(self, board: chess.Board) -> str:
        """Determine game phase based on material."""
        piece_count = len(board.piece_map())
        
        if board.fullmove_number <= 10:
            return 'opening'
        elif piece_count <= 10:
            return 'endgame'
        else:
            return 'middlegame'
    
    def process_pgn_file(self, pgn_path: str, max_games: Optional[int] = None) -> List[GameAnalysis]:
        """Process all games in a PGN file."""
        games_analyzed = []
        
        logger.info(f"Processing {pgn_path}")
        
        with open(pgn_path) as pgn_file:
            game_count = 0
            
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                game_count += 1
                if max_games and game_count > max_games:
                    break
                
                if game_count % 10 == 0:
                    logger.info(f"  Processed {game_count} games from {Path(pgn_path).name}")
                
                analysis = self.analyze_game(game)
                if analysis:
                    games_analyzed.append(analysis)
        
        logger.info(f"Completed {pgn_path}: {len(games_analyzed)} games analyzed")
        return games_analyzed


class ParameterEstimator:
    """Estimates (s, c) parameters using maximum likelihood."""
    
    def __init__(self, move_data: List[MoveAnalysis]):
        self.move_data = move_data
        self.calculator = IPRCalculator()
    
    def negative_log_likelihood(self, params: Tuple[float, float]) -> float:
        """
        Calculate negative log-likelihood for given (s, c) parameters.
        
        We want to maximize likelihood, so minimize negative log-likelihood.
        """
        s, c = params
        
        # Parameter bounds check
        if s <= 0 or c <= 0:
            return 1e10
        
        total_nll = 0.0
        
        for move in self.move_data:
            # Calculate deltas for all alternatives
            deltas = []
            move_played_idx = None
            
            for idx, (move_uci, score) in enumerate(move.alternatives):
                delta = self.calculator.scaled_difference(move.best_score, score)
                deltas.append(delta)
                
                if move_uci == move.move_played:
                    move_played_idx = idx
            
            if move_played_idx is None:
                continue
            
            # Calculate probabilities
            try:
                probs = self.calculator.normalize_probabilities(deltas, s, c)
                
                # Add log probability of played move
                prob_played = probs[move_played_idx]
                
                if prob_played > 0:
                    total_nll -= math.log(prob_played)
                else:
                    total_nll += 100  # Penalty for zero probability
                    
            except (ValueError, ZeroDivisionError, OverflowError):
                total_nll += 100
        
        return total_nll
    
    def estimate_parameters(self, method: str = 'differential_evolution') -> Tuple[float, float, float]:
        """
        Estimate (s, c) parameters.
        
        Returns:
            (s, c, negative_log_likelihood)
        """
        logger.info(f"Estimating parameters from {len(self.move_data)} moves")
        
        if method == 'differential_evolution':
            # Global optimization - more robust
            bounds = [(0.01, 0.5), (0.1, 2.0)]  # s, c bounds
            
            result = differential_evolution(
                self.negative_log_likelihood,
                bounds,
                maxiter=100,
                popsize=15,
                tol=0.01,
                workers=1,
                seed=42
            )
            
            s, c = result.x
            nll = result.fun
            
        else:
            # Local optimization - faster but may find local minimum
            initial_guess = [0.1, 0.5]
            bounds = [(0.01, 0.5), (0.1, 2.0)]
            
            result = minimize(
                self.negative_log_likelihood,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds
            )
            
            s, c = result.x
            nll = result.fun
        
        logger.info(f"Estimated parameters: s={s:.4f}, c={c:.4f}, NLL={nll:.2f}")
        
        return s, c, nll
    
    def calculate_aee(self, s: float, c: float) -> Tuple[float, float]:
        """
        Calculate Average Expected Error for given parameters.
        
        Returns:
            (AEe, sigma_AEe)
        """
        total_error = 0.0
        variance_sum = 0.0
        
        for move in self.move_data:
            deltas = []
            for move_uci, score in move.alternatives:
                delta = self.calculator.scaled_difference(move.best_score, score)
                deltas.append(delta)
            
            try:
                probs = self.calculator.normalize_probabilities(deltas, s, c)
                
                # Expected error for this position
                expected_error = sum(p * d for p, d in zip(probs[1:], deltas[1:]))
                total_error += expected_error
                
                # Variance contribution
                for i in range(1, len(probs)):
                    variance_sum += probs[i] * (1 - probs[i]) * deltas[i]
                
            except (ValueError, ZeroDivisionError):
                continue
        
        n_moves = len(self.move_data)
        aee = total_error / n_moves if n_moves > 0 else 0
        sigma_aee = math.sqrt(variance_sum / n_moves) if n_moves > 0 else 0
        
        return aee, sigma_aee


def load_buckets_from_text_file(config_path: str) -> List[BucketConfig]:
    """Load bucket configuration from a text file."""
    buckets = []
    with open(config_path) as f:
        for line in f:
            pgn_file = line.strip()
            if pgn_file:
                # Use the filename (without extension) as the bucket name
                bucket_name = Path(pgn_file).stem
                # Since Elo is not in the text file, we'll have to make some assumptions
                # or have a convention for the filenames. For now, let's assume
                # the file name contains the Elo range, e.g., "Elo_2400-2500.pgn"
                # If not, we can set a default or signal an error.
                try:
                    parts = bucket_name.split('_')
                    if len(parts) > 1 and '-' in parts[-1]:
                        elo_parts = parts[-1].split('-')
                        elo_min = int(elo_parts[0])
                        elo_max = int(elo_parts[1])
                    else:
                        # Default values if not in filename
                        elo_min = 0
                        elo_max = 3000
                except (ValueError, IndexError):
                    elo_min = 0
                    elo_max = 3000

                bucket = BucketConfig(
                    name=bucket_name,
                    elo_min=elo_min,
                    elo_max=elo_max,
                    pgn_files=[pgn_file]
                )
                buckets.append(bucket)
    return buckets


def process_bucket(bucket: BucketConfig, engine_path: str, depth: int, 
                   max_games_per_file: Optional[int] = None) -> Dict:
    """Process all games in a bucket and estimate parameters."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing bucket: {bucket.name} (Elo {bucket.elo_min}-{bucket.elo_max})")
    logger.info(f"{'='*60}")
    
    processor = GameProcessor(engine_path, depth)
    all_moves = []
    
    # Process each PGN file
    for pgn_file in bucket.pgn_files:
        if not Path(pgn_file).exists():
            logger.warning(f"File not found: {pgn_file}")
            continue
        
        games = processor.process_pgn_file(pgn_file, max_games_per_file)
        
        for game in games:
            all_moves.extend(game.moves)
    
    logger.info(f"Total moves collected: {len(all_moves)}")
    
    if len(all_moves) < 100:
        logger.warning(f"Insufficient data for bucket {bucket.name}")
        return {
            'bucket': bucket.name,
            'elo_range': [bucket.elo_min, bucket.elo_max],
            'n_moves': len(all_moves),
            'error': 'Insufficient data'
        }
    
    # Estimate parameters
    estimator = ParameterEstimator(all_moves)
    s, c, nll = estimator.estimate_parameters()
    aee, sigma_aee = estimator.calculate_aee(s, c)
    
    # Calculate IPR
    ipr = 3571 - 15413 * aee
    sigma_ipr = 15413 * sigma_aee
    
    results = {
        'bucket': bucket.name,
        'elo_range': [bucket.elo_min, bucket.elo_max],
        'elo_center': (bucket.elo_min + bucket.elo_max) / 2,
        'n_moves': len(all_moves),
        's': s,
        'c': c,
        'nll': nll,
        'aee': aee,
        'sigma_aee': sigma_aee,
        'ipr': ipr,
        'confidence_interval_2sigma': [ipr - 2*sigma_ipr, ipr + 2*sigma_ipr],
        'confidence_interval_2.8sigma': [ipr - 2.8*sigma_ipr, ipr + 2.8*sigma_ipr]
    }
    
    logger.info(f"\nResults for {bucket.name}:")
    logger.info(f"  s = {s:.4f}, c = {c:.4f}")
    logger.info(f"  AEe = {aee:.4f} ± {sigma_aee:.4f}")
    logger.info(f"  IPR = {ipr:.0f} (95% CI: [{ipr-2*sigma_ipr:.0f}, {ipr+2*sigma_ipr:.0f}])")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Estimate IPR parameters from PGN databases organized by Elo buckets'
    )
            parser.add_argument(
        'config',
        help='Text file with a list of PGN bucket files'
    )
    parser.add_argument(
        '--engine',
        default='stockfish',
        help='Path to UCI chess engine (default: stockfish)'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=13,
        help='Engine search depth (default: 13)'
    )
    parser.add_argument(
        '--max-games',
        type=int,
        help='Maximum games to process per PGN file (for testing)'
    )
            parser.add_argument(
        '--output',
        default='ipr_parameters.txt',
        help='Output text file for results'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Process buckets in parallel (experimental)'
    )
    
    args = parser.parse_args()
    
        # Load configuration
    buckets = load_buckets_from_text_file(args.config)
    logger.info(f"Loaded {len(buckets)} buckets from {args.config}")
    
    # Process buckets
    all_results = []
    
    if args.parallel:
        # Parallel processing (be careful with engine instances)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    process_bucket, 
                    bucket, 
                    args.engine, 
                    args.depth, 
                    args.max_games
                ): bucket for bucket in buckets
            }
            
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
    else:
        # Sequential processing
        for bucket in buckets:
            result = process_bucket(bucket, args.engine, args.depth, args.max_games)
            all_results.append(result)
    
    # Sort by Elo
    all_results.sort(key=lambda x: x.get('elo_center', 0))
    
    # Save results
    output_data = {
        'engine': args.engine,
        'depth': args.depth,
        'buckets': all_results,
        'summary': {
            'total_buckets': len(all_results),
            'total_moves': sum(r.get('n_moves', 0) for r in all_results)
        }
    }
    
                    with open(args.output, 'w') as f:
        # We'll write the summary table to the file
        f.write("="*80 + "\\n")
        f.write(f"{'Bucket':<15} {'Elo Range':<15} {'Moves':<8} {'s':<8} {'c':<8} {'IPR':<8}\\n")
        f.write("="*80 + "\\n")
        for result in all_results:
            if 'error' in result:
                f.write(f"{result['bucket']:<15} {result['elo_range'][0]}-{result['elo_range'][1]:<10} "
                        f"{result['n_moves']:<8} ERROR\\n")
            else:
                f.write(f"{result['bucket']:<15} "
                        f"{result['elo_range'][0]}-{result['elo_range'][1]:<10} "
                        f"{result['n_moves']:<8} "
                        f"{result['s']:<8.4f} "
                        f"{result['c']:<8.4f} "
                        f"{result['ipr']:<8.0f}\\n")
        f.write("="*80 + "\\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to {args.output}")
    logger.info(f"{'='*60}")
    
    # Print summary table
    print("\n" + "="*80)
    print(f"{'Bucket':<15} {'Elo Range':<15} {'Moves':<8} {'s':<8} {'c':<8} {'IPR':<8}")
    print("="*80)
    
    for result in all_results:
        if 'error' in result:
            print(f"{result['bucket']:<15} {result['elo_range'][0]}-{result['elo_range'][1]:<10} "
                  f"{result['n_moves']:<8} ERROR")
        else:
            print(f"{result['bucket']:<15} "
                  f"{result['elo_range'][0]}-{result['elo_range'][1]:<10} "
                  f"{result['n_moves']:<8} "
                  f"{result['s']:<8.4f} "
                  f"{result['c']:<8.4f} "
                  f"{result['ipr']:<8.0f}")
    
    print("="*80)


if __name__ == '__main__':
    main()

}
