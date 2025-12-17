#!/usr/bin/env python
import argparse
import csv
import os
import pickle
#import platformengin
import platform
import sys
from pathlib import Path

import chess
import chess.pgn
import chess.engine
import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------
# Utility: determine system resources (RAM, CPU) best-effort
# ----------------------------------------------------------

def get_system_memory_mb(default_mb=4096):
    """Return approximate total system memory in MB.
    Uses psutil if available, otherwise returns a default.
    """
    try:
        import psutil
        mem = psutil.virtual_memory().total
        return int(mem / (1024 * 1024))
    except Exception:
        return default_mb


def get_num_cores(default_cores=4):
    """Return number of CPU cores, or a default if not available."""
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except Exception:
        return default_cores


def get_default_engine_path():
    """Determines default Stockfish path based on OS."""
    system = platform.system()
    if system == "Windows":
        # Adjust this path to your actual Stockfish location
        return Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
    elif system in ("Linux", "Darwin"):
        return Path("stockfish")  # Assumes 'stockfish' is in PATH
    return Path("stockfish")


# ----------------------------------------------------------
# Engine manager (persistent Stockfish instance)
# ----------------------------------------------------------

class StockfishManager:
    def __init__(self, engine_path="stockfish", multipv=1, hash_fraction=0.8):
        self.engine_path = engine_path
        self.multipv = multipv
        self.hash_fraction = hash_fraction
        self.engine = None

    def start(self):
        if self.engine is not None:
            return

        # Compute hash size ~ 80% of system memory, but clamp to reasonable limits
        total_mb = get_system_memory_mb()
        hash_mb = int(total_mb * self.hash_fraction)
        # It is usually excessive to give more than a few GB to Stockfish in practical usage,
        # but we implement the requirement literally and then clamp to a reasonable max, say 16384 MB.
        hash_mb = max(16, min(hash_mb, 16384))

        self.engine = chess.engine.SimpleEngine.popen_uci([self.engine_path])
        # Set engine options: Hash and MultiPV
        self.engine.configure({
            "Hash": hash_mb,
            "MultiPV": self.multipv
        })

    def analyze_position(self, board, movetime_ms=100):
        """Analyze a position and return main evaluation (cp) or mate score.

        Returns a dict with keys:
        - 'score_cp': centipawn score from side to move perspective (if cp)
        - 'score_mate': mate score if available (int moves to mate, sign indicates side)
        - 'is_mate': bool
        """
        if self.engine is None:
            raise RuntimeError("Engine not started")

        # Run analysis
        limit = chess.engine.Limit(time=movetime_ms / 1000.0)
        info = self.engine.analyse(board, limit, multipv=self.multipv)

        # For MultiPV > 1, info is a list; we use the best line (index 0)
        if isinstance(info, list):
            info0 = info[0]
        else:
            info0 = info

        score = info0["score"]
        result = {
            "score_cp": None,
            "score_mate": None,
            "is_mate": False,
        }

        if score.is_mate():
            result["is_mate"] = True
            result["score_mate"] = score.mate()
        else:
            # Centipawn score relative to side to move
            result["score_cp"] = score.white().score(mate_score=100000)

        return result

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None


# ----------------------------------------------------------
# Data extraction from PGN using Stockfish
# ----------------------------------------------------------

class GameDataCollector:
    def __init__(self, engine_manager, elo_min=None, elo_max=None, max_games=None):
        self.engine_manager = engine_manager
        self.elo_min = elo_min
        self.elo_max = elo_max
        self.max_games = max_games

        self.game_elos = []  # per game average Elo
        self.game_years = []  # per game year
        self.positions_X = []
        self.targets_y = []

    def _game_in_elo_range(self, white_elo, black_elo):
        if white_elo is None or black_elo is None:
            return False
        avg = (white_elo + black_elo) / 2.0
        if self.elo_min is not None and avg < self.elo_min:
            return False
        if self.elo_max is not None and avg > self.elo_max:
            return False
        return True

    def process_pgn(self, pgn_path, movetime_ms=100, sample_every_n_plies=1):
        """Read PGN file, filter games by Elo, and evaluate positions.

        We will:
        - For each selected game, step through moves.
        - At some interval (sample_every_n_plies), analyze the current position
          with Stockfish.
        - Collect feature and target data.

        Here, as an example, we will:
        - Use the engine centipawn evaluation as 'y' (target).
        - Use a simple scalar feature: ply index (move number) as 'X'.

        You can replace this by a richer feature extraction method.
        """
        self.engine_manager.start()

        num_games_processed = 0
        with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                # Extract Elo information
                try:
                    white_elo = int(game.headers.get("WhiteElo", "0"))
                    black_elo = int(game.headers.get("BlackElo", "0"))
                except ValueError:
                    continue

                if not self._game_in_elo_range(white_elo, black_elo):
                    continue

                # Year
                date_str = game.headers.get("Date", "????.??.??")
                year = None
                try:
                    year = int(date_str.split(".")[0])
                except Exception:
                    pass

                board = game.board()

                ply_index = 0
                for move in game.mainline_moves():
                    board.push(move)
                    ply_index += 1

                    if ply_index % sample_every_n_plies != 0:
                        continue

                    info = self.engine_manager.analyze_position(board, movetime_ms=movetime_ms)

                    if info["is_mate"]:
                        # Skip mate positions as numeric target
                        continue

                    cp = info["score_cp"]
                    if cp is None:
                        continue

                    # Simple choice: X = [ply_index], y = cp
                    self.positions_X.append([ply_index])
                    self.targets_y.append(cp)

                avg_elo = (white_elo + black_elo) / 2.0
                self.game_elos.append(avg_elo)
                self.game_years.append(year)

                num_games_processed += 1
                if self.max_games is not None and num_games_processed >= self.max_games:
                    break

        return num_games_processed


# ----------------------------------------------------------
# Model fitting: s and c
# Here we implement a simple linear model: y ≈ s * x + c
# ----------------------------------------------------------

class SCModel:
    def __init__(self):
        self.s = None
        self.c = None
        self.model = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Linear regression without regularization: y = s * x + c
        reg = LinearRegression()
        reg.fit(X, y)

        # For 1D feature, coef_ is a 1-element array
        self.s = float(reg.coef_[0])
        self.c = float(reg.intercept_)
        self.model = reg

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        return self.model.predict(X)

    def mean_absolute_error(self, X, y):
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        return float(np.mean(np.abs(y_pred - y)))


# ----------------------------------------------------------
# Pickle-based restart/checkpoint
# ----------------------------------------------------------

def save_checkpoint(checkpoint_path, data_dict):
    with open(checkpoint_path, "wb") as f:
        pickle.dump(data_dict, f)


def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------
# Main program logic
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fit s and c parameters from PGN using Stockfish evaluations.")
    parser.add_argument("pgn_file", help="Input PGN file")
    parser.add_argument("--engine", default=get_default_engine_path(), help="Path to Stockfish binary")
    parser.add_argument("--multipv", type=int, default=20, help="MultiPV setting for Stockfish")
    parser.add_argument("--elo-min", type=int, default=None, help="Minimum average Elo")
    parser.add_argument("--elo-max", type=int, default=None, help="Maximum average Elo")
    parser.add_argument("--max-games", type=int, default=None, help="Max number of games to process")
    parser.add_argument("--movetime-ms", type=int, default=100, help="Movetime per position in milliseconds")
    parser.add_argument("--sample-every-n-plies", type=int, default=1, help="Sample every N plies")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint pickle file")

    args = parser.parse_args()

    pgn_path = Path(args.pgn_file)
    if not pgn_path.exists():
        print(f"PGN file not found: {pgn_path}", file=sys.stderr)
        sys.exit(1)

    base_name = pgn_path.stem
    output_csv = f"{base_name}.csv"

    # Determine CPU usage policy as requested
    total_cores = get_num_cores()
    # 60% of total cores, at least 1
    used_cores = max(1, int(total_cores * 0.6))

    # For simplicity, we use Stockfish's own threading for parallel search instead of
    # manually distributing work across multiple Python processes.
    # If desired, you can use multiprocessing to parallelize games/positions explicitly.

    # Initialize engine manager
    engine_manager = StockfishManager(
        engine_path=args.engine,
        multipv=args.multipv,
        hash_fraction=0.8,
    )

    # Set engine threads after starting

    # Checkpoint handling
    checkpoint_path = args.checkpoint or f"{base_name}_checkpoint.pkl"
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint is not None:
        print(f"Loaded checkpoint from {checkpoint_path}")
        collector = checkpoint["collector"]
        model = checkpoint.get("model", None)
    else:
        collector = GameDataCollector(
            engine_manager=engine_manager,
            elo_min=args.elo_min,
            elo_max=args.elo_max,
            max_games=args.max_games,
        )
        model = None

    try:
        # Start engine and configure threads
        engine_manager.start()
        # After start, we can set threads option (if engine supports it)
        try:
            engine_manager.engine.configure({"Threads": used_cores})
        except Exception:
            pass

        # If we haven't collected data yet, or want to resume, process PGN
        if not collector.positions_X or not collector.targets_y:
            num_games = collector.process_pgn(
                pgn_path=str(pgn_path),
                movetime_ms=args.movetime_ms,
                sample_every_n_plies=args.sample_every_n_plies,
            )
            print(f"Processed {num_games} games.")

            # Save checkpoint after data collection
            save_checkpoint(checkpoint_path, {
                "collector": collector,
                "model": model,
            })
            print(f"Checkpoint saved to {checkpoint_path}")
        else:
            print("Using data from loaded checkpoint.")

        # Fit model s, c
        X = collector.positions_X
        y = collector.targets_y

        if not X or not y:
            print("No data collected (X, y are empty). Cannot fit model.", file=sys.stderr)
            sys.exit(1)

        if model is None:
            model = SCModel()
        model.fit(X, y)

        # Compute AE_e (mean absolute error)
        ae_e = model.mean_absolute_error(X, y)

        # Aggregate game-level info
        num_games = len(collector.game_elos)
        num_moves = len(X)
        min_elo = float(np.min(collector.game_elos)) if num_games > 0 else None
        max_elo = float(np.max(collector.game_elos)) if num_games > 0 else None
        avg_elo = float(np.mean(collector.game_elos)) if num_games > 0 else None

        years = [y for y in collector.game_years if y is not None]
        first_year = int(np.min(years)) if years else None
        last_year = int(np.max(years)) if years else None

        # Save final checkpoint including model
        save_checkpoint(checkpoint_path, {
            "collector": collector,
            "model": model,
        })
        print(f"Final checkpoint with model saved to {checkpoint_path}")

    finally:
        engine_manager.close()

    # Write CSV output
    # Columns:
    # filename,MULTI_PV,Number_Games,Number_Moves,MinElo,MaxElo,AvgElo,s,c,AE_e,FirstYear,LastYear

    row = {
        "filename": base_name,
        "MULTI_PV": args.multipv,
        "Number_Games": num_games,
        "Number_Moves": num_moves,
        "MinElo": min_elo if min_elo is not None else "",
        "MaxElo": max_elo if max_elo is not None else "",
        "AvgElo": avg_elo if avg_elo is not None else "",
        "s": model.s,
        "c": model.c,
        "AE_e": ae_e,
        "FirstYear": first_year if first_year is not None else "",
        "LastYear": last_year if last_year is not None else "",
    }

    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "filename",
            "MULTI_PV",
            "Number_Games",
            "Number_Moves",
            "MinElo",
            "MaxElo",
            "AvgElo",
            "s",
            "c",
            "AE_e",
            "FirstYear",
            "LastYear",
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Output written to {output_csv}")


if __name__ == "__main__":
    main()

