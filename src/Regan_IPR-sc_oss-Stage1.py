#!/usr/bin/env python
import argparse
import csv
import os
from pathlib import Path
import sys
import pickle
import chess
import chess.pgn
import chess.engine
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to dynamically determine hash size based on system resources
def get_system_memory_mb(default_mb=4096):
    try:
        import psutil
        mem = psutil.virtual_memory().total
        return int(mem / (1024 * 1024))
    except Exception:
        return default_mb

def get_default_engine_hash(engine_path="stockfish"):
    total_mb = get_system_memory_mb()
    hash_mb = int(total_mb * 0.5)  # 50% of system memory
    return max(16, min(hash_mb, 16384))

# Function to dynamically determine the default engine path
def get_default_engine_path():
    # Adjust this based on your environment and Stockfish installation
    return "stockfish"

class SCModel:
    def __init__(self):
        self.s = None
        self.c = None

    def fit(self, X, y):
        # Placeholder for model fitting logic
        pass

    def mean_absolute_error(self, X, y):
        # Placeholder for MAE calculation
        return 0.0

class GameDataCollector:
    def __init__(self, engine_manager, elo_min=None, elo_max=None, max_games=None):
        self.engine_manager = engine_manager
        self.elo_min = elo_min
        self.elo_max = elo_max
        self.max_games = max_games
        self.positions_X = []
        self.targets_y = []
        self.game_elos = []
        self.game_years = []
        self.num_games_processed = 0

    def process_pgn(self, pgn_path, depth=1, sample_every_n_plies=1, checkpoint_path=None, save_every_n_games=100):
        with open(pgn_path) as f:
            game = chess.pgn.read_game(f)
            while game:
                if self.max_games is not None and self.num_games_processed >= self.max_games:
                    break

                # Process the game
                elo_avg = sum(game.headers.get("WhiteElo", 0), game.headers.get("BlackElo", 0)) / 2
                if self.elo_min is not None and elo_avg < self.elo_min:
                    continue
                if self.elo_max is not None and elo_avg > self.elo_max:
                    continue

                year = int(game.headers.get("Date", "0000")[0:4])
                move_number = 1

                for move in game.mainline_moves():
                    position = game.board()
                    eval_result = self.engine_manager.engine.analyse(position, chess.engine.Limit(depth=depth))

                    if move_number % sample_every_n_plies == 0:
                        self.positions_X.append(self._extract_features(position))
                        self.targets_y.append(eval_result["score"].pov(chess.WHITE).value)
                        self.game_elos.append(elo_avg)
                        self.game_years.append(year)

                    position.push(move)
                    move_number += 1

                self.num_games_processed += 1
                if checkpoint_path and self.num_games_processed % save_every_n_games == 0:
                    save_checkpoint(checkpoint_path, {
                        "collector": self,
                        "model": None,
                    })

                game = chess.pgn.read_game(f)

    def _extract_features(self, position):
        # Placeholder for feature extraction logic
        return []

class StockfishManager:
    def __init__(self, engine_path="stockfish", multipv=1, hash_fraction=None):
        self.engine_path = engine_path
        self.multipv = multipv
        self.hash_fraction = hash_fraction if hash_fraction is not None else 0.5
        self.engine = None

    def start(self):
        if self.engine is not None:
            return

        print(f"Engine path: {self.engine_path}")  # Add this line to verify the path

        hash_mb = get_default_engine_hash(self.engine_path)
        self.engine = chess.engine.SimpleEngine.popen_uci([self.engine_path])
        self.engine.configure({
            "Hash": hash_mb,
            "Threads": get_num_cores() * 0.6
        })

    def close(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

def save_checkpoint(checkpoint_path, data_dict):
    with open(checkpoint_path, "wb") as f:
        pickle.dump(data_dict, f)

def load_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)

def get_num_cores():
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except Exception:
        return 1

# Main program logic
def main():
    parser = argparse.ArgumentParser(description="Fit s and c parameters from PGN using Stockfish evaluations.")
    parser.add_argument("pgn_file", help="Input PGN file")
    parser.add_argument("--engine", default=get_default_engine_path(), help="Path to Stockfish binary")
    parser.add_argument("--multipv", type=int, default=20, help="MultiPV setting for Stockfish")
    parser.add_argument("--elo-min", type=int, default=None, help="Minimum average Elo")
    parser.add_argument("--elo-max", type=int, default=None, help="Maximum average Elo")
    parser.add_argument("--max-games", type=int, default=None, help="Max number of games to process")
    parser.add_argument("--depth", type=int, default=1, help="Stockfish search depth")
    parser.add_argument("--sample-every-n-plies", type=int, default=1, help="Sample every N plies")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint pickle file")
    parser.add_argument("--save-every-n-games", type=int, default=100, help="Save a checkpoint every N games")

    args = parser.parse_args()

    pgn_path = Path(args.pgn_file)
    if not pgn_path.exists():
        print(f"PGN file not found: {pgn_path}", file=sys.stderr)
        sys.exit(1)

    base_name = pgn_path.stem
    output_dir = pgn_path.parent
    checkpoint_path = output_dir / f"{base_name}_checkpoint.pkl"
    output_csv = output_dir / f"{base_name}.csv"

    # Determine CPU usage policy as requested
    total_cores = get_num_cores()
    used_cores = max(1, int(total_cores * 0.6))

    engine_manager = StockfishManager(
        engine_path=args.engine,
        multipv=args.multipv,
        hash_fraction=0.8,
    )

    # Checkpoint handling
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
        try:
            engine_manager.engine.configure({"Threads": used_cores})
        except Exception:
            pass

        should_process_pgn = True
        if args.max_games is not None and collector.num_games_processed >= args.max_games:
            print(f"Checkpoint indicates {collector.num_games_processed} games already processed; satisfies --max-games.")
            should_process_pgn = False

        if should_process_pgn:
            print(f"Starting/resuming PGN processing. Already processed: {collector.num_games_processed} games.")
            collector.process_pgn(
                pgn_path=str(pgn_path),
                depth=args.depth,
                sample_every_n_plies=args.sample_every_n_plies,
                checkpoint_path=checkpoint_path,
                save_every_n_games=args.save_every_n_games,
            )
            print(f"Finished PGN processing. Total games processed: {collector.num_games_processed}.")

            save_checkpoint(checkpoint_path, {
                "collector": collector,
                "model": model,
            })
            print(f"Checkpoint saved to {checkpoint_path}")
        else:
            print("Skipping PGN processing.")

        # Fit model s, c
        X = collector.positions_X
        y = collector.targets_y

        if not X or not y:
            print("No data collected (X, y are empty). Cannot fit model.", file=sys.stderr)
            sys.exit(1)

        if model is None:
            model = SCModel()
        model.fit(X, y)

        ae_e = model.mean_absolute_error(X, y)

        num_moves = len(X)
        num_games = len(collector.game_elos)

        min_elo = float(np.min(collector.game_elos)) if num_games > 0 else None
        max_elo = float(np.max(collector.game_elos)) if num_games > 0 else None
        avg_elo = float(np.mean(collector.game_elos)) if num_games > 0 else None

        years = [y for y in collector.game_years if y is not None]
        first_year = int(np.min(years)) if years else None
        last_year = int(np.max(years)) if years else None

        save_checkpoint(checkpoint_path, {
            "collector": collector,
            "model": model,
        })
        print(f"Final checkpoint with model saved to {checkpoint_path}")

    finally:
        engine_manager.close()

    row = {
        "filename": base_name,
        "MULTI_PV": args.multipv,
        "Number_Games": collector.num_games_processed,
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
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Output written to {output_csv}")

if __name__ == "__main__":
    main()

