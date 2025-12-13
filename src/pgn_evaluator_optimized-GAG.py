#!/usr/bin/env python
# V1 written in Google Antigravity, then fixed
"""
PGN Evaluator Optimized
-----------------------
Parallel processing, buffered I/O, and quick scan modes for efficient chess game analysis.
"""
import chess
import chess.engine
import chess.pgn
from pathlib import Path
from tqdm import tqdm
import sys
import multiprocessing as mp
import argparse
import time
import io
import atexit

# --- CONFIGURATION DEFAULTS ---
DEFAULT_INPUT_PGN = "./historical-gamedata.pgn"
ENGINE_PATH = Path(r"C:\Users\Public\Libraries\stockfish\stockfish-windows-x86-64-avx2.exe")
MAX_ABS_CP = 3000

# --- ANALYSIS TIME PRESETS ---
ANALYSIS_PRESETS = {
    'quick': 0.05,      # 2x faster, good for quick scans
    'normal': 0.1,      # Default, balanced speed/accuracy
    'deep': 0.2,        # 2x slower, more accurate
    'tournament': 0.5   # 5x slower, high accuracy
}

# --- QUICK SCAN MODES ---
QUICK_SCAN_MODES = {
    'critical': 'Analyze critical positions only (tactical moves, captures)',
    'sample-25': 'Analyze every 4th move (25% of positions)',
    'sample-50': 'Analyze every other move (50% of positions)',
    'opening-endgame': 'Skip middlegame (moves 20-60), analyze opening/endgame',
    'turning-points': 'Analyze only when material changes significantly'
}

# --- OPTIMIZATION SETTINGS ---
BUFFER_SIZE = 10  # Number of games to buffer before writing
DEFAULT_WRITE_CHUNK_SIZE = 50  # Write output after every N games
DEFAULT_NUM_WORKERS = max(1, int(mp.cpu_count() * 0.90))  # Use 90% of cores
USE_MULTIPROCESSING = True
ENGINE_HASH_MB = 128
ENGINE_THREADS = 1

# Global variables (will be set by main)
INPUT_PGN = None
OUTPUT_PGN = None
NUM_WORKERS = DEFAULT_NUM_WORKERS
WRITE_CHUNK_SIZE = DEFAULT_WRITE_CHUNK_SIZE
ENGINE_LIMIT = chess.engine.Limit(time=0.1)
QUICK_SCAN_MODE = None
worker_engine = None  # Global for worker processes


def init_worker():
    """Initialize worker process with its own engine instance"""
    global worker_engine
    try:
        worker_engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        configure_engine(worker_engine)
        atexit.register(worker_engine.quit)
    except Exception as e:
        pass


def configure_engine(engine):
    """Configure engine with optimal settings"""
    try:
        engine.configure({"Hash": ENGINE_HASH_MB, "Threads": ENGINE_THREADS})
    except chess.engine.EngineError:
        pass


def should_analyze_move(board, move, move_num, quick_scan_mode):
    """Determine if a move should be analyzed based on quick scan mode"""
    if not quick_scan_mode:
        return True
    
    if quick_scan_mode == 'critical':
        return (board.is_capture(move) or board.gives_check(move) or move.promotion is not None)
    elif quick_scan_mode == 'sample-25':
        return move_num % 4 == 0
    elif quick_scan_mode == 'sample-50':
        return move_num % 2 == 0
    elif quick_scan_mode == 'opening-endgame':
        return move_num <= 20 or move_num >= 60
    elif quick_scan_mode == 'turning-points':
        return board.is_capture(move)
    
    return True


def count_games(pgn_path):
    """Count total games in PGN file for progress reporting"""
    count = 0
    try:
        with open(pgn_path, encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line: break
                if line.startswith("[Event "): count += 1
    except:
        pass
    return count if count > 0 else 100  # Fallback estimate


def format_game_per_move_pair(game):
    """Format game with one move pair per line"""
    # Get headers using a dummy game to ensure correct PGN format
    dummy = chess.pgn.Game()
    dummy.headers = game.headers
    header_str = str(dummy).split("\n\n")[0]
    
    body_lines = []
    current_line = ""
    
    node = game
    while node.variations:
        next_node = node.variations[0]
        board = node.board()
        san = board.san(next_node.move)
        comment = f" {next_node.comment}" if next_node.comment else ""
        if board.turn == chess.WHITE:
            if current_line:
                body_lines.append(current_line)
            current_line = f"{board.fullmove_number}. {san}{comment}"
        else:
            current_line += f" {san}{comment}"
            
        node = next_node
        
    if current_line:
        body_lines.append(current_line)
        
    # Result
    result = game.headers.get("Result", "*")
    if body_lines:
        body_lines[-1] += f" {result}"
    else:
        body_lines.append(result)
        
    return header_str + "\n\n" + "\n".join(body_lines) + "\n\n"


def annotate_single_game(game_data):
    """Worker function for multiprocessing"""
    game_string, game_num = game_data
    
    global worker_engine
    if worker_engine is None:
        return ""
    
    engine = worker_engine

    try:
        game = chess.pgn.read_game(io.StringIO(game_string))
        if game is None: return ""

        # Skip empty games (often caused by trailing newlines or result strings)
        # We handle this here now because the raw text splitter is less strict
        if not game.mainline_moves() and game.headers.get("Event", "?") == "?":
            return ""
        
        board = game.board()
        annotated_game = chess.pgn.Game()
        annotated_game.headers = game.headers.copy()
        annotated_node = annotated_game
        
        move_num = 0
        for move in game.mainline_moves():
            board.push(move)
            move_num += 1
            
            if should_analyze_move(board, move, move_num, QUICK_SCAN_MODE):
                try:
                    info = engine.analyse(board, ENGINE_LIMIT)
                    score = info.get("score")
                    
                    if score.is_mate():
                        mate_in = score.relative.mate()
                        eval_str = f"#M{mate_in}"
                    else:
                        cp = score.white().score()
                        eval_str = f"{cp/100:.2f}"
                    
                    annotated_node = annotated_node.add_variation(move)
                    annotated_node.comment = f"{{[%eval {eval_str}]}}"
                except:
                    annotated_node = annotated_node.add_variation(move)
            else:
                annotated_node = annotated_node.add_variation(move)
        
        return format_game_per_move_pair(annotated_game)
        
    except Exception:
        return ""


def read_game_strings(pgn_path):
    """
    Generator that yields raw game strings by splitting on [Event tag.
    This is much faster than parsing every game in the main process.
    """
    with open(pgn_path, encoding="utf-8") as f:
        buffer = []
        game_num = 0
        
        for line in f:
            # Standard PGNs start new games with the Event tag
            if line.startswith("[Event \""):
                if buffer:
                    full_game_str = "".join(buffer)
                    # Only yield if it looks like a game (prevents yielding start-of-file garbage)
                    if "[Event" in full_game_str: 
                        game_num += 1
                        yield (full_game_str, game_num)
                    buffer = []
            buffer.append(line)
            
        # Yield last game
        if buffer:
            full_game_str = "".join(buffer)
            if "[Event" in full_game_str:
                game_num += 1
                yield (full_game_str, game_num)


def annotate_games_parallel():
    """Parallel processing mode with incremental output"""
    print(f"Loading games from {INPUT_PGN}...")
    print(f"Using {NUM_WORKERS} worker processes")
    print(f"Writing output every {WRITE_CHUNK_SIZE} games")
    
    total_games = count_games(INPUT_PGN)
    
    with open(OUTPUT_PGN, "w", encoding="utf-8", buffering=1) as pgn_out:
        with mp.Pool(NUM_WORKERS, initializer=init_worker) as pool:
            # Use imap (not imap_unordered) to preserve game order
            results = pool.imap(
                annotate_single_game,
                read_game_strings(INPUT_PGN),
                chunksize=5
            )
            
            games_processed = 0
            for annotated_game_str in tqdm(results, total=total_games, desc="Processing"):
                if annotated_game_str:
                    pgn_out.write(annotated_game_str)
                    games_processed += 1
                    
                    # Flush to disk every WRITE_CHUNK_SIZE games
                    if games_processed % WRITE_CHUNK_SIZE == 0:
                        pgn_out.flush()
    
    print(f"\n✅ Annotated {games_processed} games saved to: {OUTPUT_PGN}")



def annotate_games_sequential():
    """Sequential processing mode with incremental output"""
    print(f"Loading games from {INPUT_PGN}...")
    print("Sequential processing mode")
    print(f"Writing output every {WRITE_CHUNK_SIZE} games")
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        configure_engine(engine)
    except FileNotFoundError:
        print(f"Error: Engine not found at {ENGINE_PATH}")
        sys.exit(1)
    
    try:
        total_games = count_games(INPUT_PGN)
        
        with open(INPUT_PGN, encoding="utf-8") as pgn_in, \
             open(OUTPUT_PGN, "w", encoding="utf-8", buffering=1) as pgn_out:
            
            write_buffer = []
            games_processed = 0
            
            for _ in range(total_games):
                game = chess.pgn.read_game(pgn_in)
                if game is None: break
                
                # Skip empty games (often caused by trailing newlines or result strings)
                if not game.mainline_moves() and game.headers.get("Event", "?") == "?":
                    continue
                
                board = game.board()
                annotated_game = chess.pgn.Game()
                annotated_game.headers = game.headers.copy()
                annotated_node = annotated_game
                
                move_num = 0
                for move in game.mainline_moves():
                    board.push(move)
                    move_num += 1
                    
                    if should_analyze_move(board, move, move_num, QUICK_SCAN_MODE):
                        try:
                            info = engine.analyse(board, ENGINE_LIMIT)
                            score = info.get("score")
                            
                            if score.is_mate():
                                mate_in = score.relative.mate()
                                eval_str = f"#M{mate_in}"
                            else:
                                cp = score.white().score()
                                eval_str = f"{cp/100:.2f}"
                            
                            annotated_node = annotated_node.add_variation(move)
                            annotated_node.comment = f"{{[%eval {eval_str}]}}"
                        except:
                            annotated_node = annotated_node.add_variation(move)
                    else:
                        annotated_node = annotated_node.add_variation(move)
                write_buffer.append(format_game_per_move_pair(annotated_game))
                games_processed += 1
                
                if len(write_buffer) >= BUFFER_SIZE:
                    pgn_out.write("".join(write_buffer))
                    write_buffer = []
                    
                    # Flush to disk every WRITE_CHUNK_SIZE games
                    if games_processed % WRITE_CHUNK_SIZE == 0:
                        pgn_out.flush()
            
            if write_buffer:
                pgn_out.write("".join(write_buffer))
        
        print(f"\n✅ Annotated {games_processed} games saved to: {OUTPUT_PGN}")
        
    finally:
        engine.quit()


if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows
    
    parser = argparse.ArgumentParser(description='PGN Evaluator Optimized')
    
    parser.add_argument('input_file', nargs='?', default=DEFAULT_INPUT_PGN,
                        help=f'Input PGN file (default: {DEFAULT_INPUT_PGN})')
    
    parser.add_argument('-o', '--output', dest='output_file',
                        help='Output PGN file')
    
    parser.add_argument('--sequential', action='store_true',
                        help='Disable multiprocessing')
    
    parser.add_argument('--workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='Number of worker processes')
    
    # Analysis options
    parser.add_argument('--analysis-time', choices=list(ANALYSIS_PRESETS.keys()),
                        help='Analysis time preset')
    
    parser.add_argument('--time', type=float,
                        help='Custom analysis time in seconds')
    
    parser.add_argument('--quick-scan', choices=list(QUICK_SCAN_MODES.keys()),
                        help='Quick scan mode')
    
    parser.add_argument('--write-chunk', type=int, default=DEFAULT_WRITE_CHUNK_SIZE,
                        help=f'Write output to disk after every N games (default: {DEFAULT_WRITE_CHUNK_SIZE})')
    
    parser.add_argument('--cwd', help='Current working directory (internal use)')
    
    args = parser.parse_args()
    
    # Setup globals
    if args.cwd and not Path(args.input_file).is_absolute():
        INPUT_PGN = Path(args.cwd) / args.input_file
    else:
        INPUT_PGN = Path(args.input_file)
    
    if args.output_file:
        OUTPUT_PGN = Path(args.output_file)
    else:
        OUTPUT_PGN = INPUT_PGN.parent / f"{INPUT_PGN.stem}-evaluated{INPUT_PGN.suffix}"
    
    if args.sequential:
        USE_MULTIPROCESSING = False
    
    if args.workers:
        NUM_WORKERS = args.workers
    
    if args.write_chunk:
        WRITE_CHUNK_SIZE = args.write_chunk
        
    # Set analysis time
    if args.analysis_time:
        ENGINE_LIMIT = chess.engine.Limit(time=ANALYSIS_PRESETS[args.analysis_time])
        print(f"Using '{args.analysis_time}' preset")
    elif args.time:
        ENGINE_LIMIT = chess.engine.Limit(time=args.time)
        print(f"Using custom time: {args.time}s")
    
    if args.quick_scan:
        QUICK_SCAN_MODE = args.quick_scan
        print(f"Quick scan mode: {args.quick_scan}")

    # Validation
    if not INPUT_PGN.exists():
        print(f"Error: Input PGN not found at {INPUT_PGN}")
        sys.exit(1)
        
    if not ENGINE_PATH.exists():
        print(f"Error: Stockfish engine not found at {ENGINE_PATH}")
        sys.exit(1)
        
    print(f"Input: {INPUT_PGN}")
    print(f"Output: {OUTPUT_PGN}")
    
    if USE_MULTIPROCESSING and NUM_WORKERS > 1:
        annotate_games_parallel()
    else:
        annotate_games_sequential()
