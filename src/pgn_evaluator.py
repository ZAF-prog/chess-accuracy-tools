#!/usr/bin/env python
"""
Program 1: pgn_evaluator.py
----------------------------
Analyzes positions from a PGN file using Stockfish and outputs an annotated PGN
with evaluations stored as comments in the format {[%eval X.XX]}.
"""
import chess
import chess.engine
import chess.pgn
from pathlib import Path
from tqdm import tqdm
import sys
import platform
import re
import io

# --- CONFIGURATION ---
INPUT_PGN = Path("data/historical-gamedata.pgn")
OUTPUT_PGN = Path("data/annotated-gamedata.pgn")

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

ENGINE_PATH = get_engine_path()
print(f"Using Stockfish engine path: {ENGINE_PATH}")

ENGINE_LIMIT = chess.engine.Limit(time=0.1)
MAX_ABS_CP = 3000

def preprocess_pgn(input_path):
    """
    Preprocesses the PGN file to normalize formatting:
    - Replaces multiple consecutive blank lines (2+ newlines) with single blank line
    - Ensures proper game separation
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace any sequence of 2 or more newlines with exactly 2 newlines (one blank line)
    content = re.sub(r'\n{2,}', '\n\n', content)
    
    return content

def format_game_with_linebreaks(game):
    """
    Formats a game with linebreaks after each move pair (white + black).
    Each full move gets its own line.
    """
    output = []
    
    # Write headers
    for key, value in game.headers.items():
        output.append(f'[{key} "{value}"]')
    output.append('')  # Blank line after headers
    
    # Traverse the game tree and format moves
    board = game.board()
    node = game
    move_number = 1
    line_parts = []
    
    while node.variations:
        node = node.variation(0)
        move = node.move
        
        # Determine if this is white's or black's move
        if board.turn == chess.WHITE:
            # White's move - start new line with move number
            if line_parts:
                output.append(' '.join(line_parts))
                line_parts = []
            
            move_san = board.san(move)
            comment = node.comment if node.comment else ''
            if comment:
                line_parts.append(f'{move_number}. {move_san} {comment}')
            else:
                line_parts.append(f'{move_number}. {move_san}')
        else:
            # Black's move - add to current line
            move_san = board.san(move)
            comment = node.comment if node.comment else ''
            if comment:
                line_parts.append(f'{move_san} {comment}')
            else:
                line_parts.append(f'{move_san}')
            move_number += 1
        
        board.push(move)
    
    # Add any remaining moves
    if line_parts:
        output.append(' '.join(line_parts))
    
    # Add result
    result = game.headers.get('Result', '*')
    output.append('')
    output.append(result)
    
    return '\n'.join(output)

def annotate_games():
    """
    Reads games from INPUT_PGN, analyzes each position with Stockfish,
    and writes annotated PGN to OUTPUT_PGN with {[%eval]} comments.
    """
    print(f"Loading games from {INPUT_PGN}...")
    
    if not INPUT_PGN.exists():
        print(f"Error: Input PGN not found at {INPUT_PGN.resolve()}")
        sys.exit(1)

    OUTPUT_PGN.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if not ENGINE_PATH.exists():
            print(f"Error: Engine not found at {ENGINE_PATH.resolve()}. Please check the path.")
            sys.exit(1)
            
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    except Exception as e:
        print(f"Error starting Stockfish engine: {e}")
        sys.exit(1)
    
    try:
        # Preprocess the PGN content
        print("Preprocessing PGN file...")
        pgn_content = preprocess_pgn(INPUT_PGN)
        pgn_io = io.StringIO(pgn_content)
        
        with open(OUTPUT_PGN, "w", encoding="utf-8") as pgn_out:
            
            game_count = 0
            
            while True:
                # Read game from preprocessed content
                original_game = chess.pgn.read_game(pgn_io)
                if original_game is None:
                    break
                
                # Skip games with no moves
                moves = list(original_game.mainline_moves())
                if len(moves) == 0:
                    continue
                    
                game_count += 1
                
                # Create a NEW game with the same headers
                new_game = chess.pgn.Game()
                new_game.headers = original_game.headers.copy()
                
                # Build the new game tree with annotations
                board = new_game.board()
                node = new_game
                
                for move in tqdm(moves, desc=f"Game {game_count}", leave=False):
                    board.push(move)
                    
                    try:
                        info = engine.analyse(board, ENGINE_LIMIT)
                        score = info.get("score")
                        
                        if score.is_mate():
                            mate_in = score.relative.mate()
                            eval_str = f"#{mate_in}"
                        else:
                            cp = score.white().score()
                            eval_str = f"{cp/100:.2f}"
                        
                        # Add the move as a variation with comment
                        node = node.add_variation(move)
                        node.comment = f"{{[%eval {eval_str}]}}"
                        
                    except chess.engine.EngineError:
                        # Add move without comment
                        node = node.add_variation(move)
                        continue
                
                # Write formatted game with linebreaks
                formatted_game = format_game_with_linebreaks(new_game)
                pgn_out.write(formatted_game)
                pgn_out.write('\n\n')
                
            print(f"\n✅ Annotated {game_count} games. Output: {OUTPUT_PGN.resolve()}")
            
    finally:
        if 'engine' in locals():
            engine.quit()
            print("Stockfish engine closed.")

if __name__ == "__main__":
    annotate_games()
