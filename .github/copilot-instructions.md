# Chess Accuracy Tools - AI Agent Instructions

## Project Overview
This repository analyzes the relationship between chess engine accuracy scores (Stockfish evaluations) and player strength (Elo ratings). It processes PGN game files, annotates positions with engine evaluations, calculates move accuracies using custom formulas, and correlates results with historical player data.

## Architecture
- **Data Flow**: Raw PGN → Stockfish evaluation → Annotated PGN → Accuracy calculations → CSV outputs → Statistical analysis
- **Key Components**:
  - `pgn_evaluator.py`: Annotates PGN files with `[%eval X.XX]` comments using Stockfish
  - `chess_accuracy_from_pgn.py`: Parses annotated PGN to compute move accuracies based on eval deltas
  - Chessmetrics parsers: Extract historical rating data from web scrapes
- **Dependencies**: `python-chess`, `tqdm`, Stockfish binary (Ubuntu/Linux: `binaries/stockfish-ubuntu-x86-64-avx512`)

## Critical Workflows
- **Evaluation Pipeline**: Run `python src/pgn_evaluator.py` to annotate games (uses 0.1s/move limit)
- **Accuracy Calculation**: Execute `python src/chess_accuracy_from_pgn.py <annotated_pgn>` to generate player accuracy stats
- **Cross-Platform Engine**: Scripts auto-detect OS and use appropriate Stockfish binary path
- **Data Processing**: All scripts expect input/output in `data/` directory relative to repo root

## Project Conventions
- **Eval Format**: Centipawns as floats (e.g., `[%eval 0.33]`), mates as `#M6`
- **Accuracy Formula**: Custom exponential decay: `103.17 * exp(-0.0435 * win_diff) - 3.17` (clamped 0-100%)
- **Winning Chances**: Logistic conversion: `50 + 50 * (2/(1+exp(-0.003682*cp)) - 1)`
- **Path Handling**: Use `pathlib.Path` for cross-platform compatibility
- **Error Handling**: Scripts exit with `sys.exit(1)` on missing files/binaries
- **Output Structure**: CSVs with columns like `Player,Elo,Games,Avg_Accuracy,Std_Accuracy`

## Examples
- Annotate a PGN: `python src/pgn_evaluator.py` (hardcoded I/O paths in script)
- Calculate accuracies: `python src/chess_accuracy_from_pgn.py data/annotated-gamedata.pgn`
- Parse chessmetrics: `python src/parse_chessmetrics-profiles_full.py` for player rating extraction

## Key Files
- [src/pgn_evaluator.py](src/pgn_evaluator.py): Core evaluation engine integration
- [src/chess_accuracy_from_pgn.py](src/chess_accuracy_from_pgn.py): Accuracy computation logic
- [data/Capablanca-evaluated.csv](data/Capablanca-evaluated.csv): Example output format
- [binaries/stockfish-ubuntu-x86-64-avx512](binaries/stockfish-ubuntu-x86-64-avx512): Engine binary</content>
<parameter name="filePath">/workspaces/chess-accuracy-tools/.github/copilot-instructions.md