@echo off
rem Batch file to run chess_accuracy_from_pgn.py with a PGN file argument

if "%~1"=="" (
    echo Usage: %~nx0 path_to_pgn_file
    exit /b 1
)

rem Resolve the input file to absolute path from current directory
set "PGN_FILE=%~f1"

rem Change to script directory and run
pushd "%~dp0"
python "chess_accuracy_from_pgn.py" "%PGN_FILE%"
popd
