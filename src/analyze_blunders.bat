@echo off
rem Batch file to run analyze_blunders.py with a PGN file argument

if "%~1"=="" (
    echo Usage: %~nx0 path_to_pgn_file
    exit /b 1
)

rem Resolve the input file to absolute path from current directory
set "PGN_FILE=%~f1"

rem Change to script directory and run
pushd "%~dp0"
python "analyze_blunders.py" "%PGN_FILE%"
popd
