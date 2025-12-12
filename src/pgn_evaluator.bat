@echo off
rem Robust launcher for PGN Evaluator
rem Works from any directory by passing CWD to Python script

rem Get the directory where this batch file resides
set "SCRIPT_DIR=%~dp0"

rem Capture current working directory
set "ORIG_DIR=%CD%"

rem Change to the script directory (so relative imports work)
pushd "%SCRIPT_DIR%"

rem Run the Python script with all original arguments + CWD
python "pgn_evaluator_optimized-GAG.py" %* --cwd "%ORIG_DIR%"

rem Return to original directory
popd
