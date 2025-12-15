@echo off
rem Robust launcher for pare_pgn_mingames (PGN MinGames Filter)

rem Get the directory where this batch file resides
set "SCRIPT_DIR=%~dp0"

rem Run the Python script with all original arguments
rem Use quotes to handle spaces in paths
python "%SCRIPT_DIR%pare_pgn_mingames.py" %*
