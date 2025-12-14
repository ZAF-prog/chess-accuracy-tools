@echo off
rem Robust launcher for Regan_IPR_oss_PGN
rem Works from any directory by passing CWD to Python script

rem Get the directory where this batch file resides
set "SCRIPT_DIR=%~dp0"

rem Run the Python script with all original arguments
rem Use quotes to handle spaces in paths
python "%SCRIPT_DIR%Regan_IPR_oss_PGN.py" %*

