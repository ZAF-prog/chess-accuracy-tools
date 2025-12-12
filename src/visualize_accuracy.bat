@echo off
rem Launcher for visualize_accuracy.py

if "%~1"=="" (
    echo Usage: %~nx0 path_to_csv_file
    exit /b 1
)

rem Resolve the CSV file to an absolute path
set "CSV_FILE=%~f1"

rem Change to the directory containing this batch file (script location)
pushd "%~dp0"
python "visualize_accuracy.py" "%CSV_FILE%"
popd
