@echo off
setlocal
title Socratic Mentor — Learn editorial reasoning

echo [1/3] Navigating to project folder...
F:
cd "F:\ai asac review folder"

set "INPUT_FILE=%~1"
if "%INPUT_FILE%"=="" (
    echo.
    echo [!] No file provided.
    echo Drag and drop the manuscript (same as the one you reviewed) onto this batch file,
    echo or paste the path below. The script will auto-detect the ASAC_Report.json.
    set /p "INPUT_FILE=Path to manuscript: "
)

echo.
echo [2/3] Starting Socratic mentor (probing questions from the review)...
echo Target: %INPUT_FILE%
echo.

python socratic_mentor.py "%INPUT_FILE%"

echo.
echo [3/3] Done.
pause
