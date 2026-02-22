@echo off
setlocal
title Editorial Mentor Review

echo [1/3] Navigating to project folder...
F:
cd "F:\ai asac review folder"

set "INPUT_FILE=%~1"
if "%INPUT_FILE%"=="" (
    echo.
    echo [!] No file provided.
    echo Drag and drop a PDF/DOCX onto this batch file, or paste the path below:
    set /p "INPUT_FILE=Path to manuscript: "
)

echo.
echo [2/3] Starting Editorial Mentor (custom questions from editorial_mentor_config.json)...
echo Target: %INPUT_FILE%
echo.
echo Optional: to use first-pass ASAC report for context, run manually with:
echo   python editorial_mentor_review.py "path" --first-pass-report "BaseName_ASAC_Report.json"
echo.

python editorial_mentor_review.py "%INPUT_FILE%"

echo.
echo [3/3] Done.
pause
