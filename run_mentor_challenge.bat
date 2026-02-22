@echo off
setlocal
title Editorial Mentor — Challenge Your View

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
echo [2/3] Challenge mode: you will be asked a question, then the mentor will use
echo      the article (and quotes) to argue why your view might be wrong.
echo Target: %INPUT_FILE%
echo.

python editorial_mentor_review.py "%INPUT_FILE%" --challenge

echo.
echo [3/3] Done.
pause
