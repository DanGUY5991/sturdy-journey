@echo off
setlocal
title AI Journal Editor Reviewer

:: Use IPEX-LLM Ollama from this folder
set "OLLAMA_DIR=F:\ollama-ipex-llm-2.2.0-win"
set "OLLAMA_SERVE_PATH=%OLLAMA_DIR%\ollama-serve.bat"

:: 1. Navigate to the project directory
echo [1/3] Navigating to project folder...
F:
cd "F:\ai asac review folder"

:: 2. Handle input file
set "INPUT_FILE=%~1"

if "%INPUT_FILE%"=="" (
    echo.
    echo [!] No file provided.
    echo Please drag and drop a PDF/DOCX file onto this batch file
    echo or paste the full path below:
    set /p "INPUT_FILE=Path to manuscript: "
)

:: Keep quotes so paths with spaces work; pass through to Python as-is
:: (If user pasted a path without quotes, that's fine too.)

:: 3. Run the Python Reviewer
echo.
echo [2/3] Starting AI Reviewer...
echo Target: %INPUT_FILE%
echo.

python journal_editor_coach.py "%INPUT_FILE%"

echo.
echo [3/3] Process finished.
pause
