@echo off
setlocal
title Full ASAC Review (7 stages + evidence trail)

:: Use IPEX-LLM Ollama from this folder
set "OLLAMA_DIR=F:\ollama-ipex-llm-2.2.0-win"
set "OLLAMA_SERVE_PATH=%OLLAMA_DIR%\ollama-serve.bat"

echo [1/3] Navigating to project folder...
F:
cd "F:\ai asac review folder"

set "INPUT_FILE=%~1"
if "%INPUT_FILE%"=="" (
    echo.
    echo [!] No file provided.
    echo Drag and drop a PDF/DOCX onto this batch file or paste the path below:
    set /p "INPUT_FILE=Path to manuscript: "
)

echo.
echo [2/3] Running full 7-stage ASAC review (20-30 page support)...
echo Target: %INPUT_FILE%
echo.

python journal_editor_coach.py "%INPUT_FILE%"

echo.
echo [3/3] Done. Run run_socratic_learning.bat with the same file for Socratic dialogue.
pause
