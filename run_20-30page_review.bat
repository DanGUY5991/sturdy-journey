@echo off
setlocal enabledelayedexpansion
title Journal Editor Coach (20-30 Page Support)

:: Use IPEX-LLM Ollama from this folder (script will start or wait for it here)
set "OLLAMA_DIR=F:\ollama-ipex-llm-2.2.0-win"
set "OLLAMA_SERVE_PATH=%OLLAMA_DIR%\ollama-serve.bat"

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python or add it to PATH.
    pause
    exit /b 1
)
python --version

echo.
echo [2/4] Navigating to project folder...
F:
cd "F:\ai asac review folder"
if errorlevel 1 (
    echo [ERROR] Could not navigate to project folder: F:\ai asac review folder
    pause
    exit /b 1
)

:: Handle drag-and-drop: %~1 gets the file path (quotes removed by %~1)
set "INPUT_FILE=%~1"

if "%INPUT_FILE%"=="" (
    echo.
    echo [!] No file provided.
    echo Drag and drop a PDF/DOCX onto this batch file, or paste the full path below:
    set /p "INPUT_FILE=Path to manuscript: "
    if "!INPUT_FILE!"=="" (
        echo No path entered. Exiting.
        pause
        exit /b 1
    )
)

:: Verify file exists before running Python (use quotes for paths with spaces)
if not exist "!INPUT_FILE!" (
    echo [ERROR] File not found: !INPUT_FILE!
    echo Please check the path and try again.
    pause
    exit /b 1
)

echo.
echo [3/4] Running 20-30 page editor coach review (8-section analysis)...
echo Target: !INPUT_FILE!
echo Expect 45-90 minutes. Local AI (Ollama) must be running.
echo.

:: Check if Python script exists
if not exist "journal_editor_coach_20-30page.py" (
    echo [ERROR] Python script not found: journal_editor_coach_20-30page.py
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo Starting Python script (will not kill or restart Ollama)...
echo.

python journal_editor_coach_20-30page.py "!INPUT_FILE!" --no-cleanup-env --fresh
set PYTHON_EXIT=%errorlevel%

echo.
echo [4/4] Python script finished with exit code: %PYTHON_EXIT%

if %PYTHON_EXIT% equ 0 (
    echo.
    echo [SUCCESS] Check for *_editor_review.md and *_structured_quotes.json outputs.
) else (
    echo.
    echo [ERROR] Script exited with error code %PYTHON_EXIT%.
    echo Check the messages above for details.
    echo.
    echo Common fixes:
    echo   - Ollama not running: start %OLLAMA_DIR%\ollama-serve.bat first
    echo   - Model not found: run create_JournalEditorCoach_model.bat in this folder
    echo   - Or in a terminal: cd to this folder, then ollama create JournalEditorCoach -f JournalEditorCoach.Modelfile
)

pause
