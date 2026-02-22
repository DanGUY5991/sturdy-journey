@echo off
setlocal
title Create JournalEditorCoach model

set "OLLAMA_DIR=F:\ollama-ipex-llm-2.2.0-win"
:: Use the SAME ollama.exe as the server to avoid "server must be updated" version mismatch
set "OLLAMA_EXE=%OLLAMA_DIR%\ollama.exe"

echo.
echo This will create the "JournalEditorCoach" model used by the review scripts.
echo.
echo IMPORTANT: Use the ollama.exe from IPEX-LLM folder (not the one in PATH).
echo 1. Start Ollama first: run %OLLAMA_DIR%\ollama-serve.bat and wait until ready.
echo 2. Then run this .bat, or in PowerShell run the command shown at the end.
echo.
pause

F:
cd "F:\ai asac review folder"

if not exist "JournalEditorCoach.Modelfile" (
    echo [ERROR] JournalEditorCoach.Modelfile not found in current folder.
    pause
    exit /b 1
)

if not exist "%OLLAMA_EXE%" (
    echo [ERROR] Ollama client not found: %OLLAMA_EXE%
    echo Use the ollama.exe from your IPEX-LLM folder to match the server version.
    echo In PowerShell: "& 'F:\ollama-ipex-llm-2.2.0-win\ollama.exe' create JournalEditorCoach -f JournalEditorCoach.Modelfile"
    pause
    exit /b 1
)

echo.
echo Creating model (using IPEX-LLM ollama.exe so client matches server)...
echo.

"%OLLAMA_EXE%" create JournalEditorCoach -f JournalEditorCoach.Modelfile

set EXIT_CODE=%errorlevel%
echo.
if %EXIT_CODE% equ 0 (
    echo [SUCCESS] Model JournalEditorCoach created. You can now run the review .bat files.
) else (
    echo [ERROR] Create failed with exit code %EXIT_CODE%.
    echo Make sure Ollama is running from %OLLAMA_DIR% and the base model qwen2.5:14b-instruct-q5_k_m is available.
)
echo.
pause
