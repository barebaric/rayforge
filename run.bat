@echo off
setlocal

:: ==========================================================================
:: run.bat - Development Task Runner for Windows
::
:: This script provides a simple, Pixi-like interface for common development
:: tasks. It ensures commands are run inside the MSYS2/MinGW64 environment
:: and will pause on error to allow reading the output.
::
:: Usage:
::   run setup      - Installs all dependencies
::   run test       - Runs the pytest suite
::   run build      - Builds the final .exe
::   run app        - Runs the application from source
::
:: Prerequisite:
::   MSYS2 must be installed at "C:\msys64". If your path is different,
::   please edit the MSYS2_SHELL variable below.
:: ==========================================================================

set "MSYS2_SHELL=C:\msys64\msys2_shell.cmd"
set "MSYS2_ARGS=-mingw64 -no-start -here -c"

:: --- Reusable "pause on error" logic for Bash ---
set "PAUSE_ON_ERROR= || { echo; echo '*** ERROR DETECTED ***'; read -p 'Press [Enter] to close...'; exit 1; }"

:: --- Check for command ---
if "%~1"=="" (
    goto :usage
)

:: --- Command Dispatcher ---
if /i "%~1"=="setup"   goto :setup
if /i "%~1"=="test"    goto :test
if /i "%~1"=="build"   goto :build
if /i "%~1"=="app"     goto :app

echo ERROR: Unknown command "%~1".
echo.
goto :usage

:: --------------------------------------------------------------------------
:: Task Implementations
:: --------------------------------------------------------------------------

:setup
echo.
echo --- Setting up Windows Environment ---
%MSYS2_SHELL% %MSYS2_ARGS% "bash scripts/win_setup.sh%PAUSE_ON_ERROR%"
goto :eof

:test
echo.
echo --- Running Test Suite ---
%MSYS2_SHELL% %MSYS2_ARGS% "bash scripts/win_test.sh%PAUSE_ON_ERROR%"
goto :eof

:build
echo.
echo --- Building Windows Executable ---
%MSYS2_SHELL% %MSYS2_ARGS% "bash scripts/win_build.sh%PAUSE_ON_ERROR%"
goto :eof

:app
echo.
echo --- Running Rayforge Application ---
%MSYS2_SHELL% %MSYS2_ARGS% "(source .msys2_env && python -m rayforge.app %*)%PAUSE_ON_ERROR%"
goto :eof


:: --------------------------------------------------------------------------
:: Usage Information
:: --------------------------------------------------------------------------

:usage
echo Usage: run [command]
echo.
echo Available commands:
echo   setup      Install all dependencies
echo   test       Run the test suite
echo   build      Build the Windows executable
echo   app        Run the application from source
goto :eof
