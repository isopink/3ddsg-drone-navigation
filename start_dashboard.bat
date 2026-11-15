@echo off
rem Double-click: sets up Python the first time, then opens the runs dashboard.
setlocal
cd /d "%~dp0"

if exist "cf-env\Scripts\python.exe" goto run

echo First start: setting up Python in cf-env (a few minutes, needs internet)...
where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 -m venv cf-env
) else (
    python -m venv cf-env
)
if not exist "cf-env\Scripts\python.exe" (
    echo.
    echo Could not create cf-env. Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH", then double-click this file again.
    pause
    exit /b 1
)
"cf-env\Scripts\python.exe" -m pip install --upgrade pip
"cf-env\Scripts\python.exe" -m pip install -e ".[sim,live]"
if errorlevel 1 (
    echo.
    echo Installing the packages failed. Check the internet connection and try again.
    echo To start over, delete the cf-env folder.
    pause
    exit /b 1
)

:run
echo Starting the runs dashboard. Keep this window open while you experiment.
"cf-env\Scripts\python.exe" scripts\dashboard.py %*
if errorlevel 1 pause
