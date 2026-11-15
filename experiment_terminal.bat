@echo off
rem Double-click: a terminal in this folder with cf-env active, ready to fly.
cd /d "%~dp0"
if not exist "cf-env\Scripts\activate.bat" (
    echo Run setup.bat once first: it sets up Python.
    pause
    exit /b 1
)
echo.
echo   Simulation:  python scripts\fly_route.py --backend sim --rooms 2 2 2 --depth 2 "go from A to C"
echo   Real drone:  python scripts\fly_route.py --backend crazyflie --rooms 2 2 --depth 2 "from A to B"
echo.
cmd /k call "cf-env\Scripts\activate.bat"
