@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No local .venv interpreter was found.
  echo Open forecast_gui.py in PyCharm using your existing project interpreter.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "forecast_gui.py"
if errorlevel 1 pause
