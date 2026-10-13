@echo off
rem Refresh: fetch finished games, rebuild the lake, refit, cache the upcoming slate, then
rem store today's final projections for the Results tab (no odds credits; grading is separate:
rem tracking.py grade, about 200 credits per game).
cd /d "%~dp0"
set PYTHONUTF8=1
echo ==== %date% %time% >> "data\refresh.log"
".venv\Scripts\python.exe" live.py >> "data\refresh.log" 2>&1
".venv\Scripts\python.exe" tracking.py snapshot >> "data\refresh.log" 2>&1
