@echo off
rem Refresh: fetch finished games, rebuild the lake, refit, cache the upcoming slate.
cd /d "%~dp0"
set PYTHONUTF8=1
echo ==== %date% %time% >> "data\refresh.log"
".venv\Scripts\python.exe" live.py >> "data\refresh.log" 2>&1
