@echo off
REM Double-click to start the web demo. The browser opens by itself when the model is loaded (~10 s).
REM Close this window to stop the demo.
cd /d "%~dp0"
set PYTHONIOENCODING=utf8
set HF_HUB_OFFLINE=1
.venv\Scripts\python.exe app\server.py --open
pause
