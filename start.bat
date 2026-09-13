@echo off
rem Double-click launcher. The real logic is in start.ps1.
rem   start.bat          backend + frontend
rem   start.bat -Local   also start llama.cpp server
rem   start.bat -Stop    shut everything down
rem ASCII only on purpose: cmd reads .bat with the console codepage (cp950 here),
rem so non-ASCII comments would come out garbled.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
echo.
pause
