@echo off
setlocal
REM Bypass Windows PowerShell execution policy for this run only.
REM Double-click this file, or from PowerShell:  .\restore_rag_dump.cmd
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Unblock-File -LiteralPath '%~dp0restore_rag_dump.ps1' -ErrorAction SilentlyContinue"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0restore_rag_dump.ps1" %*
exit /b %ERRORLEVEL%
