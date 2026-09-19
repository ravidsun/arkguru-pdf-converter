@echo off
setlocal
REM Bypass Windows PowerShell execution policy for this run only.
REM Double-click this file, or from PowerShell:  .\setup_windows_pg.cmd
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Unblock-File -LiteralPath '%~dp0setup_windows_pg.ps1' -ErrorAction SilentlyContinue"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows_pg.ps1" %*
exit /b %ERRORLEVEL%
