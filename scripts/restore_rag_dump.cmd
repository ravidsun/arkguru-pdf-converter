@echo off
setlocal EnableExtensions
REM Download-only flow (no git checkout):
REM   Invoke-WebRequest -Uri ".../restore_rag_dump.cmd" -OutFile .\restore_rag_dump.cmd
REM   .\restore_rag_dump.cmd
REM This launcher fetches restore_rag_dump.ps1 when it is not next to the .cmd.
set "DIR=%~dp0"
set "PS1=%DIR%restore_rag_dump.ps1"
set "URL_MAIN=https://raw.githubusercontent.com/ravidsun/arkguru-pdf-converter/refs/heads/main/scripts/restore_rag_dump.ps1"
set "URL_BRANCH=https://raw.githubusercontent.com/ravidsun/arkguru-pdf-converter/refs/heads/cursor/restore-rag-dump-e7c4/scripts/restore_rag_dump.ps1"

findstr /C:"user-agent curl/8.5.0" "%PS1%" >nul 2>nul
if errorlevel 1 (
  echo [restore_rag_dump] restore_rag_dump.ps1 is missing or outdated; downloading
  if exist "%PS1%" del /q "%PS1%"
)

if not exist "%PS1%" (
  echo [restore_rag_dump] restore_rag_dump.ps1 missing; downloading
  curl.exe -fsSL -o "%PS1%" "%URL_MAIN%"
  if errorlevel 1 (
    if exist "%PS1%" del /q "%PS1%"
    curl.exe -fsSL -o "%PS1%" "%URL_BRANCH%"
  )
  if errorlevel 1 (
    if exist "%PS1%" del /q "%PS1%"
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri '%URL_MAIN%' -OutFile '%PS1%' -UseBasicParsing } catch { Invoke-WebRequest -Uri '%URL_BRANCH%' -OutFile '%PS1%' -UseBasicParsing }"
  )
)

if not exist "%PS1%" (
  echo [restore_rag_dump] failed to download restore_rag_dump.ps1
  exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Unblock-File -LiteralPath '%PS1%' -ErrorAction SilentlyContinue"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
exit /b %ERRORLEVEL%
