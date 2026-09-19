#Requires -Version 5.1
<#
.SYNOPSIS
  Download and install native PostgreSQL 16 on Windows for arkguru.

.DESCRIPTION
  Idempotent. Safe to re-run. Does not open port 5432 on the LAN.

  1. Installs PostgreSQL 16 via winget if psql is missing
  2. Adds C:\Program Files\PostgreSQL\16\bin to the user PATH
  3. Starts the postgresql-x64-16 Windows service
  4. Creates role rag / password change-me / database rag on 127.0.0.1:5432
  5. CREATE EXTENSION vector (needs pgvector; see the printed nmake steps if this fails)
  6. Writes gitignored .env with PG_DSN

  Windows often blocks .ps1 files (execution policy). Use one of:

    .\setup_windows_pg.cmd
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\setup_windows_pg.ps1

  Prefer an elevated PowerShell (Right-click -> Run as administrator).

  If this file lives in arkguru-pdf-converter\scripts, .env is written to the repo
  root. If you downloaded the script alone, .env is written to the current directory.

.PARAMETER SuperPassword
  Password for the postgres superuser. EDB/winget unattended default is "postgres".

.PARAMETER SuperUser
  Superuser name. Default postgres.

.PARAMETER PgRoot
  PostgreSQL 16 install directory.
#>
[CmdletBinding()]
param(
    [string]$SuperUser = 'postgres',
    [string]$SuperPassword = 'postgres',
    [string]$PgRoot = 'C:\Program Files\PostgreSQL\16',
    [string]$HostName = '127.0.0.1',
    [int]$Port = 5432,
    [string]$AppUser = 'rag',
    [string]$AppPassword = 'change-me',
    [string]$AppDb = 'rag'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PgBin = Join-Path $PgRoot 'bin'
$Dsn = "postgresql://${AppUser}:${AppPassword}@${HostName}:${Port}/${AppDb}"
$ServiceName = 'postgresql-x64-16'

function Write-Log {
    param([string]$Message)
    Write-Host "[setup_windows_pg] $Message"
}

function Test-IsAdministrator {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Find-Psql {
    $direct = Join-Path $PgBin 'psql.exe'
    if (Test-Path -LiteralPath $direct) { return $direct }
    $cmd = Get-Command psql -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-RepoRoot {
    if ($PSScriptRoot) {
        $parent = Split-Path -Parent $PSScriptRoot
        $hereIsScripts = (Split-Path -Leaf $PSScriptRoot) -eq 'scripts'
        $marker = Join-Path $parent 'compose.yaml'
        if ($hereIsScripts -and (Test-Path -LiteralPath $marker)) {
            return $parent
        }
    }
    return (Get-Location).Path
}

function Invoke-Psql {
    param(
        [Parameter(Mandatory = $true)][string]$PsqlPath,
        [Parameter(Mandatory = $true)][string]$User,
        [Parameter(Mandatory = $true)][string]$Database,
        [Parameter(Mandatory = $true)][string]$Sql,
        [string]$Password,
        [switch]$TuplesOnly,
        [switch]$IgnoreError
    )
    $prev = $env:PGPASSWORD
    if ($Password) { $env:PGPASSWORD = $Password }
    $psqlArgs = @('-h', $HostName, '-p', "$Port", '-U', $User, '-d', $Database, '-v', 'ON_ERROR_STOP=1')
    if ($TuplesOnly) { $psqlArgs += @('-tA') }
    $psqlArgs += @('-c', $Sql)
    try {
        $output = & $PsqlPath @psqlArgs 2>&1
        $code = $LASTEXITCODE
        if ($code -ne 0 -and -not $IgnoreError) {
            throw "psql failed ($code): $output"
        }
        return [pscustomobject]@{ ExitCode = $code; Output = $output }
    }
    finally {
        if ($null -eq $prev) { Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue }
        else { $env:PGPASSWORD = $prev }
    }
}

$onWindows = ($env:OS -eq 'Windows_NT') -or ($PSVersionTable.PSVersion.Major -ge 6 -and $IsWindows)
if (-not $onWindows) {
    throw 'This script installs native Windows PostgreSQL. Run it on your PC, not in Cloud Agent / Linux / macOS.'
}

if (-not (Test-IsAdministrator)) {
    Write-Warning 'Not running as Administrator. winget / the EDB installer / the Windows service usually need an elevated PowerShell. Re-run: Right-click PowerShell -> Run as administrator.'
}

$repoRoot = Get-RepoRoot
Write-Log "repo/env dir: $repoRoot"

$psql = Find-Psql
if (-not $psql) {
    Write-Log 'psql not found; installing PostgreSQL 16 via winget'
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw 'winget is not available. Install App Installer from https://aka.ms/getwinget or install PostgreSQL 16 from https://www.postgresql.org/download/windows/ then re-run this script.'
    }
    winget install --id PostgreSQL.PostgreSQL.16 --exact --accept-package-agreements --accept-source-agreements --disable-interactivity
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
    $psql = Find-Psql
    if (-not $psql) {
        throw "psql still not found after winget. Close this window, open a new Administrator PowerShell, and re-run. Expected: $(Join-Path $PgBin 'psql.exe')"
    }
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not $userPath) { $userPath = '' }
if ($userPath -notlike "*$PgBin*") {
    $newPath = if ($userPath) { "$userPath;$PgBin" } else { $PgBin }
    [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    $env:Path += ";$PgBin"
    Write-Log "added $PgBin to user PATH"
}

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc) {
    if ($svc.Status -ne 'Running') {
        Write-Log "starting $ServiceName"
        Start-Service $ServiceName
    }
    else {
        Write-Log "$ServiceName already running"
    }
}
else {
    Write-Warning "Windows service $ServiceName not found. If PostgreSQL is installed under another name, start it in services.msc."
}

$pgIsReady = Join-Path $PgBin 'pg_isready.exe'
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    if (Test-Path -LiteralPath $pgIsReady) {
        & $pgIsReady -h $HostName -p $Port | Out-Null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    }
    else {
        $probe = Invoke-Psql -PsqlPath $psql -User $SuperUser -Database 'postgres' -Sql 'SELECT 1;' -Password $SuperPassword -IgnoreError
        if ($probe.ExitCode -eq 0) { $ready = $true; break }
    }
    Start-Sleep -Seconds 2
}
if (-not $ready) {
    throw "PostgreSQL is not accepting connections on ${HostName}:${Port}. If you set a custom superuser password during install, re-run: .\setup_windows_pg.ps1 -SuperPassword 'YOUR_PASSWORD'"
}

$role = Invoke-Psql -PsqlPath $psql -User $SuperUser -Database 'postgres' -Password $SuperPassword `
    -TuplesOnly -Sql "SELECT 1 FROM pg_roles WHERE rolname = '$AppUser';"
if (($role.Output | Out-String).Trim() -ne '1') {
    Write-Log "creating role $AppUser"
    Invoke-Psql -PsqlPath $psql -User $SuperUser -Database 'postgres' -Password $SuperPassword `
        -Sql "CREATE ROLE $AppUser LOGIN PASSWORD '$AppPassword';" | Out-Null
}
else {
    Write-Log "updating password for role $AppUser"
    Invoke-Psql -PsqlPath $psql -User $SuperUser -Database 'postgres' -Password $SuperPassword `
        -Sql "ALTER ROLE $AppUser WITH LOGIN PASSWORD '$AppPassword';" | Out-Null
}

$db = Invoke-Psql -PsqlPath $psql -User $SuperUser -Database 'postgres' -Password $SuperPassword `
    -TuplesOnly -Sql "SELECT 1 FROM pg_database WHERE datname = '$AppDb';"
if (($db.Output | Out-String).Trim() -ne '1') {
    Write-Log "creating database $AppDb"
    Invoke-Psql -PsqlPath $psql -User $SuperUser -Database 'postgres' -Password $SuperPassword `
        -Sql "CREATE DATABASE $AppDb OWNER $AppUser;" | Out-Null
}
else {
    Write-Log "database $AppDb already exists"
}

$ext = Invoke-Psql -PsqlPath $psql -User $SuperUser -Database $AppDb -Password $SuperPassword `
    -Sql 'CREATE EXTENSION IF NOT EXISTS vector;' -IgnoreError
if ($ext.ExitCode -ne 0) {
    Write-Warning @'
pgvector is not installed yet. PostgreSQL itself is running.

Install Visual Studio Build Tools with the "Desktop development with C++" workload,
then open "x64 Native Tools Command Prompt for VS" as Administrator:

  set "PGROOT=C:\Program Files\PostgreSQL\16"
  cd %TEMP%
  git clone --branch v0.8.6 https://github.com/pgvector/pgvector.git
  cd pgvector
  nmake /F Makefile.win
  nmake /F Makefile.win install

Then re-run this script, or:

  $env:PGPASSWORD = 'postgres'
  & "C:\Program Files\PostgreSQL\16\bin\psql.exe" -h 127.0.0.1 -p 5432 -U postgres -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
'@
}
else {
    $ver = Invoke-Psql -PsqlPath $psql -User $SuperUser -Database $AppDb -Password $SuperPassword `
        -Sql "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
    Write-Log ("vector extension: " + ($ver.Output | Out-String).Trim())
}

$envFile = Join-Path $repoRoot '.env'
@(
    '# Written by scripts/setup_windows_pg.ps1. Do not commit.'
    '# origin=native-windows'
    "PG_DSN=$Dsn"
) | Set-Content -Path $envFile -Encoding ascii
Write-Log "wrote $envFile"

Write-Host @"

Connection (native Windows PostgreSQL)
  host:      $HostName
  port:      $Port
  database:  $AppDb
  user:      $AppUser
  password:  $AppPassword
  PG_DSN:    $Dsn
  psql:      $psql $Dsn

Verify:
  $psql "$Dsn" -c "SELECT version();"
  $psql "$Dsn" -c "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"
"@
