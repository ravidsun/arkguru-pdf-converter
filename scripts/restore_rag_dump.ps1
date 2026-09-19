#Requires -Version 5.1
<#
.SYNOPSIS
  Restore the portable rag corpus into local Windows PostgreSQL.

.DESCRIPTION
  The Windows setup script creates an empty rag database. This loads the
  complete 2026-09-15 snapshot: 60 sources, 109163 chunks, 109163 embeddings.

  powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\restore_rag_dump.ps1
  .\restore_rag_dump.cmd

  Requires PostgreSQL 16 + pgvector (run setup_windows_pg.ps1 first).
#>
[CmdletBinding()]
param(
    [string]$DumpPath = '',
    [string]$DumpUrl = 'https://filebin.net/arkguru-complete/arkguru_rag_complete_20260915.dump',
    [string]$ExpectedSha256 = '90b21d209c211b48378cf951e349c8ea954cd65f4804385d654659c4d222968c',
    [int]$ExpectedChunks = 109163,
    [int]$ExpectedEmbeddings = 109163,
    [int]$ExpectedSources = 60,
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
$ExpectedSha256 = $ExpectedSha256.ToLowerInvariant()

function Write-Log {
    param([string]$Message)
    Write-Host "[restore_rag_dump] $Message"
}

function Convert-NativeOutput {
    param($Output)
    if ($null -eq $Output) { return '' }
    $parts = foreach ($item in @($Output)) {
        if ($item -is [System.Management.Automation.ErrorRecord]) { $item.ToString() }
        else { [string]$item }
    }
    return ($parts -join "`n").Trim()
}

function Find-PgExe {
    param([Parameter(Mandatory = $true)][string]$Name)
    $direct = Join-Path $PgBin $Name
    if (Test-Path -LiteralPath $direct) { return $direct }
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Invoke-PgTool {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$Password,
        [switch]$IgnoreError
    )
    $prev = $env:PGPASSWORD
    $prevEap = $ErrorActionPreference
    if ($Password) { $env:PGPASSWORD = $Password }
    try {
        $ErrorActionPreference = 'Continue'
        $raw = & $Exe @Arguments 2>&1
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
        $output = Convert-NativeOutput $raw
        if ($code -ne 0 -and -not $IgnoreError) {
            throw "$Exe failed ($code): $output"
        }
        return [pscustomobject]@{ ExitCode = [int]$code; Output = $output }
    }
    finally {
        $ErrorActionPreference = $prevEap
        if ($null -eq $prev) { Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue }
        else { $env:PGPASSWORD = $prev }
    }
}

$onWindows = ($env:OS -eq 'Windows_NT') -or ($PSVersionTable.PSVersion.Major -ge 6 -and $IsWindows)
if (-not $onWindows) {
    throw 'This script restores into native Windows PostgreSQL. On Linux/macOS use scripts/restore_rag_dump.sh.'
}

$psql = Find-PgExe -Name 'psql.exe'
$pgRestore = Find-PgExe -Name 'pg_restore.exe'
if (-not $psql -or -not $pgRestore) {
    throw "psql/pg_restore not found. Run scripts/setup_windows_pg.cmd first. Expected $PgBin"
}

if (-not $DumpPath) {
    $candidates = @(
        (Join-Path (Get-Location).Path 'arkguru_rag_complete_20260915.dump'),
        (Join-Path $env:TEMP 'arkguru_rag_complete_20260915.dump')
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { $DumpPath = $c; break }
    }
}
if (-not $DumpPath) {
    $DumpPath = Join-Path $env:TEMP 'arkguru_rag_complete_20260915.dump'
}

if (-not (Test-Path -LiteralPath $DumpPath)) {
    Write-Log "downloading portable dump to $DumpPath"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $DumpUrl -OutFile $DumpPath -UseBasicParsing
}

$hash = (Get-FileHash -LiteralPath $DumpPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $ExpectedSha256) {
    throw "sha256 mismatch for $DumpPath : got $hash expected $ExpectedSha256"
}
Write-Log "dump sha256 ok"

$createExt = Invoke-PgTool -Exe $psql -Password $SuperPassword -IgnoreError -Arguments @(
    '-h', $HostName, '-p', "$Port", '-U', $SuperUser, '-d', $AppDb,
    '-v', 'ON_ERROR_STOP=1', '-c', 'CREATE EXTENSION IF NOT EXISTS vector;'
)
if ($createExt.ExitCode -ne 0) {
    throw @"
vector extension is not available. Finish scripts/setup_windows_pg.ps1 (copy vector.dll) then re-run this script.
$($createExt.Output)
"@
}

Write-Log "restoring $DumpPath into ${HostName}:${Port}/${AppDb}"
Invoke-PgTool -Exe $pgRestore -Password $SuperPassword -Arguments @(
    '--no-owner', '--no-acl', '--clean', '--if-exists',
    '-h', $HostName, '-p', "$Port", '-U', $SuperUser, '-d', $AppDb,
    $DumpPath
) | Out-Null

$countSql = 'SELECT count(*)::text FROM chunks UNION ALL SELECT count(*)::text FROM chunk_embeddings UNION ALL SELECT count(DISTINCT source_id)::text FROM chunks;'
$counts = Invoke-PgTool -Exe $psql -Password $AppPassword -Arguments @(
    '-h', $HostName, '-p', "$Port", '-U', $AppUser, '-d', $AppDb,
    '-v', 'ON_ERROR_STOP=1', '-tA', '-c', $countSql
)
$lines = @($counts.Output -split '\r?\n' | Where-Object { $_ -ne '' })
if ($lines.Count -lt 3) {
    throw "could not read restore counts: $($counts.Output)"
}
$gotChunks = [int]$lines[0]
$gotEmb = [int]$lines[1]
$gotSrc = [int]$lines[2]
Write-Log "chunks=$gotChunks embeddings=$gotEmb sources=$gotSrc"

if ($gotChunks -ne $ExpectedChunks -or $gotEmb -ne $ExpectedEmbeddings -or $gotSrc -ne $ExpectedSources) {
    throw "count mismatch. expected chunks=$ExpectedChunks embeddings=$ExpectedEmbeddings sources=$ExpectedSources"
}

Write-Host @"

Restore ok (portable rag dump 2026-09-15)
  host:        $HostName
  port:        $Port
  database:    $AppDb
  chunks:      $gotChunks
  embeddings:  $gotEmb
  sources:     $gotSrc
  PG_DSN:      postgresql://${AppUser}:${AppPassword}@${HostName}:${Port}/${AppDb}
"@
