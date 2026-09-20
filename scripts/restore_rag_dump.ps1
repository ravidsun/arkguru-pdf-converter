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
    [long]$ExpectedBytes = 62509059,
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
    [string]$AppDb = 'rag',
    [string]$PgvectorVersion = '0.8.6',
    [switch]$SkipHnsw
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

function Test-IsAdministrator {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
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

function Get-PortableDump {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$ExpectedSha,
        [long]$ExpectedSize = 62509059
    )
    if (Test-Path -LiteralPath $Path) {
        $existing = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($existing -eq $ExpectedSha) {
            Write-Log "dump sha256 ok (cached)"
            return
        }
        Write-Log "cached dump has wrong sha256 ($existing); deleting"
        Remove-Item -LiteralPath $Path -Force
    }

    Write-Log "downloading portable dump to $Path"
    # user-agent curl/8.5.0
    # Filebin returns an HTML interstitial to the WindowsPowerShell user-agent
    # (sha256 6ba1004fd99133af779ba7ce1c66465a0cbc2834b755145c4a8e088250c872db).
    # curl.exe / a curl User-Agent gets the real 62509059-byte dump.
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $curl.Source @('-L', '--fail', '--retry', '3', '--user-agent', 'curl/8.5.0', '-o', $Path, $Url)
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
        $ErrorActionPreference = $prevEap
        if ($code -ne 0) { throw "curl.exe failed ($code) downloading $Url" }
    }
    else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $wc = New-Object System.Net.WebClient
        $wc.Headers.Add('User-Agent', 'curl/8.5.0')
        $wc.DownloadFile($Url, $Path)
    }

    if (-not (Test-Path -LiteralPath $Path)) { throw "download produced no file: $Path" }
    $len = (Get-Item -LiteralPath $Path).Length
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $ExpectedSha) {
        Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
        throw "sha256 mismatch after download ($len bytes, expected $ExpectedSize). Filebin served HTML instead of the dump. got $hash expected $ExpectedSha"
    }
    Write-Log "dump sha256 ok"
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

function Install-PgvectorWindows {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Version
    )
    $url = "https://github.com/andreiramani/pgvector_pgsql_windows/releases/download/${Version}_16/vector.v${Version}-pg16.zip"
    $zip = Join-Path $env:TEMP "vector.v${Version}-pg16.zip"
    $extract = Join-Path $env:TEMP "pgvector-win-pg16"
    Write-Log "downloading unofficial Windows pgvector $Version"
    Write-Log $url
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $curl.Source @('-L', '--fail', '--retry', '3', '--user-agent', 'curl/8.5.0', '-o', $zip, $url)
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
        $ErrorActionPreference = $prevEap
        if ($code -ne 0) { throw "curl.exe failed ($code) downloading $url" }
    }
    else {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $wc = New-Object System.Net.WebClient
        $wc.Headers.Add('User-Agent', 'curl/8.5.0')
        $wc.DownloadFile($url, $zip)
    }
    if (Test-Path -LiteralPath $extract) {
        Remove-Item -LiteralPath $extract -Recurse -Force
    }
    Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force
    $dll = Get-ChildItem -Path $extract -Recurse -Filter 'vector.dll' | Select-Object -First 1
    if (-not $dll) { throw "pgvector zip did not contain vector.dll: $zip" }
    $packRoot = $dll.Directory.Parent.FullName
    $libDest = Join-Path $Root 'lib'
    $extDest = Join-Path $Root 'share\extension'
    Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $libDest 'vector.dll') -Force
    Copy-Item -Path (Join-Path $packRoot 'share\extension\*') -Destination $extDest -Force
    Write-Log "installed vector.dll and vector.control into $Root"
}

Get-PortableDump -Path $DumpPath -Url $DumpUrl -ExpectedSha $ExpectedSha256 -ExpectedSize $ExpectedBytes

$createExt = Invoke-PgTool -Exe $psql -Password $SuperPassword -IgnoreError -Arguments @(
    '-h', $HostName, '-p', "$Port", '-U', $SuperUser, '-d', $AppDb,
    '-v', 'ON_ERROR_STOP=1', '-c', 'CREATE EXTENSION IF NOT EXISTS vector;'
)
if ($createExt.ExitCode -ne 0) {
    Write-Log 'vector.control missing; installing Windows pgvector files'
    if (-not (Test-IsAdministrator)) {
        throw @'
vector extension is not available (no vector.control under C:\Program Files\PostgreSQL\16\share\extension).
Open an elevated PowerShell (Run as administrator) and run:

  $zip = Join-Path $env:TEMP 'vector.v0.8.6-pg16.zip'
  curl.exe -L --fail -o $zip 'https://github.com/andreiramani/pgvector_pgsql_windows/releases/download/0.8.6_16/vector.v0.8.6-pg16.zip'
  $dest = Join-Path $env:TEMP 'pgvector-win'
  if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
  Expand-Archive $zip -DestinationPath $dest -Force
  Copy-Item "$dest\lib\vector.dll" 'C:\Program Files\PostgreSQL\16\lib\' -Force
  Copy-Item "$dest\share\extension\*" 'C:\Program Files\PostgreSQL\16\share\extension\' -Force
  $env:PGPASSWORD = 'postgres'
  & 'C:\Program Files\PostgreSQL\16\bin\psql.exe' -h 127.0.0.1 -p 5432 -U postgres -d rag -c 'CREATE EXTENSION IF NOT EXISTS vector;'

Then re-run .\restore_rag_dump.cmd from the repo folder.
'@
    }
    Install-PgvectorWindows -Root $PgRoot -Version $PgvectorVersion
    $createExt = Invoke-PgTool -Exe $psql -Password $SuperPassword -IgnoreError -Arguments @(
        '-h', $HostName, '-p', "$Port", '-U', $SuperUser, '-d', $AppDb,
        '-v', 'ON_ERROR_STOP=1', '-c', 'CREATE EXTENSION IF NOT EXISTS vector;'
    )
    if ($createExt.ExitCode -ne 0) {
        throw "CREATE EXTENSION vector still failed after copying files. $($createExt.Output)"
    }
}

function Invoke-PgRestoreStreaming {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    $prevPass = $env:PGPASSWORD
    $prevOpt = $env:PGOPTIONS
    $prevEap = $ErrorActionPreference
    $env:PGPASSWORD = $SuperPassword
    # Default maintenance_work_mem is 64MB; HNSW on 109k 1024-d rows thrashes for a long time.
    $env:PGOPTIONS = '-c maintenance_work_mem=1GB'
    try {
        $ErrorActionPreference = 'Continue'
        Write-Log ("pg_restore " + ($Arguments -join ' '))
        & $pgRestore @Arguments
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
        if ($code -ne 0) { throw "pg_restore failed ($code)" }
    }
    finally {
        $ErrorActionPreference = $prevEap
        if ($null -eq $prevPass) { Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue }
        else { $env:PGPASSWORD = $prevPass }
        if ($null -eq $prevOpt) { Remove-Item Env:PGOPTIONS -ErrorAction SilentlyContinue }
        else { $env:PGOPTIONS = $prevOpt }
    }
}

Write-Log "loading table data into ${HostName}:${Port}/${AppDb} (a few minutes)"
Invoke-PgRestoreStreaming -Arguments @(
    '--verbose', '--no-owner', '--no-acl', '--clean', '--if-exists',
    '--section=pre-data', '--section=data',
    '-h', $HostName, '-p', "$Port", '-U', $SuperUser, '-d', $AppDb,
    $DumpPath
)
Write-Log 'table data loaded; next step is indexes (HNSW is the slow part, 10-20 min on a laptop)'

if ($SkipHnsw) {
    $list = Join-Path $env:TEMP 'arkguru_rag_restore.list'
    $toc = Invoke-PgTool -Exe $pgRestore -Password $SuperPassword -Arguments @('-l', $DumpPath)
    $kept = @($toc.Output -split '\r?\n' | Where-Object { $_ -notmatch 'chunk_embeddings_hnsw_idx' })
    Set-Content -Path $list -Value $kept -Encoding ascii
    Write-Log 'SkipHnsw: restoring pkeys/GIN only'
    Invoke-PgRestoreStreaming -Arguments @(
        '--verbose', '--no-owner', '--no-acl',
        '-L', $list,
        '-h', $HostName, '-p', "$Port", '-U', $SuperUser, '-d', $AppDb,
        $DumpPath
    )
}
else {
    Invoke-PgRestoreStreaming -Arguments @(
        '--verbose', '--no-owner', '--no-acl',
        '--section=post-data',
        '-h', $HostName, '-p', "$Port", '-U', $SuperUser, '-d', $AppDb,
        $DumpPath
    )
}

$grantSql = @"
GRANT ALL ON SCHEMA public TO $AppUser;
GRANT ALL ON ALL TABLES IN SCHEMA public TO $AppUser;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO $AppUser;
ALTER TABLE IF EXISTS chunks OWNER TO $AppUser;
ALTER TABLE IF EXISTS chunk_embeddings OWNER TO $AppUser;
"@
Invoke-PgTool -Exe $psql -Password $SuperPassword -Arguments @(
    '-h', $HostName, '-p', "$Port", '-U', $SuperUser, '-d', $AppDb,
    '-v', 'ON_ERROR_STOP=1', '-c', $grantSql
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
