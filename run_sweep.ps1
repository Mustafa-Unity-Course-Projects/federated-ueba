# Every reported experiment, at five seeds, plus the centralized baseline.
# Runs from a normal Windows terminal.
#
# SAFE TO RE-RUN. Each (experiment, seed) pair records `done` and the pipeline
# version in its summary, and the runner skips a pair only when both are present
# and the version matches. So after a shutdown, running this again resumes:
# finished pairs are skipped, the pair that was interrupted is redone from
# scratch, and the rest continue. Nothing has to be deleted by hand.
#
# The experiment list is deliberately not written here. It lives in
# [tool.fueba.sweep] experiments, which names the set the thesis reports; the
# runner reads it and excludes the smoke tests itself. A second list in this
# script would be one more thing to forget to update.
#
# An experiment left out of that list is not gone, only not swept by default:
#
#   .\run_sweep.ps1 -Only "delta-ef-0.1"
#
#   .\run_sweep.ps1                        the full sweep
#   .\run_sweep.ps1 -Seeds "1"             one seed
#   .\run_sweep.ps1 -Only "baseline"       one experiment, for a quick check
#   .\run_sweep.ps1 -SkipCentralized       federated only
#   .\run_sweep.ps1 -Mode eval             rescore, do not train
#
# `-Mode eval` exists for the one operation that would otherwise be ruinous.
# Changing a scoring default changes what every recorded number means, so it
# requires a PIPELINE_VERSION bump; but `already_done` keys off that version, so
# after a bump every finished run looks unfinished and the default `-Mode full`
# would retrain all of them. `-Mode eval` reuses the saved checkpoints, rescores
# them, and rewrites the summaries at the new version. It also leaves the
# communication logs untouched, so the megabyte figures survive a rescore.
#
# To stop it: close this window, then kill what it left behind. Killing the shell
# is not enough on its own, because flower-simulation and the Ray workers survive
# it and keep writing into the same directories:
#
#   Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { taskkill /F /PID $_.ProcessId /T }

param(
    [string]$Seeds = "1,2,3,4,5",
    [string]$Only = "",
    [string]$Log = "sweep.log",
    [ValidateSet("full", "train", "eval")]
    [string]$Mode = "full",
    [switch]$SkipCentralized
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

# Python block-buffers stdout when it is a pipe rather than a console, so without
# this the log stays empty for minutes at a time and everything still in the
# buffer is lost if the run is killed. On a sweep measured in days that makes the
# log useless for watching progress and useless for diagnosing a crash, which is
# the only reason it exists. Inherited by the per-experiment subprocesses too.
$env:PYTHONUNBUFFERED = "1"
# The runner's output contains emoji and Turkish text; without this the child
# writes them through the legacy console codepage and raises on the first one.
$env:PYTHONIOENCODING = "utf-8"

$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Output "Interpreter not found at $Py"
    exit 1
}

# One handle, opened once, held for the whole sweep.
#
# Not Add-Content per line. Add-Content reopens the file for every write and
# takes an exclusive lock while it does, so anything reading the log at that
# instant makes the write fail. A `tail -f` on this file was enough: the sweep
# kept running but the log froze after 798 bytes and every subsequent line was
# lost to an IOException. FileShare::ReadWrite below is the fix, and it is the
# whole point of opening the stream by hand rather than using a cmdlet.
#
# Not Tee-Object either. On Windows PowerShell 5.1 Tee-Object writes UTF-16 and
# has no -Encoding parameter, so a log teed here and appended to elsewhere ends
# up half UTF-16 and half UTF-8: unreadable to grep, and twice the size. One
# 124 MB sweep log was lost to exactly that, which is why every line in this
# script goes through the one function below.
if (-not [System.IO.Path]::IsPathRooted($Log)) { $Log = Join-Path $PSScriptRoot $Log }
$stream = [System.IO.FileStream]::new(
    $Log,
    [System.IO.FileMode]::Append,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::ReadWrite)
# UTF8Encoding($false): no byte order mark. A BOM appended mid-file on every
# resume would sit in the middle of the log as stray bytes.
$script:LogWriter = [System.IO.StreamWriter]::new($stream, [System.Text.UTF8Encoding]::new($false))
# Without this the tail of the log is whatever was in the buffer when the window
# was closed, which is exactly the run that most needs reading.
$script:LogWriter.AutoFlush = $true

function Write-ToLog {
    param([Parameter(ValueFromPipeline = $true)][string]$Message)
    # Console and file both, on purpose: the console so progress is visible in
    # the window, the file so it survives the window being closed.
    Write-Output $Message
    $script:LogWriter.WriteLine($Message)
}

Set-Alias Write-Log Write-ToLog

function Close-Log {
    # Called before every exit rather than left to process teardown, so that a
    # run started from a session that outlives the script does not keep the file
    # handle open behind it.
    param([int]$Code)
    if ($script:LogWriter) { $script:LogWriter.Dispose(); $script:LogWriter = $null }
    exit $Code
}

Write-Log "=== SWEEP START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

$federatedArgs = @("federated_insider_detection.py", "--experiment", "all",
                   "--mode", $Mode, "--seeds", $Seeds)
if ($Only -ne "") { $federatedArgs += @("--only", $Only) }

# 2>&1 merges the child's stderr into the pipeline so the log holds everything,
# which matters here because Flower writes all of its round-by-round output to
# stderr. PowerShell wraps each of those lines in an ErrorRecord, so ToString()
# unwraps them; without it every INFO line arrives with a five-line stack trace
# attached and the log becomes unreadable.
#
# Exit status is read from $LASTEXITCODE rather than $?, because that merge makes
# $? false for a native command even when it returned 0.
& $Py $federatedArgs 2>&1 | ForEach-Object { Write-ToLog $_.ToString() }
$federatedExit = $LASTEXITCODE

if ($federatedExit -ne 0) {
    Write-Log "=== FEDERATED RUNNER EXITED $federatedExit; STOPPING $(Get-Date -Format 'HH:mm:ss') ==="
    Write-Log "Re-run this script to resume: finished experiments are skipped."
    Close-Log $federatedExit
}

if (-not $SkipCentralized) {
    foreach ($s in $Seeds.Split(",")) {
        $seed = $s.Trim()
        Write-Log "=== CENTRALIZED seed $seed $(Get-Date -Format 'HH:mm:ss') ==="
        # PowerShell has no inline VAR=value prefix, so the environment variable
        # is set for the whole session and the child inherits it.
        $env:SEED = $seed
        & $Py train_centralized.py 2>&1 | ForEach-Object { Write-ToLog $_.ToString() }
        if ($LASTEXITCODE -ne 0) {
            Write-Log "=== CENTRALIZED seed $seed EXITED $LASTEXITCODE; STOPPING ==="
            Close-Log $LASTEXITCODE
        }
    }
    Remove-Item Env:\SEED -ErrorAction SilentlyContinue
}

Write-Log "=== FINAL COMPARISON $(Get-Date -Format 'HH:mm:ss') ==="
& $Py analysis/compare_experiments.py 2>&1 | ForEach-Object { Write-ToLog $_.ToString() }

Write-Log "=== SWEEP DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Close-Log 0
