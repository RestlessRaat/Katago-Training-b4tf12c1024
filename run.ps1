param(
    [Parameter(Position=0)][string]$Action = 'doctor',
    [switch]$Smoke,
    [int]$Games = 0,
    [int]$Epochs = 0,
    [string]$Label = 'train',
    [string]$Checkpoint = '',
    [string]$Model = '',
    [int]$Visits = 128
)
$ErrorActionPreference = 'Stop'
$trainingPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$trainingArgs = @((Join-Path $PSScriptRoot 'scripts\pipeline.py'), $Action)
if ($Smoke) { $trainingArgs += '--smoke' }
if ($Games -gt 0) { $trainingArgs += @('--games', $Games) }
if ($Epochs -gt 0) { $trainingArgs += @('--epochs', $Epochs) }
if ($Label -ne 'train') { $trainingArgs += @('--label', $Label) }
if ($Checkpoint) { $trainingArgs += @('--checkpoint', $Checkpoint) }
if ($Model) { $trainingArgs += @('--model', $Model) }
if ($Visits -ne 128) { $trainingArgs += @('--visits', $Visits) }
& $trainingPython @trainingArgs
if ($LASTEXITCODE -ne 0) { throw "KataGo pipeline failed: $LASTEXITCODE" }
