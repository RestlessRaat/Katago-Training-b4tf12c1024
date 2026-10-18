param(
    [string]$Python = 'python',
    [string]$ConverterPython = (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe')
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $Python -m venv --system-site-packages .venv
        if ($LASTEXITCODE -ne 0) { throw 'Training venv creation failed' }
    }
    $trainPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    & $trainPython -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch required"'
    if ($LASTEXITCODE -ne 0) { throw 'Install a CUDA PyTorch build in .venv first; see README' }
    & $trainPython -m pip install -r requirements-local.txt
    if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed' }
    & $trainPython scripts/bootstrap.py
    if ($LASTEXITCODE -ne 0) { throw 'Official asset verification failed' }
    if (-not (Test-Path -LiteralPath 'models\pretrained\b6c96-tf\tensors.npz')) {
        if (-not (Test-Path -LiteralPath '.venv-convert\Scripts\python.exe')) {
            & $ConverterPython -m venv .venv-convert
            if ($LASTEXITCODE -ne 0) { throw 'Converter venv requires Python 3.10' }
        }
        & .\.venv-convert\Scripts\python.exe -m pip install -r requirements-convert.txt
        if ($LASTEXITCODE -ne 0) { throw 'Converter dependency installation failed' }
        & .\.venv-convert\Scripts\python.exe scripts/extract_tf_checkpoint.py models/pretrained/b6c96-tf/b6c96-s175395328-d26788732/saved_model/variables/variables models/pretrained/b6c96-tf/tensors.npz
        if ($LASTEXITCODE -ne 0) { throw 'Checkpoint extraction failed' }
    }
    if (-not (Test-Path -LiteralPath 'models\pretrained\b6c96-pytorch\initial.ckpt')) {
        & $trainPython scripts/import_b6.py
        if ($LASTEXITCODE -ne 0) { throw 'Pretrained weight migration failed' }
    }
    & $trainPython scripts/pipeline.py configure
    if ($LASTEXITCODE -ne 0) { throw 'Configuration failed' }
    & $trainPython scripts/pipeline.py doctor
    if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed' }
    Write-Output 'Setup verified. Run: .\run.ps1 cycle -Smoke'
} finally {
    Pop-Location
}
