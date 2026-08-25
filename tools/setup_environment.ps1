param(
    [ValidateSet("gpu", "cpu", "data")]
    [string]$InstallMode = "gpu",

    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$virtualEnv = Join-Path $projectRoot ".venv"
$virtualPython = Join-Path $virtualEnv "Scripts\python.exe"
$fullRequirements = Join-Path $projectRoot "requirements.txt"
$dataRequirements = Join-Path $projectRoot "requirements-data.txt"

Write-Host "[1/5] Kiem tra Python..."
& $PythonExe -c "import sys; assert sys.version_info[:2] == (3, 11), 'Du an yeu cau Python 3.11'; print(sys.version)"

if (-not (Test-Path -LiteralPath $virtualPython)) {
    Write-Host "[2/5] Tao moi truong ao .venv..."
    & $PythonExe -m venv $virtualEnv
}
else {
    Write-Host "[2/5] Su dung .venv hien co."
}

Write-Host "[3/5] Cap nhat cong cu cai dat..."
& $virtualPython -m pip install --upgrade "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1"
if ($LASTEXITCODE -ne 0) {
    throw "Khong the cap nhat pip/setuptools/wheel."
}

if ($InstallMode -eq "gpu") {
    Write-Host "[4/5] Cai PyTorch 2.5.1 / Torchvision 0.20.1 CUDA 12.1..."
    & $virtualPython -m pip install `
        "torch==2.5.1" `
        "torchvision==0.20.1" `
        --index-url "https://download.pytorch.org/whl/cu121"
    if ($LASTEXITCODE -ne 0) {
        throw "Khong the cai PyTorch CUDA."
    }
}
elseif ($InstallMode -eq "cpu") {
    Write-Host "[4/5] Cai PyTorch 2.5.1 / Torchvision 0.20.1 CPU..."
    & $virtualPython -m pip install `
        "torch==2.5.1" `
        "torchvision==0.20.1" `
        --index-url "https://download.pytorch.org/whl/cpu"
    if ($LASTEXITCODE -ne 0) {
        throw "Khong the cai PyTorch CPU."
    }
}
else {
    Write-Host "[4/5] Bo qua PyTorch cho moi truong xu ly du lieu."
}

Write-Host "[5/5] Cai cac thu vien cua du an..."
if ($InstallMode -eq "data") {
    & $virtualPython -m pip install --requirement $dataRequirements
}
else {
    & $virtualPython -m pip install --requirement $fullRequirements
}
if ($LASTEXITCODE -ne 0) {
    throw "Khong the cai thu vien cua du an."
}

Write-Host "`nKiem tra moi truong:"
& $virtualPython (Join-Path $PSScriptRoot "check_environment.py")

Write-Host "`nHoan tat. Kich hoat moi truong bang:"
Write-Host ".\.venv\Scripts\Activate.ps1"
