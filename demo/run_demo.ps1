$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
}
else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand) {
        throw "Python was not found. Install Python 3.11+ and follow the setup steps in README.md."
    }
    $python = $pythonCommand.Source
}

& $python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit is unavailable in this Python environment. Install project dependencies with: python -m pip install -r requirements.txt"
}

$checkpoint = Join-Path $projectRoot "checkpoints\eeg_multimodal.pt"
if (-not (Test-Path -LiteralPath $checkpoint)) {
    Write-Warning "No trained EEG checkpoint is present. Prediction remains unavailable until the real dataset is trained."
}

Set-Location $projectRoot
Write-Host "Starting NeuroFusion AD locally. Press Ctrl+C to stop."
& $python -m streamlit run app.py
if ($LASTEXITCODE -ne 0) {
    throw "The Streamlit app exited with code $LASTEXITCODE."
}
