$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

try {
    python -m fets27_challenge.cli prepare-assets --data-root .\data\toy
}
catch {
    Write-Host "Asset preparation skipped. Install PyTorch/MONAI/NiBabel first if you need toy data or checkpoints."
}

