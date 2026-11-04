$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
$pilotPython = Join-Path $repoRoot '.pilot-venv/Scripts/python.exe'
$cudaRoot = Join-Path $repoRoot '.ppo-venv'
if (!(Test-Path -LiteralPath $pilotPython)) { throw 'Create the existing .pilot-venv training environment first.' }
& $pilotPython -m venv $cudaRoot
if ($LASTEXITCODE) { throw 'Could not create CUDA training environment.' }
$cudaPython = Join-Path $cudaRoot 'Scripts/python.exe'
& $cudaPython -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed.' }
# Reuse the existing simulator/SB3 dependencies, while CUDA torch takes precedence.
$pilotPackages = Join-Path $repoRoot '.pilot-venv/Lib/site-packages'
[System.IO.File]::WriteAllText((Join-Path $cudaRoot 'Lib/site-packages/pilot-training-dependencies.pth'), $pilotPackages + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
& $cudaPython -c "import torch, sb3_contrib; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE) { throw 'CUDA verification failed.' }
