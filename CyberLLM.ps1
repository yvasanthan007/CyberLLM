# CyberLLM Control Script (PowerShell)
# Usage examples:
#   .\CyberLLM.ps1 status          # hardware + deps + model status
#   .\CyberLLM.ps1 dataset         # regenerate/re-validate conversation datasets
#   .\CyberLLM.ps1 download-model  # fetch base model if missing (resumable)
#   .\CyberLLM.ps1 train           # LoRA gate check / smoke / full
#   .\CyberLLM.ps1 evaluate        # run all evaluations
#   .\CyberLLM.ps1 serve           # start API on 127.0.0.1:8765 (blocks)
#   .\CyberLLM.ps1 test-api        # smoke-test the running API
#   .\CyberLLM.ps1 zyra            # interactive Zyra console client
param(
    [Parameter(Position = 0)]
    [ValidateSet('status', 'dataset', 'download-model', 'train', 'evaluate', 'serve', 'test-api', 'zyra')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Py)) { Write-Error "Virtualenv python not found at $Py. Create the .venv first."; exit 1 }
Set-Location $Root

if (-not $env:HF_HOME) { $env:HF_HOME = 'D:\HuggingFace' }   # reuse existing model cache
if (-not $env:HF_HUB_DISABLE_XET) { $env:HF_HUB_DISABLE_XET = '1' }

switch ($Action) {

    'status' {
        Write-Host "== Hardware ==" -ForegroundColor Cyan
        & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        Write-Host "== Environment ==" -ForegroundColor Cyan
        & $Py -c "import sys,torch,transformers;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('transformers',transformers.__version__)"
        Write-Host "== Model cache ==" -ForegroundColor Cyan
        & $Py -c "import os;from pathlib import Path;p=Path(os.environ['HF_HOME'])/'hub'/'models--Qwen--Qwen2.5-0.5B-Instruct'/'snapshots';print('Qwen2.5-0.5B-Instruct cached:', any(x.name=='model.safetensors' for s in p.glob('*') for x in s.iterdir())) if p.exists() else print('not cached')"
    }

    'dataset' {
        & $Py scripts\preprocess_conversations.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    'download-model' {
        & $Py -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'))"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    'train' {
        # Gated: refuses safely on CPU-only torch. Optional --smoke / --full follows the action.
        $TrainArgs = @($args)
        if ($TrainArgs.Count -eq 0) { & $Py scripts\train_lora.py }
        else { & $Py scripts\train_lora.py @TrainArgs }
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    'evaluate' {
        & $Py scripts\evaluate_conversations.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Py scripts\evaluate_phishing.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Py scripts\check_llm.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    'serve' {
        Write-Host "Starting CyberLLM API on http://127.0.0.1:8765 (Ctrl+C to stop)" -ForegroundColor Green
        & $Py -m uvicorn api.main:app --host 127.0.0.1 --port 8765
    }

    'test-api' {
        & $Py scripts\test_api_live.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    'zyra' {
        & $Py examples\zyra_cyberllm_client.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}
