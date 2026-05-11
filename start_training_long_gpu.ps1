param(
    [string]$CsvPath = "metadata.csv",
    [string]$AudioDir = ".\data_tonghop",
    [string]$SaveDir = "model_training\model_voice_clone_100steps",
    [int]$MaxSamples = 1000,
    [int]$MaxSteps = 100
)

$ErrorActionPreference = "Stop"

# Longer low-VRAM GPU run for RTX 3050 4GB.
# Keeps the smoke-test output in gpu_test_model untouched.

$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Orpheus TTS - longer GPU training"
Write-Host "================================="
Write-Host "Python: $Python"
Write-Host "CSV: $CsvPath"
Write-Host "Audio: $AudioDir"
Write-Host "Output: $SaveDir"
Write-Host "Max samples: $MaxSamples"
Write-Host "Max steps: $MaxSteps"

if (-not (Test-Path $CsvPath)) {
    throw "CSV file not found: $CsvPath"
}

if (-not (Test-Path $AudioDir)) {
    throw "Audio directory not found: $AudioDir"
}

& $Python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "CUDA is not available in this Python environment."
    Write-Host "Install the CUDA PyTorch build first:"
    Write-Host "$Python -m pip install --upgrade --force-reinstall -r requirements-gpu.txt"
    exit 1
}

& $Python train.py `
    --device cuda `
    --csv_path $CsvPath `
    --audio_dir $AudioDir `
    --save_dir $SaveDir `
    --max_samples $MaxSamples `
    --max_steps $MaxSteps `
    --batch_size 1 `
    --gradient_accumulation_steps 8 `
    --load_in_4bit `
    --max_seq_length 2048 `
    --lora_rank 16 `
    --lora_alpha 16 `
    --bf16 `
    --low_vram `
    --save_lora_only

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Long GPU training completed. LoRA model saved to: $SaveDir"
} else {
    throw "Long GPU training failed."
}
