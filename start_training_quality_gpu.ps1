param(
    [string]$CsvPath = "metadata.csv",
    [string]$AudioDir = ".\data_tonghop",
    [string]$SaveDir = "model_voice_clone_quality_2k",
    [int]$MaxSamples = 7000,
    [int]$MaxSteps = 2000,
    [double]$LearningRate = 0.0001,
    [int]$WarmupSteps = 200,
    [int]$GradientAccumulationSteps = 8,
    [int]$LoraRank = 16,
    [int]$LoraAlpha = 16,
    [int]$MaxSeqLength = 2048,
    [string]$ResumeFromCheckpoint = ""
)

$ErrorActionPreference = "Stop"

# Longer quality run for RTX 3050 4GB.
# Quality comes mainly from more samples and more optimizer steps.

$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Orpheus TTS - quality GPU training"
Write-Host "=================================="
Write-Host "Python: $Python"
Write-Host "CSV: $CsvPath"
Write-Host "Audio: $AudioDir"
Write-Host "Output: $SaveDir"
Write-Host "Max samples: $MaxSamples"
Write-Host "Max steps: $MaxSteps"
Write-Host "Learning rate: $LearningRate"

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

$TrainArgs = @(
    "train.py",
    "--device", "cuda",
    "--csv_path", $CsvPath,
    "--audio_dir", $AudioDir,
    "--save_dir", $SaveDir,
    "--max_samples", $MaxSamples,
    "--max_steps", $MaxSteps,
    "--batch_size", 1,
    "--gradient_accumulation_steps", $GradientAccumulationSteps,
    "--learning_rate", $LearningRate,
    "--warmup_steps", $WarmupSteps,
    "--logging_steps", 25,
    "--save_steps", 250,
    "--save_total_limit", 3,
    "--load_in_4bit",
    "--max_seq_length", $MaxSeqLength,
    "--lora_rank", $LoraRank,
    "--lora_alpha", $LoraAlpha,
    "--bf16",
    "--low_vram",
    "--save_lora_only"
)

if ($ResumeFromCheckpoint) {
    $TrainArgs += @("--resume_from_checkpoint", $ResumeFromCheckpoint)
}

& $Python @TrainArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Quality GPU training completed. LoRA model saved to: $SaveDir"
} else {
    throw "Quality GPU training failed."
}
