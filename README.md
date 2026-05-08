# Orpheus TTS - Vietnamese Training

Training setup for fine-tuning Orpheus TTS on a Vietnamese dataset.

## Project Structure

```text
orpheus_training/
├── config.py
├── data_processor.py
├── model_manager.py
├── train.py
├── inference.py
├── metadata.csv
├── data_tonghop/
├── requirements.txt
├── requirements-gpu.txt
├── start_training_gpu.ps1
└── README.md
```

## Windows GPU Setup

Use a local virtual environment so the project does not depend on the global Python install.

```powershell
cd "D:\Ai Companion\orpheus-finetune-vi"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall -r requirements-gpu.txt
```

Install `requirements-gpu.txt` after `requirements.txt`. This keeps PyTorch CUDA-enabled and avoids pip replacing it with a CPU-only build.

Verify CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected on this machine:

```text
torch 2.10.0+cu128
cuda True
NVIDIA GeForce RTX 3050 Laptop GPU
```

## GPU Smoke Test

Run this first before training larger datasets:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_training_gpu.ps1
```

The smoke test uses conservative settings for RTX 3050 4GB:

```text
max_samples: 200
max_steps: 20
batch_size: 1
gradient_accumulation_steps: 8
load_in_4bit: true
max_seq_length: 2048
lora_rank: 16
precision: bf16
save_lora_only: true
```

The first run needs internet access to download the model from Hugging Face.

## Manual Training Commands

### Low-VRAM GPU Test

```powershell
.\.venv\Scripts\python.exe train.py `
    --device cuda `
    --csv_path "metadata.csv" `
    --audio_dir ".\data_tonghop" `
    --save_dir "gpu_test_model" `
    --max_samples 200 `
    --max_steps 20 `
    --batch_size 1 `
    --gradient_accumulation_steps 8 `
    --load_in_4bit `
    --max_seq_length 2048 `
    --lora_rank 16 `
    --lora_alpha 16 `
    --bf16 `
    --low_vram `
    --save_lora_only
```

### Increase Gradually

After the smoke test succeeds, increase slowly:

```powershell
.\.venv\Scripts\python.exe train.py `
    --device cuda `
    --csv_path "metadata.csv" `
    --audio_dir ".\data_tonghop" `
    --save_dir "model_1k" `
    --max_samples 1000 `
    --max_steps 100 `
    --batch_size 1 `
    --gradient_accumulation_steps 8 `
    --load_in_4bit `
    --max_seq_length 2048 `
    --lora_rank 16 `
    --lora_alpha 16 `
    --bf16 `
    --low_vram `
    --save_lora_only
```

### Quality Run On RTX 3050 4GB

The 20-step and 100-step runs are only smoke tests. For noticeably better Vietnamese audio, train on most of the dataset for more optimizer steps:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_training_quality_gpu.ps1
```

Default quality settings:

```text
max_samples: 7000
max_steps: 2000
batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-4
warmup_steps: 200
load_in_4bit: true
max_seq_length: 2048
lora_rank: 16
lora_alpha: 16
precision: bf16
save_lora_only: true
```

Resume a stopped run from the latest checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_training_quality_gpu.ps1 `
    -ResumeFromCheckpoint "model_voice_clone_quality_2k\checkpoint-1000"
```

### Larger GPU Example

For a high-VRAM GPU such as A100/H100:

```bash
python train.py \
    --device cuda \
    --csv_path "metadata.csv" \
    --audio_dir "/workspace/data_tonghop/" \
    --save_dir "vietnamese_model" \
    --max_samples 10000 \
    --max_steps 1000 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --bf16
```

## Inference

```powershell
.\.venv\Scripts\python.exe inference.py `
    --lora_path "model_voice_clone_quality_2k" `
    --prompt "Xin chào, tôi là mô hình TTS tiếng Việt!" `
    --load_in_4bit `
    --max_seq_length 2048 `
    --bf16 `
    --preset balanced `
    --seed 42 `
    --save_audio
```

If you trained with `--save_lora_only`, use the LoRA directory directly. If you saved a merged model, use the `_merged` directory.

## Troubleshooting

### CUDA Not Available

If `torch.cuda.is_available()` is `False` but `nvidia-smi` shows the GPU:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall -r requirements-gpu.txt
```

### Hugging Face Download Fails

The first training run downloads `unsloth/orpheus-3b-0.1-ft` or its 4-bit variant. Make sure internet access to `huggingface.co` is allowed.

### CUDA Out Of Memory

RTX 3050 4GB is very tight for Orpheus 3B. Use:

```powershell
--batch_size 1 --gradient_accumulation_steps 8 --load_in_4bit --max_seq_length 2048 --lora_rank 16 --bf16 --low_vram --save_lora_only
```

If it still OOMs, reduce `--max_seq_length` to `1024` and keep `--max_samples` small for testing.

### Check Installed Packages

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m bitsandbytes
.\.venv\Scripts\python.exe -c "import unsloth, triton; print('ok')"
```

## Dataset Format

Expected CSV columns:

```csv
uuid,sentence
audio_001.wav,Xin chào...
```

Audio files should exist under `data_tonghop/`.
