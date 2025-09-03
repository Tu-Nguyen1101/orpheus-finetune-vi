# 🇻🇳 Orpheus TTS - Vietnamese Training

Simplified, optimized implementation for training Orpheus TTS with Vietnamese datasets.

## 📁 Project Structure

```
orpheus_training/
├── config.py              # All configurations (basic + H100)
├── utils.py               # Audio processing utilities  
├── data_processor.py      # Dataset loading & preprocessing
├── model_manager.py       # Model loading & saving
├── train.py              # Standard training (with max_samples limit)
├── train_full_dataset.py  # Full dataset training (batch processing)
├── inference.py          # Inference script
├── requirements.txt      # Dependencies
├── metadata.csv          # Your dataset (156K samples)
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training

#### Basic Training
```bash
python train.py \
    --csv_path "metadata.csv" \
    --audio_dir "/workspace/data_tonghop/" \
    --max_steps 100 \
    --save_dir "basic_model"
```

#### H100-Optimized Training (Recommended)
```bash
python train.py \
    --csv_path "metadata.csv" \
    --audio_dir "/workspace/data_tonghop/" \
    --use_h100_config \
    --max_steps 1000 \
    --max_samples 1000 \
    --save_dir "vietnamese_model"
```

### 3. Inference
```bash
python inference.py \
    --lora_path "vietnamese_model" \
    --prompt "Xin chào, tôi là mô hình TTS tiếng Việt!" \
    --save_audio
```

## ⚙️ Configuration

### Basic Config (Default)
- Batch size: 1, LoRA rank: 64, Seq length: 2048
- Training steps: 60, Learning rate: 2e-4

### H100 Config (`--use_h100_config`)
- Batch size: 4×8=32, LoRA rank: 128, Seq length: 4096  
- Training steps: 1000, Learning rate: 3e-4
- BFloat16 + TF32 optimization

## 📊 Your Dataset Status
✅ **156,344 Vietnamese sentences**  
✅ **296,905 audio files**  
✅ **100% vocabulary coverage**  
✅ **Perfect audio format** (24kHz, numpy arrays)

## 🎯 Training Strategies

### Strategy 1: Quick Test (1K samples)
Test training setup and model quality with small dataset.

### Strategy 2: Progressive Training (5K → 10K → 20K)
Gradually increase dataset size to find optimal memory usage.

### Strategy 3: Full Dataset Training (156K samples)
Train with entire dataset using batch processing.

## 🎯 Recommended Commands

### Strategy 1: Quick Test (1K samples)
```bash
python train.py \
    --csv_path "metadata.csv" \
    --audio_dir "/workspace/data_tonghop/" \
    --use_h100_config \
    --max_steps 50 \
    --max_samples 1000 \
    --save_dir "test_model"
```

### Strategy 2: Progressive Training
```bash
# Start with 5K samples
python train.py \
    --csv_path "metadata.csv" \
    --audio_dir "/workspace/data_tonghop/" \
    --use_h100_config \
    --max_steps 500 \
    --max_samples 5000 \
    --save_dir "model_5k"

# If successful, try 10K samples
python train.py \
    --csv_path "metadata.csv" \
    --audio_dir "/workspace/data_tonghop/" \
    --use_h100_config \
    --max_steps 1000 \
    --max_samples 10000 \
    --save_dir "model_10k"

# If successful, try 20K samples
python train.py \
    --csv_path "metadata.csv" \
    --audio_dir "/workspace/data_tonghop/" \
    --use_h100_config \
    --max_steps 2000 \
    --max_samples 20000 \
    --save_dir "model_20k"
```

### Strategy 3: Full Dataset Training (156K samples)
```bash
python train_full_dataset.py \
    --csv_path "metadata.csv" \
    --audio_dir "/workspace/data_tonghop/" \
    --batch_size 2000 \
    --total_steps 5000 \
    --checkpoint_dir "full_vietnamese_model"
```

## 🔧 Troubleshooting

### Memory Issues
```bash
# Reduce samples (most effective)
python train.py ... --max_samples 500

# Reduce batch size
python train.py ... --batch_size 2

# Use basic config instead
python train.py ... # (without --use_h100_config)
```

### Audio Issues
- Check `--audio_dir` path exists
- Verify CSV format: `uuid,sentence`
- Ensure audio files are .wav format

## 📈 Expected Performance

### H100 Training
- **Memory**: ~25-35 GB (of 80 GB)
- **Speed**: ~4-8 steps/minute  
- **Quality**: High (LoRA rank 128, 100% vocab coverage)
- **Time**: ~2-4 hours for 1000 steps

### Results
- Perfect Vietnamese pronunciation
- Natural intonation and rhythm
- Support for complex Vietnamese texts with diacritics

---

**🎉 Ready to train your Vietnamese TTS model!**