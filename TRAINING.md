# 🚀 Orpheus TTS - Full Dataset Training

**Simplified training setup for Vietnamese TTS with 5 epochs**

## ⚡ Quick Start

### 1. Simple Training Command
```bash
python train.py \
    --csv_path "metadata.csv" \
    --audio_dir "/workspace/data_tonghop/"
```

### 2. Or Use Helper Scripts
```bash
# Option 1: Shell script
./start_training.sh

# Option 2: Python script  
python train_full.py
```

## 📋 Configuration

All configs are now in `config.py`:

```python
# Full Dataset Training (5 epochs)
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,  # Effective batch size = 32
    "num_train_epochs": 5,              # Train for 5 complete epochs
    "warmup_steps": 200,
    "learning_rate": 2e-4,
    "output_dir": "full_dataset_model",
}
```

## 🎛️ Command Line Options

```bash
python train.py \
    --csv_path "metadata.csv"           # Required: CSV metadata file
    --audio_dir "/workspace/data/"      # Required: Audio directory
    --max_samples 50000                 # Optional: Limit samples (None = full dataset)
    --batch_size 4                      # Optional: Override batch size
    --learning_rate 3e-4                # Optional: Override learning rate
    --save_dir "my_model"               # Optional: Save directory
```

## 📊 Training Output

After training completes:
- `full_dataset_model/` - LoRA adapters
- `full_dataset_model_merged/` - **Merged model for inference**

## 🎤 Inference

```bash
# Test inference
python inference.py --lora_path "full_dataset_model_merged" --prompt "Xin chào!"

# Launch Gradio web interface
python gradio_app.py
```

## 💾 Memory & Performance

- **Batch Size**: 2 (effective 32 with grad accumulation)
- **Epochs**: 5 (full dataset coverage)
- **GPU Memory**: ~10-15GB on H100
- **Training Time**: ~2-4 hours for 150k samples

## 🔧 Troubleshooting

**Out of Memory:**
```bash
python train.py --csv_path "metadata.csv" --audio_dir "/workspace/data/" --batch_size 1
```

**Partial Dataset:**
```bash
python train.py --csv_path "metadata.csv" --audio_dir "/workspace/data/" --max_samples 10000
```

**Check Training:**
```bash
tail -f full_dataset_model/trainer_state.json
```

---

**That's it! Simple and clean training setup! 🎯**
