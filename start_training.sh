#!/bin/bash

# Orpheus TTS Full Dataset Training Script
# Train Vietnamese TTS model on full dataset with 5 epochs

echo "🚀 Orpheus TTS - Full Dataset Training (5 epochs)"
echo "=================================================="

# Default configuration
CSV_PATH="metadata.csv"
AUDIO_DIR="./data_tonghop/"
SAVE_DIR="model_training/full_dataset_model"

# Check if files exist
if [ ! -f "$CSV_PATH" ]; then
    echo "❌ CSV file not found: $CSV_PATH"
    echo "💡 Please update CSV_PATH in this script"
    exit 1
fi

if [ ! -d "$AUDIO_DIR" ]; then
    echo "❌ Audio directory not found: $AUDIO_DIR" 
    echo "💡 Please update AUDIO_DIR in this script"
    exit 1
fi

echo "📋 Training Configuration:"
echo "   📂 CSV: $CSV_PATH"
echo "   🎵 Audio: $AUDIO_DIR"
echo "   📊 Dataset: Full dataset (no sample limit)"
echo "   ⚙️  Epochs: 5"
echo "   💾 Save to: $SAVE_DIR"
echo "=================================================="

# Set environment variables
export TOKENIZERS_PARALLELISM=false

# Start training
echo "🔄 Starting training..."
python train.py \
    --csv_path "$CSV_PATH" \
    --audio_dir "$AUDIO_DIR" \
    --save_dir "$SAVE_DIR"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Models saved to:"
    echo "   📦 LoRA: $SAVE_DIR/"
    echo "   🔄 Merged: ${SAVE_DIR}_merged/"
    echo ""
    echo "💡 Next steps:"
    echo "   1. Test inference: python inference.py --lora_path ${SAVE_DIR}_merged"
    echo "   2. Launch Gradio: python gradio_app.py"
else
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi
