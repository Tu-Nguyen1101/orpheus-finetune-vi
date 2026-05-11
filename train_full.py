#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple script to train Orpheus TTS on full dataset with 5 epochs
Usage: python train_full.py
"""

import os
import sys

def main():
    """Train on full Vietnamese dataset"""
    
    # Default paths - update these for your setup
    csv_path = "metadata.csv"
    audio_dir = "/workspace/data_tonghop/"
    
    print("🚀 Orpheus TTS Full Dataset Training")
    print("=" * 60)
    print("📋 Configuration:")
    print(f"   📂 CSV: {csv_path}")
    print(f"   🎵 Audio: {audio_dir}")
    print(f"   📊 Dataset: Full (no sample limit)")
    print(f"   ⚙️  Epochs: 5")
    save_dir = "model_training/full_dataset_model"
    print(f"   💾 Output: {save_dir}")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        print("💡 Update csv_path in this script to match your setup")
        return
    
    if not os.path.exists(audio_dir):
        print(f"❌ Audio directory not found: {audio_dir}")
        print("💡 Update audio_dir in this script to match your setup")
        return
    
    # Import and run training
    try:
        from train import OrpheusTrainer, resolve_save_dir
        save_dir = resolve_save_dir("full_dataset_model")
        
        print("🔄 Starting training...")
        trainer = OrpheusTrainer(device="cuda")
        
        # Train with full dataset
        training_stats = trainer.train(
            csv_path=csv_path,
            audio_dir=audio_dir,
            text_column="sentence",
            audio_column="uuid",
            max_samples=None  # Full dataset
        )
        
        # Save model
        trainer.save_model(save_dir, save_merged=True)
        
        print("\n🎉 Training completed successfully!")
        print("📁 Models saved:")
        print(f"   📦 LoRA model: {save_dir}/")
        print(f"   🔄 Merged model: {save_dir}_merged/")
        print("💡 Use merged model for inference")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return
    except Exception as e:
        print(f"❌ Training error: {e}")
        return


def check_system():
    """Check system requirements"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
        else:
            print("❌ CUDA not available")
            return False
            
        import unsloth
        print("✅ Unsloth available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


if __name__ == "__main__":
    print("🔍 Checking system requirements...")
    
    if not check_system():
        print("\n💡 Install missing dependencies and try again")
        sys.exit(1)
    
    print("\n✅ System ready for training")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Training stopped by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        raise
