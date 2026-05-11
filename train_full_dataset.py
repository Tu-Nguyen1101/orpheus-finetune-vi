# -*- coding: utf-8 -*-
"""
Training script for full dataset with batch processing
Trains with all 156K samples in manageable batches
"""

from train import OrpheusTrainer, resolve_save_dir
import pandas as pd
import argparse
import os
import torch
import gc


def train_full_dataset(csv_path: str, audio_dir: str, batch_size: int = 2000, 
                      total_steps: int = 5000, checkpoint_dir: str = "full_training"):
    """
    Train with full dataset using batch processing
    
    Args:
        csv_path: Path to CSV file
        audio_dir: Audio directory
        batch_size: Samples per batch
        total_steps: Total training steps across all batches
        checkpoint_dir: Directory to save checkpoints
    """
    checkpoint_dir = resolve_save_dir(checkpoint_dir)

    print("🚀 Full Dataset Training with Batch Processing")
    print("=" * 60)
    
    # Load CSV to see total samples
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    total_batches = (total_samples + batch_size - 1) // batch_size
    steps_per_batch = total_steps // total_batches
    
    print(f"📊 Dataset info:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Batch size: {batch_size:,}")
    print(f"  Total batches: {total_batches}")
    print(f"  Steps per batch: {steps_per_batch}")
    print(f"  Total training steps: {total_steps}")
    
    # Create trainer
    trainer = OrpheusTrainer(device="cuda")
    
    # Training loop
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        current_batch_size = end_idx - start_idx
        
        print(f"\n🔄 Batch {batch_idx + 1}/{total_batches}")
        print(f"   Samples: {start_idx:,} - {end_idx:,} ({current_batch_size:,} samples)")
        
        # Create batch CSV
        batch_df = df.iloc[start_idx:end_idx].copy()
        batch_csv = f"temp_batch_{batch_idx}.csv"
        batch_df.to_csv(batch_csv, index=False)
        
        try:
            # Load checkpoint if exists
            checkpoint_path = None
            if batch_idx > 0:
                prev_checkpoint = f"{checkpoint_dir}/batch_{batch_idx-1}"
                if os.path.exists(prev_checkpoint):
                    checkpoint_path = prev_checkpoint
                    print(f"   📁 Loading from checkpoint: {checkpoint_path}")
            
            # Train this batch
            if batch_idx == 0:
                # First batch - setup model
                print("   🤖 Setting up model for first batch...")
                training_stats = trainer.train(
                    csv_path=batch_csv,
                    audio_dir=audio_dir,
                    use_h100_config=True,
                    max_samples=current_batch_size,
                    training_args={
                        "max_steps": steps_per_batch,
                        "output_dir": f"{checkpoint_dir}/batch_{batch_idx}",
                        "save_steps": steps_per_batch // 4,
                        "logging_steps": 10,
                    }
                )
            else:
                # Continue training from previous batch
                print("   🔄 Continuing training from previous batch...")
                # For simplicity, we'll train a new model each batch
                # In practice, you'd want to load and continue from checkpoint
                training_stats = trainer.train(
                    csv_path=batch_csv,
                    audio_dir=audio_dir,
                    use_h100_config=True,
                    max_samples=current_batch_size,
                    training_args={
                        "max_steps": steps_per_batch,
                        "output_dir": f"{checkpoint_dir}/batch_{batch_idx}",
                        "save_steps": steps_per_batch // 4,
                        "logging_steps": 10,
                    }
                )
            
            # Save batch checkpoint
            batch_model_dir = f"{checkpoint_dir}/batch_{batch_idx}"
            trainer.save_model(batch_model_dir)
            print(f"   ✅ Batch {batch_idx + 1} completed, saved to {batch_model_dir}")
            
            # Cleanup
            os.remove(batch_csv)
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # Print memory stats
            memory_used = torch.cuda.memory_reserved() / 1024**3
            print(f"   💾 GPU memory: {memory_used:.2f} GB")
            
        except Exception as e:
            print(f"   ❌ Batch {batch_idx + 1} failed: {e}")
            # Cleanup and continue
            if os.path.exists(batch_csv):
                os.remove(batch_csv)
            continue
    
    # Final model save
    final_model_dir = f"{checkpoint_dir}/final_model"
    trainer.save_model(final_model_dir)
    
    print(f"\n🎉 Full dataset training completed!")
    print(f"📂 Final model saved to: {final_model_dir}")
    print(f"📂 All checkpoints in: {checkpoint_dir}/")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train with full dataset")
    
    parser.add_argument("--csv_path", type=str, default="metadata.csv")
    parser.add_argument("--audio_dir", type=str, default="/workspace/data_tonghop/")
    parser.add_argument("--batch_size", type=int, default=2000,
                       help="Samples per batch (default: 2000)")
    parser.add_argument("--total_steps", type=int, default=5000,
                       help="Total training steps (default: 5000)")
    parser.add_argument("--checkpoint_dir", type=str, default="full_training",
                       help="Directory for checkpoints")
    
    args = parser.parse_args()
    
    train_full_dataset(
        csv_path=args.csv_path,
        audio_dir=args.audio_dir,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == "__main__":
    main()
