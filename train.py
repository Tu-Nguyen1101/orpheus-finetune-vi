# -*- coding: utf-8 -*-
"""
Training script for Orpheus TTS model
"""

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from typing import Optional, Dict, Any
import argparse
import os

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import TRAINING_CONFIG, MODEL_CONFIG, LORA_CONFIG
from model_manager import create_model_manager
from data_processor import create_data_processor, print_tokenization_info
from utils import print_memory_stats


class OrpheusTrainer:
    """
    Training class for Orpheus TTS model
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the trainer
        
        Args:
            device: Device to use for training
        """
        self.device = device
        self.model_manager = create_model_manager()
        self.data_processor = None
        self.trainer = None
        self.training_stats = None
        
    def setup_model(self, **model_kwargs):
        """
        Setup model for training
        
        Args:
            **model_kwargs: Additional model configuration arguments
        """
        print("Setting up model for training...")
        model, tokenizer = self.model_manager.prepare_for_training(**model_kwargs)
        
        # Create data processor with tokenizer
        self.data_processor = create_data_processor(tokenizer, self.device)
        
        return model, tokenizer
    
    def prepare_dataset(self, 
                       csv_path: str,
                       audio_dir: str,
                       text_column: str = "sentence",
                       audio_column: str = "uuid",
                       max_samples: int = 999999999):
        """
        Prepare the training dataset from CSV
        
        Args:
            csv_path: Path to CSV metadata file
            audio_dir: Directory containing audio files
            text_column: Column name for text data in CSV
            audio_column: Column name for audio filenames in CSV
            max_samples: Maximum samples to load (large number for full dataset)
            
        Returns:
            Processed dataset
        """
        if self.data_processor is None:
            raise ValueError("Model must be setup first to initialize data processor")
        
        print("Preparing training dataset...")
        print_tokenization_info()
        
        # Use CSV dataset (full dataset training)
        print("🗂️ Loading CSV dataset for full training...")
        dataset = self.data_processor.process_csv_dataset(
            csv_path=csv_path,
            audio_dir=audio_dir,
            text_column=text_column,
            audio_column=audio_column,
            max_samples=max_samples
        )
        
        return dataset
    
    def create_trainer(self, 
                      dataset,
                      training_args: Optional[Dict[str, Any]] = None) -> Trainer:
        """
        Create the Trainer instance
        
        Args:
            dataset: Training dataset
            training_args: Training arguments override
            
        Returns:
            Trainer instance
        """
        # Use config values as defaults, override with provided args
        config = TRAINING_CONFIG.copy()
        if training_args:
            config.update(training_args)
        
        print("Creating trainer with configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        training_arguments = TrainingArguments(**config)
        
        # Ensure tokenizer is available
        if self.model_manager.tokenizer is None:
            raise ValueError("Tokenizer must be loaded before creating trainer")
        
        # Configure data collator for variable-length sequences
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.model_manager.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        self.trainer = Trainer(
            model=self.model_manager.model,
            train_dataset=dataset,
            args=training_arguments,
            data_collator=data_collator,
        )
        
        return self.trainer
    
    def train(self, 
             csv_path: str,
             audio_dir: str,
             text_column: str = "sentence",
             audio_column: str = "uuid",
             training_args: Optional[Dict[str, Any]] = None,
             model_kwargs: Optional[Dict[str, Any]] = None,
             max_samples: Optional[int] = None):
        """
        Complete training pipeline for full dataset with 5 epochs
        
        Args:
            csv_path: Path to CSV metadata file
            audio_dir: Directory containing audio files
            text_column: Column name for text data in CSV
            audio_column: Column name for audio filenames in CSV
            training_args: Training arguments override
            model_kwargs: Model configuration arguments
            max_samples: Maximum samples to load (None for full dataset)
        """
        # Setup for full dataset training (5 epochs)
        print("🚀 Setting up full dataset training with 5 epochs")
        
        # Use default configs
        default_model_kwargs = {**MODEL_CONFIG, **LORA_CONFIG}
        if model_kwargs:
            default_model_kwargs.update(model_kwargs)
        model_kwargs = default_model_kwargs
        
        # Use default training config
        if training_args is None:
            training_args = TRAINING_CONFIG.copy()
        else:
            default_training_args = TRAINING_CONFIG.copy()
            default_training_args.update(training_args)
            training_args = default_training_args
        
        # Setup model
        model_kwargs = model_kwargs or {}
        self.setup_model(**model_kwargs)
        
        # Prepare dataset (full dataset if max_samples is None)
        dataset = self.prepare_dataset(
            csv_path=csv_path,
            audio_dir=audio_dir,
            text_column=text_column,
            audio_column=audio_column,
            max_samples=max_samples or 999999999  # Very large number for full dataset
        )
        
        # Create trainer
        self.create_trainer(dataset, training_args)
        
        # Show memory stats before training
        print_memory_stats("Before training")
        
        # Start training
        print("Starting training...")
        if self.trainer is None:
            raise ValueError("Trainer must be created before training")
        self.training_stats = self.trainer.train()
        
        # Show memory and time stats after training
        self.print_training_stats()
        
        return self.training_stats
    
    def print_training_stats(self):
        """Print training statistics"""
        if self.training_stats is None:
            print("No training statistics available")
            return
        
        print_memory_stats("After training")
        
        # Calculate memory usage
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        
        # Print training time
        runtime = self.training_stats.metrics['train_runtime']
        print(f"\nTraining completed!")
        print(f"Training time: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
        print(f"Peak memory usage: {used_memory} GB ({used_percentage}% of {max_memory} GB)")
        
        # Print other metrics if available
        if 'train_loss' in self.training_stats.metrics:
            print(f"Final training loss: {self.training_stats.metrics['train_loss']:.4f}")
    
    def save_model(self, output_dir: Optional[str] = None, save_merged: bool = True):
        """
        Save the trained model
        
        Args:
            output_dir: Directory to save the model
            save_merged: Whether to save merged model (better for inference)
        """
        if self.model_manager.model is None:
            raise ValueError("No model to save")
        
        if save_merged:
            # Save merged model for easier inference
            merged_dir = f"{output_dir}_merged" if output_dir else "merged_model"
            print(f"💾 Saving merged model for inference to {merged_dir}...")
            try:
                self.model_manager.save_merged_model(merged_dir, "merged_16bit")
                print(f"✅ Merged model saved to {merged_dir}")
            except Exception as e:
                print(f"⚠️  Failed to save merged model: {e}")
                print("Falling back to LoRA-only save...")
                self.model_manager.save_lora_model(output_dir)
        else:
            # Save LoRA adapters only
            self.model_manager.save_lora_model(output_dir)
        
        print(f"Model saved successfully")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        return self.model_manager.get_model_info()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Orpheus TTS model on full dataset (5 epochs)")
    
    # Required dataset arguments
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path to CSV metadata file")
    parser.add_argument("--audio_dir", type=str, required=True,
                       help="Directory containing audio files")
    
    # Optional dataset arguments
    parser.add_argument("--text_column", type=str, default="sentence",
                       help="Column name for text data in CSV (default: sentence)")
    parser.add_argument("--audio_column", type=str, default="uuid",
                       help="Column name for audio filenames in CSV (default: uuid)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to load (None for full dataset)")
    
    # Optional training overrides
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override training batch size")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save the trained model (default: full_dataset_model)")
    
    # Device argument
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training (default: cuda)")
    
    return parser.parse_args()


def main():
    """Main training function for full dataset with 5 epochs"""
    args = parse_args()
    
    print("🚀 Orpheus TTS Full Dataset Training")
    print("=" * 50)
    print(f"📂 CSV Path: {args.csv_path}")
    print(f"🎵 Audio Dir: {args.audio_dir}")
    print(f"📊 Max Samples: {'Full Dataset' if args.max_samples is None else args.max_samples}")
    print(f"⚙️  Epochs: 5")
    print("=" * 50)
    
    # Create trainer
    trainer = OrpheusTrainer(device=args.device)
    
    # Prepare training kwargs (only override if specified)
    training_kwargs = {}
    if args.batch_size:
        training_kwargs["per_device_train_batch_size"] = args.batch_size
    if args.learning_rate:
        training_kwargs["learning_rate"] = args.learning_rate
    
    try:
        # Start training
        training_stats = trainer.train(
            csv_path=args.csv_path,
            audio_dir=args.audio_dir,
            text_column=args.text_column,
            audio_column=args.audio_column,
            training_args=training_kwargs if training_kwargs else None,
            max_samples=args.max_samples
        )
        
        # Save model
        save_dir = args.save_dir or "full_dataset_model"
        trainer.save_model(save_dir, save_merged=True)
        
        print("\n" + "=" * 50)
        print("🎉 Full Dataset Training Completed Successfully!")
        print(f"💾 Model saved to: {save_dir}")
        print(f"🔧 Merged model for inference: {save_dir}_merged")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
