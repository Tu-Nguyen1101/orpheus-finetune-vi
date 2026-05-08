# -*- coding: utf-8 -*-
"""
Training script for Orpheus TTS model
"""

import torch
from typing import Optional, Dict, Any
import argparse
import os

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import TRAINING_CONFIG, MODEL_CONFIG, LORA_CONFIG
from model_manager import create_model_manager
from data_processor import create_data_processor, print_tokenization_info
from utils import print_memory_stats


def get_transformers_training_classes():
    """Import Transformers training classes lazily to make preflight errors clearer."""
    try:
        from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
    except Exception as exc:
        raise ImportError(
            "Transformers training dependencies failed to import. Reinstall the project "
            "requirements before running training."
        ) from exc

    return TrainingArguments, Trainer, DataCollatorForSeq2Seq


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
                      training_args: Optional[Dict[str, Any]] = None) -> Any:
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
        
        TrainingArguments, Trainer, DataCollatorForSeq2Seq = get_transformers_training_classes()
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
             max_samples: Optional[int] = None,
             resume_from_checkpoint: Optional[str] = None):
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
            resume_from_checkpoint: Optional checkpoint directory to resume from
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
        if resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            self.training_stats = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
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
        
        # Print training time
        runtime = self.training_stats.metrics['train_runtime']
        print(f"\nTraining completed!")
        print(f"Training time: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")

        # Calculate memory usage when CUDA is available
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                       help="Override gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Override maximum training steps")
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="Override warmup steps")
    parser.add_argument("--logging_steps", type=int, default=None,
                       help="Override logging interval")
    parser.add_argument("--save_steps", type=int, default=None,
                       help="Override checkpoint save interval")
    parser.add_argument("--save_total_limit", type=int, default=None,
                       help="Override number of checkpoints to keep")
    parser.add_argument("--num_train_epochs", type=float, default=None,
                       help="Override number of training epochs")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save the trained model (default: full_dataset_model)")
    parser.add_argument("--save_lora_only", action="store_true",
                       help="Save LoRA adapters only; skip merged 16-bit model save")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from a Trainer checkpoint directory")

    # Optional model overrides
    parser.add_argument("--max_seq_length", type=int, default=None,
                       help="Override model sequence length")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load the base model in 4-bit mode")
    parser.add_argument("--lora_rank", type=int, default=None,
                       help="Override LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None,
                       help="Override LoRA alpha")

    # Precision / GPU memory controls
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 training precision")
    parser.add_argument("--bf16", action="store_true",
                       help="Use BF16 training precision when supported")
    parser.add_argument("--low_vram", action="store_true",
                       help="Use conservative settings for small GPUs")
    parser.add_argument("--no_auto_low_vram", action="store_true",
                       help="Do not auto-enable low-VRAM settings on GPUs with <=6GB VRAM")
    
    # Device argument
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training (default: cuda)")
    
    return parser.parse_args()


def validate_device(device: str) -> Dict[str, Any]:
    """Validate the requested training device and return basic device info."""
    device = device.lower()

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but this Python environment is using CPU-only PyTorch. "
                "Install the CUDA PyTorch build first, then rerun training."
            )

        device_index = 0
        if ":" in device:
            device_index = int(device.split(":", 1)[1])

        props = torch.cuda.get_device_properties(device_index)
        memory_gb = props.total_memory / 1024**3
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"✅ CUDA device: {props.name} ({memory_gb:.1f} GB VRAM)")
        print(f"✅ BF16 supported: {bf16_supported}")

        return {
            "type": "cuda",
            "name": props.name,
            "memory_gb": memory_gb,
            "bf16_supported": bf16_supported,
        }

    if device != "cpu":
        raise ValueError(f"Unsupported device: {device}")

    print("⚠️  Training on CPU. This will be very slow.")
    return {"type": "cpu", "name": "CPU", "memory_gb": 0, "bf16_supported": False}


def apply_precision_defaults(training_kwargs: Dict[str, Any],
                             device_info: Dict[str, Any],
                             fp16_requested: bool,
                             bf16_requested: bool) -> None:
    """Set a compatible mixed precision mode."""
    if device_info["type"] != "cuda":
        training_kwargs["bf16"] = False
        training_kwargs["fp16"] = False
        training_kwargs["tf32"] = False
        return

    if bf16_requested and device_info["bf16_supported"]:
        training_kwargs["bf16"] = True
        training_kwargs["fp16"] = False
        return

    if bf16_requested and not device_info["bf16_supported"]:
        print("⚠️  BF16 was requested but this GPU does not support it. Falling back to FP16.")

    if fp16_requested or not device_info["bf16_supported"]:
        training_kwargs["bf16"] = False
        training_kwargs["fp16"] = True
        return

    training_kwargs["bf16"] = True
    training_kwargs["fp16"] = False


def apply_low_vram_defaults(training_kwargs: Dict[str, Any],
                            model_kwargs: Dict[str, Any]) -> None:
    """Use defaults that have a realistic chance on 4-6GB GPUs."""
    training_kwargs.setdefault("per_device_train_batch_size", 1)
    training_kwargs.setdefault("gradient_accumulation_steps", 8)
    training_kwargs.setdefault("dataloader_num_workers", 0)
    training_kwargs.setdefault("logging_steps", 10)
    training_kwargs.setdefault("save_steps", 100)
    training_kwargs.setdefault("save_total_limit", 2)
    training_kwargs.setdefault("bf16", False)
    training_kwargs.setdefault("fp16", True)

    model_kwargs.setdefault("load_in_4bit", True)
    model_kwargs.setdefault("max_seq_length", 2048)
    model_kwargs.setdefault("r", 16)
    model_kwargs.setdefault("lora_alpha", 16)
    model_kwargs.setdefault("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


def main():
    """Main training function for full dataset with 5 epochs"""
    args = parse_args()
    device_info = validate_device(args.device)
    
    print("🚀 Orpheus TTS Full Dataset Training")
    print("=" * 50)
    print(f"📂 CSV Path: {args.csv_path}")
    print(f"🎵 Audio Dir: {args.audio_dir}")
    print(f"📊 Max Samples: {'Full Dataset' if args.max_samples is None else args.max_samples}")
    print(f"⚙️  Epochs: {args.num_train_epochs or TRAINING_CONFIG.get('num_train_epochs')}")
    if args.max_steps is not None:
        print(f"🔢 Max steps: {args.max_steps}")
    print("=" * 50)
    
    # Create trainer
    trainer = OrpheusTrainer(device=args.device)
    
    # Prepare training kwargs (only override if specified)
    save_dir = args.save_dir or "full_dataset_model"
    training_kwargs = {"output_dir": save_dir}
    model_kwargs = {}

    if args.batch_size:
        training_kwargs["per_device_train_batch_size"] = args.batch_size
    if args.gradient_accumulation_steps:
        training_kwargs["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.learning_rate:
        training_kwargs["learning_rate"] = args.learning_rate
    if args.max_steps:
        training_kwargs["max_steps"] = args.max_steps
    if args.warmup_steps is not None:
        training_kwargs["warmup_steps"] = args.warmup_steps
    if args.logging_steps is not None:
        training_kwargs["logging_steps"] = args.logging_steps
    if args.save_steps is not None:
        training_kwargs["save_steps"] = args.save_steps
    if args.save_total_limit is not None:
        training_kwargs["save_total_limit"] = args.save_total_limit
    if args.num_train_epochs:
        training_kwargs["num_train_epochs"] = args.num_train_epochs

    if args.max_seq_length:
        model_kwargs["max_seq_length"] = args.max_seq_length
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if args.lora_rank:
        model_kwargs["r"] = args.lora_rank
    if args.lora_alpha:
        model_kwargs["lora_alpha"] = args.lora_alpha

    auto_low_vram = (
        device_info["type"] == "cuda"
        and device_info["memory_gb"] <= 6
        and not args.no_auto_low_vram
    )
    if args.low_vram or auto_low_vram:
        print("⚠️  Low-VRAM GPU settings enabled.")
        apply_low_vram_defaults(training_kwargs, model_kwargs)

    apply_precision_defaults(training_kwargs, device_info, args.fp16, args.bf16)
    
    try:
        # Start training
        training_stats = trainer.train(
            csv_path=args.csv_path,
            audio_dir=args.audio_dir,
            text_column=args.text_column,
            audio_column=args.audio_column,
            training_args=training_kwargs if training_kwargs else None,
            model_kwargs=model_kwargs if model_kwargs else None,
            max_samples=args.max_samples,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        # Save model
        trainer.save_model(save_dir, save_merged=not args.save_lora_only)
        
        print("\n" + "=" * 50)
        print("🎉 Full Dataset Training Completed Successfully!")
        print(f"💾 Model saved to: {save_dir}")
        if not args.save_lora_only:
            print(f"🔧 Merged model for inference: {save_dir}_merged")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
