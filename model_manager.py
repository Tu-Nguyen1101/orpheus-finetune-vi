# -*- coding: utf-8 -*-
"""
Model management module for Orpheus TTS
Handles model loading, initialization, and saving
"""

# Import unsloth first to avoid warnings
import unsloth
from unsloth import FastLanguageModel
import torch
from typing import Tuple, Optional

from config import MODEL_CONFIG, LORA_CONFIG, MODEL_PATHS
from utils import get_gpu_memory_stats, print_memory_stats


class OrpheusModelManager:
    """
    Model manager for Orpheus TTS model
    """
    
    def __init__(self):
        """Initialize the model manager"""
        self.model = None
        self.tokenizer = None
        self.is_inference_mode = False
        
    def load_base_model(self, 
                       model_name: Optional[str] = None,
                       max_seq_length: Optional[int] = None,
                       dtype: Optional[str] = None,
                       load_in_4bit: Optional[bool] = None,
                       token: Optional[str] = None) -> Tuple[any, any]:
        """
        Load the base model and tokenizer
        
        Args:
            model_name: Name of the model to load
            max_seq_length: Maximum sequence length
            dtype: Data type for the model
            load_in_4bit: Whether to load in 4bit mode
            token: HuggingFace token for gated models
            
        Returns:
            Tuple of (model, tokenizer)
        """
        print("Loading base model...")
        print_memory_stats("Before model loading")
        
        # Use config values as defaults
        model_name = model_name or MODEL_CONFIG["model_name"]
        max_seq_length = max_seq_length or MODEL_CONFIG["max_seq_length"]
        dtype = dtype if dtype is not None else MODEL_CONFIG["dtype"]
        load_in_4bit = load_in_4bit if load_in_4bit is not None else MODEL_CONFIG["load_in_4bit"]
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token,
        )
        
        # Set pad token if not already set - important for data collation
        from config import TOKEN_CONFIG
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = TOKEN_CONFIG["pad_token"]
            self.tokenizer.pad_token = self.tokenizer.decode(TOKEN_CONFIG["pad_token"])
        
        print(f"Successfully loaded model: {model_name}")
        print_memory_stats("After model loading")
        
        return self.model, self.tokenizer
    
    def add_lora_adapters(self,
                         r: Optional[int] = None,
                         target_modules: Optional[list] = None,
                         lora_alpha: Optional[int] = None,
                         lora_dropout: Optional[float] = None,
                         bias: Optional[str] = None,
                         use_gradient_checkpointing: Optional[str] = None,
                         random_state: Optional[int] = None,
                         use_rslora: Optional[bool] = None,
                         loftq_config: Optional[dict] = None):
        """
        Add LoRA adapters to the model
        
        Args:
            r: LoRA rank
            target_modules: Target modules for LoRA
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            bias: Bias setting for LoRA
            use_gradient_checkpointing: Gradient checkpointing setting
            random_state: Random state for reproducibility
            use_rslora: Whether to use rank stabilized LoRA
            loftq_config: LoftQ configuration
        """
        if self.model is None:
            raise ValueError("Base model must be loaded first")
        
        print("Adding LoRA adapters...")
        print_memory_stats("Before LoRA")
        
        # Use config values as defaults
        config = LORA_CONFIG
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r or config["r"],
            target_modules=target_modules or config["target_modules"],
            lora_alpha=lora_alpha or config["lora_alpha"],
            lora_dropout=lora_dropout if lora_dropout is not None else config["lora_dropout"],
            bias=bias or config["bias"],
            use_gradient_checkpointing=use_gradient_checkpointing or config["use_gradient_checkpointing"],
            random_state=random_state or config["random_state"],
            use_rslora=use_rslora if use_rslora is not None else config["use_rslora"],
            loftq_config=loftq_config or config["loftq_config"],
        )
        
        print("LoRA adapters added successfully")
        print_memory_stats("After LoRA")
    
    def prepare_for_training(self, **kwargs) -> Tuple[any, any]:
        """
        Load model and prepare for training
        
        Args:
            **kwargs: Additional arguments for model loading and LoRA configuration
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Extract model loading arguments
        model_args = {k: v for k, v in kwargs.items() if k in [
            'model_name', 'max_seq_length', 'dtype', 'load_in_4bit', 'token'
        ]}
        
        # Extract LoRA arguments
        lora_args = {k: v for k, v in kwargs.items() if k in [
            'r', 'target_modules', 'lora_alpha', 'lora_dropout', 'bias',
            'use_gradient_checkpointing', 'random_state', 'use_rslora', 'loftq_config'
        ]}
        
        # Load base model
        self.load_base_model(**model_args)
        
        # Add LoRA adapters
        self.add_lora_adapters(**lora_args)
        
        return self.model, self.tokenizer
    
    def prepare_for_inference(self):
        """Prepare model for inference"""
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        print("Preparing model for inference...")
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference
        self.is_inference_mode = True
        print("Model ready for inference")
    
    def save_lora_model(self, output_dir: Optional[str] = None):
        """
        Save LoRA adapters locally
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        output_dir = output_dir or MODEL_PATHS["lora_output"]
        
        print(f"Saving LoRA model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("LoRA model saved successfully")
    
    def push_lora_to_hub(self, repo_name: str, token: str):
        """
        Push LoRA adapters to Hugging Face Hub
        
        Args:
            repo_name: Repository name on Hugging Face Hub
            token: Hugging Face token
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        print(f"Pushing LoRA model to Hub: {repo_name}...")
        self.model.push_to_hub(repo_name, token=token)
        self.tokenizer.push_to_hub(repo_name, token=token)
        print("LoRA model pushed to Hub successfully")
    
    def save_merged_model(self, 
                         output_dir: str,
                         save_method: str = "merged_16bit"):
        """
        Save merged model (base + LoRA) in specified format
        
        Args:
            output_dir: Directory to save the model
            save_method: Save method - "merged_16bit", "merged_4bit", or "lora"
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        if save_method not in ["merged_16bit", "merged_4bit", "lora"]:
            raise ValueError("save_method must be 'merged_16bit', 'merged_4bit', or 'lora'")
        
        print(f"Saving merged model to {output_dir} with method {save_method}...")
        
        if save_method == "lora":
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        else:
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method=save_method)
        
        print("Merged model saved successfully")
    
    def push_merged_to_hub(self, 
                          repo_name: str, 
                          token: str,
                          save_method: str = "merged_16bit"):
        """
        Push merged model to Hugging Face Hub
        
        Args:
            repo_name: Repository name on Hugging Face Hub
            token: Hugging Face token
            save_method: Save method - "merged_16bit", "merged_4bit", or "lora"
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        if save_method not in ["merged_16bit", "merged_4bit", "lora"]:
            raise ValueError("save_method must be 'merged_16bit', 'merged_4bit', or 'lora'")
        
        print(f"Pushing merged model to Hub: {repo_name} with method {save_method}...")
        
        if save_method == "lora":
            self.model.push_to_hub(repo_name, token=token)
            self.tokenizer.push_to_hub(repo_name, token=token)
        else:
            self.model.push_to_hub_merged(repo_name, self.tokenizer, save_method=save_method, token=token)
        
        print("Merged model pushed to Hub successfully")
    
    def load_lora_model(self, lora_path: str):
        """
        Load a saved LoRA model
        
        Args:
            lora_path: Path to the saved LoRA model
        """
        print(f"Loading LoRA model from {lora_path}...")
        
        import os
        
        print("🔍 Analyzing checkpoint directory...")
        
        # Check for PEFT/LoRA checkpoint structure
        if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
            print("✅ Detected PEFT/LoRA checkpoint with adapter_model.safetensors")
            
            try:
                # Load base model first
                print("📂 Loading base model...")
                self.load_base_model(max_seq_length=4096)
                
                # Load LoRA adapters using PEFT
                print("🔧 Loading LoRA adapters...")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(
                    self.model, 
                    lora_path,
                    is_trainable=False  # Set to inference mode
                )
                
                # Load tokenizer from checkpoint if available
                tokenizer_path = os.path.join(lora_path, "tokenizer.json")
                if os.path.exists(tokenizer_path):
                    print("📝 Loading tokenizer from checkpoint...")
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(lora_path)
                
                print("✅ PEFT model loaded successfully with trained weights")
                
            except Exception as peft_e:
                print(f"❌ Failed to load PEFT adapters: {peft_e}")
                print("🔄 Falling back to base model only...")
                self.load_base_model(max_seq_length=4096)
        
        elif os.path.exists(os.path.join(lora_path, "pytorch_model.bin")):
            print("📦 Detected pytorch_model.bin - attempting direct load...")
            
            try:
                # Try loading as complete model
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=lora_path,
                    max_seq_length=4096,
                    dtype=None,
                    load_in_4bit=False,
                )
                print("✅ Successfully loaded complete model")
                
            except Exception as e:
                print(f"❌ Failed to load as complete model: {e}")
                print("🔄 Falling back to base model...")
                self.load_base_model(max_seq_length=4096)
        
        else:
            print("❓ Unknown checkpoint format - using base model")
            self.load_base_model(max_seq_length=4096)
        
        # Always ensure pad token is set
        from config import TOKEN_CONFIG
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = TOKEN_CONFIG["pad_token"]
            self.tokenizer.pad_token = self.tokenizer.decode(TOKEN_CONFIG["pad_token"])
        
        print("✅ Model loading completed")
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        memory_stats = get_gpu_memory_stats()
        
        return {
            "model_loaded": True,
            "inference_mode": self.is_inference_mode,
            "memory_stats": memory_stats,
            "model_type": type(self.model).__name__,
        }


def create_model_manager() -> OrpheusModelManager:
    """
    Factory function to create a model manager
    
    Returns:
        OrpheusModelManager instance
    """
    return OrpheusModelManager()
