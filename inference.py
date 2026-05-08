# -*- coding: utf-8 -*-
"""
Inference script for Orpheus TTS model
"""

import torch
from typing import List, Optional, Union
import argparse
import numpy as np
from IPython.display import display, Audio
import torchaudio

from config import INFERENCE_CONFIG, INFERENCE_PRESETS, DEFAULT_PROMPTS
from model_manager import create_model_manager
from utils import (
    load_snac_model, 
    prepare_inference_input, 
    process_generated_tokens, 
    redistribute_codes,
    print_memory_stats
)


class OrpheusInference:
    """
    Inference class for Orpheus TTS model
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the inference engine
        
        Args:
            device: Device to use for inference
        """
        self.device = device
        self.model_manager = create_model_manager()
        self.snac_model = None
        
    def load_model(self, 
                   model_name: Optional[str] = None,
                   lora_path: Optional[str] = None,
                   use_fp16: bool = False,
                   **model_kwargs):
        """
        Load model for inference
        
        Args:
            model_name: Name of the base model to load
            lora_path: Path to LoRA adapters (if any)
            use_fp16: Use FP16 for faster inference
            **model_kwargs: Additional model configuration arguments
        """
        print(f"Loading model for inference (FP16: {use_fp16})...")
        
        if lora_path:
            # Check if this is a checkpoint directory or LoRA directory
            import os
            if os.path.exists(os.path.join(lora_path, "trainer_state.json")):
                # This is a training checkpoint - extract the model
                print(f"Detected training checkpoint at {lora_path}")
                model_path = lora_path
            else:
                # This is a LoRA adapter directory
                model_path = lora_path
            
            try:
                # Load the trained model directly from checkpoint
                self.model_manager.load_lora_model(
                    model_path,
                    model_name=model_name,
                    **model_kwargs,
                )
            except Exception as e:
                print(f"Failed to load from checkpoint: {e}")
                print("Loading base model + applying LoRA...")
                # Fallback to base model
                self.model_manager.prepare_for_training(model_name=model_name or "unsloth/orpheus-3b-0.1-ft", **model_kwargs)
        else:
            self.model_manager.load_base_model(model_name=model_name, **model_kwargs)
        
        # Convert to FP16 if requested
        if use_fp16 and self.model_manager.model is not None:
            print("🚀 Converting model to FP16 for faster inference...")
            self.model_manager.model = self.model_manager.model.half()
        
        # Prepare for inference
        self.model_manager.prepare_for_inference()
        
        print("Model loaded and ready for inference")
        
    def initialize_snac_model(self):
        """Initialize SNAC model for audio decoding"""
        if self.snac_model is None:
            print("Loading SNAC model...")
            self.snac_model = load_snac_model("cpu")  # Keep on CPU to save GPU memory
            print("SNAC model loaded successfully")
    
    def generate_audio(self, 
                      prompts: Union[str, List[str]],
                      chosen_voice: Optional[str] = None,
                      seed: Optional[int] = None,
                      **generation_kwargs) -> List[torch.Tensor]:
        """
        Generate audio from text prompts
        
        Args:
            prompts: Text prompt(s) to generate audio for
            chosen_voice: Voice name for multi-speaker models  
            seed: Random seed for consistent generation
            **generation_kwargs: Additional generation arguments
            
        Returns:
            List of generated audio tensors
        """
        if self.model_manager.model is None:
            raise ValueError("Model must be loaded first")
        
        # Set seed for consistent generation
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            print(f"🎲 Set random seed: {seed}")
        
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        print(f"Generating audio for {len(prompts)} prompt(s)...")
        
        # Initialize SNAC model if needed
        self.initialize_snac_model()
        
        # Prepare inference input
        input_ids, attention_mask = prepare_inference_input(
            prompts, self.model_manager.tokenizer, chosen_voice
        )
        
        # Move to device with proper dtype
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Ensure FP16 compatibility
        if hasattr(self.model_manager.model, 'dtype') and self.model_manager.model.dtype == torch.float16:
            print("🚀 Using FP16 inference mode")
        
        # Prepare generation config
        generation_config = INFERENCE_CONFIG.copy()
        generation_config.update(generation_kwargs)
        generation_config["eos_token_id"] = generation_config.get("eos_token_id", 128258)
        
        # Fix potential past_key_values issue
        generation_config["use_cache"] = False  # Disable cache to avoid NoneType error
        generation_config["pad_token_id"] = self.model_manager.tokenizer.pad_token_id
        
        print("Generating tokens...")
        print_memory_stats("Before generation")
        
        # Generate with error handling
        try:
            with torch.inference_mode():
                generated_ids = self.model_manager.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_config
                )
        except Exception as gen_e:
            print(f"Generation failed with cache: {gen_e}")
            print("Retrying with simplified config...")
            
            # Simplified generation config
            simple_config = {
                "max_new_tokens": generation_config.get("max_new_tokens", 1200),
                "do_sample": True,
                "temperature": generation_config.get("temperature", 0.6),
                "pad_token_id": self.model_manager.tokenizer.pad_token_id,
                "eos_token_id": generation_config.get("eos_token_id", 128258),
                "use_cache": False,
            }
            
            with torch.inference_mode():
                generated_ids = self.model_manager.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **simple_config
                )
        
        print_memory_stats("After generation")
        
        # Process generated tokens to extract audio codes
        code_lists = process_generated_tokens(generated_ids)
        
        # Convert codes back to audio
        print("Converting codes to audio...")
        audio_samples = []
        for code_list in code_lists:
            if len(code_list) > 0:
                audio_tensor = redistribute_codes(code_list, self.snac_model)
                audio_samples.append(audio_tensor)
            else:
                print("Warning: Empty code list, skipping...")
                audio_samples.append(torch.zeros(1, 1000))  # Empty audio
        
        print(f"Generated {len(audio_samples)} audio sample(s)")
        return audio_samples
    
    def generate_and_play(self, 
                         prompts: Union[str, List[str]],
                         chosen_voice: Optional[str] = None,
                         sample_rate: int = 24000,
                         **generation_kwargs):
        """
        Generate audio and display for playback
        
        Args:
            prompts: Text prompt(s) to generate audio for
            chosen_voice: Voice name for multi-speaker models
            sample_rate: Sample rate for audio playback
            **generation_kwargs: Additional generation arguments
        """
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Generate audio
        audio_samples = self.generate_audio(prompts, chosen_voice, **generation_kwargs)
        
        if len(prompts) != len(audio_samples):
            print("Warning: Number of prompts and samples do not match")
            return
        
        # Display audio for each prompt
        for i, (prompt, samples) in enumerate(zip(prompts, audio_samples)):
            print(f"\nPrompt {i+1}: {prompt}")
            if hasattr(samples, 'detach'):
                audio_data = samples.detach().squeeze().to("cpu").numpy()
            else:
                audio_data = samples
            
            try:
                display(Audio(audio_data, rate=sample_rate))
            except Exception as e:
                print(f"Error displaying audio: {e}")
                print(f"Audio shape: {audio_data.shape if hasattr(audio_data, 'shape') else 'unknown'}")
    
    def save_audio(self, 
                   audio_samples: List[torch.Tensor],
                   output_paths: List[str],
                   sample_rate: int = 24000):
        """
        Save generated audio to files
        
        Args:
            audio_samples: List of audio tensors
            output_paths: List of output file paths
            sample_rate: Sample rate for audio files
        """
        if len(audio_samples) != len(output_paths):
            raise ValueError("Number of audio samples and output paths must match")
        
        for i, (samples, output_path) in enumerate(zip(audio_samples, output_paths)):
            try:
                if hasattr(samples, 'detach'):
                    audio_data = samples.detach().squeeze().to("cpu")
                else:
                    audio_data = torch.tensor(samples).squeeze()
                
                # Ensure audio_data is 2D (channels, samples)
                if audio_data.dim() == 1:
                    audio_data = audio_data.unsqueeze(0)
                
                try:
                    torchaudio.save(output_path, audio_data, sample_rate)
                except Exception as torchaudio_error:
                    try:
                        import soundfile as sf

                        audio_np = audio_data.detach().cpu().numpy()
                        if audio_np.ndim == 2:
                            audio_np = audio_np.T
                        sf.write(output_path, audio_np, sample_rate)
                    except Exception as soundfile_error:
                        raise RuntimeError(
                            f"torchaudio.save failed: {torchaudio_error}; "
                            f"soundfile fallback failed: {soundfile_error}"
                        ) from soundfile_error
                print(f"Saved audio {i+1} to {output_path}")
                
            except Exception as e:
                print(f"Error saving audio {i+1} to {output_path}: {e}")
    
    def batch_inference(self, 
                       prompts: List[str],
                       output_dir: str,
                       chosen_voice: Optional[str] = None,
                       file_prefix: str = "generated_audio",
                       **generation_kwargs) -> List[str]:
        """
        Perform batch inference and save results
        
        Args:
            prompts: List of text prompts
            output_dir: Directory to save audio files
            chosen_voice: Voice name for multi-speaker models
            file_prefix: Prefix for output files
            **generation_kwargs: Additional generation arguments
            
        Returns:
            List of output file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate audio
        audio_samples = self.generate_audio(prompts, chosen_voice, **generation_kwargs)
        
        # Create output paths
        output_paths = []
        for i in range(len(audio_samples)):
            output_path = os.path.join(output_dir, f"{file_prefix}_{i+1:03d}.wav")
            output_paths.append(output_path)
        
        # Save audio files
        self.save_audio(audio_samples, output_paths)
        
        return output_paths
    
    def cleanup(self):
        """Clean up resources"""
        # Move SNAC model to CPU to free GPU memory
        if self.snac_model is not None:
            self.snac_model.to("cpu")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Cleanup completed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Orpheus TTS inference")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default=None,
                       help="Name of the base model to use")
    parser.add_argument("--lora_path", type=str, default=None,
                       help="Path to LoRA adapters")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4bit mode")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length for model loading")
    parser.add_argument("--fp16", action="store_true",
                       help="Load/convert model in FP16")
    parser.add_argument("--bf16", action="store_true",
                       help="Load model in BF16")
    
    # Input arguments
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for generation")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="File containing prompts (one per line)")
    parser.add_argument("--chosen_voice", type=str, default=None,
                       help="Voice name for multi-speaker models")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=None,
                       help="Maximum new tokens to generate")
    parser.add_argument("--preset", type=str, choices=sorted(INFERENCE_PRESETS.keys()), default=None,
                       help="Generation preset. Explicit generation flags override it")
    parser.add_argument("--temperature", type=float, default=None,
                       help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=None,
                       help="Repetition penalty")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for repeatable generation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="generated_audio",
                       help="Output directory for audio files")
    parser.add_argument("--file_prefix", type=str, default="generated_audio",
                       help="Prefix for output files")
    parser.add_argument("--sample_rate", type=int, default=24000,
                       help="Sample rate for audio output")
    
    # Device argument
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference")
    
    # Mode arguments
    parser.add_argument("--save_audio", action="store_true",
                       help="Save audio to files instead of playing")
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()
    
    # Create inference engine
    inference = OrpheusInference(device=args.device)
    
    # Prepare model kwargs
    model_kwargs = {}
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if args.max_seq_length:
        model_kwargs["max_seq_length"] = args.max_seq_length
    if args.bf16:
        model_kwargs["dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["dtype"] = torch.float16
    
    # Load model
    inference.load_model(
        model_name=args.model_name,
        lora_path=args.lora_path,
        use_fp16=args.fp16,
        **model_kwargs
    )
    
    # Prepare prompts
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Use default prompts
        prompts = DEFAULT_PROMPTS
        print("Using default prompts:")
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}: {prompt}")
    
    # Prepare generation kwargs
    generation_kwargs = INFERENCE_PRESETS.get(args.preset, {}).copy()
    if args.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        generation_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        generation_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.seed is not None:
        generation_kwargs["seed"] = args.seed
    
    try:
        if args.save_audio:
            # Batch inference and save
            output_paths = inference.batch_inference(
                prompts=prompts,
                output_dir=args.output_dir,
                chosen_voice=args.chosen_voice,
                file_prefix=args.file_prefix,
                **generation_kwargs
            )
            print(f"\nGenerated {len(output_paths)} audio files:")
            for path in output_paths:
                print(f"  {path}")
        else:
            # Generate and play
            inference.generate_and_play(
                prompts=prompts,
                chosen_voice=args.chosen_voice,
                sample_rate=args.sample_rate,
                **generation_kwargs
            )
        
        print("\nInference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    finally:
        # Cleanup
        inference.cleanup()


if __name__ == "__main__":
    main()
