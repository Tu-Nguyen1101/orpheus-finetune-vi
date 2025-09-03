# -*- coding: utf-8 -*-
"""
Data processing module for Orpheus TTS training
Handles dataset loading, preprocessing, and tokenization
"""

from datasets import load_dataset, Dataset
from typing import Dict, Any, Optional, List, Generator
import torch
from snac import SNAC
import pandas as pd
import librosa
import os
import numpy as np
from collections import Counter
import psutil
import gc

from config import DATASET_CONFIG, TOKEN_CONFIG
from utils import tokenise_audio, remove_duplicate_frames, load_snac_model


class OrpheusDataProcessor:
    """
    Data processor for Orpheus TTS model training
    """
    
    def __init__(self, tokenizer, device: str = "cuda"):
        """
        Initialize the data processor
        
        Args:
            tokenizer: Model tokenizer
            device: Device to use for processing
        """
        self.tokenizer = tokenizer
        self.device = device
        self.snac_model = None
        self.dataset = None
        self.ds_sample_rate = None
    
    def get_memory_usage_gb(self) -> float:
        """Get current system memory usage in GB"""
        return psutil.virtual_memory().used / (1024**3)
    
    def get_available_memory_gb(self) -> float:
        """Get available system memory in GB"""
        return psutil.virtual_memory().available / (1024**3)
    
    def should_trigger_gc(self, threshold_gb: float = 2.0) -> bool:
        """Check if we should trigger garbage collection"""
        return self.get_available_memory_gb() < threshold_gb
    
    def load_dataset(self, dataset_name: Optional[str] = None, split: Optional[str] = None) -> Dataset:
        """
        Load the training dataset
        
        Args:
            dataset_name: Name of the dataset to load
            split: Dataset split to use
            
        Returns:
            Loaded dataset
        """
        dataset_name = dataset_name or DATASET_CONFIG["dataset_name"]
        split = split or DATASET_CONFIG["split"]
        
        print(f"Loading dataset: {dataset_name}, split: {split}")
        self.dataset = load_dataset(dataset_name, split=split)
        
        # Get sample rate from first audio sample
        try:
            if hasattr(self.dataset, '__len__') and len(self.dataset) > 0 and "audio" in self.dataset[0]:
                self.ds_sample_rate = self.dataset[0]["audio"]["sampling_rate"]
                print(f"Dataset sample rate: {self.ds_sample_rate}")
        except (IndexError, KeyError, TypeError):
            print("Warning: Could not determine sample rate from dataset")
            self.ds_sample_rate = 24000  # Default sample rate
        
        return self.dataset
    
    def load_csv_dataset_streaming(self, csv_path: str, audio_dir: str, 
                                   text_column: str = "sentence", 
                                   audio_column: str = "uuid",
                                   target_sample_rate: int = 24000,
                                   max_samples: Optional[int] = None,
                                   chunk_size: int = 1000) -> Dataset:
        """
        Memory-efficient streaming CSV dataset loader
        
        Args:
            csv_path: Path to CSV metadata file
            audio_dir: Directory containing audio files
            text_column: Column name for text data
            audio_column: Column name for audio filenames
            target_sample_rate: Target sample rate for audio
            max_samples: Maximum number of samples to load (None for all)
            chunk_size: Process data in chunks to manage memory
            
        Returns:
            Loaded dataset
        """
        print(f"🚀 Loading CSV dataset with streaming (memory-efficient mode)")
        print(f"CSV path: {csv_path}")
        print(f"Audio directory: {audio_dir}")
        print(f"Chunk size: {chunk_size}")
        
        # Read CSV metadata only (lightweight)
        df = pd.read_csv(csv_path)
        total_samples = len(df)
        print(f"Found {total_samples} entries in CSV")
        
        # Apply sample limit
        if max_samples and total_samples > max_samples:
            print(f"🔥 Limiting to {max_samples} samples (from {total_samples} total)")
            df = df.head(max_samples)
            total_samples = max_samples
        
        # Check vocabulary coverage with sample
        sample_texts = df[text_column].head(100).tolist()
        self.check_vocabulary_coverage(sample_texts)
        
        # Create lightweight metadata-only dataset first
        metadata_list = []
        for idx, row in df.iterrows():
            audio_filename = row[audio_column]
            text = row[text_column]
            audio_path = os.path.join(audio_dir, audio_filename)
            
            # Only include if audio file exists
            if os.path.exists(audio_path):
                metadata_list.append({
                    "text": text,
                    "audio_path": audio_path,
                    "sampling_rate": target_sample_rate
                })
            else:
                print(f"Warning: Audio file not found: {audio_path}")
        
        print(f"Valid audio files found: {len(metadata_list)}")
        self.ds_sample_rate = target_sample_rate
        
        # Convert to dataset - this is lightweight as we're only storing paths
        dataset = Dataset.from_list(metadata_list)
        
        # Clean up pandas dataframe
        del df, metadata_list
        gc.collect()
        
        return dataset
    
    def load_csv_dataset(self, csv_path: str, audio_dir: str, 
                        text_column: str = "sentence", 
                        audio_column: str = "uuid",
                        target_sample_rate: int = 24000,
                        max_samples: int = 1000) -> Dataset:
        """
        Load dataset from CSV metadata and audio directory
        
        Args:
            csv_path: Path to CSV metadata file
            audio_dir: Directory containing audio files
            text_column: Column name for text data
            audio_column: Column name for audio filenames
            target_sample_rate: Target sample rate for audio
            max_samples: Maximum number of samples to load (to avoid OOM)
            
        Returns:
            Loaded dataset
        """
        # For large datasets, use streaming approach
        if max_samples > 5000:
            return self.load_csv_dataset_streaming(
                csv_path, audio_dir, text_column, audio_column, 
                target_sample_rate, max_samples
            )
        
        print(f"Loading CSV dataset from: {csv_path}")
        print(f"Audio directory: {audio_dir}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} entries in CSV")
        
        # Limit samples to avoid OOM
        if len(df) > max_samples:
            print(f"🔥 Limiting to {max_samples} samples to avoid OOM (from {len(df)} total)")
            df = df.head(max_samples)
        
        # Check vocabulary coverage before processing
        self.check_vocabulary_coverage(df[text_column].tolist())
        
        # Create dataset
        dataset_list = []
        failed_count = 0
        
        for idx, row in df.iterrows():
            try:
                audio_filename = row[audio_column]
                text = row[text_column]
                
                # Construct full audio path
                audio_path = os.path.join(audio_dir, audio_filename)
                
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file not found: {audio_path}")
                    failed_count += 1
                    continue
                
                # Load audio
                audio_data = self.load_audio_file(audio_path, target_sample_rate)
                
                # Create dataset entry
                entry = {
                    "text": text,
                    "audio": {
                        "array": audio_data,
                        "sampling_rate": target_sample_rate
                    }
                }
                
                dataset_list.append(entry)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                failed_count += 1
                continue
        
        print(f"Successfully loaded {len(dataset_list)} samples")
        if failed_count > 0:
            print(f"Failed to load {failed_count} samples")
        
        # Convert to Dataset with memory optimization
        print(f"Converting to Dataset format...")
        self.dataset = Dataset.from_list(dataset_list)
        self.ds_sample_rate = target_sample_rate
        
        # Clear intermediate data to save memory
        del dataset_list
        import gc
        gc.collect()
        
        return self.dataset
    
    def load_audio_file(self, audio_path: str, target_sample_rate: int = 24000) -> np.ndarray:
        """
        Load and resample audio file
        
        Args:
            audio_path: Path to audio file
            target_sample_rate: Target sample rate
            
        Returns:
            Audio data as numpy array
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return np.zeros(target_sample_rate, dtype=np.float32)
            
            # Load audio with librosa
            audio_data, sr = librosa.load(audio_path, sr=target_sample_rate, mono=True)
            
            # Ensure it's a numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # Ensure it's float32
            audio_data = audio_data.astype(np.float32)
            
            # Check if audio is too short or empty
            if len(audio_data) == 0:
                print(f"Empty audio file: {audio_path}")
                return np.zeros(target_sample_rate, dtype=np.float32)
            
            return audio_data
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silence if loading fails
            return np.zeros(target_sample_rate, dtype=np.float32)  # 1 second of silence
    
    def check_vocabulary_coverage(self, texts: List[str], sample_size: int = 100) -> Dict[str, Any]:
        """
        Check tokenizer vocabulary coverage for Vietnamese text
        
        Args:
            texts: List of text samples
            sample_size: Number of samples to analyze
            
        Returns:
            Dictionary with coverage statistics
        """
        print("Checking vocabulary coverage for Vietnamese text...")
        
        # Sample texts for analysis
        sample_texts = texts[:sample_size] if len(texts) > sample_size else texts
        
        total_tokens = 0
        unk_tokens = 0
        all_tokens = []
        
        for text in sample_texts:
            # Tokenize text
            tokens = self.tokenizer.tokenize(text)
            all_tokens.extend(tokens)
            total_tokens += len(tokens)
            
            # Count UNK tokens
            unk_count = sum(1 for token in tokens if token == self.tokenizer.unk_token)
            unk_tokens += unk_count
        
        # Calculate statistics
        coverage_rate = ((total_tokens - unk_tokens) / total_tokens) * 100 if total_tokens > 0 else 0
        avg_tokens_per_text = total_tokens / len(sample_texts) if sample_texts else 0
        
        # Most common tokens
        token_counts = Counter(all_tokens)
        most_common = token_counts.most_common(20)
        
        stats = {
            "total_texts_analyzed": len(sample_texts),
            "total_tokens": total_tokens,
            "unk_tokens": unk_tokens,
            "coverage_rate": coverage_rate,
            "avg_tokens_per_text": avg_tokens_per_text,
            "most_common_tokens": most_common
        }
        
        print(f"Vocabulary Coverage Analysis:")
        print(f"  Total texts analyzed: {stats['total_texts_analyzed']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  UNK tokens: {stats['unk_tokens']}")
        print(f"  Coverage rate: {stats['coverage_rate']:.2f}%")
        print(f"  Average tokens per text: {stats['avg_tokens_per_text']:.2f}")
        
        if coverage_rate < 95:
            print(f"⚠️  WARNING: Low vocabulary coverage ({coverage_rate:.2f}%)")
            print("Consider expanding vocabulary or using a different tokenizer for Vietnamese")
        else:
            print(f"✅ Good vocabulary coverage ({coverage_rate:.2f}%)")
        
        print(f"Most common tokens: {[token for token, count in most_common[:10]]}")
        
        return stats
    
    def initialize_snac_model(self):
        """Initialize the SNAC model for audio tokenization"""
        if self.snac_model is None:
            print("Loading SNAC model...")
            self.snac_model = load_snac_model(self.device)
            print("SNAC model loaded successfully")
    
    def add_codes(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add audio codes to a dataset example (supports both pre-loaded and streaming)
        
        Args:
            example: Dataset example
            
        Returns:
            Example with added codes_list
        """
        # Always initialize codes_list to None
        codes_list = None

        try:
            # Handle streaming dataset (audio_path) vs pre-loaded (audio array)
            if "audio_path" in example:
                # Streaming mode: load audio on-demand
                audio_path = example["audio_path"]
                sample_rate = example.get("sampling_rate", self.ds_sample_rate or 24000)
                
                # Load audio file
                audio_array = self.load_audio_file(audio_path, sample_rate)
                codes_list = tokenise_audio(audio_array, sample_rate, self.snac_model)
                
                # Trigger garbage collection if memory is low
                if self.should_trigger_gc():
                    gc.collect()
                    print(f"💾 Memory cleanup triggered. Available: {self.get_available_memory_gb():.1f}GB")
            
            else:
                # Legacy mode: pre-loaded audio
                answer_audio = example.get("audio")
                if answer_audio and "array" in answer_audio:
                    audio_array = answer_audio["array"]
                    sample_rate = self.ds_sample_rate or 24000
                    codes_list = tokenise_audio(audio_array, sample_rate, self.snac_model)
        
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            # Keep codes_list as None if we fail
        
        example["codes_list"] = codes_list
        return example
    
    def process_duplicate_frames(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove duplicate frames from audio codes
        
        Args:
            example: Dataset example with codes_list
            
        Returns:
            Example with processed codes_list
        """
        if example["codes_list"] is not None:
            try:
                processed_codes, removed_frames = remove_duplicate_frames(example["codes_list"])
                example["codes_list"] = processed_codes
            except Exception as e:
                print(f"Error processing duplicate frames: {e}")
                example["codes_list"] = None
        
        return example
    
    def create_input_ids(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create input IDs from text and audio codes
        
        Args:
            example: Dataset example
            
        Returns:
            Example with input_ids, labels, and attention_mask
        """
        # Determine whether to include the source field
        text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]

        text_ids = self.tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(TOKEN_CONFIG["end_of_text"])

        example["text_tokens"] = text_ids
        input_ids = (
            [TOKEN_CONFIG["start_of_human"]]
            + example["text_tokens"]
            + [TOKEN_CONFIG["end_of_human"]]
            + [TOKEN_CONFIG["start_of_ai"]]
            + [TOKEN_CONFIG["start_of_speech"]]
            + example["codes_list"]
            + [TOKEN_CONFIG["end_of_speech"]]
            + [TOKEN_CONFIG["end_of_ai"]]
        )
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)

        return example
    
    def filter_valid_examples(self, dataset: Dataset) -> Dataset:
        """
        Filter dataset to keep only valid examples with codes
        
        Args:
            dataset: Input dataset
            
        Returns:
            Filtered dataset
        """
        print("Filtering dataset for valid examples...")
        
        # Filter out examples with None codes_list
        filtered_dataset = dataset.filter(lambda x: x["codes_list"] is not None)
        print(f"After filtering None codes: {len(filtered_dataset)} examples")
        
        # Filter out examples with empty codes_list
        filtered_dataset = filtered_dataset.filter(lambda x: len(x["codes_list"]) > 0)
        print(f"After filtering empty codes: {len(filtered_dataset)} examples")
        
        return filtered_dataset
    
    def clean_dataset_columns(self, dataset: Dataset) -> Dataset:
        """
        Remove unnecessary columns from dataset
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with only required columns
        """
        columns_to_keep = DATASET_CONFIG["columns_to_keep"]
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        
        if columns_to_remove:
            print(f"Removing columns: {columns_to_remove}")
            dataset = dataset.remove_columns(columns_to_remove)
        
        return dataset
    
    def process_dataset(self, 
                       dataset_name: Optional[str] = None, 
                       split: Optional[str] = None) -> Dataset:
        """
        Complete dataset processing pipeline
        
        Args:
            dataset_name: Name of the dataset to load
            split: Dataset split to use
            
        Returns:
            Processed dataset ready for training
        """
        # Load dataset
        dataset = self.load_dataset(dataset_name, split)
        
        # Initialize SNAC model
        self.initialize_snac_model()
        
        # Process audio to codes
        print("Processing audio to codes...")
        dataset = dataset.map(self.add_codes, remove_columns=["audio"])
        
        # Filter valid examples
        dataset = self.filter_valid_examples(dataset)
        
        # Remove duplicate frames
        print("Removing duplicate frames...")
        dataset = dataset.map(self.process_duplicate_frames)
        
        # Filter again after duplicate frame removal
        dataset = self.filter_valid_examples(dataset)
        
        # Create input IDs
        print("Creating input IDs...")
        dataset = dataset.map(self.create_input_ids, remove_columns=["text", "codes_list"])
        
        # Clean up columns
        dataset = self.clean_dataset_columns(dataset)
        
        print(f"Final dataset size: {len(dataset)} examples")
        
        # Clean up SNAC model to save memory
        if self.snac_model is not None:
            self.snac_model.to("cpu")
            print("Moved SNAC model to CPU to save memory")
        
        return dataset
    
    def process_csv_dataset(self, 
                           csv_path: str, 
                           audio_dir: str,
                           text_column: str = "sentence",
                           audio_column: str = "uuid",
                           target_sample_rate: int = 24000,
                           max_samples: int = 1000) -> Dataset:
        """
        Complete CSV dataset processing pipeline
        
        Args:
            csv_path: Path to CSV metadata file
            audio_dir: Directory containing audio files
            text_column: Column name for text data
            audio_column: Column name for audio filenames
            target_sample_rate: Target sample rate for audio
            max_samples: Maximum number of samples to load (to avoid OOM)
            
        Returns:
            Processed dataset ready for training
        """
        # Load CSV dataset
        dataset = self.load_csv_dataset(csv_path, audio_dir, text_column, audio_column, target_sample_rate, max_samples)
        
        # Initialize SNAC model
        self.initialize_snac_model()
        
        # Process audio to codes
        print("Processing audio to codes...")
        print(f"💾 RAM usage before processing: {self.get_memory_usage_gb():.1f}GB")
        
        # For streaming datasets, don't remove audio_path - we need it for lazy loading
        if "audio_path" in dataset.column_names:
            # Streaming mode - process with memory monitoring
            dataset = dataset.map(
                self.add_codes, 
                remove_columns=["audio_path", "sampling_rate"],
                batched=False,  # Process one by one to manage memory
                desc="Processing audio to codes"
            )
        else:
            # Legacy mode with pre-loaded audio
            dataset = dataset.map(self.add_codes, remove_columns=["audio"])
        
        print(f"💾 RAM usage after processing: {self.get_memory_usage_gb():.1f}GB")
        
        # Force garbage collection
        gc.collect()
        
        # Filter valid examples
        dataset = self.filter_valid_examples(dataset)
        
        # Remove duplicate frames
        print("Removing duplicate frames...")
        dataset = dataset.map(self.process_duplicate_frames)
        
        # Filter again after duplicate frame removal
        dataset = self.filter_valid_examples(dataset)
        
        # Create input IDs
        print("Creating input IDs...")
        dataset = dataset.map(self.create_input_ids, remove_columns=["text", "codes_list"])
        
        # Clean up columns
        dataset = self.clean_dataset_columns(dataset)
        
        print(f"Final dataset size: {len(dataset)} examples")
        
        # Clean up SNAC model to save memory
        if self.snac_model is not None:
            self.snac_model.to("cpu")
            print("Moved SNAC model to CPU to save memory")
        
        return dataset


def print_tokenization_info():
    """Print information about tokenization setup"""
    tok_info = '''*** Dataset Tokenization Information ***
If you are training a multi-speaker model (e.g., canopylabs/orpheus-3b-0.1-ft),
ensure that the dataset includes a "source" field and format the input accordingly:
- Single-speaker: f"{example['text']}"
- Multi-speaker: f"{example['source']}: {example['text']}"
'''
    print(tok_info)


def create_data_processor(tokenizer, device: str = "cuda") -> OrpheusDataProcessor:
    """
    Factory function to create a data processor
    
    Args:
        tokenizer: Model tokenizer
        device: Device to use for processing
        
    Returns:
        OrpheusDataProcessor instance
    """
    return OrpheusDataProcessor(tokenizer, device)
