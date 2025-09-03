# -*- coding: utf-8 -*-
"""
Utility functions for audio processing and token manipulation
"""

import torch
import torchaudio.transforms as T
from snac import SNAC
from typing import List, Tuple, Optional
import locale
import numpy as np

from config import AUDIO_CONFIG, TOKEN_CONFIG, AUDIO_OFFSETS

# Ensure UTF-8 encoding
locale.getpreferredencoding = lambda: "UTF-8"


def load_snac_model(device: str = "cuda") -> SNAC:
    """
    Load the SNAC model for audio tokenization
    
    Args:
        device: Device to load the model on
        
    Returns:
        SNAC model instance
    """
    snac_model = SNAC.from_pretrained(AUDIO_CONFIG["snac_model_name"])
    return snac_model.to(device)


def tokenise_audio(waveform, ds_sample_rate: int, snac_model: SNAC) -> List[int]:
    """
    Tokenize audio waveform using SNAC model
    
    Args:
        waveform: Audio waveform as numpy array or torch tensor
        ds_sample_rate: Original sample rate of the dataset
        snac_model: SNAC model instance
        
    Returns:
        List of audio tokens
    """
    # Convert to torch tensor if it's numpy array
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    elif not isinstance(waveform, torch.Tensor):
        # Handle other types (list, etc.)
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    # Ensure it's float32
    waveform = waveform.to(dtype=torch.float32)
    
    # Add batch dimension if needed
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Resample to target sample rate
    resample_transform = T.Resample(
        orig_freq=ds_sample_rate, 
        new_freq=AUDIO_CONFIG["target_sample_rate"]
    )
    waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0)
    
    # Move to same device as SNAC model
    try:
        # Try to get device from model parameters
        device = next(snac_model.parameters()).device
        waveform = waveform.to(device)
    except:
        # Fallback to CUDA if available
        if torch.cuda.is_available():
            waveform = waveform.to("cuda")
        else:
            waveform = waveform.to("cpu")

    # Generate the codes from SNAC
    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    all_codes = []
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + TOKEN_CONFIG["audio_offset"])
        all_codes.append(codes[1][0][2*i].item() + TOKEN_CONFIG["audio_offset"] + AUDIO_OFFSETS["layer_2_first"])
        all_codes.append(codes[2][0][4*i].item() + TOKEN_CONFIG["audio_offset"] + AUDIO_OFFSETS["layer_3_first"])
        all_codes.append(codes[2][0][(4*i)+1].item() + TOKEN_CONFIG["audio_offset"] + AUDIO_OFFSETS["layer_3_second"])
        all_codes.append(codes[1][0][(2*i)+1].item() + TOKEN_CONFIG["audio_offset"] + AUDIO_OFFSETS["layer_2_second"])
        all_codes.append(codes[2][0][(4*i)+2].item() + TOKEN_CONFIG["audio_offset"] + AUDIO_OFFSETS["layer_3_third"])
        all_codes.append(codes[2][0][(4*i)+3].item() + TOKEN_CONFIG["audio_offset"] + AUDIO_OFFSETS["layer_3_fourth"])

    return all_codes


def remove_duplicate_frames(codes_list: List[int]) -> Tuple[List[int], int]:
    """
    Remove duplicate frames from audio codes
    
    Args:
        codes_list: List of audio codes
        
    Returns:
        Tuple of (filtered codes, number of removed frames)
    """
    if len(codes_list) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = codes_list[:7]
    removed_frames = 0

    for i in range(7, len(codes_list), 7):
        current_first = codes_list[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(codes_list[i:i+7])
        else:
            removed_frames += 1

    return result, removed_frames


def redistribute_codes(code_list: List[int], snac_model: SNAC) -> torch.Tensor:
    """
    Redistribute codes back to audio format and decode
    
    Args:
        code_list: List of processed audio codes
        snac_model: SNAC model for decoding
        
    Returns:
        Decoded audio tensor
    """
    layer_1 = []
    layer_2 = []
    layer_3 = []
    
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1] - AUDIO_OFFSETS["layer_2_first"])
        layer_3.append(code_list[7*i+2] - AUDIO_OFFSETS["layer_3_first"])
        layer_3.append(code_list[7*i+3] - AUDIO_OFFSETS["layer_3_second"])
        layer_2.append(code_list[7*i+4] - AUDIO_OFFSETS["layer_2_second"])
        layer_3.append(code_list[7*i+5] - AUDIO_OFFSETS["layer_3_third"])
        layer_3.append(code_list[7*i+6] - AUDIO_OFFSETS["layer_3_fourth"])
    
    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0)
    ]

    # Move codes to same device as SNAC model
    try:
        device = next(snac_model.parameters()).device
        codes = [c.to(device) for c in codes]
    except:
        # Fallback to CPU
        codes = [c.to("cpu") for c in codes]

    audio_hat = snac_model.decode(codes)
    return audio_hat


def process_generated_tokens(generated_ids: torch.Tensor) -> List[List[int]]:
    """
    Process generated tokens to extract audio codes
    
    Args:
        generated_ids: Generated token IDs from the model
        
    Returns:
        List of processed code lists
    """
    token_to_find = TOKEN_CONFIG["start_of_speech"]
    token_to_remove = TOKEN_CONFIG["end_of_speech"]

    # Find the last occurrence of start_of_speech token
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
    else:
        cropped_tensor = generated_ids

    # Remove end_of_speech tokens
    processed_rows = []
    for row in cropped_tensor:
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    # Process codes
    code_lists = []
    for row in processed_rows:
        row_length = row.size(0)
        new_length = (row_length // 7) * 7
        trimmed_row = row[:new_length]
        trimmed_row = [t.item() - TOKEN_CONFIG["audio_offset"] for t in trimmed_row]
        code_lists.append(trimmed_row)

    return code_lists


def prepare_inference_input(prompts: List[str], tokenizer, chosen_voice: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare input for inference
    
    Args:
        prompts: List of text prompts
        tokenizer: Model tokenizer
        chosen_voice: Voice name for multi-speaker models
        
    Returns:
        Tuple of (input_ids, attention_mask)
    """
    # Format prompts with voice if specified
    prompts_ = [(f"{chosen_voice}: " + p) if chosen_voice else p for p in prompts]

    all_input_ids = []
    for prompt in prompts_:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        all_input_ids.append(input_ids)

    # Add special tokens
    start_token = torch.tensor([[TOKEN_CONFIG["start_of_human"]]], dtype=torch.int64)
    end_tokens = torch.tensor([[TOKEN_CONFIG["end_of_text"], TOKEN_CONFIG["end_of_human"]]], dtype=torch.int64)

    all_modified_input_ids = []
    for input_ids in all_input_ids:
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
        all_modified_input_ids.append(modified_input_ids)

    # Pad sequences
    max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])
    
    all_padded_tensors = []
    all_attention_masks = []
    
    for modified_input_ids in all_modified_input_ids:
        padding = max_length - modified_input_ids.shape[1]
        padded_tensor = torch.cat([
            torch.full((1, padding), TOKEN_CONFIG["pad_token"], dtype=torch.int64), 
            modified_input_ids
        ], dim=1)
        
        attention_mask = torch.cat([
            torch.zeros((1, padding), dtype=torch.int64), 
            torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)
        ], dim=1)
        
        all_padded_tensors.append(padded_tensor)
        all_attention_masks.append(attention_mask)

    all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)

    return all_padded_tensors, all_attention_masks


def get_gpu_memory_stats() -> dict:
    """
    Get current GPU memory statistics
    
    Returns:
        Dictionary with memory statistics
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    gpu_stats = torch.cuda.get_device_properties(0)
    current_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    return {
        "gpu_name": gpu_stats.name,
        "current_memory_gb": current_memory,
        "max_memory_gb": max_memory,
        "memory_usage_percent": round(current_memory / max_memory * 100, 3)
    }


def print_memory_stats(stage: str = ""):
    """
    Print GPU memory statistics
    
    Args:
        stage: Stage description for logging
    """
    stats = get_gpu_memory_stats()
    if "error" in stats:
        print(f"[{stage}] {stats['error']}")
        return
        
    print(f"[{stage}] GPU = {stats['gpu_name']}. Max memory = {stats['max_memory_gb']} GB.")
    print(f"[{stage}] {stats['current_memory_gb']} GB of memory reserved ({stats['memory_usage_percent']}%).")
