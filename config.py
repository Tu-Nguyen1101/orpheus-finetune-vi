# -*- coding: utf-8 -*-
"""
Configuration file for Orpheus TTS model training
Optimized for full dataset training with 5 epochs
"""

# Base Model Configuration
MODEL_CONFIG = {
    "model_name": "unsloth/orpheus-3b-0.1-ft",
    "max_seq_length": 4096,
    "dtype": None,
    "load_in_4bit": False,
    "full_finetuning": True
}

# LoRA Configuration  
LORA_CONFIG = {
    "r": 128,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head"
    ],
    "lora_alpha": 128,
    "lora_dropout": 0,
    "bias": "none",
    "use_gradient_checkpointing": "unsloth",
    "random_state": 42,
    "use_rslora": True,
    "loftq_config": None,
}

# Full Dataset Training Configuration (5 epochs)
TRAINING_CONFIG = {
    "per_device_train_batch_size": 32,
    "gradient_accumulation_steps": 1,  # Effective batch size = 32
    "num_train_epochs": 2,  # Train for 5 complete epochs
    "warmup_steps": 200,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "logging_steps": 50,
    "save_steps": 500,
    "optim": "adamw_8bit",
    "bf16": True,
    "tf32": True,
    "dataloader_num_workers": 8,
    "seed": 42,
    "output_dir": "model_training/full_dataset_model",
    "report_to": "none",
    "save_total_limit": 5,              # Giữ 5 checkpoint gần nhất
    "load_best_model_at_end": False,    # Không cần eval dataset riêng  
    "eval_strategy": "no",        # Không eval (chỉ dùng training loss)
    "dataloader_pin_memory": True,
    "remove_unused_columns": False,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
}

# Dataset Configuration
DATASET_CONFIG = {
    "columns_to_keep": ["input_ids", "labels", "attention_mask"],
}

# Audio Processing Configuration
AUDIO_CONFIG = {
    "snac_model_name": "hubertsiuzdak/snac_24khz",
    "target_sample_rate": 24000,
}

# Token Configuration
TOKEN_CONFIG = {
    "tokeniser_length": 128256,
    "start_of_text": 128000,
    "end_of_text": 128009,
    "start_of_speech": 128257,
    "end_of_speech": 128258,
    "start_of_human": 128259,
    "end_of_human": 128260,
    "start_of_ai": 128261,
    "end_of_ai": 128262,
    "pad_token": 128263,
    "audio_offset": 128266,
}

# Audio Token Offsets  
AUDIO_OFFSETS = {
    "layer_1": 0,
    "layer_2_first": 4096,
    "layer_3_first": 2 * 4096,
    "layer_3_second": 3 * 4096,
    "layer_2_second": 4 * 4096,
    "layer_3_third": 5 * 4096,
    "layer_3_fourth": 6 * 4096,
}

# Inference Configuration
INFERENCE_CONFIG = {
    "max_new_tokens": 1200,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "use_cache": False,  # Disabled for stability
}

# Inference presets. CLI values override these presets.
INFERENCE_PRESETS = {
    # More stable output for short Vietnamese prompts and low-VRAM GPUs.
    "stable": {
        "max_new_tokens": 900,
        "temperature": 0.45,
        "top_p": 0.85,
        "repetition_penalty": 1.08,
    },
    # Good first choice for normal testing.
    "balanced": {
        "max_new_tokens": 1000,
        "temperature": 0.55,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    },
    # More variation, but may produce less consistent pronunciation.
    "expressive": {
        "max_new_tokens": 1200,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.02,
    },
}

# Vietnamese prompts for testing
VIETNAMESE_PROMPTS = [
    "Xin chào, tôi là Orpheus. Tôi có thể nói tiếng Việt rất tự nhiên.",
    "Hôm nay là một ngày đẹp trời ở Việt Nam.",
    "Cảm ơn bạn đã sử dụng mô hình Orpheus TTS để tạo giọng nói tiếng Việt.",
    "Đây là bài test để kiểm tra chất lượng giọng nói tiếng Việt.",
]

# Default English prompts
DEFAULT_PROMPTS = [
    "Hey there my name is Elise, <giggles> and I'm a speech generation model that can sound like a person.",
]

# Model save/load paths
MODEL_PATHS = {
    "lora_output": "lora_model",
    "merged_16bit": "model_16bit",
    "merged_4bit": "model_4bit",
}
