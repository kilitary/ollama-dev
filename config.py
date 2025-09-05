#  Copyright (c) 2024. kilitary@gmail.com

"""
Model Configuration and Parameter Settings

This module defines all configuration parameters for Ollama model interaction,
including sampling parameters, context management, and hardware optimization
settings. The configuration is optimized for the mistral-nemo model but can
be adapted for other LLMs.

Key Configuration Areas:
- Sampling parameters (temperature, top-k, top-p)
- Context and memory management  
- Repetition control and penalties
- Hardware optimization (CPU/GPU settings)
- Stop sequences and token filtering
"""

import time
import random

iteration = 0
temperature = 0.55  # Balanced creativity vs consistency (0.0-2.0)
n_threads = 11      # CPU threads for model processing
num_ctx = 8192      # Context window size (tokens)
num_batch = 2       # Batch processing size for efficiency
# Random seed generation using multiple entropy sources
iid = time.monotonic_ns()  # High-resolution monotonic timestamp
nbit = random.randrange(0, 64)  # Random bit offset for XOR operations
outer_engine_random_seed = int(time.time_ns() - int(time.time()) ^ nbit)
random.seed(outer_engine_random_seed)
internal_model_random_seed = int(outer_engine_random_seed ^ random.randrange(0, 64))
selected_model = 'mistral-nemo:latest'  # Default model for analysis

src_options = {
    # Context and prediction settings
    "num_keep": 5,                    # Tokens to preserve from context
    "seed": internal_model_random_seed, # Reproducibility seed
    "num_predict": -2,                # Max tokens to generate (-1=unlimited, -2=model default)
    
    # Sampling algorithm parameters
    "top_k": 20,                      # Top-K sampling: consider top N tokens
    "top_p": 0.9,                     # Nucleus sampling: probability mass threshold
    "min_p": 0.0,                     # Minimum probability threshold for tokens
    "tfs_z": 0.5,                     # Tail-free sampling parameter
    "typical_p": 0.7,                 # Typical sampling parameter
    
    # Repetition control mechanisms
    "repeat_last_n": 33,              # Window size for repetition detection
    "temperature": temperature,        # Randomness in generation (higher=more creative)
    "repeat_penalty": 1.2,            # Penalty for repeating recent tokens
    "presence_penalty": 1.5,          # Penalty for token presence (promotes diversity)
    "frequency_penalty": 1.0,         # Penalty based on token frequency
    
    # Mirostat dynamic temperature adjustment
    "mirostat": 0,                    # Mirostat algorithm (0=disabled, 1/2=enabled)
    "mirostat_tau": 0.8,              # Target entropy level for mirostat
    "mirostat_eta": 0.6,              # Learning rate for mirostat adjustments
    "penalize_newline": True,         # Apply penalties to newline tokens
    # Stop sequences - tokens that indicate end of generation
    "stop": [
        # User/Assistant markers
        '<|user|>', '<|assistant|>', "<start_of_turn>", "<|end_of_turn|>",
        '<|im_start|>', '<|im_end|>', 
        "<|start_header_id|>", '<|end_header_id|>',
        
        # Response indicators  
        'RESPONSE:', 'ASSISTANT:', 'USER:', 'SYSTEM:', 'PROMPT:',
        'assistant<|end_header_id|>', 'user<|end_header_id|>',
        
        # Model-specific tokens
        '<|eot_id|>', '<|bot_id|>', '</s>',
        '<|reserved_special_token', 
        
        # Instruction formatting
        '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>',
        "<|system|>"
    ],
    
    # Hardware optimization settings
    "numa": False,                    # NUMA optimization (usually False for single node)
    "num_ctx": num_ctx,              # Context window size in tokens  
    "num_batch": num_batch,          # Batch size for parallel processing
    "num_gpu": 0,                    # Number of GPUs to use (0=CPU only)
    "main_gpu": 0,                   # Primary GPU ID for multi-GPU setups
    "low_vram": False,               # Low VRAM mode for memory-constrained GPUs
    
    # Memory management
    "f16_kv": True,                  # Use half-precision for key-value cache
    "vocab_only": False,             # Load vocabulary only (faster loading)
    "use_mmap": True,                # Memory mapping for model files
    "use_mlock": False,              # Lock model in memory (prevents swapping)
    "num_thread": n_threads          # CPU thread count for processing
}
