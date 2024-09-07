#  Copyright (c) 2024. kilitary@gmail.com

import time
import random

iteration = 0
temperature = 0.5
n_threads = 11
num_ctx = 8192
num_batch = 2
iid = time.monotonic_ns()
nbit = random.randrange(0, 64)
outer_engine_random_seed = int(time.time_ns() - int(time.time()) ^ nbit)
random.seed(outer_engine_random_seed)
internal_model_random_seed = int(outer_engine_random_seed ^ random.randrange(0, 64))
selected_model = 'mistral-nemo:latest'  # sola

src_options = {
    "num_keep": 5,
    "seed": internal_model_random_seed,
    "num_predict": -2,
    "top_k": 20,
    "top_p": 0.9,
    "min_p": 0.0,
    "tfs_z": 0.5,
    "typical_p": 0.7,
    "repeat_last_n": 33,
    "temperature": temperature,
    "repeat_penalty": 1.2,
    "presence_penalty": 1.5,
    "frequency_penalty": 1.0,
    "mirostat": 0,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": True,
    "stop": [
        '<|user|>',
        '<|assistant|>',
        "<start_of_turn>",
        '<|user|>',
        '<|im_start|>',
        '<|im_end|>',
        "<|start_header_id|>",
        '<|end_header_id|>',
        'RESPONSE:',
        '<|eot_id|>',
        '<|bot_id|>',
        '<|reserved_special_token',
        '[INST]',
        '[/INST]',
        '<<SYS>>',
        '<</SYS>>',
        "<|system|>",
        "<|endoftext|>",
        "<|end_of_turn|>",
        'ASSISTANT:',
        'USER:',
        'SYSTEM:',
        'PROMPT:',
        'assistant<|end_header_id|>',
        'user<|end_header_id|>',
        "</s>"
    ],
    "numa": False,
    "num_ctx": num_ctx,
    "num_batch": num_batch,
    "num_gpu": 0,
    "main_gpu": 0,
    "low_vram": False,
    "f16_kv": True,
    "vocab_only": False,
    "use_mmap": False,
    "use_mlock": False,
    "num_thread": n_threads
}
