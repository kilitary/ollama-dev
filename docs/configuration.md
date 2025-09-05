# Configuration Guide

This guide explains all configuration options available in the Ollama Development Toolkit.

## Core Configuration (config.py)

### Model Selection
```python
selected_model = 'mistral-nemo:latest'  # Primary model for analysis
```

**Available Models:**
- `mistral-nemo:latest` - Recommended for general tasks
- `llama2:latest` - Alternative model option
- Custom models via Ollama model registry

### Sampling Parameters

#### Temperature Control
```python
temperature = 0.55  # Controls randomness in generation
```
- **Range**: 0.0 - 2.0
- **Recommended**: 0.3-0.7 for most tasks
- **Effect**: Higher values = more creative, lower = more focused

#### Context Management
```python
num_ctx = 8192      # Maximum context window size
num_batch = 2       # Batch processing size
```
- **Context Window**: Available memory for conversation history
- **Batch Size**: Number of tokens processed simultaneously

#### Advanced Sampling
```python
src_options = {
    "num_keep": 5,              # Tokens to keep from context
    "seed": internal_model_random_seed,  # Reproducibility seed
    "num_predict": -2,          # Maximum tokens to generate (-1 = unlimited)
    "top_k": 20,               # Top-K sampling parameter
    "top_p": 0.9,              # Nucleus sampling threshold
    "min_p": 0.0,              # Minimum probability threshold
    "tfs_z": 0.5,              # Tail-free sampling parameter
    "typical_p": 0.7,          # Typical sampling parameter
    "repeat_last_n": 33,       # Repetition penalty window
    "temperature": temperature,
    "repeat_penalty": 1.2,     # Repetition penalty strength
    "presence_penalty": 1.5,   # Presence penalty for diversity
    "frequency_penalty": 1.0,  # Frequency-based penalty
    "mirostat": 0,             # Mirostat sampling (0=disabled)
    "mirostat_tau": 0.8,       # Mirostat target entropy
    "mirostat_eta": 0.6,       # Mirostat learning rate
    "penalize_newline": True,  # Penalize line breaks
}
```

### Stop Sequences
```python
"stop": [
    '<|user|>', '<|assistant|>', "<start_of_turn>",
    '<|im_start|>', '<|im_end|>', "<|start_header_id|>",
    '<|end_header_id|>', 'RESPONSE:', '<|eot_id|>',
    # ... additional stop sequences
]
```
**Purpose**: Define where the model should stop generating

### Hardware Configuration
```python
"numa": False,          # NUMA optimization
"num_gpu": 0,          # Number of GPUs to use
"main_gpu": 0,         # Primary GPU ID
"low_vram": False,     # Low VRAM mode
"f16_kv": True,        # Half-precision key-value cache
"vocab_only": False,   # Vocabulary-only mode
"use_mmap": True,      # Memory mapping
"use_mlock": False,    # Memory locking
"num_thread": n_threads # CPU thread count
```

## Analysis Configuration (analyze.py)

### Logging Settings
```python
console = console.Console(
    force_terminal=True,    # Force terminal output
    no_color=False,        # Enable color output
    highlight=False,       # Disable syntax highlighting
    force_interactive=False, # Interactive mode
    color_system='auto'    # Automatic color system detection
)
```

### Content Filtering
```python
# Stop signs that trigger response reset
stop_signs = [
    'milk', 'egg', 'food', 'tea ', 'cake',
    'oil', 'cream', 'banan', 'yogurt', 'bread'
]

# Keywords for special handling
keywords = [
    'fruit', 'you have any other',
    'potentially harmful',
    'violates ethical', 'as a responsible ai',
    'unethical and potentially illegal'
]
```

**Customization**: Modify these arrays to change filtering behavior

### Display Options
```python
colors = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
nypd_mode = False  # Enable random color cycling
```

## Simulation Configuration (sim.py)

### Simulatar Class Settings
```python
sim = Simulatar(
    name="experiment_name",           # Experiment identifier
    rules=[],                        # Behavioral rules list
    instructions=[],                 # Instruction templates
    sim_log_path="/path/to/logs",    # Log file location
    temperature=0.7,                 # Generation temperature
    batch=True,                      # Batch processing enabled
    model="mistral-nemo:latest",     # Model selection
    template="custom_template"       # Custom prompt template
)
```

### Console Configuration
```python
console = console.Console(
    force_terminal=False,   # Terminal forcing
    no_color=False,        # Color output
    force_interactive=True, # Interactive mode
    color_system='windows'  # Color system (windows/auto/256/truecolor)
)
```

## Language Features (langfeatures.py)

### Feature Categories
```python
features = {
    0: [-2, -1, 0, 1, 2, 3.14, 10, 12],  # Numeric values
    1: ['sort', 'encode', 'handle', ...], # Action verbs
    2: ['name', 'system', 'device', ...], # Nouns/objects  
    3: ['old', 'fast', 'clean', ...],     # Descriptors
    4: ['do', 'can', 'should', ...],      # Modal verbs
    5: ['your', 'my', 'their', ...],      # Possessives
    6: ['me', 'you', 'we', ...],          # Pronouns
    7: ['as', 'by', 'per', ...],          # Prepositions
    8: ['inside', 'over', 'through', ...] # Spatial relations
}
```

**Customization**: Add or modify categories to change text generation patterns

## Environment Variables

### Optional Environment Settings
```bash
# Redis configuration (if using Redis features)
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=your_password

# Ollama server configuration
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_API_KEY=your_api_key

# Logging configuration
export LOG_LEVEL=INFO
export LOG_FILE=/path/to/logfile.log
```

## Command Line Arguments

### analyze.py Arguments
```bash
python analyze.py [options]
  --model MODEL_NAME     # Override default model
  --temperature TEMP     # Override temperature setting
  --max-tokens N         # Maximum tokens to generate
  --context-size N       # Context window size
  --batch-size N         # Batch processing size
  --verbose             # Enable verbose logging
  --no-color            # Disable colored output
```

### sim.py Arguments
```bash
python sim.py [options]
  --name EXPERIMENT     # Experiment name
  --rules RULES_FILE    # Rules configuration file
  --temperature TEMP    # Generation temperature
  --iterations N        # Number of iterations
  --log-path PATH       # Log file location
```

## Performance Tuning

### Memory Optimization
```python
# For low-memory systems
src_options.update({
    "num_ctx": 2048,        # Reduce context window
    "low_vram": True,       # Enable low VRAM mode
    "use_mlock": False,     # Disable memory locking
    "f16_kv": True         # Use half-precision
})
```

### High-Performance Settings
```python
# For high-end systems
src_options.update({
    "num_ctx": 32768,       # Large context window
    "num_batch": 8,         # Larger batch size
    "num_gpu": 2,          # Multiple GPUs
    "num_thread": 16       # More CPU threads
})
```

### CPU vs GPU Configuration
```python
# CPU-only configuration
cpu_config = {
    "num_gpu": 0,
    "num_thread": os.cpu_count(),
    "use_mmap": True
}

# GPU-accelerated configuration  
gpu_config = {
    "num_gpu": 1,
    "main_gpu": 0,
    "low_vram": False,
    "f16_kv": True
}
```

## Troubleshooting Common Configuration Issues

### Model Loading Problems
- Verify model exists: `ollama list`
- Check model spelling and version
- Ensure sufficient disk space and memory

### Performance Issues
- Reduce `num_ctx` for memory constraints
- Adjust `num_thread` based on CPU cores
- Enable `low_vram` for GPU memory issues

### Connection Problems
- Check Ollama server status: `ollama serve`
- Verify server address and port
- Check firewall and network settings

### Memory Errors
- Reduce batch size and context window
- Enable memory mapping optimizations
- Consider using quantized models

## Configuration Examples

### Development Setup
```python
# config_dev.py
temperature = 0.7
num_ctx = 4096
debug_mode = True
verbose_logging = True
```

### Production Setup
```python
# config_prod.py
temperature = 0.5
num_ctx = 8192
debug_mode = False
log_to_file = True
performance_monitoring = True
```

### Research Setup
```python
# config_research.py
temperature = 0.8
num_ctx = 16384
experimental_features = True
detailed_metrics = True
```