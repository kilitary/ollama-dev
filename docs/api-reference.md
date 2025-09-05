# API Reference

This document provides detailed API documentation for the main functions and classes in the Ollama Development Toolkit.

## Core Functions (analyze.py)

### `update_model(model_name=None)`
Download and verify availability of an Ollama model.

**Parameters:**
- `model_name` (str, optional): Name of the Ollama model to verify/download

**Returns:**
- None

**Raises:**
- `Exception`: If model download fails or model is not available

**Example:**
```python
update_model('mistral-nemo:latest')  # Download and verify model
```

### `slog(msg="", end="\n", justify="full", style=None)`
Structured logging function with Rich formatting support.

**Parameters:**
- `msg` (str): Message to log with Rich markup support
- `end` (str): String appended after message (default: "\n")
- `justify` (str): Text justification ('left', 'center', 'right', 'full')
- `style` (str, optional): Rich style specification

**Returns:**
- None

**Examples:**
```python
slog("Analysis complete", style="green")
slog("[red]Error:[/red] Model not found", justify="center")  
slog("Processing...", end="", style="yellow")
```

## Simulation Framework (sim.py)

### Class `Simulatar`
LLM Simulation Framework for Controlled Experiments.

**Constructor Parameters:**
- `name` (str): Unique identifier for the experiment
- `rules` (list): Behavioral rules governing the simulation
- `instructions` (list): Prompt templates and system instructions
- `sim_log_path` (str): File path for experiment logs
- `temperature` (float): Model temperature for generation
- `batch` (bool, default=True): Enable batch processing
- `model` (str, optional): Ollama model name to use
- `template` (str, optional): Custom prompt template

**Example:**
```python
sim = Simulatar(
    name="experiment_1",
    rules=[],
    instructions=["Analyze the following data..."],
    sim_log_path="/logs",
    temperature=0.7
)
```

#### `log(msg="", end="\n", flush=True, justify=None)`
Enhanced logging method with file persistence and console output.

**Parameters:**
- `msg` (str): Message to log (supports Rich markup)
- `end` (str): String appended after message
- `flush` (bool): Force flush output streams
- `justify` (str): Text justification for console output

**Features:**
- Dual-output logging (console + file)
- Automatic Rich markup stripping for files
- Experiment-specific log files

#### `write_context(context)`
Persist conversation context to Redis storage.

**Parameters:**
- `context` (list): Context tokens/data to store

**Storage:**
- Uses Redis list structure
- Key: 'sim.context.ids'
- Supports distributed context sharing

#### `read_context()`
Retrieve stored conversation context from Redis.

**Returns:**
- list: Retrieved context data

#### `delete_context()`
Clear stored context from Redis storage.

**Returns:**
- bool: Success status

#### `execute()`
Execute the simulation experiment with configured parameters.

**Returns:**
- dict: Experiment results and metrics

## Configuration (config.py)

### Global Variables

#### Model Configuration
```python
selected_model = 'mistral-nemo:latest'  # Default model
temperature = 0.55                      # Generation temperature
num_ctx = 8192                         # Context window size
num_batch = 2                          # Batch processing size
n_threads = 11                         # CPU thread count
```

#### Sampling Options (`src_options`)
Dictionary containing all model sampling parameters:

```python
src_options = {
    # Core parameters
    "temperature": float,        # Randomness (0.0-2.0)
    "top_k": int,               # Top-K sampling
    "top_p": float,             # Nucleus sampling
    "num_ctx": int,             # Context window size
    
    # Repetition control
    "repeat_penalty": float,     # Repetition penalty
    "presence_penalty": float,   # Presence penalty
    "frequency_penalty": float,  # Frequency penalty
    
    # Hardware settings
    "num_gpu": int,             # GPU count
    "num_thread": int,          # CPU threads
    "use_mmap": bool,           # Memory mapping
    # ... additional options
}
```

## Language Features (langfeatures.py)

### `features` Dictionary
Categorized language features for dynamic text generation.

**Structure:**
```python
features = {
    0: list,  # Numeric values
    1: list,  # Action verbs  
    2: list,  # Nouns/objects
    3: list,  # Descriptive adjectives
    4: list,  # Modal verbs
    5: list,  # Possessive pronouns
    6: list,  # Personal pronouns
    7: list,  # Prepositions
    8: list   # Spatial relations
}
```

**Usage:**
```python
from langfeatures import features

# Get random action verb
import random
action = random.choice(features[1])

# Get numeric value
value = random.choice(features[0])
```

## Utility Functions (jam.py)

### `rail_vol(a=0)`
Volume processing algorithm with dynamic range adjustment.

**Parameters:**
- `a` (float): Audio amplitude value

**Returns:**
- float: Processed volume value

### `rail_freq_len(d=0, x=0, y=0, a=0, xx=0)`
Multi-dimensional frequency analysis algorithm.

**Parameters:**
- `d`, `x`, `y`, `a`, `xx` (float): Frequency domain parameters

**Returns:**
- Complex frequency analysis result

### `rails_run(ai=0)`
Execute comprehensive audio processing pipeline.

**Parameters:**
- `ai` (int): Audio input identifier

**Returns:**
- dict: Processing results and metrics

## HTTP Testing (httptest.py)

### `main()`
Main HTTP testing function for endpoint validation.

**Features:**
- Automated endpoint testing
- Response validation
- Performance metrics collection

**Usage:**
```bash
python httptest.py
```

## Error Handling

### Common Exceptions
- `ConnectionError`: Ollama server connection issues
- `ModelNotFoundError`: Requested model not available
- `ContextOverflowError`: Context window size exceeded
- `RedisConnectionError`: Redis storage connection failed

### Exception Handling Patterns
```python
try:
    response = client.chat(model=model_name, messages=messages)
except ConnectionError:
    slog("Failed to connect to Ollama server", style="red")
except Exception as e:
    slog(f"Unexpected error: {e}", style="red")
```

## Integration Examples

### Basic Analysis Workflow
```python
from analyze import slog, update_model
from config import selected_model, src_options

# Setup
update_model('mistral-nemo:latest')
slog("Starting analysis", style="green")

# Process data
result = analyze_text(input_data, model=selected_model, options=src_options)
slog(f"Analysis complete: {result}", style="cyan")
```

### Simulation Experiment
```python
from sim import Simulatar

# Create experiment
sim = Simulatar(
    name="test_experiment",
    rules=[],
    instructions=["Analyze the following..."],
    sim_log_path="/experiments",
    temperature=0.7
)

# Run experiment
results = sim.execute()
sim.log(f"Results: {results}")
```

### Custom Configuration
```python
from config import src_options

# Modify for creative tasks
creative_options = src_options.copy()
creative_options.update({
    "temperature": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1
})

# Use custom options
response = client.generate(model=model, options=creative_options)
```

## Performance Considerations

### Memory Management
- Monitor context window usage with `num_ctx`
- Use `low_vram: True` for GPU memory constraints
- Enable `use_mmap: True` for large models

### CPU Optimization
- Set `num_thread` to match CPU core count
- Adjust `num_batch` based on available RAM
- Use `numa: True` for multi-socket systems

### GPU Acceleration
- Set `num_gpu` to available GPU count
- Configure `main_gpu` for primary device
- Use `f16_kv: True` for memory efficiency

## Best Practices

### Model Selection
- Use `mistral-nemo:latest` for general tasks
- Consider model size vs. performance trade-offs
- Test model availability before production use

### Parameter Tuning
- Start with default parameters
- Adjust temperature based on task creativity needs
- Monitor repetition penalties for coherence

### Error Recovery
- Implement proper exception handling
- Use fallback models for reliability
- Monitor context window overflow

### Logging and Monitoring
- Use structured logging with `slog()`
- Monitor performance metrics
- Implement experiment tracking for research