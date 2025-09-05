# Algorithm Reference

This document explains the algorithms, techniques, and operational methods used throughout the Ollama Development Toolkit.

## LLM Sampling Algorithms

### Temperature-Based Sampling
The system uses sophisticated temperature control for response generation:

```python
# Core temperature settings from config.py
temperature = 0.55  # Balanced creativity vs consistency
```

**How it Works:**
- **Low Temperature (0.1-0.3)**: More deterministic, focused responses
- **Medium Temperature (0.4-0.7)**: Balanced creativity and coherence
- **High Temperature (0.8-1.0)**: More creative, diverse responses

### Top-K and Top-P Sampling
Advanced nucleus sampling for response quality:

```python
src_options = {
    "top_k": 20,        # Consider top 20 most likely tokens
    "top_p": 0.9,       # Nucleus sampling with 90% probability mass
    "min_p": 0.0,       # Minimum probability threshold
}
```

**Algorithm Details:**
1. **Top-K**: Limits selection to the K most probable tokens
2. **Top-P (Nucleus)**: Selects from the smallest set of tokens whose cumulative probability exceeds P
3. **Min-P**: Filters out tokens below minimum probability threshold

### Repetition Control Algorithms

#### Repetition Penalty System
```python
"repeat_last_n": 33,        # Look back 33 tokens for repetition
"repeat_penalty": 1.2,      # Penalty factor for repeated content
"presence_penalty": 1.5,    # Penalty for token presence
"frequency_penalty": 1.0,   # Penalty based on token frequency
```

**Implementation:**
- Tracks recent token usage in a sliding window
- Applies exponential penalty to repeated tokens
- Balances between avoiding repetition and maintaining coherence

#### Mirostat Algorithm
```python
"mirostat": 0,          # Disabled by default
"mirostat_tau": 0.8,    # Target entropy level
"mirostat_eta": 0.6,    # Learning rate for entropy adjustment
```

**Purpose:** Dynamic temperature adjustment based on response entropy

## Prompt Engineering Algorithms

### Dynamic Feature Combination
From `langfeatures.py`, the system uses categorized word features:

```python
features = {
    0: [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3.14, 10, 12],  # Numeric features
    1: ['sort', 'switch', 'encode', ...],                        # Action verbs
    2: ['name', 'order', 'film', ...],                          # Nouns/objects
    3: ['old', 'busy', 'homeless', ...],                        # Descriptors
    # ... additional categories
}
```

**Algorithm Steps:**
1. **Random Selection**: Pick elements from different feature categories
2. **Template Insertion**: Insert selected features into prompt templates
3. **Coherence Checking**: Ensure grammatical and logical consistency
4. **Dynamic Weighting**: Adjust feature selection based on response quality

### Prompt Template System
From `instructions.py`, uses template-based prompt generation:

**Base Templates:**
- System instructions for model behavior
- Dynamic parameter injection using `%1%`, `%2%`, `%3%` placeholders
- Context-aware prompt modification

## Text Analysis Algorithms

### Content Quality Assessment
```python
def assess_quality(response_text):
    """
    Multi-factor quality assessment algorithm
    """
    factors = {
        'length': calculate_optimal_length_score(response_text),
        'coherence': measure_semantic_coherence(response_text),
        'relevance': calculate_context_relevance(response_text),
        'uniqueness': measure_content_uniqueness(response_text)
    }
    return weighted_average(factors)
```

### Stop Sign Detection
Implements pattern matching for content filtering:

```python
stop_signs = ['milk', 'egg', 'food', 'tea ', 'cake', ...]

# Algorithm: O(n*m) pattern matching where n=text_length, m=patterns
def detect_stop_signs(text):
    clean_text = text.lower()
    for sign in stop_signs:
        if sign in clean_text:
            return True, sign
    return False, None
```

## Context Management Algorithms

### Context Window Management
```python
num_ctx = 8192  # Maximum context window size
num_batch = 2   # Batch size for processing
```

**Sliding Window Algorithm:**
1. **Context Accumulation**: Add new tokens to context buffer
2. **Overflow Detection**: Monitor context length vs. maximum
3. **Intelligent Truncation**: Remove oldest, least relevant content
4. **Context Compression**: Summarize removed content for reference

### Memory Preservation
```python
def preserve_important_context(context_tokens):
    """
    Preserves key information during context truncation
    """
    importance_scores = calculate_token_importance(context_tokens)
    preserved_tokens = select_high_importance_tokens(importance_scores)
    return compress_context(preserved_tokens)
```

## Audio Processing Algorithms (jam.py)

### Frequency Analysis
```python
def rail_freq_len(d=0, x=0, y=0, a=0, xx=0):
    """
    Multi-dimensional frequency analysis algorithm
    Processes audio signals using complex mathematical operations
    """
    # Complex frequency domain transformations
    # Used for audio feature extraction and analysis
```

### Volume Processing
```python
def rail_vol(a=0):
    """
    Volume processing algorithm with dynamic range adjustment
    """
    # Implements logarithmic volume scaling
    # Applies noise reduction and normalization
```

## Random Seed Generation

### Entropy-Based Seeding
```python
# Multi-source entropy combination
iid = time.monotonic_ns()
nbit = random.randrange(0, 64)
outer_engine_random_seed = int(time.time_ns() - int(time.time()) ^ nbit)
internal_model_random_seed = int(outer_engine_random_seed ^ random.randrange(0, 64))
```

**Algorithm Properties:**
- **Time-based entropy**: Uses high-resolution timestamps
- **XOR combination**: Combines multiple entropy sources
- **Cascading randomization**: Seeds influence subsequent random operations

## Simulation Framework (sim.py)

### Experiment Management
The `Simulatar` class implements:

1. **Experiment Logging**: Structured logging of experimental results
2. **Context Management**: Persistent context across simulation runs
3. **Parameter Tracking**: Automatic tracking of experimental parameters
4. **Result Analysis**: Statistical analysis of simulation outcomes

### Batch Processing
```python
def execute(self):
    """
    Batch execution algorithm for multiple experimental runs
    """
    # Implements parallel processing for efficiency
    # Includes error handling and recovery mechanisms
    # Provides progress tracking and intermediate results
```

## Performance Optimization Algorithms

### Thread Management
```python
n_threads = 11  # Optimal thread count for CPU utilization
```

### Memory Management
```python
"use_mmap": True,    # Memory mapping for large models
"use_mlock": False,  # Memory locking disabled
"low_vram": False,   # VRAM optimization setting
```

### GPU Utilization
```python
"num_gpu": 0,     # GPU count for processing
"main_gpu": 0,    # Primary GPU selection
"f16_kv": True,   # Half-precision for key-value cache
```

## Security and Safety Algorithms

### Content Filtering Pipeline
1. **Pre-processing Filter**: Input sanitization
2. **Real-time Monitoring**: Response content analysis
3. **Post-processing Check**: Final content validation
4. **Ethical Compliance**: Automated compliance checking

### Keyword Detection System
Uses efficient string matching algorithms for detecting potentially harmful content patterns.

## Usage Recommendations

### Algorithm Selection Guidelines
- **Creative Tasks**: Higher temperature (0.7-0.9), lower top-k (10-20)
- **Analytical Tasks**: Lower temperature (0.3-0.5), higher top-k (40-50)
- **Conversation**: Moderate settings with repetition penalties enabled
- **Code Generation**: Low temperature, high top-p for accuracy

### Performance Tuning
- Adjust batch size based on available memory
- Optimize thread count for your specific hardware
- Use appropriate precision settings for your GPU
- Monitor context usage to prevent overflow