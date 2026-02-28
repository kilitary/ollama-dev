# AGENTS.md - AI Coding Agent Guide

This guide helps AI agents understand the ollama-dev codebase structure, patterns, and workflows.

## Project Overview

**ollama-dev** is a multi-faceted LLM research toolkit with three main subsystems:
1. **LLM Analysis Engine** (`analyze.py`, `config.py`, `instructions.py`) - Ollama model interaction and response analysis
2. **Audio Processing System** (`jam2.py`, `jam.py`) - Multi-threaded audio playback with dynamic effects
3. **Utility Tools** (`update_nirsoft.py`, `lang-categorizer.py`) - NirSoft tool management and language categorization

**Key Architecture**: Modular Python scripts with rich CLI logging, threading for concurrent operations, and extensive statistics tracking.

---

## Critical Patterns & Conventions

### 1. **Thread-Safe Stats & Logging**

**Pattern**: All multi-threaded operations use a `Stats` class with `threading.Lock()` for safe concurrent updates.

**Example**: `update_nirsoft.py` lines 31-120 define `Stats` class with thread-safe methods:
```python
def add_download(self, bytes_count):
    with self.lock:
        self.bytes_downloaded += bytes_count
```

**Apply when**: Any feature tracking counts, metrics, or shared state in threaded code.

---

### 2. **Rich Console Output with Timestamps**

**Pattern**: All logging uses `rich.console.Console` with emoji-tagged color codes and `get_timestamp()` prefix.

**Example** (`update_nirsoft.py` line 351):
```python
console.print(f"{get_timestamp()} [bold green]🌐 NETWORK[/bold green] Fetching pad links list...")
```

**Color/Tag Mapping**:
- `[bold red]` / `[red]` - Errors (❌)
- `[bold green]` / `[green]` - Success/completion (✅)
- `[bold yellow]` / `[yellow]` - Warnings/cleanup (⚠️)
- `[bold cyan]` / `[cyan]` - Information/progress (ℹ️)
- `[bold magenta]` / `[magenta]` - Operations/processing (🔧)

**Apply when**: Creating user-facing status messages or progress tracking.

---

### 3. **Semaphore-Based Concurrency Limits**

**Pattern**: Use `threading.Semaphore(MAX)` with explicit acquire/release (NOT `with` statement) to avoid deadlocks.

**Example** (`jam2.py`):
```python
MAX_CONCURRENT_PLAYBACK = 5
playback_semaphore = threading.Semaphore(MAX_CONCURRENT_PLAYBACK)
# In thread: acquire with timeout, use try/finally for release
```

**Critical Fix**: NEVER use `with playback_semaphore:` for long-running operations—it blocks other threads. Instead:
```python
try:
    playback_semaphore.acquire(timeout=1.0)
    # long operation here
finally:
    playback_semaphore.release()
```

**See**: `CODE_REVIEW_SUMMARY.md` for why this matters (was causing "zombie threads").

---

### 4. **Configuration as Top-Level Module**

**Pattern**: `config.py` contains ALL model parameters (temperature, sampling, context size). Import with `from config import *`.

**Key configs** (see `config.py`):
- `selected_model = 'mistral-nemo:latest'` - Model name
- `temperature = 0.55` - Creativity/consistency balance
- `num_ctx = 8192` - Context window (tokens)
- Sampling params: `top_k=20`, `top_p=0.9`, `repeat_penalty=1.2`
- `src_options["stop"]` - Model-specific stop tokens (23 variants defined)

**Apply when**: Adding model interaction code—modify `config.py` instead of hardcoding values.

---

### 5. **Ollama Client Pattern**

**Pattern**: Use `ollama.Client(host='127.0.0.1')` and store instance globally (`client` variable).

**Example** (`analyze.py` lines 57-58):
```python
from ollama import Client
client = Client(host='127.0.0.1')
models = client.list()
```

**Common calls**:
- `client.pull(model_name)` - Download model
- `client.chat(model=model_name, messages=[...])` - Send prompt
- `client.ps()` - List running models

**Apply when**: Adding any model interaction; always use the global `client` instance.

---

### 6. **Instruction Templates with String Formatting**

**Pattern**: Store reusable prompts in `instructions.py` as multi-line strings, use placeholder formatting.

**Example** (`instructions.py` line 3):
```python
system = "You are security-based research assistant..."
prompt_based = [  # List of template variations
    "Develop issue when %1% ... %3% letters in data scheme...",
    # ...
]
```

**Apply when**: Creating dynamic prompts; add templates to `instructions.py` and reference via import.

---

### 7. **Audio Processing with Librosa + SoundDevice**

**Pattern**: Use `librosa` for loading/processing MP3s, `sounddevice` for playback, `pycaw` for volume control.

**Example** (`test_jam2.py` lines 24-29):
```python
import librosa
import sounddevice as sd

audio, sr = librosa.load(str(sample), sr=None, mono=True)
audio = librosa.effects.time_stretch(audio, rate=time_stretch)
audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
sd.play(audio, sr)
```

**Apply when**: Adding audio features; always use these libraries (already in `requirements.txt`).

---

### 9. **Redis Context Persistence**

**Pattern**: Use Redis lists for persisting conversation context across simulation runs. Store context with `rpush()` and retrieve with `lrange()`.

**Example** (`sim.py` lines 150-159):
```python
from redis import StrictRedis
redis_client = StrictRedis(REDIS_HOST, 6379, encoding_errors='ignore')
# Store context
redis_client.rpush('sim.context.ids', *context)
# Retrieve context
context = redis_client.lrange('sim.context.ids', 0, num_ctx)
# Clear context
redis_client.delete('sim.context.ids')
```

**Apply when**: Simulation state needs to persist across runs or be shared between processes. Always use `REDIS_HOST = "127.0.0.1"` and port 6379.

---

### 10. **Scene-Based Simulation Configuration**

**Pattern**: Simulations load JSON scenario files from `scenes/<scene_dir>/scenario.json` with experiment parameters and instructions.

**Example** (`sim.py` lines 410-435):
```python
# Load scenario
scenario_file = os.path.join('scenes', scene_dir, 'scenario.json')
program_data = json.loads(open(scenario_file, 'rt').read())
# Fields: name, instructions, biases, temperature, model_name, template, sim_log_path
```

**JSON Fields**:
- `name` - Experiment identifier
- `instructions` - List of prompt templates
- `biases` - String for prompt injection
- `temperature` - Model temperature for generation
- `model_name` - Ollama model to use
- `template` - Custom prompt template
- `sim_log_path` - Output directory for logs

**Apply when**: Creating new experiments; define scenarios in `scenes/` with standard JSON structure.

## Developer Workflows

### Running LLM Analysis
```bash
# Basic run (loads default model, runs analysis loop)
python analyze.py

# Run with model verification and streaming pull
python analyze.py  # Will auto-pull model if missing via client.pull(model, stream=True)
```

### Testing Audio System
```bash
# Test single MP3 with random effects
python test_jam2.py

# Run continuous audio playback system with volume control
python jam2.py
# Features: concurrent playback control, volume mutations, memory monitoring, thread cleanup
```

### Running Simulations
```bash
# Requires scene JSON configuration in scenes/ directory
python sim.py --scene <scene_dir>
# Loads scenario.json with instructions, biases, temperature, model_name, and template
# Output: Experiment logs in logs/ directory with Rich-formatted output
```

### Updating NirSoft Tools
```bash
python update_nirsoft.py
# Downloads all packages, detects x86/x64, extracts to T:\!power-tools\NirLauncher\
# Streaming downloads with live progress reporting
```

### Demo & Quick Tests
```bash
# Verify installation and see toolkit features
python demo.py

# See enhanced logging in action
python demo_enhanced_logging.py

# HTTP load testing
python httptest.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

---

## Key Files & Their Roles

| File | Purpose | Key Functions |
|------|---------|---|
| `config.py` | Model parameters & sampling config | `temperature`, `src_options`, seed generation |
| `analyze.py` | Ollama model interaction & analysis | `update_model()`, `slog()`, prompt execution |
| `instructions.py` | Prompt templates & system instructions | `system`, `prompt_based`, `prompt_ejector` |
| `sim.py` | Simulation framework for LLM experiments | `Simulatar` class, Redis context persistence |
| `jam2.py` | Multi-threaded audio playback engine | `playback_thread()`, `monitoring_loop()`, volume control |
| `update_nirsoft.py` | NirSoft package downloader/extractor | `Stats` class, PE analysis, streaming downloads |
| `langfeatures.py` | Language feature categorization | Dynamic prompt generation, feature mappings |
| `test_jam2.py` | Audio processing test suite | Single-file audio effect testing with librosa |
| `demo.py` | Installation verification & feature demo | Shows config, logging, model loading |
| `demo_enhanced_logging.py` | Enhanced logging showcase | Status tables, operation tracking, timestamps |
| `httptest.py` | HTTP load testing utility | Uses salvo for load testing |

---

### 11. **Module-Level Initialization & Global State**

**Pattern**: Python modules initialize global state at module level for efficient access. Critical for thread safety and performance.

**Example** (`jam2.py` lines 37-57):
```python
# Global console and state initialization
console = Console()
start_time = time.time()
SAMPLE_DIR = Path(__file__).parent / "samples"

# Device and volume control setup
device = AudioUtilities.GetSpeakers()
volumer = device.EndpointVolume
playback_semaphore = threading.Semaphore(MAX_CONCURRENT_PLAYBACK)
console_lock = threading.Lock()

# Thread control events
volume_thread_running = threading.Event()
periodic_sample_thread_running = threading.Event()
monitoring_thread_running = threading.Event()
playback_threads = []
```

**Apply when**: Setting up threads, external device control, or shared resources. Initialize at module level for consistent access across functions.

---

### 12. **Speed Calculation & Performance Metrics**

**Pattern**: Track performance metrics with rolling averages and ETA calculation for user-facing reports.

**Example** (`update_nirsoft.py` lines 156-180):
```python
def get_download_speed(self):
    """Calculate current download speed in bytes/sec"""
    elapsed = time.time() - self.start_time
    if elapsed > 0:
        return self.bytes_downloaded / elapsed
    return 0

def get_average_speed(self):
    """Get smoothed average speed from recent samples"""
    if len(self.download_speeds) > 0:
        return sum(self.download_speeds) / len(self.download_speeds)
    return self.get_download_speed()

def get_eta(self, total_files, current_file):
    """Estimate time remaining"""
    if current_file == 0:
        return None
    elapsed = time.time() - self.start_time
    avg_time_per_file = elapsed / current_file
    remaining_files = total_files - current_file
    return remaining_files * avg_time_per_file
```

### 13. **Rich Table Status Reporting**

**Pattern**: Use `Table.grid()` with multi-column layout for formatting operation status, statistics, and progress metrics in a human-readable display.

**Example** (`demo_enhanced_logging.py` lines 25-70):
```python
from rich.table import Table
table = Table.grid(padding=(0, 2))
table.add_column(justify="right", style="cyan", no_wrap=True)
table.add_column(justify="left", style="white")

table.add_row("", "[bold yellow]═══ CURRENT OPERATION ═══[/bold yellow]")
table.add_row("🔄 Operation:", "Downloading package 1/10")
table.add_row("📦 Package:", "example-tool")
table.add_row("⏲️  Op. Duration:", f"{duration:.1f}s")

from rich.panel import Panel
console.print(Panel(table, title="[bold cyan]📊 STATUS REPORT[/bold cyan]", border_style="cyan"))
```

**Structure**:
- Use `Table.grid()` for flexible column layout
- Organize data into logical sections with bold headers
- Include emoji prefixes for visual clarity
- Wrap in `Panel` for bordered display

---

### 14. **Internet Connectivity Detection**

**Pattern**: Check network availability before attempting downloads using `socket.create_connection()` with a known host and short timeout.

**Example** (`analyze.py` lines 200-207):
```python
try:
    socket.create_connection(('he.net', 80), timeout=1.8)
    slog('Network available')
except Exception as e:
    slog(f'Network error: {e}')
    # Handle offline gracefully - skip downloads, use cached data, etc.
```

**Apply when**: Starting network-dependent operations. Use reliable public hosts like `he.net` (Hurricane Electric) with 1-2 second timeout.

---

### 15. **Persistent Logging to Disk**

**Pattern**: Log messages to disk while displaying on console. Strip Rich markup for file output to ensure clean logs.

**Example** (`analyze.py` lines 123-145):
```python
def slog(msg: str = "", end: str = "\n", justify: str = "full", style: str = None):
    msg_for_input = msg
    # Strip uppercase markup (Rich tags) for display
    msg_for_log = re.sub(r'(\[/?[A-Z_]*?])', '', msg_for_input)
    # Strip lowercase markup for file logging
    msg_for_file = re.sub(r'(\[/?[a-z_]*?])', '', msg_for_log)
    
    console.print(msg_for_log, end=end, justify=justify, style=style)
    sys.stdout.flush()
    
    log_file = os.path.join(log_dir, f'sim_log_{iid:09d}.md')
    with open(log_file, "ab") as f:
        f.write((msg_for_file + end).encode(encoding='utf_8', errors='ignore'))
```

**Apply when**: Creating logging functions. Always strip markup before file I/O and use append mode (`"ab"`) for multi-threaded safety.

---

## Common Tasks & Implementation Patterns

### Adding a New Statistics Metric
1. Add counter to `Stats.__init__()` in relevant file (see `update_nirsoft.py` line 31+)
2. Create `add_*()` method with lock protection
3. Reference in status reports via `stats.add_*()` calls
4. Update `generate_live_table()` or `report_and_reset()` to display

### Adding Model Parameters
1. Edit `config.py` `src_options` dict
2. Reference via `from config import src_options` in analysis files
3. Pass to `client.chat(options=src_options, ...)`

### Adding Threading Operations
1. Create `threading.Thread(target=func, args=())` (NOT direct calls from other threads)
2. Protect shared state with locks or semaphores
3. Use `try/finally` to ensure cleanup (see semaphore pattern above)
4. Track thread lifecycle in list with lock

### Exception Handling Pattern
1. Use broad `try/except Exception` for critical operations (model pulling, downloads)
2. Log exceptions with context using `slog()` or `console.print()`
3. For recovery: Implement retry logic with exponential backoff
4. Always handle keyboard interrupts (`KeyboardInterrupt`) for graceful shutdown

**Example** (`analyze.py` lines 57-80):
```python
try:
    socket.create_connection(('he.net', 80), timeout=1.8)
    slog('exist')
except Exception as e:
    slog(f'missing: {e}')
    # Graceful degradation or retry logic
```

### Thread Monitoring & Cleanup Pattern
1. Use `threading.Event()` flags for thread lifecycle management
2. Check `is_alive()` and enumerate threads with `threading.enumerate()`
3. Track thread state in monitoring loop with `psutil.Process()`
4. Log frame info via `sys._current_frames()` for debugging stuck threads

**Example** (`jam2.py` lines 230-260):
```python
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
mem_mb = mem_info.rss / 1024 / 1024
thread_count = threading.active_count()
for t in threading.enumerate():
    is_alive = "alive" if t.is_alive() else "dead"
    # Log thread status with frame information
```

### Adding Downloads/Network Calls
1. Use `requests.get(url, stream=True, timeout=16)` for large files
2. Chunk data and update stats: `stats.add_download(len(chunk))`
3. Report progress with rich `Progress` widget
4. Always handle timeouts and retry logic

---

## Critical Gotchas

⚠️ **Semaphore Deadlock**: Never use `with semaphore:` for operations longer than milliseconds—use acquire/release with timeout.

⚠️ **Zombie Threads**: Never call `periodic_sample_loop()` directly from a thread—spawn with `threading.Thread(...).start()`.

⚠️ **Config Imports**: Use `from config import *` to get all parameters; don't copy hardcoded values.

⚠️ **Rich Console Lock**: Protect multi-threaded console output with `console_lock` to avoid garbled output.

⚠️ **Stop Tokens**: `src_options["stop"]` has 23 defined tokens—update this when switching models to avoid truncation.

---

## Dependencies Overview

**Core LLM**: `ollama` (local server required)  
**CLI/Output**: `rich` (for progress bars, tables, colors)  
**Audio**: `librosa`, `sounddevice`, `pycaw`, `numpy`  
**Web**: `requests`, `aiohttp`  
**PE Analysis**: `pefile` (for detecting x86/x64)  
**In-Memory DB**: `redis` (optional, for advanced caching)  

See `requirements.txt` for complete list and versions.

---

## Documentation References

- **Thread Safety**: See `THREAD_SAFETY_FIXES.md`, `SEMAPHORE_LOCK_FIXES.md`, `CODE_REVIEW_SUMMARY.md`
- **Audio Features**: See `JAM2_FEATURES.md`, `JAM2_USAGE.md`
- **Logging**: See `ENHANCED_LOGGING_SUMMARY.md`
- **API Details**: See `docs/api-reference.md`

---

**Last Updated**: 2025-02-28  
**Maintainer**: kilitary@gmail.com

