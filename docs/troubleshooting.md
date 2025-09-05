# Troubleshooting Guide

This guide helps resolve common issues encountered when using the Ollama Development Toolkit.

## Installation Issues

### Python Dependencies
**Problem:** Package installation fails with version conflicts
```
ERROR: Could not find a version that satisfies the requirement pywin32==306
```

**Solution:**
```bash
# Skip platform-specific packages
pip install --ignore-installed ollama rich redis

# Or create a filtered requirements file
grep -v "pywin32\|pywinpty" requirements.txt > requirements_linux.txt
pip install -r requirements_linux.txt
```

### Ollama Server Connection
**Problem:** Cannot connect to Ollama server
```
ConnectionError: Failed to connect to Ollama server
```

**Solutions:**
1. **Check Ollama service status:**
   ```bash
   ollama serve
   # Or check if running: ps aux | grep ollama
   ```

2. **Verify server address:**
   ```python
   # In config or environment
   OLLAMA_HOST = "http://localhost:11434"
   ```

3. **Test connection manually:**
   ```bash
   curl http://localhost:11434/api/version
   ```

### Redis Connection Issues
**Problem:** Redis connection fails
```
redis.exceptions.ConnectionError: Connection refused
```

**Solutions:**
1. **Install and start Redis:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   sudo systemctl start redis-server
   
   # macOS
   brew install redis
   brew services start redis
   ```

2. **Check Redis configuration:**
   ```python
   # Verify connection parameters
   REDIS_HOST = "localhost"
   REDIS_PORT = 6379
   ```

3. **Test Redis connection:**
   ```bash
   redis-cli ping
   # Should return: PONG
   ```

## Model Issues

### Model Not Found
**Problem:** Specified model doesn't exist
```
Model 'mistral-nemo:latest' not found
```

**Solutions:**
1. **List available models:**
   ```bash
   ollama list
   ```

2. **Download missing model:**
   ```bash
   ollama pull mistral-nemo:latest
   ```

3. **Use alternative model:**
   ```python
   selected_model = 'llama2:latest'  # or another available model
   ```

### Model Download Fails
**Problem:** Model download interrupted or fails
```
download error: connection timeout
```

**Solutions:**
1. **Check internet connection and retry:**
   ```python
   update_model('mistral-nemo:latest')
   ```

2. **Use smaller model:**
   ```python
   selected_model = 'mistral:7b'  # Smaller, faster download
   ```

3. **Download manually:**
   ```bash
   ollama pull mistral-nemo:latest --verbose
   ```

## Performance Issues

### Memory Problems
**Problem:** Out of memory errors during generation
```
CUDA out of memory / RAM allocation failed
```

**Solutions:**
1. **Reduce context window:**
   ```python
   src_options["num_ctx"] = 2048  # Reduced from 8192
   ```

2. **Enable low VRAM mode:**
   ```python
   src_options["low_vram"] = True
   ```

3. **Reduce batch size:**
   ```python
   src_options["num_batch"] = 1
   ```

4. **Use CPU instead of GPU:**
   ```python
   src_options["num_gpu"] = 0
   ```

### Slow Generation
**Problem:** Very slow text generation
```
Generation taking several minutes per response
```

**Solutions:**
1. **Optimize thread count:**
   ```python
   n_threads = os.cpu_count()  # Use all available cores
   src_options["num_thread"] = n_threads
   ```

2. **Enable memory mapping:**
   ```python
   src_options["use_mmap"] = True
   ```

3. **Reduce generation length:**
   ```python
   src_options["num_predict"] = 512  # Limit response length
   ```

4. **Use smaller model:**
   ```python
   selected_model = 'mistral:7b'  # Instead of larger models
   ```

### High CPU Usage
**Problem:** Excessive CPU usage affecting system
```
100% CPU usage, system becomes unresponsive
```

**Solutions:**
1. **Limit thread count:**
   ```python
   n_threads = os.cpu_count() // 2  # Use half of available cores
   ```

2. **Add processing delays:**
   ```python
   import time
   time.sleep(0.1)  # Small delay between operations
   ```

3. **Use GPU acceleration:**
   ```python
   src_options["num_gpu"] = 1
   src_options["num_thread"] = 4  # Reduce CPU threads
   ```

## Response Quality Issues

### Repetitive Responses
**Problem:** Model generates repetitive or looping text
```
"The system the system the system..."
```

**Solutions:**
1. **Increase repetition penalty:**
   ```python
   src_options["repeat_penalty"] = 1.3  # Increased from 1.2
   src_options["repeat_last_n"] = 64    # Increased window
   ```

2. **Adjust presence penalty:**
   ```python
   src_options["presence_penalty"] = 1.8  # Increased from 1.5
   ```

3. **Modify sampling parameters:**
   ```python
   src_options["top_k"] = 40      # Increased diversity
   src_options["top_p"] = 0.95    # Higher nucleus threshold
   ```

### Poor Response Quality
**Problem:** Incoherent or irrelevant responses
```
Responses don't make sense or are off-topic
```

**Solutions:**
1. **Lower temperature:**
   ```python
   temperature = 0.3  # More focused responses
   ```

2. **Adjust sampling parameters:**
   ```python
   src_options["top_k"] = 20      # More selective
   src_options["top_p"] = 0.85    # Lower threshold
   ```

3. **Improve prompt engineering:**
   ```python
   # Add context and constraints to prompts
   prompt = "Please provide a clear, specific answer to: " + original_prompt
   ```

### Context Loss
**Problem:** Model loses track of conversation context
```
Model doesn't remember earlier parts of conversation
```

**Solutions:**
1. **Increase context window:**
   ```python
   num_ctx = 16384  # Doubled context size
   ```

2. **Preserve important context:**
   ```python
   src_options["num_keep"] = 10  # Keep more initial tokens
   ```

3. **Implement context management:**
   ```python
   # Manual context preservation
   important_context = extract_key_points(conversation)
   ```

## Logging and Output Issues

### No Color Output
**Problem:** Rich formatting doesn't show colors
```
Plain text output instead of colored formatting
```

**Solutions:**
1. **Force color mode:**
   ```python
   console = console.Console(
       force_terminal=True,
       color_system='256'  # or 'truecolor'
   )
   ```

2. **Check terminal compatibility:**
   ```bash
   echo $TERM
   # Should show color-capable terminal
   ```

3. **Disable color if needed:**
   ```python
   console = console.Console(no_color=True)
   ```

### Log File Issues
**Problem:** Cannot write to log files
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
1. **Check file permissions:**
   ```bash
   chmod 755 /path/to/log/directory
   ```

2. **Use accessible directory:**
   ```python
   sim_log_path = os.path.expanduser("~/logs")  # User home directory
   ```

3. **Create directory if needed:**
   ```python
   os.makedirs(log_directory, exist_ok=True)
   ```

## Configuration Issues

### Invalid Configuration
**Problem:** Configuration parameters cause errors
```
ValueError: temperature must be between 0.0 and 2.0
```

**Solutions:**
1. **Validate parameters:**
   ```python
   # Temperature validation
   temperature = max(0.0, min(2.0, temperature))
   
   # Context size validation
   num_ctx = max(1, min(32768, num_ctx))
   ```

2. **Use default values:**
   ```python
   # Reset to safe defaults
   from config import *  # Reload default configuration
   ```

3. **Check parameter ranges:**
   ```python
   # Recommended ranges
   temperature: 0.1 - 1.5
   top_k: 1 - 100
   top_p: 0.1 - 1.0
   num_ctx: 512 - 32768
   ```

## Network and Connectivity

### Firewall Issues
**Problem:** Cannot connect to external services
```
ConnectionError: Connection timed out
```

**Solutions:**
1. **Check firewall settings:**
   ```bash
   # Allow Ollama port
   sudo ufw allow 11434
   ```

2. **Test connectivity:**
   ```bash
   telnet localhost 11434
   ```

3. **Configure proxy if needed:**
   ```bash
   export HTTP_PROXY=http://proxy:8080
   export HTTPS_PROXY=http://proxy:8080
   ```

### DNS Resolution
**Problem:** Cannot resolve hostnames
```
socket.gaierror: [Errno -2] Name or service not known
```

**Solutions:**
1. **Use IP addresses:**
   ```python
   OLLAMA_HOST = "http://127.0.0.1:11434"
   ```

2. **Check DNS configuration:**
   ```bash
   nslookup localhost
   ```

## Development and Debugging

### Import Errors
**Problem:** Cannot import required modules
```
ImportError: No module named 'ollama'
```

**Solutions:**
1. **Check Python path:**
   ```python
   import sys
   print(sys.path)
   ```

2. **Install missing packages:**
   ```bash
   pip install ollama rich redis
   ```

3. **Use virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

### Debugging Tips
**Enable verbose logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug output
slog(f"Debug: {variable_value}", style="dim")
```

**Check system resources:**
```python
import psutil

# Monitor memory usage
memory = psutil.virtual_memory()
slog(f"Memory usage: {memory.percent}%")

# Monitor CPU usage
cpu = psutil.cpu_percent()
slog(f"CPU usage: {cpu}%")
```

**Profile performance:**
```python
import time

start_time = time.time()
# ... your code here ...
end_time = time.time()

slog(f"Execution time: {end_time - start_time:.2f}s")
```

## Getting Help

### Community Resources
- **Ollama Documentation**: https://ollama.ai/docs
- **Rich Documentation**: https://rich.readthedocs.io/
- **Redis Documentation**: https://redis.io/documentation

### Diagnostic Information
When reporting issues, include:

```python
import sys, platform, ollama

# System information
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.architecture()}")

# Ollama client info
try:
    client = ollama.Client()
    print(f"Ollama version: {client._client.version()}")
except:
    print("Ollama not available")

# Available models
try:
    models = ollama.list()
    print(f"Available models: {[m['name'] for m in models['models']]}")
except:
    print("Cannot list models")
```

### Creating Minimal Reproduction
When reporting bugs, create a minimal example:

```python
import ollama

# Minimal example that reproduces the issue
try:
    client = ollama.Client()
    response = client.chat(
        model='mistral-nemo:latest',
        messages=[{'role': 'user', 'content': 'Hello'}]
    )
    print(response['message']['content'])
except Exception as e:
    print(f"Error: {e}")
```

This helps identify the root cause quickly and enables faster resolution.