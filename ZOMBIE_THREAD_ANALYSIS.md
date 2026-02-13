# Zombie Thread Analysis and Prevention

## What Are Zombie Threads?

In this codebase, zombie threads are playback threads that:
1. Never complete their execution
2. Continue to hold resources (semaphore slots, memory, thread handles)
3. Stay in the thread list but don't respond to termination signals
4. Accumulate over time, eventually exhausting system resources

## Root Causes Identified

### 1. The Recursive Blocking Call Bug ⚠️ MOST CRITICAL

**Location**: Original line 615 in `playback_thread()`

```python
# ZOMBIE CREATOR:
def playback_thread(...):
    with sd.OutputStream(...) as stream:
        while position < len(audio_copy):
            if not triggered and position >= trigger_position:
                periodic_sample_loop()  # ← THIS LINE CREATES ZOMBIES!
```

**Why this creates zombies**:
1. `periodic_sample_loop()` contains `while periodic_sample_thread_running.is_set():`
2. This is an **infinite loop** that never returns
3. The playback thread calls it and **blocks forever**
4. The thread never reaches the `finally` block to clean up
5. Semaphore is never released → deadlock
6. playback_count is never decremented → counter corruption

**Lifecycle of a zombie thread**:
```
Time 0: Thread starts → acquires semaphore (count: 1/5)
Time 1: Loads and processes audio
Time 2: Starts playback
Time 3: Reaches 2/3 mark, calls periodic_sample_loop()
Time 4+: STUCK in infinite loop, never exits
Time 5+: Thread is "alive" but doing nothing useful
Time 6+: Cleanup tries to kill it, but it may be in I/O wait
Time 7+: Thread becomes a zombie - shows in thread list but non-functional
```

**Fix applied**:
```python
# CORRECT:
if not triggered and position >= trigger_position:
    triggered = True
    elapsed_trigger_event.set()
    # Spawn independent thread - does NOT block
    with playback_count_lock:
        if playback_count < MAX_CONCURRENT_PLAYBACK:
            threading.Thread(target=play_random_sample, daemon=True).start()
```

---

### 2. Semaphore Never Released 🔒

**Problem**: If a thread crashes or hangs before reaching the semaphore release, that slot is lost forever.

**Original code**:
```python
with playback_semaphore:  # Acquire
    playback_count += 1
    ... playback code ...
    playback_count -= 1  # This is INSIDE the with block
# Semaphore released HERE - but only if no exception!
```

**Issue**: If thread crashes in playback code:
- Exception propagates up
- `with` block cleanup may fail
- Semaphore never released
- One less slot available forever
- After 5 crashes, no more threads can start

**Fix applied**:
```python
semaphore_acquired = playback_semaphore.acquire(blocking=True, timeout=5.0)
if not semaphore_acquired:
    return  # Skip if can't acquire
try:
    # ... do work ...
finally:
    # GUARANTEED to release, even on exception
    playback_semaphore.release()
    playback_count -= 1
```

---

### 3. Thread List Accumulation 📈

**Problem**: Dead threads were not properly removed from `playback_threads` list

**Original cleanup code**:
```python
playback_threads[:] = [t for t in playback_threads if t.is_alive()]
active_count = len(playback_threads)
if active_count > 3:  # Then try to kill excess
```

**Issue**: 
- Cleanup only happened when count exceeded threshold
- Dead threads could accumulate below threshold
- No logging to detect gradual buildup

**Fix applied**:
```python
# Clean up dead threads FIRST, ALWAYS
alive_threads = [t for t in playback_threads if t.is_alive()]
dead_count = len(playback_threads) - len(alive_threads)
if dead_count > 0:
    log_message(f"Cleaned up {dead_count} finished thread(s)")
playback_threads[:] = alive_threads

# THEN check if we need to kill excess
active_count = len(playback_threads)
if active_count > MAX_CONCURRENT_PLAYBACK:
    # Kill oldest threads
```

---

## Zombie Thread Symptoms

### Memory Symptoms
- Gradual memory increase over time
- Memory doesn't decrease when samples finish
- `psutil` shows growing RSS even after playback stops

### Thread Symptoms
- Thread count increases beyond MAX_CONCURRENT_PLAYBACK
- `threading.enumerate()` shows many threads
- Thread names show old timestamps

### Performance Symptoms
- New samples take longer to start
- Audio stuttering or gaps
- CPU usage stays high even when "idle"

### Semaphore Symptoms
- Log shows "Failed to acquire semaphore" even when thread count is low
- playback_count doesn't match actual playing samples
- New samples stop starting entirely

---

## Zombie Prevention Strategies

### 1. Never Call Blocking Functions in Worker Threads ✅
```python
# BAD:
def worker():
    do_work()
    blocking_infinite_loop()  # Never returns!

# GOOD:
def worker():
    do_work()
    # Spawn new thread for next task
    threading.Thread(target=next_task, daemon=True).start()
```

### 2. Always Use try/finally for Cleanup ✅
```python
# BAD:
acquire_resource()
do_work()
release_resource()  # Might not execute on exception

# GOOD:
try:
    acquire_resource()
    do_work()
finally:
    release_resource()  # ALWAYS executes
```

### 3. Set Timeouts on Blocking Operations ✅
```python
# BAD:
semaphore.acquire()  # Could wait forever

# GOOD:
if not semaphore.acquire(timeout=5.0):
    log_error("Timeout acquiring semaphore")
    return  # Give up gracefully
```

### 4. Use Daemon Threads for Background Work ✅
```python
# All worker threads should be daemon
thread = threading.Thread(target=worker, daemon=True)
thread.start()
```

**Why**: Daemon threads are killed when main thread exits, preventing zombie process.

### 5. Monitor and Clean Up Regularly ✅
```python
# Run cleanup every second
def cleanup_loop():
    while running:
        time.sleep(1.0)
        # Remove dead threads
        threads[:] = [t for t in threads if t.is_alive()]
        # Kill excess threads
        if len(threads) > MAX_ALLOWED:
            kill_oldest_threads()
```

---

## Testing for Zombie Threads

### Test 1: Long Running Test
```bash
# Run for 10 minutes and monitor thread count
python jam2.py
# In another terminal:
while true; do ps -p <PID> -o nlwp; sleep 5; done
```

**Expected**: Thread count oscillates but stays bounded (< 20 total)
**Zombie symptom**: Thread count continuously increases

### Test 2: Stress Test
```python
# Modify to start samples rapidly
while True:
    play_random_sample()
    time.sleep(0.01)  # Very short delay
```

**Expected**: Semaphore limits to MAX_CONCURRENT_PLAYBACK
**Zombie symptom**: Thread count exceeds limit significantly

### Test 3: Error Injection
```python
# Modify to randomly raise exceptions
def playback_thread(...):
    try:
        if random.random() < 0.1:  # 10% failure rate
            raise RuntimeError("Simulated error")
        # ... normal playback ...
    finally:
        # Verify cleanup still happens
```

**Expected**: Semaphore and count stay consistent despite errors
**Zombie symptom**: Count/semaphore diverge, threads accumulate

### Test 4: Cleanup Verification
```python
# Monitor the cleanup thread's output
# Should see regular cleanup messages
```

**Expected**: "Cleaned up N finished thread(s)" appears regularly
**Zombie symptom**: No cleanup messages, or count always 0

---

## Manual Zombie Detection

### Check Active Threads
```python
import threading
import sys

# Get all frames
frames = sys._current_frames()
for thread in threading.enumerate():
    frame = frames.get(thread.ident)
    if frame:
        print(f"{thread.name}: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
```

**Zombie indicators**:
- Thread stuck at same line number across multiple checks
- Thread in I/O wait but shouldn't be
- Thread in `periodic_sample_loop` when it should be in `playback_thread`

### Check Semaphore State
```python
# The semaphore's internal counter should equal:
# MAX_CONCURRENT_PLAYBACK - playback_count

# If they diverge, there's a leak
expected_available = MAX_CONCURRENT_PLAYBACK - playback_count
actual_available = playback_semaphore._value  # Internal, but useful for debug

if expected_available != actual_available:
    print("SEMAPHORE LEAK DETECTED!")
```

---

## Recovery Strategies

### Automatic Recovery (Implemented) ✅
1. Cleanup thread removes dead threads every second
2. Cleanup thread kills excess threads beyond limit
3. Graceful semaphore timeout prevents infinite waiting

### Manual Recovery
If zombies accumulate despite fixes:

```python
# Nuclear option: Kill all playback threads
import ctypes
with playback_threads_lock:
    for thread in playback_threads:
        if thread.is_alive():
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread.ident),
                ctypes.py_object(SystemExit)
            )
    playback_threads.clear()
    playback_count = 0
    # Release all semaphore slots
    while playback_semaphore._value < MAX_CONCURRENT_PLAYBACK:
        playback_semaphore.release()
```

⚠️ **Warning**: This is dangerous and can cause segfaults. Only use as last resort.

---

## Summary

The zombie thread issue was caused by:
1. **Blocking recursive call** - calling infinite loop from worker thread
2. **Semaphore mismanagement** - not using try/finally for cleanup
3. **Inadequate cleanup** - not removing dead threads regularly

All three issues have been fixed with:
1. **Async task spawning** - use threading.Thread.start() instead of direct calls
2. **Proper resource management** - try/finally with timeouts
3. **Regular cleanup** - dedicated cleanup thread runs every second

The code now has robust zombie prevention and should maintain stable thread count indefinitely.
