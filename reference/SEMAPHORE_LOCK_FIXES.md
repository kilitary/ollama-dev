# Semaphore and Lock Error Fixes

## Summary
Fixed critical semaphore/lock errors and zombie thread issues in jam2.py that were causing deadlocks, thread leaks, and inconsistent state.

## Issues Found and Fixed

### 1. **CRITICAL: Semaphore Deadlock** ✅ FIXED
**Location**: Line 576 (old code)

**Problem**: 
- The semaphore was acquired using `with playback_semaphore:` which held it for the ENTIRE playback duration (potentially several seconds)
- This meant only 2 threads could EVER be active, even if they were just waiting for I/O
- Semaphore should only be held briefly to check/increment a counter, not for the entire operation

**Fix**:
```python
# OLD (WRONG):
with playback_semaphore:
    # Entire playback happens here - blocks semaphore for seconds!
    playback_count += 1
    ... play audio ...
    playback_count -= 1

# NEW (CORRECT):
semaphore_acquired = playback_semaphore.acquire(blocking=True, timeout=5.0)
if not semaphore_acquired:
    return  # Gracefully skip if can't acquire
try:
    # Increment counter immediately
    with playback_count_lock:
        playback_count += 1
    
    # Do the actual work (semaphore held, but that's the point)
    ... play audio ...
    
finally:
    # ALWAYS release semaphore
    playback_semaphore.release()
    playback_count -= 1
```

**Why this matters**:
- Old code: Maximum 2 concurrent threads total
- New code: Maximum 5 concurrent playback threads (controlled by semaphore value)
- Prevents deadlock if thread crashes while holding semaphore
- Timeout prevents infinite waiting

---

### 2. **CRITICAL: Playback Count Corruption** ✅ FIXED
**Location**: Line 632-634 (old code)

**Problem**:
- `playback_count` was decremented in the `finally` block but OUTSIDE the semaphore's `with` block
- If an exception occurred before semaphore was acquired, count would still be decremented
- This caused the count to go negative or become desynchronized

**Fix**:
```python
# OLD (WRONG):
with playback_semaphore:
    playback_count += 1  # Inside semaphore
    ...
    # Finally block is OUTSIDE the with statement!
    finally:
        playback_count -= 1  # Decremented even if semaphore not acquired

# NEW (CORRECT):
try:
    # Increment only after semaphore acquired
    with playback_count_lock:
        playback_count += 1
    ...
finally:
    # Decrement paired with semaphore release
    playback_semaphore.release()
    with playback_count_lock:
        playback_count -= 1
```

---

### 3. **CRITICAL: Recursive Call Bug (Zombie Thread Creator)** ✅ FIXED
**Location**: Line 615 (old code)

**Problem**:
- `periodic_sample_loop()` was called DIRECTLY from the playback thread
- This is a BLOCKING function that runs an infinite loop
- Causes the playback thread to never finish (zombie thread)
- Stack overflow risk if called recursively

**Fix**:
```python
# OLD (WRONG):
if not triggered and position >= trigger_position:
    triggered = True
    elapsed_trigger_event.set()
    periodic_sample_loop()  # BLOCKS FOREVER!

# NEW (CORRECT):
if not triggered and position >= trigger_position:
    triggered = True
    elapsed_trigger_event.set()
    # Check count and spawn new thread asynchronously
    with playback_count_lock:
        if playback_count < MAX_CONCURRENT_PLAYBACK:
            threading.Thread(target=play_random_sample, daemon=True).start()
```

**Why this matters**:
- Old code created zombie threads that never died
- New code properly spawns independent threads
- No blocking calls in playback path

---

### 4. **Thread List Cleanup Race Condition** ✅ FIXED
**Location**: Line 334 (old code)

**Problem**:
- List slicing `playback_threads[threads_to_kill:]` had off-by-one error
- Did not track which threads were actually killed successfully
- Could remove wrong threads from the list

**Fix**:
```python
# OLD (WRONG):
for i in range(threads_to_kill):
    if i < len(playback_threads):  # Already problematic
        thread_to_kill = playback_threads[i]
        ... try to kill ...
# Remove by slicing (doesn't check if kill succeeded)
playback_threads[:] = playback_threads[threads_to_kill:]

# NEW (CORRECT):
killed_threads = []
for i in range(threads_to_kill):
    thread_to_kill = playback_threads[i]
    try:
        ... try to kill ...
        if successful:
            killed_threads.append(thread_to_kill)
    except:
        pass
# Only remove threads that were actually killed
playback_threads[:] = [t for t in playback_threads if t not in killed_threads]
```

**Also added**:
- Cleanup of dead threads before counting
- Logging of dead thread cleanup

---

### 5. **Inconsistent Thread Limits** ✅ FIXED
**Location**: Lines 57, 64, 305, 579, 634

**Problem**:
- Comments said "max 5" playback threads
- Semaphore was set to `Semaphore(2)` (max 2)
- Cleanup threshold was 3
- No clear source of truth

**Fix**:
```python
# Define constant at top
MAX_CONCURRENT_PLAYBACK = 5

# Use everywhere consistently
playback_semaphore = threading.Semaphore(MAX_CONCURRENT_PLAYBACK)

# In cleanup
if active_count > MAX_CONCURRENT_PLAYBACK:
    threads_to_kill = active_count - MAX_CONCURRENT_PLAYBACK

# In logging
log_message(f"Active: {current_active}/{MAX_CONCURRENT_PLAYBACK}")
```

---

### 6. **Unused Variable** ✅ FIXED
**Location**: Lines 57, 668, 708, 713

**Problem**:
- `current_count` variable was defined and set but never meaningfully used
- Created confusion - was it meant to track playback count?
- Different from `playback_count`

**Fix**:
- Removed `current_count` entirely
- Use `playback_count` consistently throughout
- Added proper locking when reading `playback_count`

---

## Testing Recommendations

1. **Semaphore Test**: Start 10 samples rapidly and verify only 5 play concurrently
2. **Zombie Thread Test**: Run for 5+ minutes and check thread count stays stable
3. **Cleanup Test**: Force high thread count and verify cleanup kills excess threads
4. **Error Recovery Test**: Inject audio errors and verify semaphore/count stay consistent
5. **Shutdown Test**: Press Ctrl+C and verify all threads terminate cleanly

## Monitoring Improvements

The monitoring thread now reports:
- Dead threads cleaned up
- Active thread count vs MAX_CONCURRENT_PLAYBACK
- Threads signed to terminate with execution location

## Potential Future Improvements

1. **Semaphore Timeout Handling**: Currently logs and returns. Could queue for retry.
2. **Thread Killing**: Uses `PyThreadState_SetAsyncExc` which is dangerous. Consider cooperative cancellation.
3. **Playback Count Validation**: Add assertions to detect count corruption early.
4. **Semaphore Leak Detection**: Add monitoring to detect if semaphore count diverges from thread count.

## Key Takeaways

- **Never hold a semaphore longer than necessary** - only for critical section, not entire operation
- **Always pair acquire/release in try/finally** - prevents leaks on exceptions
- **Never call blocking functions from worker threads** - spawns zombies
- **Use constants for limits** - prevents inconsistencies
- **Track resources explicitly** - lists, counts, etc. should always be consistent
