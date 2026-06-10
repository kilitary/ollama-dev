# Thread Runtime Monitoring Feature

## Summary
Added thread runtime tracking to the monitoring loop in jam2.py. Each thread now displays how long it has been running in seconds.

## Changes Made

### 1. Added Thread Start Time Tracking
```python
thread_start_times = {}  # Track when threads started (thread_id -> start_time)
thread_start_times_lock = threading.Lock()
```

### 2. Updated monitoring_loop()
- Calculates runtime for each thread by comparing current time to start time
- Displays runtime in thread status output: `ThreadName(daemon,alive,5.3s)`
- Automatically cleans up start times for dead threads

### 3. Updated Thread Functions
All thread functions now record their start time:
- `playback_thread()` - Records start time and cleans up on exit
- `volume_control_loop()` - Records start time on startup
- `monitoring_loop()` - Records start time on startup  
- `thread_cleanup_loop()` - Records start time on startup

## Output Format

### Before:
```
🔷 Thread details: MainThread(main,alive)
                   Thread-1(daemon,alive)
                   Thread-2(daemon,alive)
```

### After:
```
🔷 Thread details: MainThread(main,alive)
                   Thread-1(daemon,alive,12.5s)
                   Thread-2(daemon,alive,3.2s)
```

## Benefits

1. **Zombie Thread Detection**: Long-running threads stand out (e.g., 300s+ for a playback thread)
2. **Performance Monitoring**: See how long threads are taking to complete
3. **Debug Information**: Identify stuck threads at a glance
4. **Resource Tracking**: Correlate thread lifetime with memory usage

## Thread Lifecycle Example

```
Time 0.0s: Thread starts → recorded in thread_start_times
Time 2.5s: Monitoring shows "Playback-sample.mp3(daemon,alive,2.5s)"
Time 5.0s: Monitoring shows "Playback-sample.mp3(daemon,alive,5.0s)"
Time 6.2s: Thread finishes → removed from thread_start_times
```

## Automatic Cleanup

The monitoring loop automatically cleans up start times for dead threads to prevent memory leaks:

```python
with thread_start_times_lock:
    active_thread_ids = {t.ident for t in all_threads if t.is_alive()}
    dead_thread_ids = set(thread_start_times.keys()) - active_thread_ids
    for dead_id in dead_thread_ids:
        del thread_start_times[dead_id]
```

## Edge Cases Handled

1. **Thread fails to acquire semaphore**: Start time is recorded and then cleaned up immediately
2. **Thread crashes**: Finally block ensures start time is removed
3. **Monitoring thread itself**: Shows its own runtime
4. **Thread created before feature added**: Shows no runtime (missing from dict)

## Usage

Just run the program normally. The monitoring thread will automatically display runtimes every 2 seconds.

```bash
python jam2.py
```

Look for the thread details section in the output to see runtimes.
