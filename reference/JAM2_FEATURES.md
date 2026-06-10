# JAM2 Features Summary

## Overview
JAM2 is an advanced audio playback system with multiple concurrent sample playback, dynamic effects, and automatic triggering mechanisms.

## Current Configuration
- **Max Concurrent Samples**: 5
- **Periodic Trigger Interval**: 0.1 to 0.2 seconds (random)
- **2/3 Elapsed Trigger**: Enabled
- **Volume Control**: Random (10% - 50%)

## Main Features

### 1. Multi-Threaded Sample Playback
- Up to 5 samples can play concurrently
- Thread-safe with semaphore limiting (`playback_semaphore = Semaphore(5)`)
- Each playback runs in its own thread with proper memory isolation

### 2. Triple Trigger System

#### A. Periodic Trigger Thread
```python
periodic_sample_loop()
```
- **Independent background thread** that runs continuously
- Queues new samples every **0.1 to 0.2 seconds** (random interval)
- Respects the 5 concurrent sample limit
- Logs when max samples reached

#### B. 2/3 Elapsed Trigger
```
playback_thread() - line 354-359
```
- Each playing sample monitors its playback position
- When a sample reaches **66.7% (2/3) completion**:
  - Sets `elapsed_trigger_event`
  - Main loop detects and queues new sample
  - Allows smooth overlapping of samples

#### C. Zero-Playback Fallback
- If no samples are playing (`current_count == 0`), immediately starts one
- Ensures continuous audio output

### 3. Audio Processing Pipeline

#### Safe Time Stretching
```python
safe_time_stretch(audio, rate)
```
- Rate range: 0.5x to 2.0x
- Random application: 0.7x to 1.5x
- Memory-safe with array copying
- Error handling with fallback to original

#### Safe Pitch Shifting
```python
safe_pitch_shift(audio, sr, n_steps)
```
- Semitone range: -12 to +12
- Random application: -8 to +8 semitones
- Memory-safe with array copying
- Error handling with fallback to original

#### Playback Rate Variation
- Range: 0.6x to 1.4x
- Applied on top of time stretching

### 4. Audio Quality Controls

#### Initial Volume Check
```python
check_initial_volume(audio, sr, duration=0.1, threshold=0.01)
```
- Checks first 100ms of audio
- RMS threshold: 0.01
- Skips samples with low initial volume

#### Silence Trimming
```python
trim_silence_end(audio, sr, threshold=0.01, min_duration=0.1)
```
- Removes silent portions from end
- Maintains minimum 100ms duration
- 50ms buffer after last audible sample

### 5. Memory Management (0xC0000005 Fixes)

#### Thread-Safe Audio Copying
- Each playback thread gets its own copy: `audio_copy = np.array(audio_data, copy=True)`
- Prevents cross-thread memory corruption

#### Explicit Cleanup
```python
del audio_copy
gc.collect()
```
- Called after each playback
- Called after audio processing steps
- Prevents memory leaks

#### Conservative Parameters
- File size: 512 bytes to 2MB
- Sample rate: 8000 Hz to 192000 Hz
- Retry logic: Up to 10 attempts per sample

### 6. Dynamic Volume Control
```python
volume_control_loop()
```
- Independent background thread
- Random volume: 10% to 50%
- Update interval: 0.01 to 0.1 seconds
- Uses Windows Audio API (pycaw)

### 7. Chunked Playback
```
sd.OutputStream() - 100ms chunks
```
- Allows mid-playback triggers (2/3 elapsed)
- Smooth audio output
- Better memory management

## Thread Architecture

```
Main Thread (jam_loop)
├── Volume Control Thread
│   └── Continuous volume changes (0.01-0.1s intervals)
│
├── Periodic Sample Thread
│   └── Queue new samples (0.1-0.2s intervals)
│
└── Playback Threads (max 5)
    ├── Chunked audio output
    ├── 2/3 elapsed trigger
    └── Memory cleanup
```

## Event Flow

```
Program Start
    ↓
Start Volume Thread
    ↓
Start Periodic Thread
    ↓
Main Loop:
    ├─→ Periodic thread triggers every 0.1-0.2s
    │       └─→ Queues new sample if < 5 active
    │
    ├─→ Sample reaches 2/3 completion
    │       └─→ Triggers event → Queues new sample
    │
    └─→ If 0 samples playing
            └─→ Immediately start one
```

## File Requirements
- **Location**: `P:\ollama-dev\samples\*.mp3`
- **Size**: 512 bytes to 2MB
- **Initial volume**: RMS > 0.01 in first 100ms
- **Format**: MP3, mono converted automatically

## Log Messages
| Color i  Meaning         | Example |
|-------|------------------|---------|
| **[bold green]** | Playback started | `▶ Playback started \| Active: 3/5` |
| **[bold yellow]** | Playback finished | `■ Playback finished \| Active: 2/5` |
| **[cyan]** | Info/Trigger     | `Periodic trigger - queueing new sample` |
| **[magenta]** | Processing       | `Playing: sample.mp3 \| Rate: 1.23x` |
| **[yellow]** | Warning          | `Periodic trigger skipped - max samples reached` |
| **[red]** | Error            | `Playback error: ...` |

## Performance Characteristics
- **Sample preparation**: ~100-500ms (includes librosa processing)
- **Playback latency**: ~100ms (chunk size)
- **Memory per sample**: ~500KB - 2MB
- **CPU usage**: Moderate (5 concurrent librosa processes + playback)

## Safety Features
1. ✅ Thread-safe playback counter with locks
2. ✅ Semaphore limiting to prevent resource exhaustion
3. ✅ Memory copying to prevent cross-thread corruption
4. ✅ Explicit garbage collection
5. ✅ Error handling at every processing step
6. ✅ Sample rate validation
7. ✅ File size constraints
8. ✅ Retry logic for sample loading

## Usage
```bash
python jam2.py
```
- Press **Ctrl+C** to stop
- Volume resets to 50% on exit
- All threads terminate gracefully
