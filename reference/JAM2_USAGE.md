# JAM2 - Random Volume & Audio Chaos Generator

## Overview
JAM2 is a Python script that creates audio chaos by randomly changing Windows system volume and playing MP3 samples with randomized pitch and speed variations.

## Features

### 1. **Random Volume Control**
- Changes master volume every 0.01-0.2 seconds
- Volume range: 0.0 to 0.6 (0% to 60%)
- Uses pycaw library for Windows audio control

### 2. **Randomized MP3 Playback**
- Plays random samples from the `samples/` directory
- **Pitch Shifting**: -12 to +12 semitones (±1 octave)
- **Time Stretching**: 0.5x to 2.0x speed (independent of pitch)
- **Playback Rate**: 0.5x to 2.5x additional speed variation
- **Continuous playback**: No empty timeframes between samples
- Each playback is unique with different pitch, speed, and rate

### 3. **Fallback System**
- If no MP3 files are found, plays random beeps
- Beep frequency: 200-2000 Hz
- Beep duration: 10-250 ms

## Technologies Used

- **librosa**: Professional audio signal processing library for pitch shifting and time stretching
- **sounddevice**: Low-latency audio playback
- **numpy**: Audio array manipulation
- **pycaw**: Windows Core Audio API for volume control
- **rich**: Colorful console output

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- librosa>=0.11.0
- sounddevice>=0.5.0
- numpy>=2.3.0
- pycaw>=20240210
- rich>=13.7.1
- comtypes>=1.4.4

## Usage

1. **Add MP3 samples** to the `samples/` directory
2. **Run the script**:
   ```bash
   python jam2.py
   ```
3. **Stop with Ctrl+C** (automatically resets volume to 50%)

## Example Output

```
Starting JAM mode...
Press Ctrl+C to stop
Found 102 MP3 sample(s)
Volume: 0.42
Playing: sample.mp3 | Pitch: +7.3st | Speed: 1.45x | Rate: 2.13x
Volume: 0.15
Playing: another.mp3 | Pitch: -4.2st | Speed: 0.87x | Rate: 0.68x
Volume: 0.58
Playing: sound.mp3 | Pitch: +11.8st | Speed: 1.92x | Rate: 1.47x
... (continuous playback)
```

## How It Works

### Audio Processing Pipeline

1. **Load MP3**: librosa loads the audio file in mono
2. **Time Stretch**: Applies speed change (0.5x-2.0x) without affecting pitch
3. **Pitch Shift**: Applies pitch change (-12 to +12 semitones) independently
4. **Playback Rate**: Adjusts sample rate for additional speed variation (0.5x-2.5x)
5. **Playback**: Plays the processed audio in a background thread (non-blocking)
6. **Volume Change**: Randomly adjusts system volume continuously
7. **Continuous Loop**: Immediately triggers next sample (no gaps)

### Technical Details

- **Pitch Shifting**: Uses librosa's phase vocoder for high-quality pitch shifts
- **Time Stretching**: Uses librosa's time stretching algorithm (preserves pitch)
- **Threading**: Audio playback runs in daemon threads to prevent blocking
- **Non-blocking**: Script continues immediately after starting playback

## Safety Features

- Maximum volume capped at 60% to prevent hearing damage
- Automatic volume reset to 50% on exit
- Error handling for file loading and playback issues
- Graceful shutdown with Ctrl+C

## Customization

Edit `jam2.py` to customize:

```python
# Change sleep time between actions (continuous playback)
sleep_time = random.uniform(0.01, 0.2)  # Current: 10-200ms

# Change volume range
new_volume = random.uniform(0.0, 0.6)  # Current: 0-60%

# Change pitch range
pitch_shift_semitones = random.uniform(-12, 12)  # Current: ±1 octave

# Change time stretch range
time_stretch_rate = random.uniform(0.5, 2.0)  # Current: half to double speed

# Change playback rate range (NEW!)
playback_rate = random.uniform(0.5, 2.5)  # Current: half to 2.5x speed
```

## Warning ⚠️

This script will:
- Randomly change your system volume
- Play audio samples at random pitches and speeds
- Create potentially chaotic audio output

**Use responsibly and be mindful of your audio setup!**

## License

Copyright (c) 2025 kilitary@gmail.com

---

**Enjoy the chaos! 🎵🎲**
