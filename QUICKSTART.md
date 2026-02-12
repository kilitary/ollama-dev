# JAM2 - Quick Start Guide

## ✅ Implementation Complete!

### What's Been Built

A Python script that creates audio chaos with:
- **Random volume changes** every 0.01-0.2 seconds (0% to 60%)
- **Random pitch shifting** using professional audio processing (-12 to +12 semitones)
- **Random speed changes** independent of pitch (0.5x to 2.0x)
- **Random playback rate** for additional speed variation (0.5x to 2.5x)
- **Continuous playback** with no empty timeframes
- **102 MP3 samples** already available for playback

### Key Improvements from Original Request

1. ✅ **Infinite loop** - continuously runs until Ctrl+C
2. ✅ **Random volume** - changes Windows master volume randomly
3. ✅ **Random timing** - 0.01 to 0.2 second intervals
4. ✅ **MP3 playback** - replaced beeps with MP3 samples
5. ✅ **Pitch manipulation** - professional pitch shifting (-12 to +12 semitones)
6. ✅ **Speed variation** - independent time stretching (0.5x to 2.0x)
7. ✅ **Playback rate** - additional random rate change (0.5x to 2.5x)
8. ✅ **Continuous playback** - no empty timeframes between samples

### Technologies Used

- **librosa** - Professional audio signal processing library
  - High-quality pitch shifting using phase vocoder
  - Time stretching without pitch artifacts
  
- **sounddevice** - Low-latency audio playback
  - Direct audio output
  - Non-blocking playback
  
- **pycaw** - Windows Core Audio API
  - System volume control
  - Real-time volume adjustment
  
- **rich** - Beautiful console output
  - Colorful status messages
  - Clear visual feedback

### Files Created

1. **jam2.py** - Main script (131 lines)
2. **test_jam2.py** - Test script to verify functionality
3. **JAM2_USAGE.md** - Comprehensive usage documentation
4. **samples/README.txt** - Quick reference for the samples directory

### Quick Test

```bash
# Test single sample with random pitch/speed/rate
python test_jam2.py
```

Output:
```
Testing with: ak47_clipinfate.mp3
Sample rate: 44100 Hz
Duration: 0.83 seconds
Pitch shift: -5.4 semitones
Time stretch: 1.06x
Playback rate: 1.86x
Processing audio...
Final duration: 0.43 seconds
Playing modified audio...
✓ Test complete!
```

### Run the Full Script

```bash
# Start the audio chaos!
python jam2.py
```

Expected output:
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
... (continuous playback with no gaps)
```

### Stop the Script

Press **Ctrl+C** to stop
- Volume automatically resets to 50%
- Clean shutdown
- All threads terminated

### Safety Features

✅ Max volume capped at 60% (prevents hearing damage)
✅ Auto volume reset on exit
✅ Error handling for file operations
✅ Graceful keyboard interrupt handling
✅ Background threading prevents freezing

### Customization

Edit these values in `jam2.py`:

```python
# Line 117: Sleep interval (continuous playback)
sleep_time = random.uniform(0.01, 0.2)  # Change to (0.2, 0.8) for slower

# Line 120: Volume range  
new_volume = random.uniform(0.0, 0.6)   # Change to (0.0, 1.0) for full range

# Line 61: Pitch range
pitch_shift_semitones = random.uniform(-12, 12)  # Change to (-24, 24) for 2 octaves

# Line 64: Time stretch range
time_stretch_rate = random.uniform(0.5, 2.0)     # Change to (0.25, 4.0) for extreme

# Line 67: Playback rate range (NEW!)
playback_rate = random.uniform(0.5, 2.5)         # Change to (0.25, 4.0) for extreme speed
```

### Troubleshooting

**No audio?**
- Check that MP3 files are in `samples/` directory
- Verify sounddevice installation: `pip install sounddevice`

**Errors loading MP3?**
- Install ffmpeg (librosa dependency for MP3 support)
- Check file isn't corrupted

**Volume not changing?**
- Run as administrator if needed
- Check pycaw installation: `pip install pycaw`

### Next Steps

1. Add more MP3 samples to `samples/` directory
2. Run `python jam2.py` to start the chaos
3. Adjust parameters to your preference
4. Press Ctrl+C when done

**Enjoy the randomized audio madness! 🎵🎲🔊**
