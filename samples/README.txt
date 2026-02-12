JAM2 SAMPLES DIRECTORY
=====================

Place your MP3 files in this directory for random playback.

The jam2.py script will:
1. Randomly change Windows master volume every 0.02-0.2 seconds (0.0 to 0.6)
2. Play random MP3 samples from this directory with:
   - Random pitch shift (-12 to +12 semitones = ±1 octave)
   - Random time stretch (0.5x to 2.0x - slower to faster)
3. If no MP3 files are found, it will play random beeps instead (200-2000 Hz)

Technologies used:
- librosa: Professional audio processing with pitch shifting and time stretching
- sounddevice: Low-latency audio playback
- pycaw: Windows volume control

To use:
-------
1. Copy your MP3 files (.mp3) into this directory
2. Run: python jam2.py
3. Press Ctrl+C to stop (volume will reset to 50%)

Each playback will have completely randomized pitch and speed,
making the same sample sound different every time!

Example output:
Playing: sample.mp3 | Pitch: +7.3 semitones | Speed: 1.45x
Volume: 0.42

Enjoy the chaos!




