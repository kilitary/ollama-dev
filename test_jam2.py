"""
Quick test script for JAM2 functionality
Tests a single MP3 file with random pitch, speed, and playback rate
"""
import librosa
import sounddevice as sd
import random
from pathlib import Path
from rich.console import Console

console = Console()

# Get a sample file
SAMPLE_DIR = Path(__file__).parent / "samples"
mp3_files = list(SAMPLE_DIR.glob("*.mp3"))

if mp3_files:
    sample = random.choice(mp3_files)
    console.print(f"[green]Testing with: {sample.name}[/green]")

    # Load audio
    audio, sr = librosa.load(str(sample), sr=None, mono=True)
    console.print(f"[cyan]Sample rate: {sr} Hz[/cyan]")
    console.print(f"[cyan]Duration: {len(audio)/sr:.2f} seconds[/cyan]")

    # Random parameters
    pitch_shift = random.uniform(-12, 12)
    time_stretch = random.uniform(0.5, 2.0)
    playback_rate = random.uniform(0.5, 2.5)

    console.print(f"[yellow]Pitch shift: {pitch_shift:+.1f} semitones[/yellow]")
    console.print(f"[yellow]Time stretch: {time_stretch:.2f}x[/yellow]")
    console.print(f"[yellow]Playback rate: {playback_rate:.2f}x[/yellow]")

    # Process
    console.print("[magenta]Processing audio...[/magenta]")
    audio_stretched = librosa.effects.time_stretch(audio, rate=time_stretch)
    audio_final = librosa.effects.pitch_shift(audio_stretched, sr=sr, n_steps=pitch_shift)

    # Calculate final playback characteristics
    adjusted_sr = int(sr * playback_rate)
    final_duration = len(audio_final) / adjusted_sr
    console.print(f"[cyan]Final duration: {final_duration:.2f} seconds[/cyan]")

    console.print("[green]Playing modified audio...[/green]")
    sd.play(audio_final, adjusted_sr)
    sd.wait()

    console.print("[bold green]✓ Test complete![/bold green]")
else:
    console.print("[red]No MP3 files found in samples directory[/red]")
