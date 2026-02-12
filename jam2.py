#  Copyright (c) 2025. kilitary@gmail.com
import sys
import traceback

# location: russia
# date: 2020-03-01
# so liar

# d at e: la ter
# secret: se c ret
# for: quiery
# type: help
# variant: 1-2-3-4

# uninstruct io ns
import winsound
# import
import random
import time
import math
import threading
import os
import glob
import gc
from pathlib import Path

from rich import print as rprint
from rich.console import Console
# import "audo control"
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import librosa
import sounddevice as sd
import numpy as np

console = Console()
start_time = time.time()
last_message_time = start_time  # Track time of last log message
# Sample directory
SAMPLE_DIR = Path(__file__).parent / "samples"
SAMPLE_DIR.mkdir(exist_ok=True)

# volume control: get speakers
device = AudioUtilities.GetSpeakers()
volumer = device.EndpointVolume
get_mute = volumer.GetMute
set_mute = volumer.SetMute
volume_stated = max(0.4, abs(int(volumer.GetMasterVolumeLevelScalar())) + 0.01)

# Shared playback/volume state
volume_thread_running = threading.Event()
periodic_sample_thread_running = threading.Event()
playback_count = 0
playback_count_lock = threading.Lock()
playback_semaphore = threading.Semaphore(5)
console_lock = threading.Lock()
elapsed_trigger_event = threading.Event()  # Signal to queue new sample when 2/3 of current sample elapsed

# Emoji banner for startup logs
ASCII_ART_BANNER = "🎵 🎧 JAM2 🎸 🎹"

# Icon rules inferred from rich color tags in log messages
ICON_RULES = [
    ("[bold red]", "❌"),
    ("[red]", "❌"),
    ("[bold yellow]", "⚠️"),
    ("[yellow]", "⚠️"),
    ("[bold green]", "✅"),
    ("[green]", "✅"),
    ("[cyan]", "ℹ️"),
    ("[magenta]", "🔄"),
]


def _infer_icon(message):
    for tag, icon in ICON_RULES:
        if tag in message:
            return icon
    return ""


def log_banner():
    """Print a small ASCII art banner once at startup."""
    console.print(ASCII_ART_BANNER)


def log_message(message, icon=""):
    """Prepends absolute and relative time deltas to a log message with optional icon."""
    global last_message_time
    current_time = time.time()
    absolute_delta = current_time - start_time
    relative_delta = current_time - last_message_time
    resolved_icon = icon or _infer_icon(message)
    icon_prefix = f"{resolved_icon} " if resolved_icon else ""
    with console_lock:
        console.print(f"[dim][{absolute_delta:8.3f}s[/dim] [cyan]*{relative_delta:7.3f}s[/cyan]] {icon_prefix}{message}")
        last_message_time = current_time


def log_exception(e, context=""):
    """Log exception with line number and traceback information."""
    tb = traceback.extract_tb(e.__traceback__)

    # Print main error message
    log_message(f"[red]{context}{type(e).__name__}: {e}[/red]")

    # Print formatted stack trace
    if tb:
        log_message("[red]Stack trace:[/red]")

        # Get the actual traceback object for frame inspection
        tb_obj = e.__traceback__
        frame_locals_list = []

        # Collect locals from each frame
        current_tb = tb_obj
        while current_tb is not None:
            frame_locals_list.append(current_tb.tb_frame.f_locals.copy())
            current_tb = current_tb.tb_next

        # Format each frame
        for idx, frame in enumerate(tb):
            filename = os.path.basename(frame.filename)
            line_no = frame.lineno
            func_name = frame.name

            # Get function parameters from locals if available
            params_str = ""
            return_str = ""

            if idx < len(frame_locals_list):
                locals_dict = frame_locals_list[idx]

                # Extract function parameters (exclude special variables)
                params = []
                for key, value in locals_dict.items():
                    if not key.startswith('_') and key not in ['self', 'cls']:
                        # Shorten the value representation
                        value_str = str(value)
                        if len(value_str) > 30:
                            value_str = value_str[:27] + "..."
                        params.append(f"{key}={value_str}")

                if params:
                    params_str = ", ".join(params[:3])  # Limit to first 3 params
                    if len(locals_dict) > 3:
                        params_str += ", ..."

            # Format: line_number:func_name(params)-> return_value
            trace_line = f"  {filename}:{line_no}:{func_name}({params_str})"
            log_message(f"[red]{trace_line}[/red]")

            # Show the actual code line if available
            if frame.line:
                log_message(f"[dim red]    > {frame.line.strip()}[/dim red]")

    if random.randint(1, 6) <= 2:
        log_message("[red]вще похуй[/red]")


def volume_control_loop():
    global volume_stated
    """Thread function for continuous volume changes"""
    log_message(f"[cyan]Volume control thread started: maxvol={volume_stated}[/cyan]")
    try:
        while volume_thread_running.is_set():
            # Random volume level (0.0 to 0.6)
            new_volume = random.randrange(int(volume_stated * 100)) * 0.01
            log_message(f"[yellow]Setting volume: {new_volume:.2f}[/yellow]")
            volumer.SetMasterVolumeLevelScalar(new_volume, None)

            # Random sleep time between 0.01 and 0.1 seconds
            sleep_time = random.uniform(0.01, 0.1)
            time.sleep(sleep_time)
    except Exception as e:
        log_exception(e, "Volume control error: ")
        os.abort()


def periodic_sample_loop():
    """Thread function for playing random samples at regular intervals (0.1-0.2s)"""
    log_message(f"[yellow]Periodic sample thread started[/yellow]")
    try:
        while periodic_sample_thread_running.is_set():
            # Check if we can add another sample (max 5)
            with playback_count_lock:
                can_play = playback_count < 5

            if can_play:
                log_message(f"[cyan]Periodic trigger - queueing new sample[/cyan]")
                play_random_sample()
            else:
                log_message(f"[yellow]Periodic trigger skipped - max samples reached ({playback_count}/5)[/yellow]")

            # Random sleep time between 0.1 and 0.2 seconds
            sleep_time = random.uniform(0.1, 0.2)
            time.sleep(sleep_time)
    except Exception as e:
        log_exception(e, "Periodic sample error: ")


def get_mp3_files():
    """Get all MP3 files from the samples directory"""
    mp3_files = list(SAMPLE_DIR.glob("*.mp3"))
    return mp3_files


def safe_time_stretch(audio, rate):
    """Safely apply time stretching with error handling"""
    try:
        # Clamp time stretch rate to safe values
        rate = max(0.5, min(2.0, rate))

        # Create a copy to avoid memory issues
        audio_copy = np.array(audio, dtype=np.float32, copy=True)

        # Apply time stretching with conservative settings
        stretched = librosa.effects.time_stretch(audio_copy, rate=rate)

        # Ensure result is contiguous
        stretched = np.ascontiguousarray(stretched, dtype=np.float32)

        return stretched
    except Exception as e:
        log_exception(e, "Time stretch error: ")
        return audio


def safe_pitch_shift(audio, sr, n_steps):
    """Safely apply pitch shifting with error handling"""
    try:
        # Clamp pitch shift to safe range
        n_steps = max(-12, min(12, n_steps))

        # Create a copy to avoid memory issues
        audio_copy = np.array(audio, dtype=np.float32, copy=True)

        # Apply pitch shifting
        shifted = librosa.effects.pitch_shift(audio_copy, sr=sr, n_steps=n_steps)

        # Ensure result is contiguous
        shifted = np.ascontiguousarray(shifted, dtype=np.float32)

        return shifted
    except Exception as e:
        log_exception(e, "Pitch shift error: ")
        return audio


def check_initial_volume(audio, sr, duration_seconds=0.1, threshold=0.01):
    """Check if audio has sufficient volume in the first duration_seconds"""
    try:
        # Calculate number of samples for the specified duration
        num_samples = int(sr * duration_seconds)

        # Get the first chunk of audio
        initial_chunk = audio[:num_samples]
        if initial_chunk.size == 0:
            log_message("[yellow]Empty audio buffer during volume check[/yellow]")
            return False

        # Calculate RMS (root mean square) energy
        rms_energy = np.sqrt(np.mean(initial_chunk ** 2))

        # Check if volume is above threshold
        if rms_energy < threshold:
            log_message(f"[yellow]Low initial volume detected (RMS: {rms_energy:.4f})[/yellow]")
            return False

        log_message(f"[green]Good initial volume (RMS: {rms_energy:.4f})[/green]")
        return True

    except Exception as e:
        log_exception(e, "Volume check error: ")
        return True  # Allow on error


def trim_silence_end(audio, sr, threshold=0.01, min_duration=0.1):
    """Trim silent audio data from the end of the sample"""
    try:
        # Calculate RMS energy for the entire audio
        if audio.size == 0:
            log_message("[yellow]Empty audio buffer during trim[/yellow]")
            return audio

        rms_energy = np.sqrt(np.mean(audio ** 2))

        # Calculate silence threshold (percentage of max RMS)
        silence_threshold = rms_energy * threshold

        # Find the last sample above the silence threshold
        # Work backwards from the end
        for i in range(len(audio) - 1, -1, -1):
            if abs(audio[i]) > silence_threshold:
                # Found last audible sample, add a small buffer
                buffer_samples = int(sr * 0.05)  # 50ms buffer
                end_index = min(i + buffer_samples, len(audio))

                trimmed_audio = audio[:end_index]
                trimmed_duration = len(trimmed_audio) / sr
                min_duration_samples = int(sr * min_duration)

                # Ensure minimum duration
                if len(trimmed_audio) < min_duration_samples:
                    trimmed_audio = audio[:min_duration_samples]
                    trimmed_duration = len(trimmed_audio) / sr

                original_duration = len(audio) / sr
                removed_duration = original_duration - trimmed_duration

                log_message(f"[cyan]Trimmed silence: {removed_duration:.3f}s (original: {original_duration:.3f}s, trimmed: {trimmed_duration:.3f}s)[/cyan]")
                return np.ascontiguousarray(trimmed_audio, dtype=np.float32)

        # If no audible samples found, return first min_duration
        min_samples = int(sr * min_duration)
        return np.ascontiguousarray(audio[:min_samples], dtype=np.float32)

    except Exception as e:
        log_exception(e, "Trim silence error: ")
        return audio


def prepare_random_sample():
    """Prepare a random MP3 sample with random pitch and speed"""
    mp3_files = get_mp3_files()
    if not mp3_files:
        return None

    done = False
    attempts = 0
    max_attempts = 2

    while not done and attempts < max_attempts:
        attempts += 1
        sample = random.choice(mp3_files)
        try:
            log_message(f'[cyan]Selected sample: {sample.name}[/cyan]')
            audio_offset = 0

            # Load audio with error handling
            audio, sr = librosa.load(str(sample), sr=None, mono=True, offset=audio_offset)

            # Check file size
            if audio.size >= 1024 * 1024 * 2:
                log_message(f"[yellow]Skipping {sample.name} - file too large[/yellow]")
                continue
            if audio.size < 1024 * 4:
                log_message(f"[yellow]Skipping {sample.name} - file too small[/yellow]")
                continue

            log_message(f'[green]Loaded: {sample.name} | size: {audio.size} samples | sample rate: {sr} Hz[/green]')

            # Check if audio has sufficient volume in the first 0.1 seconds
            # if not check_initial_volume(audio, sr, duration_seconds=0.1, threshold=0.01):
            #     log_message(f"[yellow]Skipping {sample.name} - low initial volume[/yellow]")
            #     continue

            done = True

            # Random parameters with controlled ranges
            pitch_shift_semitones = random.uniform(-8, 8)
            time_stretch_rate = random.uniform(0.7, 1.5)
            playback_rate = random.uniform(0.6, 1.4)
            log_message(f'[green]Pitch shift: {pitch_shift_semitones:+.1f} semitones | Time stretch: {time_stretch_rate:.2f}x | Playback rate: {playback_rate:.2f}x[/green]')

            # Ensure audio is contiguous float32
            audio = np.ascontiguousarray(audio, dtype=np.float32)

            # Apply effects with safe functions
            log_message(f"[magenta]Applying time stretch: {time_stretch_rate:.2f}x[/magenta]")
            audio_stretched = safe_time_stretch(audio, time_stretch_rate)
            log_message(f'[magenta]Applied time stretch | new size: {audio_stretched.size} samples[/magenta]')

            # Clean up original audio
            del audio
            #gc.collect()

            # Apply pitch shifting
            log_message(f"[magenta]Applying pitch shift: {pitch_shift_semitones:+.1f}st[/magenta]")
            audio_final = safe_pitch_shift(audio_stretched, sr, pitch_shift_semitones)

            # Clean up intermediate
            del audio_stretched
            #gc.collect()

            # Trim silence from the end
            # log_message(f"[magenta]Trimming silence from end[/magenta]")
            # audio_final = trim_silence_end(audio_final, sr, threshold=0.01, min_duration=0.1)

            return {
                'audio': audio_final,
                'sr': sr,
                'playback_rate': playback_rate,
                'name': sample.name
            }

        except Exception as e:
            log_exception(e, f"Error processing {sample.name}: ")

    if not done:
        log_message("[yellow]Failed to prepare sample after max attempts[/yellow]")
    return None


# Global variable to track when next sample should be queued
# next_sample_queue_event = threading.Event()


def playback_thread(audio_data, sample_rate, rate, sample_name):
    """Thread function for audio playback with 2/3 elapsed trigger for new samples"""
    global playback_count

    # Acquire semaphore to limit concurrent playback threads (max 5)
    with playback_semaphore:
        with playback_count_lock:
            playback_count += 1
            log_message(f"[bold green]▶ Playback started | Active: {playback_count}/5[/bold green]")

        try:
            # Make a copy of audio data to avoid cross-thread corruption
            audio_copy = np.array(audio_data, dtype=np.float32, copy=True)

            adjusted_sr = int(sample_rate * rate)
            # Ensure adjusted sample rate is valid
            adjusted_sr = max(8000, min(192000, adjusted_sr))

            duration_seconds = len(audio_copy) / adjusted_sr

            log_message(
                f"[magenta]Playing: {sample_name} | Rate: {rate:.2f}x | Duration: {duration_seconds:.3f}s[/magenta]")

            # Chunked playback with 2/3 elapsed trigger
            chunk_duration = 0.1  # 100ms chunks
            chunk_size = int(adjusted_sr * chunk_duration)
            trigger_position = int(len(audio_copy) * 2 / 3)  # 2/3 of total length
            triggered = False  # Only trigger once per sample

            # Use sounddevice OutputStream for chunked playback
            position = 0

            with sd.OutputStream(samplerate=adjusted_sr, channels=1, dtype='float32') as stream:
                while position < len(audio_copy):
                    # Get next chunk
                    end_position = min(position + chunk_size, len(audio_copy))
                    chunk = audio_copy[position:end_position]

                    # Check if we've passed 2/3 of the sample and haven't triggered yet
                    if not triggered and position >= trigger_position:
                        elapsed_pct = (position / len(audio_copy)) * 100
                        log_message(f"[cyan]2/3 elapsed in {sample_name} ({elapsed_pct:.1f}%) - triggering next sample[/cyan]")
                        triggered = True
                        elapsed_trigger_event.set()  # Signal to queue new sample

                    # Write chunk to output stream
                    stream.write(chunk.reshape(-1, 1))
                    position = end_position

        except Exception as e:
            log_exception(e, "Playback error: ")
        finally:
            # Clean up audio data
            try:
                del audio_copy
                gc.collect()
            except:
                pass

            with playback_count_lock:
                playback_count -= 1
                log_message(f"[bold yellow]■ Playback finished | Active: {playback_count}/5[/bold yellow]")



def play_random_sample():
    """Play a random MP3 sample with random pitch and speed or beep if no samples found"""
    mp3_files = get_mp3_files()
    if mp3_files:
        sample_data = prepare_random_sample()
        if sample_data:
            # Play in a thread to allow main loop to continue
            thread = threading.Thread(
                target=playback_thread,
                args=(sample_data['audio'].copy(), sample_data['sr'], sample_data['playback_rate'], sample_data['name']),
                daemon=True
            )
            thread.start()


            # Clean up sample data
            del sample_data
            gc.collect()
    else:
        # Fallback to beep
        freq = random.randint(200, 2000)
        duration = random.randint(10, 250)
        log_message(f"[yellow]Beep: {freq}Hz, {duration}ms[/yellow]")
        winsound.Beep(freq, duration)


def jam_loop():
    """Infinite loop that plays samples continuously (max 5 at a time)"""
    log_message("[bold red]Starting JAM mode...[/bold red]")
    log_message(f'audioDevice->{device.FriendlyName}')
    log_message("[yellow]Press Ctrl+C to stop[/yellow]")

    mp3_files = get_mp3_files()
    if mp3_files:
        log_message(f"[green]Found {len(mp3_files)} MP3 sample(s)[/green]")
    else:
        log_message(f"[yellow]No MP3 samples found in {SAMPLE_DIR}[/yellow]")
        log_message(f"[yellow]Add .mp3 files to the samples directory for random playback[/yellow]")

    # Start volume control thread
    volume_thread_running.set()
    volume_thread = threading.Thread(target=volume_control_loop, daemon=True)
    volume_thread.start()

    # Start periodic sample thread
    # periodic_sample_thread_running.set()
    # periodic_thread = threading.Thread(target=periodic_sample_loop, daemon=True)
    # periodic_thread.start()

    try:
        while True:
            # Check if we can add another sample (max 5) - atomic check and call
            with playback_count_lock:
                can_play = playback_count < 5
                current_count = playback_count

            # Check if 2/3 elapsed trigger was fired by any playing sample
            elapsed_triggered = elapsed_trigger_event.is_set()
            if elapsed_triggered:
                elapsed_trigger_event.clear()
                if can_play:
                    log_message(f"[cyan]Queueing new sample (2/3 elapsed trigger)[/cyan]")
                    play_random_sample()

            # Start first sample if nothing is playing
            elif current_count == 0:
                play_random_sample()

            # Small delay to prevent busy-waiting
            time.sleep(0.05)

    except KeyboardInterrupt:
        # Stop volume control thread
        volume_thread_running.clear()
        # Stop periodic sample thread
        # periodic_sample_thread_running.clear()
        # log_message("\n[bold green]JAM mode stopped.[/bold green]")
        # Reset volume to 50%
        volumer.SetMasterVolumeLevelScalar(0.5, None)
        log_message("[green]Volume reset to 50%[/green]")


if __name__ == "__main__":
    log_banner()
    jam_loop()
