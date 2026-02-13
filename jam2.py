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
import psutil

from rich import print as rprint
from rich.console import Console
# import "audo control"
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import librosa
import sounddevice as sd
import numpy as np

only_volume_mut = False  # Set to True to only run volume control without playback

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
min_vol = 1110
max_vol = 0
volume_stated = max(0.6, int(volumer.GetMasterVolumeLevelScalar()) + 0.1)

# Shared playback/volume state
volume_thread_running = threading.Event()
periodic_sample_thread_running = threading.Event()
monitoring_thread_running = threading.Event()
thread_cleanup_running = threading.Event()
playback_count = 0
playback_count_lock = threading.Lock()
MAX_CONCURRENT_PLAYBACK = 5  # Maximum number of concurrent playback threads
playback_semaphore = threading.Semaphore(MAX_CONCURRENT_PLAYBACK)
console_lock = threading.Lock()
elapsed_trigger_event = threading.Event()  # Signal to queue new sample when 2/3 of current sample elapsed
playback_threads = []  # Track active playback threads
playback_threads_lock = threading.Lock()

# Emoji banner for startup logs
ASCII_ART_BANNER = "🎵 🎧 JAM2 🎸 🎹"

# Icon rules inferred from rich color tags in log messages
ICON_RULES = [
    ("[bold red]", "🚑"),
    ("[red]", "❌"),
    ("[bold yellow]", "🪱"),
    ("[yellow]", "🐍"),
    ("[bold green]", "✅"),
    ("[green]", "🐩"),
    ("[cyan]", "🪶"),
    ("[magenta]", "🐀"),
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

    if random.randint(1, 4) < 2:
        log_message("[red]вще похуй[/red]")


def volume_control_loop():
    global volume_stated, min_vol, max_vol

    """Thread function for continuous volume changes"""
    log_message(f"[cyan]Volume control thread started: maxvol={volume_stated}[/cyan]")
    try:
        while volume_thread_running.is_set():
            # Random volume level (0.0 to 0.6)
            new_volume = random.randrange(int(volume_stated * 100)) * 0.01

            if new_volume <= 0:
                new_volume = 0.01

            if new_volume > max_vol:
                max_vol = new_volume
            if new_volume < min_vol:
                min_vol = new_volume
            log_message(f"[yellow]Setting volume: {new_volume:.2f} (min: {min_vol:.2f}, max: {max_vol:.2f})[/yellow]")
            volumer.SetMasterVolumeLevelScalar(new_volume, None)

            # Random sleep time between 0.01 and 0.1 seconds
            sleep_time = random.uniform(0.1, 0.4)
            time.sleep(sleep_time)
    except Exception as e:
        log_exception(e, "Volume control error: ")
        os.abort()


def monitoring_loop():
    """Thread function for monitoring memory and CPU usage every 5 seconds"""
    log_message(f"[cyan]Monitoring thread started[/cyan]")
    process = psutil.Process(os.getpid())

    try:
        while monitoring_thread_running.is_set():
            # Get memory info
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024  # Convert to MB

            # Get CPU usage (cumulative)
            cpu_percent = process.cpu_percent(interval=0.1)

            # Get thread count
            thread_count = threading.active_count()

            # Enumerate all threads and check termination flags
            all_threads = threading.enumerate()
            thread_status = []
            threads_to_terminate = []

            for t in all_threads:
                thread_name = t.name if t.name else f"Thread-{t.ident}"
                is_alive = "alive" if t.is_alive() else "dead"
                is_daemon = "daemon" if t.daemon else "main"
                thread_status.append(f"{thread_name}({is_daemon},{is_alive})")

                # Check if this thread is signed to terminate
                should_terminate = False
                if "volume" in thread_name.lower() and not volume_thread_running.is_set():
                    should_terminate = True
                elif "periodic" in thread_name.lower() and not periodic_sample_thread_running.is_set():
                    should_terminate = True
                elif "monitoring" in thread_name.lower() and not monitoring_thread_running.is_set():
                    should_terminate = True
                elif "cleanup" in thread_name.lower() and not thread_cleanup_running.is_set():
                    should_terminate = True

                if should_terminate and t.is_alive():
                    # Get frame information for this thread
                    try:
                        frame = sys._current_frames().get(t.ident)
                        if frame:
                            func_name = frame.f_code.co_name
                            line_no = frame.f_lineno
                            filename = os.path.basename(frame.f_code.co_filename)
                            ip_pointer = id(frame.f_code)  # Instruction pointer approximation
                            threads_to_terminate.append(
                                f"{thread_name} @ {filename}:{line_no} in {func_name}() [IP:0x{ip_pointer:X}]"
                            )
                        else:
                            threads_to_terminate.append(f"{thread_name} [no frame info]")
                    except Exception as e:
                        threads_to_terminate.append(f"{thread_name} [error: {e}]")

            # Check which thread control flags are set to terminate
            termination_flags = []
            if not volume_thread_running.is_set():
                termination_flags.append("volume")
            if not periodic_sample_thread_running.is_set():
                termination_flags.append("periodic")
            if not monitoring_thread_running.is_set():
                termination_flags.append("monitoring")
            if not thread_cleanup_running.is_set():
                termination_flags.append("cleanup")

            termination_status = f" | Terminating: {', '.join(termination_flags)}" if termination_flags else ""

            # Log the monitoring data
            log_message(
                f"[bold green]Memory: {mem_mb:.2f} MB | CPU: {cpu_percent:.1f}% | Threads: {thread_count}{termination_status}[/bold green]",
                icon="📊"
            )

            # Log thread details
            log_message(f"[dim green]🩳 Thread details: " + f'{"\n\t 🥣".join(thread_status)}' + "[/dim green]")

            # Log threads signed to terminate with their current execution position
            if threads_to_terminate:
                log_message(f"[bold red]Threads signed to terminate:[/bold red]")
                for term_info in threads_to_terminate:
                    log_message(f"[red]  └─ {term_info}[/red]")

            # Sleep for 5 seconds
            time.sleep(2.0)

    except Exception as e:
        log_exception(e, "Monitoring error: ")


def thread_cleanup_loop():
    """Thread function to cleanup excess playback threads every second"""
    log_message(f"[cyan]Thread cleanup loop started[/cyan]")
    import ctypes

    try:
        while thread_cleanup_running.is_set():
            time.sleep(1.0)

            with playback_threads_lock:
                # Remove dead threads from the list
                alive_threads = [t for t in playback_threads if t.is_alive()]
                dead_count = len(playback_threads) - len(alive_threads)

                if dead_count > 0:
                    log_message(f"[dim yellow]Cleaned up {dead_count} finished thread(s)[/dim yellow]")

                playback_threads[:] = alive_threads
                active_count = len(playback_threads)

                if active_count > MAX_CONCURRENT_PLAYBACK:
                    # Kill oldest threads (first in list) to bring count down to MAX_CONCURRENT_PLAYBACK
                    threads_to_kill = active_count - MAX_CONCURRENT_PLAYBACK
                    log_message(
                        f"[bold red]🔪 Killing {threads_to_kill} excess playback thread(s) ({active_count} -> {MAX_CONCURRENT_PLAYBACK})[/bold red]")

                    killed_threads = []
                    for i in range(threads_to_kill):
                        thread_to_kill = playback_threads[i]
                        try:
                            thread_id = thread_to_kill.ident
                            if thread_id:
                                # Terminate thread forcefully
                                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                    ctypes.c_long(thread_id),
                                    ctypes.py_object(SystemExit)
                                )
                                if res > 1:
                                    # If it returns a number greater than one, you're in trouble
                                    ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
                                    log_message(f"[red]Failed to kill thread {thread_id}[/red]")
                                else:
                                    log_message(f"[yellow]Killed thread {thread_id}[/yellow]")
                                    killed_threads.append(thread_to_kill)
                        except Exception as e:
                            log_exception(e, "Thread kill error: ")

                    # Remove killed threads from list
                    playback_threads[:] = [t for t in playback_threads if t not in killed_threads]

    except Exception as e:
        log_exception(e, "Thread cleanup error: ")


def periodic_sample_loop():
    """Thread function for playing random samples at regular intervals (0.1-0.2s)"""
    log_message(f"[yellow]Periodic sample thread started[/yellow]")
    try:
        while periodic_sample_thread_running.is_set():
            # Check if we can add another sample (max 5)
            with playback_count_lock:
                can_play = playback_count <= 3

            if can_play:
                log_message(f"[cyan]Periodic trigger - queueing new sample[/cyan]")
                play_random_sample()
            else:
                log_message(f"[yellow]Periodic trigger skipped - max samples reached ({playback_count}/5)[/yellow]")

            # Random sleep time between 0.1 and 0.2 seconds
            sleep_time = random.uniform(0.01, 0.4)
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
            if audio.size >= 1024 * 1024 * 6:
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
            log_message(
                f'[green]Pitch shift: {pitch_shift_semitones:+.1f} semitones | Time stretch: {time_stretch_rate:.2f}x | Playback rate: {playback_rate:.2f}x[/green]')

            # Ensure audio is contiguous float32
            audio = np.ascontiguousarray(audio, dtype=np.float32)

            # Apply effects with safe functions
            log_message(f"[magenta]Applying time stretch: {time_stretch_rate:.2f}x[/magenta]")
            audio_stretched = safe_time_stretch(audio, time_stretch_rate)
            log_message(f'[magenta]Applied time stretch | new size: {audio_stretched.size} samples[/magenta]')

            # Clean up original audio
            del audio
            # gc.collect()

            # Apply pitch shifting
            log_message(f"[magenta]Applying pitch shift: {pitch_shift_semitones:+.1f}st[/magenta]")
            audio_final = safe_pitch_shift(audio_stretched, sr, pitch_shift_semitones)

            # Clean up intermediate
            del audio_stretched
            # gc.collect()

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

    # Try to acquire semaphore to limit concurrent playback threads
    semaphore_acquired = playback_semaphore.acquire(blocking=True, timeout=5.0)

    if not semaphore_acquired:
        log_message(f"[red]Failed to acquire semaphore for {sample_name} - skipping playback[/red]")
        return

    try:
        with playback_count_lock:
            playback_count += 1
            current_active = playback_count

        log_message(
            f"[bold green]▶ Playback started: {sample_name} | Active: {current_active}/{MAX_CONCURRENT_PLAYBACK}[/bold green]")

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
                        # Signal to queue new sample (don't call the function directly - that's a bug!)
                        elapsed_trigger_event.set()
                        # Queue a new sample if we're below the limit
                        with playback_count_lock:
                            if playback_count < MAX_CONCURRENT_PLAYBACK:
                                # Use threading to avoid blocking this playback
                                threading.Thread(target=play_random_sample, daemon=True).start()

                    # Write chunk to output stream
                    stream.write(chunk.reshape(-1, 1))
                    position = end_position

        except Exception as e:
            log_exception(e, "Playback error: ")
        finally:
            # Clean up audio data
            try:
                del audio_copy
                if random.randint(1, 5) < 3:
                    gc.collect()
            except:
                pass

    finally:
        # CRITICAL: Always release semaphore and decrement count, even on error
        playback_semaphore.release()

        with playback_count_lock:
            playback_count -= 1
            current_active = playback_count

        log_message(
            f"[bold yellow]■ Playback finished: {sample_name} | Active: {current_active}/{MAX_CONCURRENT_PLAYBACK}[/bold yellow]")


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

            # Track this thread
            with playback_threads_lock:
                playback_threads.append(thread)

            thread.start()

            # Clean up sample data
            del sample_data
            # gc.collect()
    else:
        # Fallback to beep
        freq = random.randint(200, 2000)
        duration = random.randint(10, 250)
        log_message(f"[yellow]Beep: {freq}Hz, {duration}ms[/yellow]")
        winsound.Beep(freq, duration)


def jam_loop():
    """Infinite loop that plays samples continuously (max concurrent based on MAX_CONCURRENT_PLAYBACK)"""
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

    # Start monitoring thread
    monitoring_thread_running.set()
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()

    # Start thread cleanup thread
    thread_cleanup_running.set()
    cleanup_thread = threading.Thread(target=thread_cleanup_loop, daemon=True)
    cleanup_thread.start()

    # Start periodic sample thread
    # periodic_sample_thread_running.set()
    # periodic_thread = threading.Thread(target=periodic_sample_loop, daemon=True)
    # periodic_thread.start()

    try:
        # Kick off initial playback
        play_random_sample()

        while True:
            if only_volume_mut:
                time.sleep(1)
                continue

            # Check if we need to queue more samples
            with playback_count_lock:
                current_active = playback_count

            # If no samples are playing, start one
            if current_active == 0:
                log_message("[yellow]No samples playing - starting new sample[/yellow]")
                play_random_sample()

            # Small delay to prevent busy-waiting
            time.sleep(0.05)

    except KeyboardInterrupt:
        log_message("\n[bold red]Shutting down...[/bold red]")

        # Stop volume control thread
        volume_thread_running.clear()
        # Stop monitoring thread
        monitoring_thread_running.clear()
        # Stop thread cleanup thread
        thread_cleanup_running.clear()
        # Stop periodic sample thread
        # periodic_sample_thread_running.clear()

        # Wait a bit for threads to finish
        time.sleep(0.5)

        # Reset volume to 50%
        volumer.SetMasterVolumeLevelScalar(0.5, None)
        log_message("[green]Volume reset to 50%[/green]")
        log_message("[bold green]JAM mode stopped.[/bold green]")


if __name__ == "__main__":
    log_banner()
    jam_loop()
