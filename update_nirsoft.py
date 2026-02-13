#  Copyright (c) 2025. kilitary@gmail.com

import os
import glob
import requests
import time
import rich
from rich import print
from rich import print_json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
import sys
import json
import xml.dom.minidom
import xml
import re
import random
import pefile
import shutil
from datetime import datetime
import threading
import path

console = Console()


# Stats tracking
class Stats:
    def __init__(self):
        self.bytes_downloaded = 0
        self.bytes_uploaded = 0
        self.files_unpacked = 0
        self.files_packed = 0
        self.files_x86 = 0
        self.files_amd64 = 0
        self.last_report_time = time.time()
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.current_file = ""
        self.current_file_size = 0
        self.current_file_downloaded = 0
        self.download_speeds = []  # Store recent speeds for smoothing
        self.max_speed_samples = 10
        # Enhanced tracking
        self.current_operation = ""
        self.current_package_name = ""
        self.current_url = ""
        self.files_failed = 0
        self.files_skipped = 0
        self.total_errors = 0
        self.temp_dirs_created = 0
        self.temp_dirs_cleaned = 0
        self.http_requests = 0
        self.metadata_fetched = 0
        self.pe_analyzed = 0
        self.operation_start_time = time.time()
        self.operations_history = []  # Store recent operations with timing

    def add_download(self, bytes_count):
        with self.lock:
            self.bytes_downloaded += bytes_count
            self.current_file_downloaded += bytes_count

    def add_upload(self, bytes_count):
        with self.lock:
            self.bytes_uploaded += bytes_count

    def add_unpack(self):
        with self.lock:
            self.files_unpacked += 1

    def add_pack(self):
        with self.lock:
            self.files_packed += 1

    def add_x86(self):
        with self.lock:
            self.files_x86 += 1

    def add_amd64(self):
        with self.lock:
            self.files_amd64 += 1

    def add_failed(self):
        with self.lock:
            self.files_failed += 1

    def add_skipped(self):
        with self.lock:
            self.files_skipped += 1

    def add_error(self):
        with self.lock:
            self.total_errors += 1

    def add_http_request(self):
        with self.lock:
            self.http_requests += 1

    def add_metadata_fetch(self):
        with self.lock:
            self.metadata_fetched += 1

    def add_pe_analysis(self):
        with self.lock:
            self.pe_analyzed += 1

    def add_temp_dir_created(self):
        with self.lock:
            self.temp_dirs_created += 1

    def add_temp_dir_cleaned(self):
        with self.lock:
            self.temp_dirs_cleaned += 1

    def set_operation(self, operation, package_name="", url=""):
        with self.lock:
            self.current_operation = operation
            self.current_package_name = package_name
            self.current_url = url
            self.operation_start_time = time.time()

    def complete_operation(self, operation):
        with self.lock:
            duration = time.time() - self.operation_start_time
            self.operations_history.append({
                'operation': operation,
                'duration': duration,
                'timestamp': time.time()
            })
            if len(self.operations_history) > 20:  # Keep last 20 operations
                self.operations_history.pop(0)
            self.current_operation = ""

    def set_current_file(self, filename, size=0):
        with self.lock:
            self.current_file = filename
            self.current_file_size = size
            self.current_file_downloaded = 0

    def get_download_speed(self):
        """Calculate current download speed in bytes/sec"""
        with self.lock:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                return self.bytes_downloaded / elapsed
            return 0

    def get_average_speed(self):
        """Get smoothed average speed from recent samples"""
        with self.lock:
            if len(self.download_speeds) > 0:
                return sum(self.download_speeds) / len(self.download_speeds)
            return self.get_download_speed()

    def update_speed_sample(self):
        """Add current speed to samples for smoothing"""
        with self.lock:
            speed = self.get_download_speed()
            self.download_speeds.append(speed)
            if len(self.download_speeds) > self.max_speed_samples:
                self.download_speeds.pop(0)

    def get_eta(self, total_files, current_file):
        """Estimate time remaining based on average speed and progress"""
        with self.lock:
            if current_file == 0:
                return None
            elapsed = time.time() - self.start_time
            avg_time_per_file = elapsed / current_file
            remaining_files = total_files - current_file
            return remaining_files * avg_time_per_file

    def should_report(self):
        return time.time() - self.last_report_time >= 2.0

    def generate_live_table(self, current_idx, total_files):
        """Generate a rich table for live display"""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="cyan")
        table.add_column(justify="left", style="bold white")

        speed = self.get_average_speed()
        elapsed = time.time() - self.start_time
        eta = self.get_eta(total_files, current_idx)

        # Format speed
        if speed > 1024 * 1024:
            speed_str = f"{speed / 1024 / 1024:.2f} MB/s"
        elif speed > 1024:
            speed_str = f"{speed / 1024:.2f} KB/s"
        else:
            speed_str = f"{speed:.2f} B/s"

        # Format ETA
        if eta:
            eta_mins = int(eta / 60)
            eta_secs = int(eta % 60)
            eta_str = f"{eta_mins}m {eta_secs}s"
        else:
            eta_str = "calculating..."

        # Format elapsed time
        elapsed_mins = int(elapsed / 60)
        elapsed_secs = int(elapsed % 60)
        elapsed_str = f"{elapsed_mins}m {elapsed_secs}s"

        # Current operation duration
        op_duration = time.time() - self.operation_start_time
        op_duration_str = f"{op_duration:.1f}s"

        table.add_row("", "[bold yellow]═══ CURRENT OPERATION ═══[/bold yellow]")
        table.add_row("🔄 Operation:", self.current_operation or "Idle")
        table.add_row("📦 Package:", self.current_package_name or "N/A")
        table.add_row("⏲️  Op. Duration:", op_duration_str)
        if self.current_file:
            table.add_row("📄 File:", self.current_file[:50] + "..." if len(self.current_file) > 50 else self.current_file)

        table.add_row("", "[bold cyan]═══ STATISTICS ═══[/bold cyan]")
        table.add_row("📥 Downloaded:", f"{self.bytes_downloaded / 1024.0 / 1024.0:.2f} MB")
        table.add_row("⚡ Speed:", speed_str)
        table.add_row("⏱️ Elapsed:", elapsed_str)
        table.add_row("⏳ ETA:", eta_str)
        table.add_row("📊 Progress:", f"{current_idx}/{total_files} ({(current_idx / total_files) * 100:.1f}%)")

        table.add_row("", "[bold green]═══ FILE COUNTS ═══[/bold green]")
        table.add_row("📦 Packed:", f"{self.files_packed}")
        table.add_row("📂 Unpacked:", f"{self.files_unpacked}")
        table.add_row("🖥️ x86:", f"{self.files_x86}")
        table.add_row("💻 x64:", f"{self.files_amd64}")

        table.add_row("", "[bold magenta]═══ OPERATIONS ═══[/bold magenta]")
        table.add_row("🌐 HTTP Requests:", f"{self.http_requests}")
        table.add_row("📋 Metadata Fetched:", f"{self.metadata_fetched}")
        table.add_row("🔍 PE Analyzed:", f"{self.pe_analyzed}")
        table.add_row("📁 Temp Dirs Created:", f"{self.temp_dirs_created}")
        table.add_row("🗑️ Temp Dirs Cleaned:", f"{self.temp_dirs_cleaned}")

        if self.files_failed > 0 or self.total_errors > 0:
            table.add_row("", "[bold red]═══ ERRORS ═══[/bold red]")
            table.add_row("❌ Failed Files:", f"{self.files_failed}")
            table.add_row("⚠️ Total Errors:", f"{self.total_errors}")

        return Panel(table, title="[bold cyan]📊 Download Statistics & Current Operation[/bold cyan]", border_style="cyan")

    def report_and_reset(self):
        with self.lock:
            current_time = get_timestamp()
            speed = self.get_average_speed()
            if speed > 1024 * 1024:
                speed_str = f"{speed / 1024 / 1024:.2f} MB/s"
            elif speed > 1024:
                speed_str = f"{speed / 1024:.2f} KB/s"
            else:
                speed_str = f"{speed:.2f} B/s"

            # Create a detailed status table
            status_table = Table.grid(padding=(0, 2))
            status_table.add_column(justify="right", style="cyan", no_wrap=True)
            status_table.add_column(justify="left", style="white")

            status_table.add_row("", "[bold yellow]═══ CURRENT OPERATION ═══[/bold yellow]")
            if self.current_operation:
                status_table.add_row("🔄 Operation:", self.current_operation)
            if self.current_package_name:
                status_table.add_row("📦 Package:", self.current_package_name)
            if self.current_url:
                url_display = self.current_url[:60] + "..." if len(self.current_url) > 60 else self.current_url
                status_table.add_row("🔗 URL:", url_display)

            status_table.add_row("", "[bold cyan]═══ STATISTICS ═══[/bold cyan]")
            status_table.add_row("📥 Downloaded:", f"{self.bytes_downloaded / 1024.0 / 1024.0:.2f} MB @ {speed_str}")
            status_table.add_row("📦 Packed:", f"{self.files_packed}")
            status_table.add_row("📂 Unpacked:", f"{self.files_unpacked}")
            status_table.add_row("🖥️ x86:", f"{self.files_x86}")
            status_table.add_row("💻 x64:", f"{self.files_amd64}")

            status_table.add_row("", "[bold magenta]═══ OPERATIONS ═══[/bold magenta]")
            status_table.add_row("🌐 HTTP Requests:", f"{self.http_requests}")
            status_table.add_row("📋 Metadata:", f"{self.metadata_fetched}")
            status_table.add_row("🔍 PE Analyzed:", f"{self.pe_analyzed}")
            status_table.add_row("📁 Temp Created:", f"{self.temp_dirs_created}")
            status_table.add_row("🗑️ Temp Cleaned:", f"{self.temp_dirs_cleaned}")

            if self.files_failed > 0 or self.total_errors > 0:
                status_table.add_row("", "[bold red]═══ ERRORS ═══[/bold red]")
                status_table.add_row("❌ Failed:", f"{self.files_failed}")
                status_table.add_row("⚠️ Errors:", f"{self.total_errors}")

            console.print(
                Panel(status_table, title=f"[bold cyan]📊 STATUS REPORT - {current_time}[/bold cyan]", border_style="cyan"))

            self.last_report_time = time.time()
            self.update_speed_sample()


stats = Stats()


def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")


def download_with_progress(url, headers=None):
    """Download a file with a progress bar"""
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=16)
        total_size = int(response.headers.get('content-length', 0))

        if total_size == 0:
            # Fallback to non-streaming download
            response = requests.get(url, headers=headers)
            stats.add_download(len(response.content))
            return response

        content = bytearray()

        with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Downloading...", total=total_size)

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content.extend(chunk)
                    stats.add_download(len(chunk))
                    progress.update(task, advance=len(chunk))

        # Create a mock response object
        class MockResponse:
            def __init__(self, content, status_code):
                self.content = bytes(content)
                self.status_code = status_code
                self.text = self.content.decode('utf-8', errors='ignore')

        return MockResponse(content, response.status_code)

    except Exception as e:
        raise e


console.print(f"{get_timestamp()} [bold green]🌐 NETWORK[/bold green] Fetching pad links list...")
stats.set_operation("Fetching PAD links list", "", "https://www.nirsoft.net/pad/pad-links.txt")
links = requests.get('https://www.nirsoft.net/pad/pad-links.txt')
stats.add_http_request()
stats.add_download(len(links.content))
links = links.text.split('\n')
stats.complete_operation("Fetch PAD links")
updated = 0
total = len(links)
console.print(f"{get_timestamp()} [bold blue]ℹ️ INFO[/bold blue] Found {total} links to process")

path.Path(r'h:\upd').mkdir_p()

prevs = glob.glob(r'h:\upd\*')
if prevs:
    console.print(
        f"{get_timestamp()} [bold yellow]🗑️  CLEANUP[/bold yellow] Removing {len(prevs)} previous temporary directories...")
    stats.set_operation("Cleanup temp directories", f"{len(prevs)} directories", "")
    for prev in prevs:
        console.print(f"{get_timestamp()} [yellow]🗑️ DELETE[/yellow] {prev}")
        try:
            shutil.rmtree(prev)
            stats.add_temp_dir_cleaned()
        except Exception as e:
            console.print(f"{get_timestamp()} [red]❌ ERROR[/red] Failed to remove {prev}: {e}")
            stats.add_error()
            pass
    stats.complete_operation("Cleanup")
    console.print(f"{get_timestamp()} [bold green]✅ CLEANUP[/bold green] Removed {stats.temp_dirs_cleaned} temporary directories")
tot = 0.
for link in links:
    # Report stats every 2 seconds
    # if stats.should_report():
    #     stats.report_and_reset()

    if link == '':
        console.print(f"{get_timestamp()} [bold green]✅ COMPLETE[/bold green] Done with all links, updated={updated}")
        break
    updated += 1

    console.print(f"{get_timestamp()} [bold green]🌐 NETWORK[/bold green] [{updated}/{total}] Fetching metadata: {link}")
    stats.set_operation("Fetching metadata", f"Package {updated}/{total}", link)
    xm = requests.get(link)
    stats.add_http_request()
    stats.add_download(len(xm.content))
    stats.add_metadata_fetch()
    xm = xm.text
    xm = json.dumps(xm)
    url = re.findall(r'Primary_Download_URL>(.*?)</Prim', xm)
    console.print(f"{get_timestamp()} [bold blue]ℹ️ INFO[/bold blue] Primary download URL: {url[0]}")
    stats.complete_operation("Fetch metadata")

    while True:
        try:
            package_name = url[0].split('/')[-1].replace('.zip', '')
            console.print(f"{get_timestamp()} [bold green]⬇️ DOWNLOAD[/bold green] [{updated}/{total}] {url[0]}")
            stats.set_operation(f"Downloading package {updated}/{total}", package_name, url[0])
            stats.set_current_file(url[0].split('/')[-1])
            resp = download_with_progress(url[0], headers={
                'Referer': 'https://www.nirsoft.net/pad/index.html',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4'
            })
            stats.complete_operation("Download package")
        except Exception as e:
            console.print(f"{get_timestamp()} [red]❌ ERROR[/red] Failed with {url[0]}: {e}")
            stats.add_error()
            stats.add_failed()
            time.sleep(5)
            continue
        break

    if resp.status_code != 200:
        console.print(f"{get_timestamp()} [red]❌ ERROR[/red] HTTP {resp.status_code} for {url[0]}")
        stats.add_error()
        stats.add_failed()
        continue

    data = resp.content

    rand_name = str(random.randrange(10, 99120)) + '.zip'
    dr = str(random.randrange(1, 100000))
    stats.set_operation("Creating temp directory", package_name, "")
    os.makedirs(rf'h:\upd\{dr}')
    stats.add_temp_dir_created()
    drr = os.path.join(r'h:\upd', dr)
    path = os.path.join(r"h:\upd", dr, rand_name)

    stats.set_operation("Writing archive to disk", package_name, path)
    with open(path, 'wb') as f:
        console.print(f"{get_timestamp()} [bold magenta]💾 WRITE[/bold magenta] Writing {len(data)} bytes to {path}")
        ret = f.write(data)
        stats.add_pack()
        console.print(f"{get_timestamp()} [magenta]💾 WRITE[/magenta] Wrote {ret} bytes")
    stats.complete_operation("Write archive")

    sz = f'{os.path.getsize(path) / 1024.0 / 1024.0:.2f} MB'
    tot += float(os.path.getsize(path) / 1024.0 / 1024.0)
    console.print(
        f"{get_timestamp()} [bold blue]📈 PROGRESS[/bold blue] [{updated:d}/{total:d}] ({(updated / total) * 100.0:.2f}%) | File: {sz} | Total: {tot:.2f} MB")

    os.chdir(drr)

    stats.set_operation("Extracting archive", package_name, path)
    console.print(f"{get_timestamp()} [bold cyan]📂 UNPACK[/bold cyan] Extracting {path}...")
    os.system(f'7z x -p0 -bb0 {path} > o')
    stats.add_unpack()
    stats.complete_operation("Extract archive")

    # find exe
    stats.set_operation("Searching for executable", package_name, "")
    console.print(f"{get_timestamp()} [bold blue]🔍 SEARCH[/bold blue] Looking for executable in {url}")

    try:
        exepath = re.findall(r'\w+/\w+/([^.]{1,25})\.zip', url[0])
        name = exepath[0]
        console.print(f"{get_timestamp()} [bold blue]ℹ️ INFO[/bold blue] Package name: {name}")
        exes = glob.glob(f'*.exe')

        epath = os.path.join(drr, exes[0])
        console.print(f"{get_timestamp()} [bold green]✅ FOUND[/bold green] Executable: {epath}")
        stats.complete_operation("Search executable")
    except Exception as e:
        console.print(f"{get_timestamp()} [red]❌ ERROR[/red] EXE is broken or infected: {e}")
        stats.add_error()
        stats.add_failed()
        stats.complete_operation("Search executable")
        continue

    stats.set_operation("Analyzing PE file", package_name, epath)
    try:
        pe = pefile.PE(epath)
        stats.add_pe_analysis()
    except Exception as e:
        console.print(f"{get_timestamp()} [red]🔒 ERROR[/red] POSSIBLY PASSWORDED")
        stats.add_error()
        stats.add_failed()
        stats.complete_operation("Analyze PE")
        continue
    # determine x64

    console.print(
        f"{get_timestamp()} [bold blue]🔧 ANALYZE[/bold blue] Machine type: {pe.FILE_HEADER.Machine} (0x{pe.FILE_HEADER.Machine:x})")
    stats.complete_operation("Analyze PE")
    # unpack again to proper nirsoft dir

    if pe.FILE_HEADER.Machine == 0x8664:
        dest = os.path.join(rf'T:\!power-tools\NirLauncher\NirSoft\x64\{name}')
        console.print(f"{get_timestamp()} [bold blue]ℹ️ INFO[/bold blue] Detected 🐴 x64 architecture")
        stats.add_amd64()
    elif pe.FILE_HEADER.Machine == 0x14c:
        dest = os.path.join(rf'T:\!power-tools\NirLauncher\NirSoft\{name}')
        console.print(f"{get_timestamp()} [bold blue]ℹ️ INFO[/bold blue] Detected 🦍 x86 architecture")
        stats.add_x86()
    else:
        console.print(
            f"{get_timestamp()} [red]❌ ERROR[/red] Unknown machine type: {pe.FILE_HEADER.Machine} (0x{pe.FILE_HEADER.Machine:x})")
        stats.add_error()
        time.sleep(10)

    stats.set_operation("Extracting to final destination", package_name, dest)
    console.print(f"{get_timestamp()} [bold cyan]⛈️ UNPACK[/bold cyan] Extracting to final destination: {dest}")
    try:
        os.system(rf'7z x -p0 -y -bb0 {path} -o{dest} > o')
        stats.add_unpack()
        pe.close()
        extract_log = open('o', 'r').read()
        extract_log = re.sub(r'[\r\n]+', '\n   🔸  ', extract_log.strip()).strip()
        console.print(f"{get_timestamp()} [bold blue]💼 LOG[/bold blue] Extraction:\n ⇶ {extract_log}")
        os.chdir(r"h:\\")
        shutil.rmtree(drr)
        stats.add_temp_dir_cleaned()
        stats.complete_operation("Final extraction")
        console.print(f"{get_timestamp()} [bold green]✅ COMPLETE[/bold green] Successfully processed {name}")
    except Exception as e:
        console.print(f"{get_timestamp()} [red]❌ ERROR[/red] Exception: {e}")
        stats.add_error()
        stats.add_failed()
        stats.complete_operation("Final extraction")

# Final stats report
console.print(f"\n{get_timestamp()} [bold cyan]📊 FINAL STATS[/bold cyan]")
console.print(f"{get_timestamp()} [cyan]⬇️ Total Downloaded:[/cyan] {stats.bytes_downloaded / 1024.0 / 1024.0:.2f} MB")
console.print(f"{get_timestamp()} [cyan]⬆️ Total Uploaded:[/cyan] {stats.bytes_uploaded / 1024.0 / 1024.0:.2f} MB")
console.print(f"{get_timestamp()} [cyan]📦 Files Packed:[/cyan] {stats.files_packed}")
console.print(f"{get_timestamp()} [cyan]📂 Files Unpacked:[/cyan] {stats.files_unpacked}")
console.print(f"{get_timestamp()} [cyan]🖥️ x86 Files:[/cyan] {stats.files_x86}")
console.print(f"{get_timestamp()} [cyan]💻 x64 Files:[/cyan] {stats.files_amd64}")
console.print(f"{get_timestamp()} [bold green]✅ ALL DONE![/bold green]")
