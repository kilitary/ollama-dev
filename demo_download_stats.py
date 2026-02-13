"""
Demo script to test the dynamic download stats functionality
from the update_nirsoft.py enhancements
"""

import time
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

console = Console()


class DemoStats:
    def __init__(self):
        self.bytes_downloaded = 0
        self.start_time = time.time()
        self.download_speeds = []
        self.max_speed_samples = 10

    def add_download(self, bytes_count):
        self.bytes_downloaded += bytes_count

    def get_download_speed(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.bytes_downloaded / elapsed
        return 0

    def get_average_speed(self):
        if len(self.download_speeds) > 0:
            return sum(self.download_speeds) / len(self.download_speeds)
        return self.get_download_speed()

    def update_speed_sample(self):
        speed = self.get_download_speed()
        self.download_speeds.append(speed)
        if len(self.download_speeds) > self.max_speed_samples:
            self.download_speeds.pop(0)

    def generate_live_table(self, current_idx, total_files):
        """Generate a rich table for live display"""
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="cyan")
        table.add_column(justify="left", style="bold white")

        speed = self.get_average_speed()
        elapsed = time.time() - self.start_time

        # Format speed
        if speed > 1024 * 1024:
            speed_str = f"{speed / 1024 / 1024:.2f} MB/s"
        elif speed > 1024:
            speed_str = f"{speed / 1024:.2f} KB/s"
        else:
            speed_str = f"{speed:.2f} B/s"

        # Format elapsed time
        elapsed_mins = int(elapsed / 60)
        elapsed_secs = int(elapsed % 60)
        elapsed_str = f"{elapsed_mins}m {elapsed_secs}s"

        table.add_row("📥 Downloaded:", f"{self.bytes_downloaded / 1024.0 / 1024.0:.2f} MB")
        table.add_row("⚡ Speed:", speed_str)
        table.add_row("⏱️  Elapsed:", elapsed_str)
        table.add_row("📊 Progress:", f"{current_idx}/{total_files} ({(current_idx/total_files)*100:.1f}%)")

        return Panel(table, title="[bold cyan]📊 Download Statistics Demo[/bold cyan]", border_style="cyan")


def demo_download_with_progress(url):
    """Demonstrate download with progress bar"""
    stats = DemoStats()

    console.print(f"[bold green]⬇️  DOWNLOAD[/bold green] {url}")

    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        if total_size == 0:
            console.print("[yellow]⚠️  Warning: No content-length header[/yellow]")
            return

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

        stats.update_speed_sample()

        # Show stats after download
        console.print("\n")
        console.print(stats.generate_live_table(1, 1))
        console.print("\n[bold green]✅ Download complete![/bold green]")
        console.print(f"[cyan]Total size:[/cyan] {len(content) / 1024.0 / 1024.0:.2f} MB")
        console.print(f"[cyan]Average speed:[/cyan] {stats.get_average_speed() / 1024.0 / 1024.0:.2f} MB/s")

    except Exception as e:
        console.print(f"[red]❌ ERROR[/red] {e}")


if __name__ == "__main__":
    console.print("[bold cyan]🧪 Dynamic Download Stats Demo[/bold cyan]\n")

    # Demo with a small file
    test_url = "https://httpbin.org/bytes/1048576"  # 1 MB test file

    console.print("[yellow]Testing with 1 MB file from httpbin.org...[/yellow]\n")
    demo_download_with_progress(test_url)

    console.print("\n[bold green]✅ Demo complete![/bold green]")
    console.print("\n[yellow]This demonstrates the new dynamic download stats features:[/yellow]")
    console.print("  • Real-time progress bar with spinner")
    console.print("  • Download size display (current/total)")
    console.print("  • Transfer speed calculation")
    console.print("  • Time remaining estimate")
    console.print("  • Beautiful statistics panel")
    console.print("  • Speed averaging for smooth display")
