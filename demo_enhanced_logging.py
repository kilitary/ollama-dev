#!/usr/bin/env python
"""
Demo script to showcase the enhanced logging features
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from datetime import datetime
import time

console = Console()

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

def demo_enhanced_logging():
    """Demonstrate the new enhanced logging features"""
    
    console.print("\n[bold cyan]═══ Enhanced Logging Demo ═══[/bold cyan]\n")
    
    # Demo 1: Current operation tracking
    console.print(f"{get_timestamp()} [bold green]🌐 NETWORK[/bold green] [1/10] Fetching metadata: https://example.com/package.xml")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [bold blue]ℹ️  INFO[/bold blue] Primary download URL: https://example.com/package.zip")
    time.sleep(0.5)
    
    # Demo 2: Status panel
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(justify="left", style="white")
    
    table.add_row("", "[bold yellow]═══ CURRENT OPERATION ═══[/bold yellow]")
    table.add_row("🔄 Operation:", "Downloading package 1/10")
    table.add_row("📦 Package:", "example-tool")
    table.add_row("🔗 URL:", "https://example.com/package.zip")
    table.add_row("⏲️  Op. Duration:", "2.3s")
    
    table.add_row("", "[bold cyan]═══ STATISTICS ═══[/bold cyan]")
    table.add_row("📥 Downloaded:", "15.67 MB @ 2.34 MB/s")
    table.add_row("⚡ Speed:", "2.34 MB/s")
    table.add_row("⏱️ Elapsed:", "5m 23s")
    table.add_row("⏳ ETA:", "45m 12s")
    table.add_row("📊 Progress:", "1/10 (10.0%)")
    
    table.add_row("", "[bold green]═══ FILE COUNTS ═══[/bold green]")
    table.add_row("📦 Packed:", "1")
    table.add_row("📂 Unpacked:", "2")
    table.add_row("🖥️ x86:", "1")
    table.add_row("💻 x64:", "0")
    
    table.add_row("", "[bold magenta]═══ OPERATIONS ═══[/bold magenta]")
    table.add_row("🌐 HTTP Requests:", "3")
    table.add_row("📋 Metadata Fetched:", "1")
    table.add_row("🔍 PE Analyzed:", "1")
    table.add_row("📁 Temp Dirs Created:", "1")
    table.add_row("🗑️ Temp Dirs Cleaned:", "0")
    
    console.print(Panel(table, title=f"[bold cyan]📊 STATUS REPORT - {get_timestamp()}[/bold cyan]", border_style="cyan"))
    time.sleep(1)
    
    # Demo 3: Operation messages
    console.print(f"\n{get_timestamp()} [bold green]⬇️  DOWNLOAD[/bold green] [1/10] https://example.com/package.zip")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [bold magenta]💾 WRITE[/bold magenta] Writing 1048576 bytes to h:\\12345\\67890.zip")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [magenta]💾 WRITE[/magenta] Wrote 1048576 bytes")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [bold blue]📈 PROGRESS[/bold blue] [1/10] (10.00%) | File: 1.00 MB | Total: 1.00 MB")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [bold cyan]📂 UNPACK[/bold cyan] Extracting h:\\12345\\67890.zip...")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [bold blue]🔍 SEARCH[/bold blue] Looking for executable in ['package.zip']")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [bold green]✅ FOUND[/bold green] Executable: h:\\12345\\tool.exe")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [bold blue]🔧 ANALYZE[/bold blue] Machine type: 34404 (0x8664)")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [bold blue]ℹ️  INFO[/bold blue] Detected x64 architecture")
    time.sleep(0.5)
    console.print(f"{get_timestamp()} [bold green]✅ COMPLETE[/bold green] Successfully processed example-tool")
    
    # Demo 4: Error example
    console.print(f"\n{get_timestamp()} [red]❌ ERROR[/red] Failed to download: Connection timeout")
    
    console.print("\n[bold cyan]═══ Demo Complete ═══[/bold cyan]\n")

if __name__ == "__main__":
    demo_enhanced_logging()
