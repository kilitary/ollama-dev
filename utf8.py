import os
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import datetime

console = Console()

SRC_DIR = Path(__file__).parent / "vk"
DST_DIR = Path(__file__).parent / "vk8"

def get_timestamp():
    return f"[dim]{datetime.datetime.now().strftime('%H:%M:%S')}[/dim]"

def convert_file(src_path: Path, dst_path: Path) -> bool:
    try:
        content = src_path.read_bytes()
        text = content.decode("windows-1251")
        # Replace charset declaration in meta tags if present
        text = text.replace('charset=windows-1251', 'charset=utf-8')
        text = text.replace('charset=Windows-1251', 'charset=utf-8')
        text = text.replace('charset=WINDOWS-1251', 'charset=utf-8')
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(text.encode("utf-8"))
        return True
    except Exception as e:
        console.print(f"{get_timestamp()} [bold red]❌ ERROR[/bold red] {src_path}: {e}")
        return False

def copy_non_html(src_path: Path, dst_path: Path):
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        console.print(f"{get_timestamp()} [bold red]❌ COPY ERROR[/bold red] {src_path}: {e}")

def main():
    console.print(f"\n{get_timestamp()} [bold cyan]ℹ️  SCAN[/bold cyan] Source : [cyan]{SRC_DIR}[/cyan]")
    console.print(f"{get_timestamp()} [bold cyan]ℹ️  SCAN[/bold cyan] Destination: [cyan]{DST_DIR}[/cyan]\n")

    html_files = list(SRC_DIR.rglob("*.html"))
    all_files  = list(SRC_DIR.rglob("*"))
    non_html   = [f for f in all_files if f.is_file() and f.suffix.lower() != ".html"]

    total     = len(html_files)
    converted = 0
    failed    = 0

    console.print(f"{get_timestamp()} [bold magenta]🔧 START[/bold magenta] Found [bold]{total}[/bold] HTML file(s) to convert")

    for idx, src in enumerate(html_files, 1):
        rel     = src.relative_to(SRC_DIR)
        dst     = DST_DIR / rel
        ok      = convert_file(src, dst)
        status  = "[bold green]✅[/bold green]" if ok else "[bold red]❌[/bold red]"
        console.print(f"{get_timestamp()} {status} [{idx}/{total}] {rel}")
        if ok:
            converted += 1
        else:
            failed += 1

    # Copy non-HTML files as-is
    console.print(f"\n{get_timestamp()} [bold yellow]⚠️  COPY[/bold yellow] Copying [bold]{len(non_html)}[/bold] non-HTML file(s) as-is...")
    for src in non_html:
        rel = src.relative_to(SRC_DIR)
        dst = DST_DIR / rel
        copy_non_html(src, dst)

    # Summary table
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(justify="left",  style="white")
    table.add_row("", "[bold yellow]═══ CONVERSION SUMMARY ═══[/bold yellow]")
    table.add_row("📂 Source dir:",      str(SRC_DIR))
    table.add_row("📂 Output dir:",      str(DST_DIR))
    table.add_row("📄 HTML files found:", str(total))
    table.add_row("✅ Converted:",        f"[bold green]{converted}[/bold green]")
    table.add_row("❌ Failed:",           f"[bold red]{failed}[/bold red]")
    table.add_row("📋 Non-HTML copied:", str(len(non_html)))
    console.print(Panel(table, title="[bold cyan]📊 RESULT[/bold cyan]", border_style="cyan"))

if __name__ == "__main__":
    main()

