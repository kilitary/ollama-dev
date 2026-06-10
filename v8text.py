import csv
import os
import time
from datetime import datetime
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

HTML_FILE = os.path.join(os.path.dirname(__file__), "v8", "comments.html")
CSV_FILE = os.path.join(os.path.dirname(__file__), "comments.csv")


def get_timestamp() -> str:
    return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]"


def extract_comments(html_path: str) -> list[tuple[str, str, str]]:
    console.print(f"{get_timestamp()} [bold cyan]ℹ️  PARSE[/bold cyan] Opening [cyan]{html_path}[/cyan]")
    t0 = time.time()

    with open(html_path, "r", encoding="utf-8") as f:
        raw = f.read()

    file_kb = len(raw) / 1024
    console.print(f"{get_timestamp()} [cyan]ℹ️  PARSE[/cyan] File size: [yellow]{file_kb:.1f} KB[/yellow] — building DOM...")

    soup = BeautifulSoup(raw, "html.parser")
    parse_elapsed = time.time() - t0
    console.print(f"{get_timestamp()} [cyan]ℹ️  PARSE[/cyan] DOM built in [yellow]{parse_elapsed:.2f}s[/yellow]")

    items = soup.select("div.item")
    console.print(f"{get_timestamp()} [bold cyan]ℹ️  PARSE[/bold cyan] Found [bold yellow]{len(items)}[/bold yellow] item blocks")

    records = []
    skipped = 0

    for idx, item in enumerate(items, 1):
        mains = item.select("div.item__main")
        tertiaries = item.select("div.item__tertiary")

        if len(mains) < 2 or len(tertiaries) < 2:
            skipped += 1
            console.print(
                f"{get_timestamp()} [yellow]⚠️  SKIP[/yellow] Item #{idx} — missing fields "
                f"(mains={len(mains)}, tertiaries={len(tertiaries)})"
            )
            continue

        comment_text = mains[0].get_text(separator=" ").strip()

        a_tag = mains[1].find("a")
        url = a_tag["href"].strip() if a_tag else mains[1].get_text().strip()

        date = tertiaries[1].get_text().strip()

        if comment_text or url:
            records.append((comment_text, date, url))
        else:
            skipped += 1
            console.print(f"{get_timestamp()} [yellow]⚠️  SKIP[/yellow] Item #{idx} — empty text and URL")

        if idx % 5000 == 0:
            console.print(
                f"{get_timestamp()} [cyan]🔧 PROGRESS[/cyan] Processed [bold]{idx}[/bold]/{len(items)} items "
                f"— [green]{len(records)}[/green] records so far"
            )

    elapsed = time.time() - t0
    console.print(
        f"{get_timestamp()} [bold green]✅ PARSE[/bold green] Extraction complete: "
        f"[green]{len(records)}[/green] records, [yellow]{skipped}[/yellow] skipped "
        f"in [yellow]{elapsed:.2f}s[/yellow]"
    )
    return records


def write_csv(records: list[tuple[str, str, str]], csv_path: str) -> None:
    console.print(
        f"{get_timestamp()} [bold magenta]🔧 WRITE[/bold magenta] Writing [bold]{len(records)}[/bold] records to [cyan]{csv_path}[/cyan]"
    )
    t0 = time.time()

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for comment_text, date, url in records:
            writer.writerow([comment_text, date, url])

    elapsed = time.time() - t0
    file_kb = os.path.getsize(csv_path) / 1024
    console.print(
        f"{get_timestamp()} [bold green]✅ WRITE[/bold green] Done in [yellow]{elapsed:.2f}s[/yellow] — "
        f"output size [yellow]{file_kb:.1f} KB[/yellow]"
    )

    # Summary table
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(justify="left", style="white")
    table.add_row("", "[bold yellow]═══ SUMMARY ═══[/bold yellow]")
    table.add_row("📄 Source:", HTML_FILE)
    table.add_row("💾 Output:", csv_path)
    table.add_row("📝 Records:", f"[bold green]{len(records)}[/bold green]")
    table.add_row("📦 File size:", f"[yellow]{file_kb:.1f} KB[/yellow]")
    table.add_row("⏱️  Duration:", f"[yellow]{elapsed:.2f}s[/yellow]")
    console.print(Panel(table, title="[bold cyan]📊 COMMENTS EXPORT[/bold cyan]", border_style="cyan"))


if __name__ == "__main__":
    console.print(f"{get_timestamp()} [bold cyan]🚀 START[/bold cyan] comments.html → comments.csv extractor")
    total_t0 = time.time()
    records = extract_comments(HTML_FILE)
    write_csv(records, CSV_FILE)
    console.print(f"{get_timestamp()} [bold green]✅ DONE[/bold green] Total time: [yellow]{time.time() - total_t0:.2f}s[/yellow]")
