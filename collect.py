import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
console_lock = threading.Lock()

VK8_DIR = Path(__file__).parent / "vk8"
OUTPUT_DIR = Path(__file__).parent / "v8"

start_time = time.time()

# EN key → (RU name, EN name)
COLLECTION_NAMES: dict[str, tuple[str, str]] = {
    "ads":          ("Реклама",                 "Ads"),
    "apps":         ("Приложения",              "Apps"),
    "audio":        ("Аудио",                   "Audio"),
    "bookmarks":    ("Закладки",                "Bookmarks"),
    "comments":     ("Комментарии",             "Comments"),
    "likes":        ("Отметки «Нравится»",      "Likes"),
    "messages":     ("Сообщения",               "Messages"),
    "other":        ("Прочее",                  "Other"),
    "payments":     ("Платежи",                 "Payments"),
    "photos":       ("Фотографии",              "Photos"),
    "profile":      ("Профиль",                 "Profile"),
    "verification": ("Верификация",             "Verification"),
    "video":        ("Видео",                   "Video"),
    "wall":         ("Стена / Записи",          "Wall"),
}

# Regex to pull every URL from href / src / action attributes
_URL_RE = re.compile(r'(?:href|src|action)\s*=\s*["\']?(https?://[^\s"\'<>]+)', re.IGNORECASE)
# Domains considered internal (no external-link flag)
_INTERNAL_DOMAINS = ("vk.com", "vkontakte.ru", "vk.me", "vk.cc")


def get_timestamp():
    return f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]"


def slog(msg: str = ""):
    with console_lock:
        console.print(f"{get_timestamp()} {msg}")


def collect_files(top_dir: Path):
    """Recursively collect all files under top_dir, sorted by path."""
    files = []
    for root, dirs, filenames in os.walk(top_dir):
        dirs.sort()
        for fname in sorted(filenames):
            files.append(Path(root) / fname)
    return files


ENCODINGS_TO_TRY = ["utf-8", "cp1251", "cp1252", "latin-1", "iso-8859-1"]


def fix_html_charset(text: str) -> str:
    """Replace any charset declaration in HTML meta tags with utf-8."""
    text = re.sub(r'(<meta\s[^>]*charset\s*=\s*")[^"]*(")', r'\g<1>utf-8\g<2>', text, flags=re.IGNORECASE)
    text = re.sub(r"(<meta\s[^>]*charset\s*=\s*')[^']*(')", r"\g<1>utf-8\g<2>", text, flags=re.IGNORECASE)
    text = re.sub(r'(content\s*=\s*["\'][^"\']*;\s*charset\s*=\s*)[^\s"\']+', r'\g<1>utf-8', text, flags=re.IGNORECASE)
    return text


def to_utf8(fpath: Path) -> tuple[bytes, str]:
    """Read a file, re-encode to UTF-8, fix charset meta tags.
    Returns (utf8_bytes, detected_encoding)."""
    raw = fpath.read_bytes()
    detected = None
    text = None
    for enc in ENCODINGS_TO_TRY:
        try:
            text = raw.decode(enc)
            detected = enc
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
        detected = "utf-8 (replace)"
    text = fix_html_charset(text)
    return text.encode("utf-8"), detected


def extract_external_links(text: str) -> set[str]:
    """Return set of external (non-VK) domains found in HTML text."""
    domains: set[str] = set()
    for url in _URL_RE.findall(text):
        m = re.match(r'https?://([^/?#\s]+)', url)
        if not m:
            continue
        host = m.group(1).lower().lstrip("www.")
        if not any(host == d or host.endswith("." + d) for d in _INTERNAL_DOMAINS):
            domains.add(host)
    return domains


def write_collection(collection_name: str, files: list, output_path: Path) -> tuple[int, set[str]]:
    """Concatenate all files (UTF-8 converted) into one output file.
    Returns (total_bytes_written, external_domains)."""
    total_bytes = 0
    enc_counts: dict[str, int] = {}
    all_ext_links: set[str] = set()

    with open(output_path, "wb") as out:
        for fpath in files:
            try:
                rel = fpath.relative_to(VK8_DIR)
                data, detected = to_utf8(fpath)
                enc_counts[detected] = enc_counts.get(detected, 0) + 1
                if detected not in ("utf-8",):
                    slog(f"  [dim]🔤 {rel}[/dim] [yellow]← {detected}[/yellow] → utf-8")

                # scan for external links while content is still text
                text_for_scan = data.decode("utf-8", errors="replace")
                all_ext_links |= extract_external_links(text_for_scan)

                header = f"\n\n<!-- === FILE: {rel} === -->\n\n"
                out.write(header.encode("utf-8"))
                out.write(data)
                total_bytes += len(data)
            except Exception as e:
                slog(f"[yellow]⚠️  Skipped [/yellow]{fpath.name}: {e}")

    enc_summary = "  ".join(f"[cyan]{enc}[/cyan]×{n}" for enc, n in sorted(enc_counts.items()))
    slog(f"  [dim]encodings detected:[/dim] {enc_summary}")
    return total_bytes, all_ext_links


def _fmt_size(b: int) -> str:
    if b >= 1_048_576:
        return f"{b / 1_048_576:.2f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def build_index(collection_meta: list[dict]) -> None:
    """Write v8/index.html — top-level summary of every collection."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows_html = ""
    for m in collection_meta:
        name_key  = m["name"]
        ru, en    = COLLECTION_NAMES.get(name_key, (name_key.capitalize(), name_key.capitalize()))
        fc        = m["file_count"]
        sz        = _fmt_size(m["bytes"])
        ext_links = m["ext_links"]
        col_file  = f"{name_key}.html"

        if ext_links:
            links_cell = (
                "<details><summary>✅ да / yes "
                f"<span class='badge'>{len(ext_links)}</span></summary>"
                "<ul>" + "".join(f"<li>{h}</li>" for h in sorted(ext_links)) + "</ul>"
                "</details>"
            )
        else:
            links_cell = "<span class='no-links'>❌ нет / no</span>"

        rows_html += f"""
        <tr>
          <td><a href="{col_file}">{ru}</a></td>
          <td>{en}</td>
          <td class="num">{fc:,}</td>
          <td class="num">{sz}</td>
          <td>{links_cell}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VK Archive — Index</title>
  <style>
    :root {{
      --bg: #1a1a2e; --surface: #16213e; --accent: #0f3460;
      --blue: #4cc9f0; --green: #4ade80; --yellow: #fbbf24;
      --red: #f87171; --text: #e2e8f0; --muted: #94a3b8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
            padding: 2rem; }}
    h1   {{ color: var(--blue); margin-bottom: .25rem; font-size: 1.8rem; }}
    .sub {{ color: var(--muted); margin-bottom: 2rem; font-size: .9rem; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface);
             border-radius: 8px; overflow: hidden; }}
    th   {{ background: var(--accent); color: var(--blue); padding: .75rem 1rem;
            text-align: left; font-size: .85rem; letter-spacing: .05em; }}
    td   {{ padding: .65rem 1rem; border-bottom: 1px solid #0f3460; vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #0f346022; }}
    td a {{ color: var(--blue); text-decoration: none; }}
    td a:hover {{ text-decoration: underline; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; color: var(--yellow); }}
    .badge {{ background: var(--accent); color: var(--blue); border-radius: 999px;
              padding: .1em .5em; font-size: .78rem; margin-left: .4em; }}
    .no-links {{ color: var(--muted); }}
    details summary {{ cursor: pointer; color: var(--green); }}
    details ul {{ margin: .4rem 0 0 1.2rem; font-size: .8rem; color: var(--muted); }}
    details ul li {{ margin: .1rem 0; word-break: break-all; }}
    footer {{ margin-top: 2rem; color: var(--muted); font-size: .8rem; text-align: center; }}
  </style>
</head>
<body>
  <h1>📦 VK Archive — Index</h1>
  <p class="sub">Сгенерировано / Generated: {generated_at}</p>
  <table>
    <thead>
      <tr>
        <th>Коллекция (RU)</th>
        <th>Collection (EN)</th>
        <th style="text-align:right">Файлов / Files</th>
        <th style="text-align:right">Размер / Size</th>
        <th>Внешние ссылки / External links</th>
      </tr>
    </thead>
    <tbody>{rows_html}
    </tbody>
  </table>
  <footer>vk8 → v8 collector · {generated_at}</footer>
</body>
</html>
"""
    index_path = OUTPUT_DIR / "index.html"
    index_path.write_text(html, encoding="utf-8")
    slog(f"[bold green]✅ INDEX[/bold green]  written → [cyan]{index_path}[/cyan]")


def main():
    if not VK8_DIR.exists():
        slog(f"[bold red]❌ ERROR[/bold red] vk8 directory not found: {VK8_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slog(f"[bold cyan]ℹ️  OUTPUT[/bold cyan] Writing collections to: [cyan]{OUTPUT_DIR}[/cyan]")

    top_dirs = sorted([d for d in VK8_DIR.iterdir() if d.is_dir()], key=lambda d: d.name)

    if not top_dirs:
        slog("[bold red]❌ ERROR[/bold red] No subdirectories found in vk8/")
        sys.exit(1)

    slog(f"[bold cyan]ℹ️  FOUND[/bold cyan] [cyan]{len(top_dirs)}[/cyan] top-level collections: "
         f"{', '.join(d.name for d in top_dirs)}")

    stats = {"collections": 0, "total_files": 0, "total_bytes": 0}
    collection_meta: list[dict] = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(), console=console, transient=False,
    ) as progress:
        main_task = progress.add_task("[bold cyan]Processing collections...", total=len(top_dirs))

        for top_dir in top_dirs:
            collection_name = top_dir.name
            output_path = OUTPUT_DIR / f"{collection_name}.html"
            progress.update(main_task, description=f"[bold cyan]📂 {collection_name}[/bold cyan]")

            files = collect_files(top_dir)
            if not files:
                slog(f"[yellow]⚠️  SKIP[/yellow] [yellow]{collection_name}[/yellow] — no files found")
                progress.advance(main_task)
                continue

            slog(f"[bold magenta]🔧 COLLECT[/bold magenta] [magenta]{collection_name}[/magenta] "
                 f"→ [cyan]{len(files)}[/cyan] files → [cyan]{output_path.name}[/cyan]")

            written_bytes, ext_links = write_collection(collection_name, files, output_path)
            stats["collections"] += 1
            stats["total_files"] += len(files)
            stats["total_bytes"] += written_bytes

            collection_meta.append({
                "name":       collection_name,
                "file_count": len(files),
                "bytes":      written_bytes,
                "ext_links":  ext_links,
            })

            kb = written_bytes / 1024
            ext_info = f", [yellow]{len(ext_links)} ext domains[/yellow]" if ext_links else ""
            slog(f"[bold green]✅ DONE[/bold green]  [green]{collection_name}[/green] "
                 f"— {len(files)} files, {kb:.1f} KB{ext_info} → [cyan]{output_path}[/cyan]")

            progress.advance(main_task)

    # Build the top-level index
    build_index(collection_meta)

    elapsed = time.time() - start_time

    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="cyan", no_wrap=True)
    table.add_column(justify="left", style="white")
    table.add_row("", "[bold yellow]═══ COLLECTION SUMMARY ═══[/bold yellow]")
    table.add_row("📁 Output dir:",   str(OUTPUT_DIR))
    table.add_row("📦 Collections:",  str(stats["collections"]))
    table.add_row("📄 Total files:",  f"{stats['total_files']:,}")
    table.add_row("💾 Total size:",   f"{stats['total_bytes'] / 1024 / 1024:.2f} MB")
    table.add_row("⏱️  Elapsed:",     f"{elapsed:.1f}s")
    console.print(Panel(table, title="[bold green]✅ COLLECT COMPLETE[/bold green]", border_style="green"))


if __name__ == "__main__":
    main()
