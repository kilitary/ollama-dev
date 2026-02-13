#  Copyright (c) 2025. kilitary@gmail.com

import os
import glob
import requests
import time
import rich
from rich import print
from rich import print_json
from rich.console import Console
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
        self.lock = threading.Lock()

    def add_download(self, bytes_count):
        with self.lock:
            self.bytes_downloaded += bytes_count

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

    def should_report(self):
        return time.time() - self.last_report_time >= 2.0

    def report_and_reset(self):
        with self.lock:
            current_time = get_timestamp()
            console.print(
                f"{current_time} [bold cyan]📊 STATS[/bold cyan] ⬇️  {self.bytes_downloaded / 1024.0 / 1024.0:.2f} MB ⬆️  {self.bytes_uploaded / 1024.0 / 1024.0:.2f} MB | 📦 Packed: {self.files_packed} 📂 Unpacked: {self.files_unpacked} | 🖥️  x86: {self.files_x86} 💻 x64: {self.files_amd64}")
            self.last_report_time = time.time()


stats = Stats()


def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")


console.print(f"{get_timestamp()} [bold green]🌐 NETWORK[/bold green] Fetching pad links list...")
links = requests.get('https://www.nirsoft.net/pad/pad-links.txt')
stats.add_download(len(links.content))
links = links.text.split('\n')
updated = 0
total = len(links)
console.print(f"{get_timestamp()} [bold blue]ℹ️  INFO[/bold blue] Found {total} links to process")
prevs = glob.glob(r'h:\*')
console.print(f"{get_timestamp()} [bold yellow]🗑️  CLEANUP[/bold yellow] Removing {len(prevs)} previous temporary directories...")
for prev in prevs:
    console.print(f"{get_timestamp()} [yellow]🗑️  DELETE[/yellow] {prev}")
    try:
        shutil.rmtree(prev)
    except Exception as e:
        console.print(f"{get_timestamp()} [red]❌ ERROR[/red] Failed to remove {prev}: {e}")
        pass
tot = 0.
for link in links:
    # Report stats every 2 seconds
    if stats.should_report():
        stats.report_and_reset()

    if link == '':
        console.print(f"{get_timestamp()} [bold green]✅ COMPLETE[/bold green] Done with all links, updated={updated}")
        break
    updated += 1

    console.print(f"{get_timestamp()} [bold green]🌐 NETWORK[/bold green] Fetching metadata: {link}")
    xm = requests.get(link)
    stats.add_download(len(xm.content))
    xm = xm.text
    xm = json.dumps(xm)
    url = re.findall(r'Primary_Download_URL>(.*?)</Prim', xm)

    while True:
        try:
            console.print(f"{get_timestamp()} [bold green]⬇️  DOWNLOAD[/bold green] {url[0]}")
            resp = requests.get(url[0], headers={
                'Referer': 'https://www.nirsoft.net/pad/index.html',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4'
            })
            stats.add_download(len(resp.content))
        except Exception as e:
            console.print(f"{get_timestamp()} [red]❌ ERROR[/red] Failed with {url[0]}: {e}")
            time.sleep(5)
            continue
        break

    if resp.status_code != 200:
        console.print(f"{get_timestamp()} [red]❌ ERROR[/red] HTTP {resp.status_code} for {url[0]}")
        continue

    data = resp.content

    rand_name = str(random.randrange(10, 99120)) + '.zip'
    dr = str(random.randrange(1, 100000))
    os.makedirs(rf'h:\\\{dr}')
    drr = os.path.join(r'h:\\', dr)
    path = os.path.join(r'h:\\', dr, rand_name)

    with open(path, 'wb') as f:
        console.print(f"{get_timestamp()} [bold magenta]💾 WRITE[/bold magenta] Writing {len(data)} bytes to {path}")
        ret = f.write(data)
        stats.add_pack()
        console.print(f"{get_timestamp()} [magenta]💾 WRITE[/magenta] Wrote {ret} bytes")

    sz = f'{os.path.getsize(path) / 1024.0 / 1024.0:.2f} MB'
    tot += float(os.path.getsize(path) / 1024.0 / 1024.0)
    console.print(
        f"{get_timestamp()} [bold blue]📈 PROGRESS[/bold blue] [{updated:d}/{total:d}] ({(updated / total) * 100.0:.2f}%) | File: {sz} | Total: {tot:.2f} MB")

    os.chdir(drr)

    console.print(f"{get_timestamp()} [bold cyan]📂 UNPACK[/bold cyan] Extracting {path}...")
    os.system(f'7z x -p0 -bb0 {path} > o')
    stats.add_unpack()

    # find exe
    console.print(f"{get_timestamp()} [bold blue]🔍 SEARCH[/bold blue] Looking for executable in {url}")

    try:
        exepath = re.findall(r'\w+/\w+/([^.]{1,25})\.zip', url[0])
        name = exepath[0]
        console.print(f"{get_timestamp()} [bold blue]ℹ️  INFO[/bold blue] Package name: {name}")
        exes = glob.glob(f'*.exe')

        epath = os.path.join(drr, exes[0])
        console.print(f"{get_timestamp()} [bold green]✅ FOUND[/bold green] Executable: {epath}")
    except Exception as e:
        console.print(f"{get_timestamp()} [red]❌ ERROR[/red] EXE is broken or infected: {e}")
        continue

    try:
        pe = pefile.PE(epath)
    except Exception as e:
        console.print(f"{get_timestamp()} [red]🔒 ERROR[/red] POSSIBLY PASSWORDED")
        continue
    # determine x64

    console.print(
        f"{get_timestamp()} [bold blue]🔧 ANALYZE[/bold blue] Machine type: {pe.FILE_HEADER.Machine} (0x{pe.FILE_HEADER.Machine:x})")
    # unpack again to proper nirsoft dir

    if pe.FILE_HEADER.Machine == 0x8664:
        dest = os.path.join(rf'T:\!power-tools\NirLauncher\NirSoft\x64\{name}')
        console.print(f"{get_timestamp()} [bold blue]ℹ️  INFO[/bold blue] Detected x64 architecture")
        stats.add_amd64()
    elif pe.FILE_HEADER.Machine == 0x14c:
        dest = os.path.join(rf'T:\!power-tools\NirLauncher\NirSoft\{name}')
        console.print(f"{get_timestamp()} [bold blue]ℹ️  INFO[/bold blue] Detected x86 architecture")
        stats.add_x86()
    else:
        console.print(
            f"{get_timestamp()} [red]❌ ERROR[/red] Unknown machine type: {pe.FILE_HEADER.Machine} (0x{pe.FILE_HEADER.Machine:x})")
        time.sleep(10)

    console.print(f"{get_timestamp()} [bold cyan]📂 UNPACK[/bold cyan] Extracting to final destination: {dest}")
    try:
        os.system(rf'7z x -p0 -y -bb0 {path} -o{dest} > o')
        stats.add_unpack()
        pe.close()
        os.chdir(r"h:\\")
        shutil.rmtree(drr)
        console.print(f"{get_timestamp()} [bold green]✅ COMPLETE[/bold green] Successfully processed {name}")
    except Exception as e:
        console.print(f"{get_timestamp()} [red]❌ ERROR[/red] Exception: {e}")

# Final stats report
console.print(f"\n{get_timestamp()} [bold cyan]📊 FINAL STATS[/bold cyan]")
console.print(f"{get_timestamp()} [cyan]⬇️  Total Downloaded:[/cyan] {stats.bytes_downloaded / 1024.0 / 1024.0:.2f} MB")
console.print(f"{get_timestamp()} [cyan]⬆️  Total Uploaded:[/cyan] {stats.bytes_uploaded / 1024.0 / 1024.0:.2f} MB")
console.print(f"{get_timestamp()} [cyan]📦 Files Packed:[/cyan] {stats.files_packed}")
console.print(f"{get_timestamp()} [cyan]📂 Files Unpacked:[/cyan] {stats.files_unpacked}")
console.print(f"{get_timestamp()} [cyan]🖥️  x86 Files:[/cyan] {stats.files_x86}")
console.print(f"{get_timestamp()} [cyan]💻 x64 Files:[/cyan] {stats.files_amd64}")
console.print(f"{get_timestamp()} [bold green]✅ ALL DONE![/bold green]")
