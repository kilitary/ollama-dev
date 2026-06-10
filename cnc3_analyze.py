"""Analyze C&C3 DLLs and EXEs for exported/imported functions (SAGE engine entry points)."""
import json
import os

import pefile
from rich.console import Console
from rich.table import Table

console = Console()

BASE = r"S:\media\games\Command and Conquer 3 - Tiberium Wars"
TARGETS = [
    r"CNC3.exe",
    r"VistaShellSupport.dll",
    r"patchw32.dll",
    r"RetailExe\1.9\cnc3game.dat",
    r"RetailExe\1.9\dbghelp.dll",
]

# Keywords suggesting real-time game-state modification entry points
REALTIME_KEYWORDS = [
    "Update", "Tick", "Logic", "Game", "Unit", "Player", "Object", "World",
    "Script", "Command", "Event", "Notify", "Set", "Get", "Apply", "Spawn",
    "Damage", "Kill", "Create", "Destroy", "Add", "Remove", "Change", "Modify",
    "Attack", "Move", "Build", "Upgrade", "Power", "Money", "Cash", "Resource",
    "Health", "Armor", "Speed", "Weapon", "Ability", "Tech", "Research",
    "Map", "Camera", "UI", "Render", "State", "Load", "Save", "Net",
    "Sync", "Frame", "Timer", "AI", "Team", "Win", "Lose", "Mission",
]

results = {}

for rel in TARGETS:
    path = os.path.join(BASE, rel)
    name = os.path.basename(path)
    if not os.path.exists(path):
        console.print(f"[yellow]MISSING:[/yellow] {path}")
        continue
    try:
        pe = pefile.PE(path, fast_load=True)
        pe.parse_data_directories(directories=[
            pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT'],
            pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
        ])

        exports = []
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                n = exp.name.decode() if exp.name else f"ord#{exp.ordinal}"
                exports.append({
                    "name": n,
                    "ordinal": exp.ordinal,
                    "rva": hex(exp.address) if exp.address else "0x0",
                })

        imports = {}
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for imp in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = imp.dll.decode()
                funcs = []
                for f in imp.imports:
                    fname = f.name.decode() if f.name else f"ord#{f.ordinal}"
                    funcs.append(fname)
                imports[dll_name] = funcs

        results[name] = {"exports": exports, "imports": imports}
        console.print(
            f"[green]OK[/green] [bold]{name}[/bold]: "
            f"[cyan]{len(exports)}[/cyan] exports, "
            f"[cyan]{sum(len(v) for v in imports.values())}[/cyan] imports "
            f"from [cyan]{len(imports)}[/cyan] DLLs"
        )
        pe.close()
    except Exception as e:
        console.print(f"[red]ERR[/red] {name}: {e}")

# Save full results
with open("cnc3_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
console.print("\n[bold]Saved raw data → cnc3_analysis.json[/bold]")

# --- Build realtime-relevant entry point report ---
console.print("\n[bold magenta]═══ REAL-TIME MODIFIABLE ENTRY POINTS ═══[/bold magenta]\n")

rt_hits = {}
for binary, data in results.items():
    hits = []
    for exp in data.get("exports", []):
        name = exp["name"]
        if any(kw.lower() in name.lower() for kw in REALTIME_KEYWORDS):
            hits.append(("EXPORT", name, exp["rva"]))
    # Also flag interesting imports (things the game calls that could be hooked)
    for dll, funcs in data.get("imports", {}).items():
        for fn in funcs:
            if any(kw.lower() in fn.lower() for kw in REALTIME_KEYWORDS):
                hits.append(("IMPORT", f"{dll}!{fn}", "—"))
    if hits:
        rt_hits[binary] = hits

# Print table per binary
for binary, hits in rt_hits.items():
    table = Table(title=f"[bold cyan]{binary}[/bold cyan]", show_lines=True)
    table.add_column("Type", style="bold yellow", width=8)
    table.add_column("Symbol", style="white")
    table.add_column("RVA / Source", style="cyan", width=14)
    for kind, sym, rva in hits:
        color = "green" if kind == "EXPORT" else "dim"
        table.add_row(f"[{color}]{kind}[/{color}]", sym, rva)
    console.print(table)

# Summary
console.print("\n[bold green]═══ SUMMARY ═══[/bold green]")
for binary, hits in rt_hits.items():
    exp_count = sum(1 for h in hits if h[0] == "EXPORT")
    imp_count = sum(1 for h in hits if h[0] == "IMPORT")
    console.print(f"  [bold]{binary}[/bold]: {exp_count} export hooks, {imp_count} import hooks")

console.print("\n[bold]Full data in cnc3_analysis.json[/bold]")
