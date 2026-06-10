C&C3 Tiberium Wars - SAGE Engine Real-Time DLL Entry Point Analyzer
Decodes mangled C++ symbols and categorizes hooks for real-time game modification.

import json
import re
import subprocess
import sys
import os

# ── Try to demangle with undname (ships with MSVC / Visual Studio) ──────────
def demangle(sym):
    if not sym.startswith('?'):
        return sym
    try:
        r = subprocess.run(['undname', sym], capture_output=True, text=True, timeout=3)
        line = r.stdout.strip()
        # undname outputs several lines; grab the last non-empty one
        parts = [l.strip() for l in line.splitlines() if l.strip()]
        if parts and not parts[-1].startswith('=='):
            return parts[-1]
    except Exception:
        pass
    # Fallback: minimal regex-based decode
    m = re.match(r'\?(\\w+)@(\\w+)@@', sym)
    if m:
        return f"{m.group(2)}::{m.group(1)}(...)"
    return sym

# ── Category classification ──────────────────────────────────────────────────
CATEGORIES = {
    "🎮 Game Logic / Units": [
        "Unit","Object","Entity","Weapon","Attack","Combat","Health","Armor",
        "Damage","Speed","Move","Position","Target","Player","Team",
    ],
    "💰 Economy / Resources": [
        "Money","Cash","Resource","Power","Credit","Supply","Tiberium","Harvest",
        "Build","Queue","Upgrade","Tech","Research",
    ],
    "🗺️ World / Map": [
        "Map","World","Terrain","Cell","Tile","Region","Area","Fog","Vision",
        "Waypoint","Path",
    ],
    "📜 Scripting / Events": [
        "Script","Event","Trigger","Notify","Command","Action","AI","Mission",
        "Objective","Condition","Timer","Timer",
    ],
    "🖥️ UI / HUD (APT Flash)": [
        "Apt","Display","Render","Frame","Animation","Sprite","Text","Button",
        "Screen","Window","UI","HUD","Menu",
    ],
    "🌐 Networking / Sync": [
        "Net","Network","Packet","Sync","Multiplayer","Lobby","Session","Host",
        "Client","Message","Broadcast",
    ],
    "📷 Camera / Viewport": [
        "Camera","View","Viewport","Zoom","Pan","Scroll","FOV",
    ],
    "💾 State / Save-Load": [
        "Load","Save","State","Checkpoint","Restore","Serialize",
    ],
    "🔧 Engine Core / Memory": [
        "Pool","Memory","Alloc","Free","GC","Ref","Count","Init","Create",
        "Destroy","Manager","System",
    ],
}

def categorize(name):
    nl = name.lower()
    for cat, kws in CATEGORIES.items():
        if any(kw.lower() in nl for kw in kws):
            return cat
    return "❓ Other"

# ── IAT hooks: game-engine imports most interesting for real-time patching ────
IMPORTANT_IAT = {
    # DirectX - intercept rendering / presentation
    "d3d9.dll": "DirectX 9 → intercept Present(), CreateDevice() for render injection",
    "dinput8.dll": "DirectInput 8 → hook GetDeviceState() to inject input",
    "dsound.dll": "DirectSound → hook volume/play for audio mods",
    "xinput1_3.dll": "XInput → gamepad injection",
    # Win32 timing / threading
    "winmm.dll": "WinMM timeGetTime() → hook game timer for speed hacks",
    "kernel32.dll": "Kernel32 → VirtualAlloc, CreateThread, GetTickCount (speed/mem)",
    "user32.dll": "User32 → SendMessage, PostMessage, WM hooks for UI control",
    # Networking
    "wsock32.dll": "WinSock → intercept send()/recv() for net traffic manipulation",
    "ws2_32.dll": "WinSock2 → same as above",
}

# ── Load analysis data ───────────────────────────────────────────────────────
data = json.load(open('cnc3_analysis.json'))

out = []
def p(*args):
    line = ' '.join(str(a) for a in args)
    out.append(line)
    print(line)

p("=" * 80)
print("C&C3 TIBERIUM WARS — SAGE ENGINE REAL-TIME ENTRY POINT REPORT")
p("=" * 80)

# ── Section 1: cnc3game.dat exports (SAGE engine) ────────────────────────────
p(\n" + "─" * 80)
print("SECTION 1: cnc3game.dat — SAGE Engine Exported Symbols (494 total)")
print("These can be called directly after injecting a DLL and resolving via GetProcAddress.")
print("─" * 80)

engine_exports = data.get('cnc3game.dat', {}).get('exports', [])
categorized = {}
for e in engine_exports:
    cat = categorize(e['name'])
    categorized.setdefault(cat, []).append(e)

for cat in sorted(categorized.keys()):
    entries = categorized[cat]
    print(f"\n  {cat} ({len(entries)} symbols)")
    for e in entries:
        decoded = demangle(e['name'])
        print(f"    ord={e['ordinal']:4d}  rva={e['rva']:12s}  {decoded}")

# ── Section 2: IAT hooks in cnc3game.dat ─────────────────────────────────────
p(\n" + "─" * 80)
print("SECTION 2: cnc3game.dat — Import Address Table (IAT) Hook Candidates")
print("Hook these with a DLL injector (e.g., detours/minhook) to intercept engine calls.")
print("─" * 80)

engine_imports = data.get('cnc3game.dat', {}).get('imports', {})
for dll_key, note in IMPORTANT_IAT.items():
    # Find matching DLL (case-insensitive)
    for dll, funcs in engine_imports.items():
        if dll.lower() == dll_key:
            print(f"\n  [{dll}]  ← {note}")
            print(f"  {len(funcs)} imported functions. Top candidates:")
            RT_KW = ["Create","Update","Tick","Set","Get","Send","Recv","Play",
                     "Post","Time","Alloc","Thread","Present","Reset","Device"]
            hits = [f for f in funcs if any(k.lower() in f.lower() for k in RT_KW)]
            for fn in hits[:20]:
                print(f"    {fn}")
            if len(hits) > 20:
                print(f"    ... and {len(hits)-20} more")
            break

# ── Section 3: patchw32.dll ───────────────────────────────────────────────────
p(\n" + "─" * 80)
print("SECTION 3: patchw32.dll — RT-Patch Live Update Library")
print("EA's real-time binary patching library. These are used by the game's auto-updater")
print("and can also be exploited to apply in-memory patches while the game is running.")
print("─" * 80)

for e in data.get('patchw32.dll', {}).get('exports', []):
    print(f"  ord={e['ordinal']:4d}  rva={e['rva']:12s}  {e['name']}")

print("\n  ⚡ KEY: RTPatchApply32@12 — applies a binary diff to a loaded module in memory.")
print("  Can be called to patch cnc3game.dat at runtime without restarting the game.")

# ── Section 4: CNC3.exe single export ────────────────────────────────────────
p(\n" + "─" * 80)
print("SECTION 4: CNC3.exe — Launcher Single Export")
print("─" * 80)
for e in data.get('CNC3.exe', {}).get('exports', []):
    print(f"  ord={e['ordinal']:4d}  rva={e['rva']:12s}  {e['name']}")

# ── Section 5: Injection Strategy Summary ────────────────────────────────────
p(\n" + "=" * 80)
print("SECTION 5: RECOMMENDED REAL-TIME MODIFICATION STRATEGIES")
print("=" * 80)

STRATEGIES = [
    ("METHOD A — GetProcAddress on cnc3game.dat",
     [
         "cnc3game.dat is loaded by CNC3.exe via LoadLibrary (it's a renamed DLL).",
         "Get the base address: GetModuleHandle('cnc3game.dat')",
         "Resolve any of the 494 exported symbols with GetProcAddress.",
         "Best targets for real-time effects:",
         "  • AptAnimationTarget::SetEventHandler  (UI scripting events)",
         "  • AptCharacterTextInst::SetTextValue   (change HUD text live)",
         "  • AptExtObject::SetVariable            (set script variables)",
         "  • AptAnimationTarget::GetDisplayList   (traverse scene graph)",
         "  • AptCIH::SetDepth / SetIsVisible      (show/hide UI elements)",
     ]
    ),
    ("METHOD B — IAT Hook (recommended for game logic)",
     [
         "Inject a DLL via CreateRemoteThread + LoadLibrary into CNC3.exe.",
         "In DllMain, find cnc3game.dat's IAT and replace function pointers.",
         "Priority hooks:",
         "  • USER32:SetTimer / KillTimer          → control game tick rate",
         "  • KERNEL32:GetTickCount                → freeze/speed time",
         "  • USER32:SendMessage / PostMessage     → inject UI commands",
         "  • KERNEL32:CreateThread                → intercept new threads",
         "  • KERNEL32:VirtualAlloc                → monitor memory allocation",
         "Tools: Microsoft Detours, MinHook, or manual IAT patching.",
     ]
    ),
    ("METHOD C — patchw32 In-Memory Binary Patching",
     [
         "patchw32.dll!RTPatchApply32@12 is already loaded in the process.",
         "Create a binary diff (RTPatch format) for specific game logic bytes.",
         "Call RTPatchApply32 at runtime to apply the diff to cnc3game.dat's mapping.",
         "Effective for: changing unit stats, weapon damage, build times embedded",
         "              in the executable's data sections.",
     ]
    ),
    ("METHOD D — SAGE Script Injection via Apt scripting API",
     [
         "cnc3game.dat exports the full APT ActionScript runtime.",
         "Inject code via AptExtObject::SetVariable to modify game-visible vars.",
         "Use AptAnimationTarget::AddActionFront to queue script actions.",
         "Intercept AptActionInterpreter::stackGetPop to monitor script execution.",
         "This is the cleanest, least crash-prone approach for UI/HUD mods.",
     ]
    ),
]

for title, steps in STRATEGIES:
    print(f"\n  ▶ {title}")
    for step in steps:
        print(f"    {step}")

print("\n" + "=" * 80)
print("Full raw data: cnc3_analysis.json")
print("=" * 80)

with open('cnc3_entrypoints.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print("\n→ Report saved to cnc3_entrypoints.txt")