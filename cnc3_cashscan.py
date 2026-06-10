"""
C&C3 — AI Player Cash Scanner (READ-ONLY)
Scans process memory for float values in typical cash ranges.
Groups close candidates that resemble player structs.
"""
import ctypes
import ctypes.wintypes as wt
import struct
import sys

# ── Win32 ────────────────────────────────────────────────────────────────────
PROCESS_VM_READ      = 0x0010
PROCESS_QUERY_INFO   = 0x0400
MEM_COMMIT           = 0x1000
PAGE_NOACCESS        = 0x01
PAGE_GUARD           = 0x100
LIST_MODULES_ALL     = 0x03

kernel32 = ctypes.windll.kernel32
psapi    = ctypes.windll.psapi

class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress",       ctypes.c_size_t),
        ("AllocationBase",    ctypes.c_size_t),
        ("AllocationProtect", wt.DWORD),
        ("RegionSize",        ctypes.c_size_t),
        ("State",             wt.DWORD),
        ("Protect",           wt.DWORD),
        ("Type",              wt.DWORD),
    ]

PID = 22756

# Cash candidates: C&C3 skirmish starting amounts + income accumulation range
CASH_MIN =    50.0
CASH_MAX = 200_000.0

# ── Open read-only ────────────────────────────────────────────────────────────
hProc = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, False, PID)
if not hProc:
    print(f"OpenProcess failed: {kernel32.GetLastError()}")
    sys.exit(1)
print(f"Opened PID {PID} (read-only handle 0x{hProc:X})")

# ── Find cnc3game.dat module base ─────────────────────────────────────────────
HMODULE  = ctypes.c_size_t   # pointer-sized on both 32/64-bit
hMods    = (HMODULE * 1024)()
cbNeeded = wt.DWORD(0)
psapi.EnumProcessModulesEx(hProc, hMods, ctypes.sizeof(hMods),
                            ctypes.byref(cbNeeded), LIST_MODULES_ALL)
n_mods = cbNeeded.value // ctypes.sizeof(HMODULE)

module_map = {}
for i in range(n_mods):
    name_buf = ctypes.create_string_buffer(512)
    psapi.GetModuleFileNameExA(hProc, ctypes.c_void_p(hMods[i]), name_buf, 512)
    mod_name = name_buf.value.decode(errors='replace').split('\\')[-1].lower()
    module_map[mod_name] = hMods[i]

print(f"Loaded modules: {n_mods}  |  cnc3game.dat base: "
      f"0x{module_map.get('cnc3game.dat', 0):X}")

# ── Collect readable heap/data regions (skip small stack regions) ─────────────
mbi    = MEMORY_BASIC_INFORMATION()
addr   = 0
regions = []
while kernel32.VirtualQueryEx(hProc, ctypes.c_void_p(addr),
                               ctypes.byref(mbi), ctypes.sizeof(mbi)):
    if (mbi.State == MEM_COMMIT
            and mbi.Protect not in (PAGE_NOACCESS,)
            and not (mbi.Protect & PAGE_GUARD)
            and mbi.RegionSize >= 4096          # skip tiny regions
            and mbi.Protect in (0x02, 0x04,     # RW / RW+copy
                                 0x20, 0x40)):   # exec+RW
        regions.append((mbi.BaseAddress, mbi.RegionSize))
    next_addr = mbi.BaseAddress + mbi.RegionSize
    if next_addr <= addr:
        break
    addr = next_addr

total_mb = sum(r[1] for r in regions) / 1024 / 1024
print(f"Scanning {len(regions)} RW regions, {total_mb:.1f} MB total...\n")

# ── Scan for floats in cash range ─────────────────────────────────────────────
READ_CHUNK = 4 * 1024 * 1024   # 4 MB chunks
read_count = ctypes.c_size_t(0)
candidates = []                  # (address, float_value)

for base, size in regions:
    offset = 0
    while offset < size:
        chunk_size = min(READ_CHUNK, size - offset)
        buf        = (ctypes.c_char * chunk_size)()
        ok = kernel32.ReadProcessMemory(hProc,
                                        ctypes.c_void_p(base + offset),
                                        buf, chunk_size,
                                        ctypes.byref(read_count))
        if not ok or read_count.value < 4:
            offset += chunk_size
            continue

        raw = bytes(buf[:read_count.value])
        # Scan every 4 bytes aligned
        for i in range(0, len(raw) - 3, 4):
            val = struct.unpack_from('<f', raw, i)[0]
            if CASH_MIN <= val <= CASH_MAX:
                # Filter: only round-ish numbers typical for cash
                # (cash is usually updated in increments, rarely irrational)
                if val == int(val) or (val * 10) == int(val * 10):
                    addr_abs = base + offset + i
                    candidates.append((addr_abs, val))
        offset += chunk_size

print(f"Raw float candidates in [{CASH_MIN:.0f}, {CASH_MAX:.0f}]: {len(candidates)}")

# ── Cluster nearby candidates (player structs keep fields close together) ──────
# Sort by value bucket to find repeating cash-like values
from collections import defaultdict
value_map = defaultdict(list)
for addr_v, val in candidates:
    bucket = round(val / 25) * 25      # bucket to nearest 25
    value_map[bucket].append((addr_v, val))

# Interesting buckets: value appears 1-20 times (too many = background noise)
print("\n─── Cash value clusters (1–20 occurrences — likely player data) ───")
print(f"  {'Value':>10}  {'Count':>6}  {'Addresses'}")
interesting = []
for bucket in sorted(value_map.keys()):
    entries = value_map[bucket]
    if 1 <= len(entries) <= 20:
        interesting.append((bucket, entries))

for bucket, entries in sorted(interesting, key=lambda x: -len(x[1]))[:40]:
    addrs = ', '.join(f"0x{a:X}" for a, _ in entries[:6])
    suffix = f" +{len(entries)-6} more" if len(entries) > 6 else ""
    print(f"  {entries[0][1]:>10.1f}  {len(entries):>6}  {addrs}{suffix}")

# ── Look for repeated cash value groups at nearby addresses ───────────────────
# (AI player structs in array → similar cash values within ~4KB of each other)
print("\n─── Address clusters (possible player struct arrays) ───")
if candidates:
    sorted_cands = sorted(candidates)
    groups = []
    group  = [sorted_cands[0]]
    for prev, cur in zip(sorted_cands, sorted_cands[1:]):
        if cur[0] - prev[0] < 0x2000:    # within 8KB
            group.append(cur)
        else:
            if 2 <= len(group) <= 12:
                groups.append(group)
            group = [cur]
    if 2 <= len(group) <= 12:
        groups.append(group)

    for g in groups[:20]:
        span = g[-1][0] - g[0][0]
        vals = [f"{v:.1f}" for _, v in g]
        print(f"  [{len(g)} floats within {span} bytes @ 0x{g[0][0]:X}]  values: {vals}")

kernel32.CloseHandle(hProc)
print("\nDone. Handle closed.")
