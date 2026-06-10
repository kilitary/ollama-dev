"""
C&C3 Cash Finder — Double-read delta method (READ-ONLY, SAFE)
Reads candidate addresses twice with a 2s gap.
Values that CHANGE between reads are live game state (cash with income).
Values that don't change are static/tables.
"""
import ctypes
import ctypes.wintypes as wt
import struct
import sys
import time
from collections import defaultdict

PROCESS_VM_READ    = 0x0010
PROCESS_QUERY_INFO = 0x0400
MEM_COMMIT         = 0x1000
PAGE_NOACCESS      = 0x01
PAGE_GUARD         = 0x100

kernel32 = ctypes.windll.kernel32

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

def read_int32(hProc, addr):
    buf = (ctypes.c_char * 4)()
    rd  = ctypes.c_size_t(0)
    ok  = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(addr),
                                     buf, 4, ctypes.byref(rd))
    if ok and rd.value == 4:
        return struct.unpack('<i', bytes(buf))[0]
    return None

def read_context(hProc, addr, n=16):
    """Read n*4 bytes before and after addr, return as list of int32s."""
    base = addr - n * 4
    size = (n * 2 + 1) * 4
    buf  = (ctypes.c_char * size)()
    rd   = ctypes.c_size_t(0)
    kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(base), buf, size, ctypes.byref(rd))
    raw  = bytes(buf[:rd.value])
    vals = []
    for i in range(0, len(raw) - 3, 4):
        vals.append((base + i, struct.unpack_from('<i', raw, i)[0]))
    return vals

# ─────────────────────────────────────────────────────────────────────────────
hProc = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, False, PID)
if not hProc:
    print(f"OpenProcess failed: {kernel32.GetLastError()}")
    sys.exit(1)
print(f"Opened PID {PID} read-only\n")

# ── Quick scan narrow range (5000–30000, appear 3–8 times in heap) ─────────────
CASH_MIN, CASH_MAX = 5000, 30000

mbi    = MEMORY_BASIC_INFORMATION()
addr   = 0
regions = []
while kernel32.VirtualQueryEx(hProc, ctypes.c_void_p(addr),
                               ctypes.byref(mbi), ctypes.sizeof(mbi)):
    if (mbi.State == MEM_COMMIT
            and mbi.Protect not in (PAGE_NOACCESS,)
            and not (mbi.Protect & PAGE_GUARD)
            and mbi.RegionSize >= 4096
            and mbi.Protect in (0x02, 0x04)
            # heap only: skip low addresses (module images)
            and mbi.BaseAddress > 0x1000000):
        regions.append((mbi.BaseAddress, mbi.RegionSize))
    nxt = mbi.BaseAddress + mbi.RegionSize
    if nxt <= addr: break
    addr = nxt

READ_CHUNK = 4 * 1024 * 1024
read_count = ctypes.c_size_t(0)
val_map    = defaultdict(list)

print(f"Scanning {len(regions)} heap regions for int32 [{CASH_MIN}–{CASH_MAX}]...")
for base, size in regions:
    offset = 0
    while offset < size:
        chunk = min(READ_CHUNK, size - offset)
        buf   = (ctypes.c_char * chunk)()
        ok    = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(base + offset),
                                           buf, chunk, ctypes.byref(read_count))
        if not ok or read_count.value < 4:
            offset += chunk
            continue
        raw = bytes(buf[:read_count.value])
        for i in range(0, len(raw) - 3, 4):
            v = struct.unpack_from('<i', raw, i)[0]
            if CASH_MIN <= v <= CASH_MAX:
                val_map[v].append(base + offset + i)
        offset += chunk

# Keep only values appearing 3–8 times (player count range)
candidates = {v: a for v, a in val_map.items() if 3 <= len(a) <= 8}
print(f"Found {len(candidates)} values with 3–8 occurrences\n")

# ── Snapshot 1 ────────────────────────────────────────────────────────────────
print("Taking snapshot 1...")
snap1 = {}
for val, addrs in candidates.items():
    for a in addrs:
        snap1[a] = read_int32(hProc, a)

# Wait for income tick (C&C3 income ~every 3–5 sec)
print("Waiting 4 seconds for income tick...")
time.sleep(4)

# ── Snapshot 2 ────────────────────────────────────────────────────────────────
print("Taking snapshot 2...")
snap2 = {}
for a in snap1:
    snap2[a] = read_int32(hProc, a)

# ── Compare: changed addresses = live cash ────────────────────────────────────
changed = {a: (snap1[a], snap2[a]) for a in snap1
           if snap2[a] is not None and snap1[a] != snap2[a]}

print(f"\n{'='*70}")
print(f"CHANGED VALUES (live game cash): {len(changed)} addresses")
print(f"{'='*70}")
print(f"  {'Address':>18}  {'Before':>8}  {'After':>8}  {'Delta':>8}")

# Group changed by proximity (addresses within 4KB = same player struct)
if changed:
    sorted_chg = sorted(changed.items())
    groups = []
    grp = [sorted_chg[0]]
    for prev, cur in zip(sorted_chg, sorted_chg[1:]):
        if cur[0] - prev[0] < 0x10000:
            grp.append(cur)
        else:
            groups.append(grp)
            grp = [cur]
    groups.append(grp)

    for gi, grp in enumerate(groups):
        print(f"\n  ── Group {gi+1} (base ~0x{grp[0][0]:X}) ──")
        for addr, (v1, v2) in grp:
            delta = v2 - v1 if v1 and v2 else '?'
            delta_str = f"+{delta}" if isinstance(delta, int) and delta > 0 else str(delta)
            print(f"  0x{addr:>16X}  {v1:>8}  {v2:>8}  {delta_str:>8}")

        # Read context around first address in group
        ctx = read_context(hProc, grp[0][0], n=8)
        if ctx:
            print(f"\n    Context around 0x{grp[0][0]:X} (±8 int32s):")
            for ca, cv in ctx:
                marker = " ◄ CASH" if abs(cv - grp[0][1][1]) < 100 else ""
                if 0 < cv < 500000:
                    print(f"      0x{ca:X}  {cv:>10}{marker}")
else:
    print("  No values changed — game may be paused or income hasn't ticked.")
    print("  Top stable candidates (most likely cash addresses):")
    stable = sorted(candidates.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for val, addrs in stable:
        print(f"    val={val:>8}  count={len(addrs)}  addrs: "
              + "  ".join(f"0x{a:X}" for a in addrs))

kernel32.CloseHandle(hProc)
print("\nHandle closed. Done.")
