"""
C&C3 Human Player Cash Finder (READ-ONLY)
Human cash is unique in memory (appears 1-2x) vs AI which clusters at similar values 3-8x.
Three snapshots capture both income AND spend events.
"""
import ctypes, ctypes.wintypes as wt, struct, time, sys
from collections import defaultdict

PROCESS_VM_READ    = 0x0010
PROCESS_QUERY_INFO = 0x0400
MEM_COMMIT         = 0x1000
PAGE_NOACCESS      = 0x01
PAGE_GUARD         = 0x100

kernel32 = ctypes.windll.kernel32

class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress",    ctypes.c_size_t),
        ("AllocationBase", ctypes.c_size_t),
        ("AllocationProtect", wt.DWORD),
        ("RegionSize",     ctypes.c_size_t),
        ("State",          wt.DWORD),
        ("Protect",        wt.DWORD),
        ("Type",           wt.DWORD),
    ]

PID = 22756

# Known AI cash addresses — exclude these
KNOWN_AI = {
    0x3091AE8, 0x41B1AE8, 0x41BE388, 0x522A88C,
    0x56F0590, 0x6113D60, 0x6353580, 0x6490088,
    0xC9F8998, 0xC9F89B8, 0xCA7FACC, 0xCBCFA18, 0xCBE96C0,
}

hProc = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, False, PID)
if not hProc:
    print(f"OpenProcess failed: {kernel32.GetLastError()}"); sys.exit(1)

# ── Collect heap RW regions (skip module images at low addresses) ─────────────
mbi    = MEMORY_BASIC_INFORMATION()
addr   = 0
regions = []
while kernel32.VirtualQueryEx(hProc, ctypes.c_void_p(addr),
                               ctypes.byref(mbi), ctypes.sizeof(mbi)):
    if (mbi.State == MEM_COMMIT
            and not (mbi.Protect & PAGE_GUARD)
            and mbi.Protect in (0x02, 0x04)
            and mbi.RegionSize >= 4096
            and mbi.BaseAddress > 0x1000000):
        regions.append((mbi.BaseAddress, mbi.RegionSize))
    nxt = mbi.BaseAddress + mbi.RegionSize
    if nxt <= addr: break
    addr = nxt

# Human cash range: allow wide window to catch any playstyle
CASH_MIN, CASH_MAX = 200, 120_000

READ_CHUNK = 4 * 1024 * 1024
rd         = ctypes.c_size_t(0)

print(f"Opened PID {PID} read-only")
print(f"Scanning {len(regions)} heap RW regions for unique int32 [{CASH_MIN}–{CASH_MAX}]...")

val_map = defaultdict(list)

for base, size in regions:
    offset = 0
    while offset < size:
        chunk = min(READ_CHUNK, size - offset)
        buf   = (ctypes.c_char * chunk)()
        ok    = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(base + offset),
                                           buf, chunk, ctypes.byref(rd))
        if not ok or rd.value < 4:
            offset += chunk; continue
        raw = bytes(buf[:rd.value])
        for i in range(0, len(raw) - 3, 4):
            v = struct.unpack_from('<i', raw, i)[0]
            if CASH_MIN <= v <= CASH_MAX:
                val_map[v].append(base + offset + i)
        offset += chunk

# Keep values appearing 1-2 times (unique = human player)
unique = {v: a for v, a in val_map.items()
          if 1 <= len(a) <= 2 and not any(x in KNOWN_AI for x in a)}

print(f"Unique-occurrence candidates: {len(unique)}\n")

# ── 3 snapshots: t=0, t=2s, t=4s ─────────────────────────────────────────────
def snap_all(addrs_flat):
    out = {}
    for a in addrs_flat:
        buf = (ctypes.c_char * 4)()
        ok  = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(a), buf, 4, ctypes.byref(rd))
        out[a] = struct.unpack('<i', bytes(buf))[0] if (ok and rd.value == 4) else None
    return out

all_addrs = [a for addrs in unique.values() for a in addrs]

print("Snapshot 1 (t=0s)…")
s0 = snap_all(all_addrs)
print("Waiting 2s…"); time.sleep(2)
print("Snapshot 2 (t=2s)…")
s1 = snap_all(all_addrs)
print("Waiting 2s…"); time.sleep(2)
print("Snapshot 3 (t=4s)…")
s2 = snap_all(all_addrs)

# ── Classify by behavior ──────────────────────────────────────────────────────
# Human cash characteristics:
#   - Consistent slow growth (harvester income ~400-3000 cr/2s)
#   - OR decrease (spent on units)
#   - Value in realistic range (200 – 120,000)
results = []
for v0, addrs in unique.items():
    for a in addrs:
        t0 = s0.get(a); t1 = s1.get(a); t2 = s2.get(a)
        if None in (t0, t1, t2): continue
        d1 = t1 - t0   # delta first 2s
        d2 = t2 - t1   # delta second 2s

        # Positive but irregular (human income tick is ~2-4 cr/s per harvester)
        # Total delta over 4s: 8-12 cr for 1 harvester, 16-24 for 2, etc.
        total_d = t2 - t0
        rate_min = total_d / 4 * 60  # cr/min if constant

        # Income rate plausible for human (1-5 harvesters = 600-6000 cr/min)
        # OR spending (negative delta at some point)
        spent   = d1 < 0 or d2 < 0
        income  = 300 <= rate_min <= 8000
        stable  = total_d == 0
        growing = total_d > 0

        if spent or income:
            results.append((a, t0, t1, t2, d1, d2, rate_min, spent, income))

# Sort: spending events first (most human-like), then by income rate
results.sort(key=lambda x: (not x[7], abs(x[8] - 1500)))

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("HUMAN PLAYER CASH CANDIDATES")
print(f"{'='*72}")
print(f"  {'Addr':>18}  {'t=0':>8}  {'t=2s':>8}  {'t=4s':>8}  "
      f"{'Δ1':>7}  {'Δ2':>7}  {'~cr/min':>8}  Note")
print("  " + "─" * 70)

shown = 0
for (a, t0, t1, t2, d1, d2, rate_min, spent, income) in results[:30]:
    spent_tag  = " 💸SPENT" if spent else ""
    income_tag = f" ✅ ~{rate_min:.0f}cr/min" if income else ""
    print(f"  0x{a:>16X}  {t0:>8}  {t1:>8}  {t2:>8}  "
          f"{d1:>+7}  {d2:>+7}  {rate_min:>8.0f}  {spent_tag}{income_tag}")
    shown += 1

if not results:
    print("  No clear unique cash found in 4s window.")
    print("  Game might be paused, or human has very high/low cash.")
    print("\n  All unique-occurrence values in range:")
    for v, addrs in sorted(unique.items(), key=lambda x: x[0]):
        print(f"    val={v:>8}  addr={'  '.join(f'0x{a:X}' for a in addrs)}")

# ── Best single guess ─────────────────────────────────────────────────────────
if results:
    best = results[0]
    a, t0, t1, t2 = best[0], best[1], best[2], best[3]
    rate = best[5]
    # Re-read one more time for freshest value
    buf = (ctypes.c_char * 4)()
    kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(a), buf, 4, ctypes.byref(rd))
    live = struct.unpack('<i', bytes(buf))[0] if rd.value == 4 else t2
    print(f"\n{'='*72}")
    print(f"  ► BEST GUESS: Human player cash = {live:,} credits")
    print(f"    Address: 0x{a:X}")
    print(f"    Income: ~{rate:.0f} cr/min  ({rate/60:.1f} cr/s)")
    print(f"{'='*72}")

kernel32.CloseHandle(hProc)
print("\nHandle closed.")
