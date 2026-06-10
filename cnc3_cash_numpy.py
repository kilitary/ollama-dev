"""
C&C3 Human Cash Finder — numpy byte-diff approach.
Stores raw region snapshots as numpy arrays, diffs in-place.
Handles 500MB process without RAM explosion.
"""
import ctypes, ctypes.wintypes as wt, struct, time, sys
import numpy as np
from collections import defaultdict

PROCESS_VM_READ    = 0x0010
PROCESS_QUERY_INFO = 0x0400
MEM_COMMIT         = 0x1000
PAGE_GUARD         = 0x100

kernel32 = ctypes.windll.kernel32

class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [("BaseAddress",ctypes.c_size_t),("AllocationBase",ctypes.c_size_t),
                ("AllocationProtect",wt.DWORD),("RegionSize",ctypes.c_size_t),
                ("State",wt.DWORD),("Protect",wt.DWORD),("Type",wt.DWORD)]

PID      = 11744
CASH_MIN = 0
CASH_MAX = 800   # wider to catch ticks during scan

hProc = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, False, PID)
if not hProc:
    print(f"OpenProcess failed: {kernel32.GetLastError()}"); sys.exit(1)

rd = ctypes.c_size_t(0)

def read4(addr):
    buf = (ctypes.c_char*4)()
    ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(int(addr)), buf, 4, ctypes.byref(rd))
    return struct.unpack('<i', bytes(buf))[0] if (ok and rd.value==4) else None

def read_region(base, size):
    buf = (ctypes.c_char * size)()
    ok  = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(base), buf, size, ctypes.byref(rd))
    n   = rd.value & ~3   # align to 4
    if not ok or n < 4: return None
    return np.frombuffer(bytes(buf[:n]), dtype=np.int32).copy()

def read_block_raw(addr, size):
    buf = (ctypes.c_char*size)()
    ok  = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(int(addr)), buf, size, ctypes.byref(rd))
    return bytes(buf[:rd.value]) if ok else b''

# ── Enumerate RW heap regions ─────────────────────────────────────────────────
mbi = MEMORY_BASIC_INFORMATION()
a   = 0
regions = []
while kernel32.VirtualQueryEx(hProc, ctypes.c_void_p(a), ctypes.byref(mbi), ctypes.sizeof(mbi)):
    if (mbi.State == MEM_COMMIT and not (mbi.Protect & PAGE_GUARD)
            and mbi.Protect in (0x02, 0x04)
            and mbi.RegionSize >= 4096 and mbi.BaseAddress > 0x1000000):
        regions.append((mbi.BaseAddress, mbi.RegionSize))
    nxt = mbi.BaseAddress + mbi.RegionSize
    if nxt <= a: break
    a = nxt

total_mb = sum(r[1] for r in regions) / 1024**2
print(f"PID {PID} — {len(regions)} RW regions ({total_mb:.0f} MB)")

# ── Phase 1: read all regions at t=0 ─────────────────────────────────────────
print("\nReading S0 (t=0)…")
snaps = {}   # base -> numpy int32 array
for base, size in regions:
    arr = read_region(base, size)
    if arr is not None:
        snaps[base] = arr
print(f"  Read {len(snaps)} regions")

print("Waiting 5s…"); time.sleep(5)

# ── Phase 2: read t=5s, compute per-region diff ──────────────────────────────
print("Reading S1 (t=5s)…")
# candidates: {addr -> (v0, v1)}
candidates = {}
for base, arr0 in snaps.items():
    size = len(arr0) * 4
    arr1 = read_region(base, size)
    if arr1 is None or len(arr1) < len(arr0):
        snaps[base] = (arr0, None)
        continue
    arr1 = arr1[:len(arr0)]
    snaps[base] = (arr0, arr1)

    # Find indices where value changed AND was in cash range in either snap
    mask_changed = arr0 != arr1
    mask_range   = (
        ((arr0 >= CASH_MIN) & (arr0 <= CASH_MAX)) |
        ((arr1 >= CASH_MIN) & (arr1 <= CASH_MAX))
    )
    hits = np.where(mask_changed & mask_range)[0]
    for idx in hits:
        abs_addr = base + idx * 4
        candidates[abs_addr] = (int(arr0[idx]), int(arr1[idx]))

print(f"  Changed & in-range candidates after S1: {len(candidates)}")

print("Waiting 5s…"); time.sleep(5)

# ── Phase 3: re-read candidate addresses ─────────────────────────────────────
print("Reading S2 (t=10s) for candidates…")
scored = []
for addr, (v0, v1) in candidates.items():
    v2 = read4(addr)
    if v2 is None: continue
    d1, d2 = v1-v0, v2-v1
    total  = v2-v0
    # cr/min if linearly extrapolated
    rate   = total / 10 * 60

    # Scoring
    in_range_all = all(CASH_MIN <= x <= CASH_MAX for x in (v0, v1, v2))
    in_range_any = any(CASH_MIN <= x <= CASH_MAX for x in (v0, v1, v2))
    income = 50 <= rate <= 8000
    spend  = d1 < 0 or d2 < 0
    gentle = abs(d1) < 10000 and abs(d2) < 10000

    sc = (income*6) + (spend*4) + (in_range_all*5) + (in_range_any*2) + (gentle*3)
    if sc > 0:
        scored.append((sc, addr, v0, v1, v2, d1, d2, rate))

scored.sort(key=lambda x: -x[0])

# De-duplicate: if multiple addresses within 4 bytes of each other, keep top
deduped = []
seen_zones = set()
for row in scored:
    zone = row[1] >> 4  # 16-byte bucket
    if zone not in seen_zones:
        seen_zones.add(zone)
        deduped.append(row)

print(f"\n{'='*72}")
print("TOP CANDIDATES")
print(f"{'='*72}")
print(f"  {'Sc':>3}  {'Addr':>12}  {'t=0':>7}  {'t=5s':>7}  {'t=10s':>7}  "
      f"{'Δ1':>7}  {'Δ2':>7}  {'cr/min':>8}")
print("  " + "─"*65)
for row in deduped[:30]:
    sc, a, v0, v1, v2, d1, d2, rm = row
    print(f"  {sc:>3}  0x{a:>10X}  {v0:>7}  {v1:>7}  {v2:>7}  "
          f"{d1:>+7}  {d2:>+7}  {rm:>8.0f}")

if not deduped:
    print("  No candidates found. Try spending cash right before/during scan.")
    kernel32.CloseHandle(hProc); sys.exit(0)

best = deduped[0]
sc, best_addr, v0, v1, v2, d1, d2, rm = best
best_addr = int(best_addr)
live = read4(best_addr) or v2
print(f"\n► BEST: 0x{best_addr:X}  live={live}  score={sc}  ~{rm:.0f} cr/min")

# ── Struct dump ───────────────────────────────────────────────────────────────
BEFORE, AFTER = 256, 512
raw = read_block_raw(best_addr - BEFORE, BEFORE + AFTER)

print(f"\n{'='*72}")
print(f"STRUCT DUMP  0x{best_addr-BEFORE:X}  cash@+0x{BEFORE:X}  ({len(raw)} bytes)")
print(f"{'='*72}")
print(f"  {'off':>7}  {'addr':>12}  {'i32':>10}  {'hex':>10}  note")
print("  " + "─"*55)

for i in range(0, len(raw)-3, 4):
    vi  = struct.unpack_from('<i', raw, i)[0]
    vu  = struct.unpack_from('<I', raw, i)[0]
    vf  = struct.unpack_from('<f', raw, i)[0]
    ca  = best_addr - BEFORE + i
    rel = i - BEFORE
    near = abs(rel) <= 128

    note = ""
    if rel == 0:                      note = "◄◄◄ CASH"
    elif vi == -1:                    note = "sentinel"
    elif 0x400000 <= vu <= 0x0FFFFFFF: note = f"ptr"
    elif 1 <= vi <= 8:                note = f"small({vi})"
    elif 100 <= vi <= 60000:          note = "resource"
    elif 0 < vf < 100000 and abs(vf-round(vf)) < 1: note = f"f~{vf:.0f}"

    sig = bool(note) or (1 <= vi <= 500000)
    if not (near or sig): continue

    rel_s = "CASH" if rel==0 else f"{rel:+d}"
    print(f"  {rel_s:>+7}  0x{ca:>10X}  {vi:>10}  0x{vu:08X}  {note}")

# ── Hex dump (±64 bytes around cash) ─────────────────────────────────────────
print(f"\n{'='*72}")
print(f"HEX DUMP ±64 bytes")
print(f"{'='*72}")
ctx_raw = read_block_raw(best_addr - 64, 128+4)
for row_off in range(0, len(ctx_raw), 16):
    abs_r = best_addr - 64 + row_off
    rb    = ctx_raw[row_off:row_off+16]
    hex_p = ' '.join(f"{b:02X}" for b in rb)
    asc_p = ''.join(chr(b) if 32<=b<127 else '.' for b in rb)
    mark  = "◄" if best_addr-64+row_off <= best_addr < best_addr-64+row_off+16 else " "
    print(f"  {mark} 0x{abs_r:08X}  {hex_p:<47}  {asc_p}")

kernel32.CloseHandle(hProc)
print("\nHandle closed.")
