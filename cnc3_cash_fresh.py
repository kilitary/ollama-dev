"""
C&C3 Human Cash Struct Finder - Fresh scan, delta-narrowing approach.
No hardcoded address. Three snapshots pin cash by income tick behavior.
Then dumps full struct around confirmed address.
"""
import ctypes
import ctypes.wintypes as wt
import struct
import sys
import time
from collections import defaultdict

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFO = 0x0400
MEM_COMMIT = 0x1000
PAGE_GUARD = 0x100

kernel32 = ctypes.windll.kernel32


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_size_t),
        ("AllocationBase", ctypes.c_size_t),
        ("AllocationProtect", wt.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wt.DWORD),
        ("Protect", wt.DWORD),
        ("Type", wt.DWORD),
    ]


PID = 11744
# Wide range to catch cash even if it ticks up slightly during scan
CASH_MIN = 0
CASH_MAX = 2000
READ_CHUNK = 4 * 1024 * 1024

hProc = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, False, PID)
if not hProc:
    print(f"OpenProcess failed: {kernel32.GetLastError()}");
    sys.exit(1)

rd = ctypes.c_size_t(0)


def read4(addr):
    buf = (ctypes.c_char * 4)()
    ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(addr), buf, 4, ctypes.byref(rd))
    return struct.unpack('<i', bytes(buf))[0] if (ok and rd.value == 4) else None


def read_block(addr, size):
    buf = (ctypes.c_char * size)()
    ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(addr), buf, size, ctypes.byref(rd))
    return bytes(buf[:rd.value]) if ok else b''


# ── Collect RW heap regions ───────────────────────────────────────────────────
mbi = MEMORY_BASIC_INFORMATION()
addr = 0
regions = []
while kernel32.VirtualQueryEx(
        hProc, ctypes.c_void_p(addr),
        ctypes.byref(mbi), ctypes.sizeof(mbi)
):
    if (mbi.State == MEM_COMMIT
            and not (mbi.Protect & PAGE_GUARD)
            and mbi.Protect in (0x02, 0x04)
            and mbi.RegionSize >= 4096
            and mbi.BaseAddress > 0x1000000):
        regions.append((mbi.BaseAddress, mbi.RegionSize))
    nxt = mbi.BaseAddress + mbi.RegionSize
    if nxt <= addr:
        break
    addr = nxt

print(f"PID {PID} — {len(regions)} RW regions")


# ── Full snapshot into dict addr->value ──────────────────────────────────────
def full_scan():
    vmap = {}
    for base, size in regions:
        offset = 0
        while offset < size:
            chunk = min(READ_CHUNK, size - offset)
            buf = (ctypes.c_char * chunk)()
            ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(base + offset),
                                            buf, chunk, ctypes.byref(rd))
            if not ok or rd.value < 4:
                offset += chunk
                continue
            raw = bytes(buf[:rd.value])
            for i in range(0, len(raw) - 3, 4):
                v = struct.unpack_from('<i', raw, i)[0]
                if CASH_MIN <= v <= CASH_MAX:
                    vmap[base + offset + i] = v
            offset += chunk
    return vmap


print(f"\nSnapshot S0 (t=0)…")
s0 = full_scan()
print(f"  {len(s0)} addresses in [{CASH_MIN}–{CASH_MAX}]")

print(f"Waiting 4s…");
time.sleep(4)
print(f"Snapshot S1 (t=4s)…")
s1 = full_scan()
print(f"  {len(s1)} addresses in range")

print(f"Waiting 4s…");
time.sleep(4)
print(f"Snapshot S2 (t=8s)…")
s2 = full_scan()
print(f"  {len(s2)} addresses in range")

# ── Intersect: addresses present in all 3, value changed at least once ────────
common = set(s0) & set(s1) & set(s2)
print(f"\nCommon to all 3 snapshots: {len(common)}")

changed = {a for a in common if s0[a] != s1[a] or s1[a] != s2[a]}
stable = {a for a in common if s0[a] == s1[a] == s2[a]}
grew = {a for a in changed if s2[a] > s0[a]}
shrank = {a for a in changed if s2[a] < s0[a]}

print(f"  Changed:  {len(changed)}")
print(f"  Growing:  {len(grew)}")
print(f"  Shrinking:{len(shrank)}")
print(f"  Stable:   {len(stable)}")

# ── Score candidates ──────────────────────────────────────────────────────────
scored = []
for a in changed:
    t0, t1, t2 = s0[a], s1[a], s2[a]
    d1, d2 = t1 - t0, t2 - t1
    total = t2 - t0
    rate_min = total / 8 * 60  # cr/min

    # Human cash: slow steady income 200-6000 cr/min OR any spend event
    income_ok = 100 <= rate_min <= 8000
    spend_ok = d1 < 0 or d2 < 0
    in_range = all(CASH_MIN <= x <= 20000 for x in (t0, t1, t2))
    score = (income_ok * 6) + (spend_ok * 5) + (in_range * 3) + (abs(total) < 50000) * 2

    if score > 0:
        scored.append((score, a, t0, t1, t2, d1, d2, rate_min))

scored.sort(key=lambda x: -x[0])

print(f"\n{'=' * 72}")
print("TOP CANDIDATES")
print(f"{'=' * 72}")
print(f"  {'Sc':>3}  {'Addr':>12}  {'t=0':>7}  {'t=4s':>7}  {'t=8s':>7}  {'Δ1':>7}  {'Δ2':>7}  {'cr/min':>8}")
print("  " + "─" * 65)
for row in scored[:25]:
    sc, a, t0, t1, t2, d1, d2, rm = row
    print(f"  {sc:>3}  0x{a:>10X}  {t0:>7}  {t1:>7}  {t2:>7}  {d1:>+7}  {d2:>+7}  {rm:>8.0f}")

if not scored:
    print("  No changing candidates found.")
    print("\n  Stable unique values (1-3 occurrences, possible cash):")
    # count occurrences across all 3
    all_vals = defaultdict(list)
    for a in stable:
        all_vals[s0[a]].append(a)
    for v, addrs in sorted(all_vals.items()):
        if 1 <= len(addrs) <= 3:
            print(f"    val={v:>6}  n={len(addrs)}  addrs={'  '.join(f'0x{a:X}' for a in addrs)}")
    sys.exit(0)

best = scored[0]
sc, best_addr, t0, t1, t2, d1, d2, rm = best
live = read4(best_addr) or t2

print(f"\n► BEST: 0x{best_addr:X}  live={live}  score={sc}  ~{rm:.0f} cr/min")

# ── Struct dump ───────────────────────────────────────────────────────────────
BEFORE = 256
AFTER = 512
dump_start = best_addr - BEFORE
raw = read_block(dump_start, BEFORE + AFTER)

print(f"\n{'=' * 72}")
print(f"STRUCT DUMP  0x{dump_start:X}  cash@+0x{BEFORE:X}")
print(f"{'=' * 72}")
print(f"  {'off':>+6}  {'addr':>12}  {'i32':>10}  {'u32':>10}  {'f32':>10}  {'hex':>10}  note")
print("  " + "─" * 72)


def guess(rel, vi, vu, vf, raw4):
    if rel == 0:            return "◄◄◄ CASH"
    if vi == -1:            return "sentinel -1"
    if vi == 0:             return ""
    if 0x400000 <= vu <= 0x0FFFFFFF: return f"ptr 0x{vu:X}"
    if 1 <= vi <= 8:        return f"small({vi})"
    if 100 <= vi <= 60000:  return f"resource/score"
    if 0 < vf < 100000 and abs(vf - round(vf)) < 1: return f"float~{vf:.0f}"
    return ""


for i in range(0, len(raw) - 3, 4):
    abs_a = dump_start + i
    vi = struct.unpack_from('<i', raw, i)[0]
    vu = struct.unpack_from('<I', raw, i)[0]
    vf = struct.unpack_from('<f', raw, i)[0]
    rel = i - BEFORE
    near = abs(rel) <= 128
    g = guess(rel, vi, vu, vf, raw[i:i + 4])
    sig = bool(g) or (1 <= vi <= 500000) or (0x400000 <= vu <= 0x0FFFFFFF)
    if not (near or sig): continue
    rel_s = f"{rel:+d}" if rel != 0 else "CASH"
    print(f"  {rel_s:>+6}  0x{abs_a:>10X}  {vi:>10}  {vu:>10}  {vf:>10.2f}  0x{vu:08X}  {g}")

# ── Hex dump ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 72}")
print("HEX DUMP (16 bytes/row)")
print(f"{'=' * 72}")
for row_off in range(0, len(raw), 16):
    abs_r = dump_start + row_off
    row_b = raw[row_off:row_off + 16]
    hex_p = ' '.join(f"{b:02X}" for b in row_b)
    asc_p = ''.join(chr(b) if 32 <= b < 127 else '.' for b in row_b)
    mark = "◄" if dump_start + row_off <= best_addr < dump_start + row_off + 16 else " "
    print(f"  {mark} 0x{abs_r:08X}  {hex_p:<47}  {asc_p}")

# ── Context window: ±32 dwords with guesses ────────────────────────────────
print(f"\n{'=' * 72}")
print("CONTEXT ±32 dwords")
print(f"{'=' * 72}")
ctx_raw = read_block(best_addr - 128, 260)
print(f"  {'rel':>+6}  {'addr':>12}  {'i32':>10}  {'hex':>10}  note")
print("  " + "─" * 52)
for i in range(0, len(ctx_raw) - 3, 4):
    ca = best_addr - 128 + i
    vi = struct.unpack_from('<i', ctx_raw, i)[0]
    vu = struct.unpack_from('<I', ctx_raw, i)[0]
    vf = struct.unpack_from('<f', ctx_raw, i)[0]
    rel = ca - best_addr
    g = guess(rel, vi, vu, vf, ctx_raw[i:i + 4])
    print(f"  {rel:>+6}  0x{ca:>10X}  {vi:>10}  0x{vu:08X}  {g}")

kernel32.CloseHandle(hProc)
print("\nHandle closed.")
