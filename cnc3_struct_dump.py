"""
C&C3 Human Player Cash Struct Finder (READ-ONLY)
- Scans for unique int32 in [0, 499] range
- Multi-snapshot confirms the live cash address
- Dumps ±512 bytes around confirmed address as full struct
"""
import ctypes, ctypes.wintypes as wt, struct, time, sys
from collections import defaultdict

PROCESS_VM_READ    = 0x0010
PROCESS_QUERY_INFO = 0x0400
MEM_COMMIT         = 0x1000
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

PID        = 11744
CASH_MIN   = 0
CASH_MAX   = 499
READ_CHUNK = 4 * 1024 * 1024

# Previously confirmed address — always include in candidates
HINT_ADDR  = 0x3091AE4

hProc = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, False, PID)
if not hProc:
    print(f"OpenProcess failed: {kernel32.GetLastError()}"); sys.exit(1)

rd = ctypes.c_size_t(0)

def read4(addr):
    buf = (ctypes.c_char * 4)()
    ok  = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(addr), buf, 4, ctypes.byref(rd))
    return struct.unpack('<i', bytes(buf))[0] if (ok and rd.value == 4) else None

def read_block(addr, size):
    buf = (ctypes.c_char * size)()
    ok  = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(addr), buf, size, ctypes.byref(rd))
    return bytes(buf[:rd.value]) if ok else b''

# ── Collect RW heap regions ───────────────────────────────────────────────────
mbi     = MEMORY_BASIC_INFORMATION()
addr    = 0
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

print(f"Opened PID {PID} — scanning {len(regions)} RW regions for int32 [{CASH_MIN}–{CASH_MAX}]...")

# ── Initial scan ──────────────────────────────────────────────────────────────
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

# Unique occurrences only (1–2 = human player)
unique = {v: a for v, a in val_map.items() if 1 <= len(a) <= 2}
# Always add hint address
hint_val = read4(HINT_ADDR)
if hint_val is not None and HINT_ADDR not in [a for addrs in unique.values() for a in addrs]:
    if CASH_MIN <= hint_val <= CASH_MAX:
        unique[hint_val] = unique.get(hint_val, []) + [HINT_ADDR]
    else:
        print(f"[hint] 0x{HINT_ADDR:X} current value = {hint_val} (outside range, will still track)")
        unique[hint_val] = [HINT_ADDR]

all_addrs = list({a for addrs in unique.values() for a in addrs})
if HINT_ADDR not in all_addrs:
    all_addrs.append(HINT_ADDR)

print(f"Unique candidates: {len(all_addrs)}  (incl. hint 0x{HINT_ADDR:X})")

# ── 3 snapshots ───────────────────────────────────────────────────────────────
def snap(addrs):
    return {a: read4(a) for a in addrs}

print("\nSnapshot 1 (t=0)…")
s0 = snap(all_addrs)
print("Waiting 3s…"); time.sleep(3)
print("Snapshot 2 (t=3s)…")
s1 = snap(all_addrs)
print("Waiting 3s…"); time.sleep(3)
print("Snapshot 3 (t=6s)…")
s2 = snap(all_addrs)

# ── Score each candidate ──────────────────────────────────────────────────────
scored = []
for a in all_addrs:
    t0, t1, t2 = s0.get(a), s1.get(a), s2.get(a)
    if None in (t0, t1, t2): continue
    d1, d2 = t1 - t0, t2 - t1
    total  = t2 - t0
    changed = d1 != 0 or d2 != 0
    in_range_all = all(CASH_MIN <= x <= CASH_MAX for x in (t0, t1, t2))
    # score: prefer changed, in-range, close to hint
    score = (
        (10 if changed else 0) +
        (5  if in_range_all else 0) +
        (3  if a == HINT_ADDR else 0) +
        (2  if abs(d1) < 5000 and abs(d2) < 5000 else 0)  # gentle delta = real cash
    )
    scored.append((score, a, t0, t1, t2, d1, d2))

scored.sort(key=lambda x: -x[0])

print(f"\n{'='*80}")
print("CANDIDATES  (sorted by confidence)")
print(f"{'='*80}")
print(f"  {'Score':>5}  {'Addr':>12}  {'t=0':>7}  {'t=3s':>7}  {'t=6s':>7}  {'Δ1':>7}  {'Δ2':>7}")
print("  " + "─" * 60)
for sc, a, t0, t1, t2, d1, d2 in scored[:20]:
    hint = " ◄ HINT" if a == HINT_ADDR else ""
    print(f"  {sc:>5}  0x{a:>10X}  {t0:>7}  {t1:>7}  {t2:>7}  {d1:>+7}  {d2:>+7}{hint}")

# ── Pick best confirmed address ───────────────────────────────────────────────
best_addr = None
for sc, a, t0, t1, t2, d1, d2 in scored:
    if sc >= 3:
        best_addr = a; break
if best_addr is None and scored:
    best_addr = scored[0][1]

print(f"\n► Best candidate: 0x{best_addr:X}  current={read4(best_addr)}")

# ── Struct dump ───────────────────────────────────────────────────────────────
DUMP_BEFORE = 256   # bytes before cash addr
DUMP_AFTER  = 512   # bytes after  cash addr
dump_start  = best_addr - DUMP_BEFORE
dump_size   = DUMP_BEFORE + DUMP_AFTER
raw = read_block(dump_start, dump_size)

print(f"\n{'='*80}")
print(f"STRUCT DUMP  0x{dump_start:X} … 0x{dump_start+dump_size:X}  ({len(raw)} bytes)")
print(f"Cash offset: +0x{best_addr - dump_start:X} (0x{best_addr:X})")
print(f"{'='*80}")
print(f"  {'Offset':>6}  {'Address':>12}  {'int32':>12}  {'uint32':>12}  {'float32':>12}  {'hex':>10}  note")
print("  " + "─" * 80)

for i in range(0, len(raw) - 3, 4):
    abs_addr = dump_start + i
    vi = struct.unpack_from('<i', raw, i)[0]
    vu = struct.unpack_from('<I', raw, i)[0]
    vf = struct.unpack_from('<f', raw, i)[0]
    rel_off  = i - DUMP_BEFORE   # offset from cash address

    note = ""
    if abs_addr == best_addr:
        note = "  ◄◄◄ CASH"
    elif 0 <= vf <= 200_000 and abs(vf - round(vf)) < 0.01:
        note = f"  ~ float {vf:.1f}"
    elif 1 <= vi <= 200_000:
        note = f"  (int)"
    elif vi < 0 and vi > -100_000:
        note = f"  (neg)"

    # Only print if interesting (near cash, or has meaningful value)
    near_cash   = abs(rel_off) <= 128
    meaningful  = (1 <= vi <= 1_000_000) or (1 <= vf <= 200_000 and abs(vf - round(vf)) < 0.1)
    is_ptr      = 0x00400000 <= vu <= 0x0FFFFFFF
    if near_cash or meaningful or is_ptr:
        rel_str  = f"{rel_off:+d}" if rel_off != 0 else "  CASH"
        ptr_note = "  [PTR]" if is_ptr and not note else ""
        print(f"  {rel_str:>6}  0x{abs_addr:>10X}  {vi:>12}  {vu:>12}  {vf:>12.2f}  "
              f"0x{vu:>08X}{note}{ptr_note}")

# ── Hex dump ─────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("HEX DUMP  (16 bytes/row, cash row highlighted)")
print(f"{'='*80}")
cash_row_start = best_addr & ~0xF
for row_addr in range(dump_start & ~0xF, dump_start + dump_size, 16):
    off  = row_addr - dump_start
    if off >= len(raw) or off < -16: continue
    row_bytes = raw[max(0,off):min(len(raw),off+16)]
    if not row_bytes: continue
    hex_part  = ' '.join(f"{b:02X}" for b in row_bytes)
    asc_part  = ''.join(chr(b) if 32 <= b < 127 else '.' for b in row_bytes)
    marker    = "◄" if row_addr <= best_addr < row_addr + 16 else " "
    print(f"  {marker} 0x{row_addr:08X}  {hex_part:<47}  {asc_part}")

# ── Adjacent int32 context around cash (tight struct view) ───────────────────
print(f"\n{'='*80}")
print("TIGHT CONTEXT  (±32 dwords from cash addr)")
print(f"{'='*80}")
ctx_start = best_addr - 32 * 4
ctx_size  = 65 * 4
ctx_raw   = read_block(ctx_start, ctx_size)
known_ranges = {
    # value range → guessed field label
    (0, 1):        "bool/flag",
    (1, 8):        "player_index/slot",
    (100, 50000):  "cash/resource",
    (0, 500):      "small_counter",
    (500, 20000):  "power/score",
}
print(f"  {'Rel':>6}  {'Addr':>12}  {'i32':>10}  {'f32':>10}  guess")
print("  " + "─" * 55)
for i in range(0, len(ctx_raw) - 3, 4):
    ca  = ctx_start + i
    vi  = struct.unpack_from('<i', ctx_raw, i)[0]
    vf  = struct.unpack_from('<f', ctx_raw, i)[0]
    rel = ca - best_addr
    rel_dw = rel // 4
    guess = ""
    if ca == best_addr:
        guess = "◄◄◄ CASH"
    elif vi == 0:
        guess = "zero"
    elif vi == -1 or vi == 0xFFFFFFFF:
        guess = "all-ones/invalid"
    elif 0 < vi < 16:
        guess = f"small int ({vi})"
    elif 100 <= vi <= 50000:
        guess = f"resource/score ({vi})"
    elif 0 < vf < 200000 and abs(vf - round(vf)) < 0.5:
        guess = f"float~{vf:.0f}"
    elif 0x400000 <= vi <= 0x10000000:
        guess = f"ptr 0x{vi:X}"
    print(f"  {rel:>+6}  0x{ca:>10X}  {vi:>10}  {vf:>10.2f}  {guess}")

kernel32.CloseHandle(hProc)
print("\nHandle closed.")

