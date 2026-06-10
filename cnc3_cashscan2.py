"""
C&C3 AI Cash - Refined int32 scan + struct proximity analysis (READ-ONLY)
Looks for integer cash values appearing 2-8 times (= player count range).
"""
import ctypes
import ctypes.wintypes as wt
import struct
import sys
from collections import defaultdict

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFO = 0x0400
MEM_COMMIT = 0x1000
PAGE_NOACCESS = 0x01
PAGE_GUARD = 0x100
LIST_MODULES_ALL = 0x03

kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi


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


PID = 22756

hProc = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, False, PID)
if not hProc:
    print(f"OpenProcess failed: {kernel32.GetLastError()}")
    sys.exit(1)

# ── Find cnc3game.dat module boundaries ───────────────────────────────────────
HMODULE = ctypes.c_size_t
hMods = (HMODULE * 1024)()
cbNeeded = wt.DWORD(0)
psapi.EnumProcessModulesEx(
    hProc, hMods, ctypes.sizeof(hMods),
    ctypes.byref(cbNeeded), LIST_MODULES_ALL
)
n_mods = cbNeeded.value // ctypes.sizeof(HMODULE)

cnc3_base = cnc3_end = 0
for i in range(n_mods):
    name_buf = ctypes.create_string_buffer(512)
    psapi.GetModuleFileNameExA(hProc, ctypes.c_void_p(hMods[i]), name_buf, 512)
    name = name_buf.value.decode(errors='replace').split('\\')[-1].lower()
    if name == 'cnc3game.dat':
        cnc3_base = hMods[i]
        # get module size from PE header
        pe_hdr = (ctypes.c_char * 256)()
        rd = ctypes.c_size_t(0)
        kernel32.ReadProcessMemory(
            hProc, ctypes.c_void_p(cnc3_base),
            pe_hdr, 256, ctypes.byref(rd)
        )
        raw = bytes(pe_hdr[:rd.value])
        if raw[:2] == b'MZ':
            e_lfanew = struct.unpack_from('<I', raw, 0x3C)[0]
            hdr = (ctypes.c_char * 512)()
            kernel32.ReadProcessMemory(
                hProc, ctypes.c_void_p(cnc3_base + e_lfanew),
                hdr, 512, ctypes.byref(rd)
            )
            h = bytes(hdr[:rd.value])
            if h[:4] == b'PE\x00\x00':
                img_size = struct.unpack_from('<I', h, 0x50)[0]
                cnc3_end = cnc3_base + img_size
        break

print(
    f"cnc3game.dat: 0x{cnc3_base:X} → 0x{cnc3_end:X}  "
    f"({(cnc3_end - cnc3_base) // 1024 // 1024} MB)"
)

# ── Collect RW data regions ───────────────────────────────────────────────────
mbi = MEMORY_BASIC_INFORMATION()
addr = 0
regions = []
while kernel32.VirtualQueryEx(
        hProc, ctypes.c_void_p(addr),
        ctypes.byref(mbi), ctypes.sizeof(mbi)
):
    if (mbi.State == MEM_COMMIT
            and mbi.Protect not in (PAGE_NOACCESS,)
            and not (mbi.Protect & PAGE_GUARD)
            and mbi.RegionSize >= 4096
            and mbi.Protect in (0x02, 0x04)):  # RW only (not exec pages)
        regions.append((mbi.BaseAddress, mbi.RegionSize))
    nxt = mbi.BaseAddress + mbi.RegionSize
    if nxt <= addr: break
    addr = nxt

total_mb = sum(r[1] for r in regions) / 1024 / 1024
print(f"Scanning {len(regions)} RW data regions ({total_mb:.1f} MB) for int32 cash...\n")

# ── Scan for int32 cash values ────────────────────────────────────────────────
# C&C3 AI income ticks at ~4 credits/sec; typical range in active game:
CASH_MIN = 1000
CASH_MAX = 75000

READ_CHUNK = 4 * 1024 * 1024
read_count = ctypes.c_size_t(0)
value_map = defaultdict(list)  # cash_value → [addresses]

for base, size in regions:
    offset = 0
    while offset < size:
        chunk = min(READ_CHUNK, size - offset)
        buf = (ctypes.c_char * chunk)()
        ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(base + offset),
                                        buf, chunk, ctypes.byref(read_count))
        if not ok or read_count.value < 4:
            offset += chunk
            continue
        raw = bytes(buf[:read_count.value])
        for i in range(0, len(raw) - 3, 4):
            v = struct.unpack_from('<i', raw, i)[0]
            if CASH_MIN <= v <= CASH_MAX:
                value_map[v].append(base + offset + i)
        offset += chunk

# ── Filter: values appearing 2–8 times (= small player group) ─────────────────
print("─── int32 cash candidates appearing 2–8 times ───")
print(f"  {'Cash':>8}  {'N':>3}  Addresses")

player_candidates = []
for val in sorted(value_map.keys()):
    addrs = value_map[val]
    if 2 <= len(addrs) <= 8:
        player_candidates.append((val, addrs))

# Sort by count descending, then by value
player_candidates.sort(key=lambda x: (-len(x[1]), x[0]))

printed = 0
for val, addrs in player_candidates:
    addr_str = '  '.join(f"0x{a:X}" for a in addrs)
    print(f"  {val:>8}  {len(addrs):>3}  {addr_str}")
    printed += 1
    if printed >= 60:
        print(f"  ... ({len(player_candidates) - printed} more)")
        break

# ── Check if any candidates are in heap (not module image) ───────────────────
print("\n─── Candidates NOT in cnc3game.dat image (heap/dynamic) ───")
print(f"  {'Cash':>8}  {'N':>3}  Addresses")
heap_cands = []
for val, addrs in player_candidates:
    heap_addrs = [a for a in addrs if not (cnc3_base <= a < cnc3_end)]
    if 2 <= len(heap_addrs) <= 8:
        heap_cands.append((val, heap_addrs))

heap_cands.sort(key=lambda x: (-len(x[1]), x[0]))
for val, addrs in heap_cands[:40]:
    addr_str = '  '.join(f"0x{a:X}" for a in addrs)
    print(f"  {val:>8}  {len(addrs):>3}  {addr_str}")

# ── Look for N same-value int32s tightly packed (struct array) ─────────────────
print("\n─── Tightly packed equal int32s (≤256 bytes apart) — player cash array ───")
for val, addrs in heap_cands:
    sorted_a = sorted(addrs)
    for j in range(len(sorted_a) - 1):
        gap = sorted_a[j + 1] - sorted_a[j]
        if gap <= 256:
            print(f"  cash={val}  gap={gap} bytes  "
                  f"0x{sorted_a[j]:X} → 0x{sorted_a[j + 1]:X}")

kernel32.CloseHandle(hProc)
print("\nDone.")
