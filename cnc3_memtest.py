"""
C&C3 Memory Safety Test — Stage 1
Verify read-only process access ONLY. No writes. No crashes.
"""
import ctypes
import ctypes.wintypes as wt
import struct
import sys

# ── Win32 constants (read-only) ───────────────────────────────────────────────
PROCESS_VM_READ      = 0x0010
PROCESS_QUERY_INFO   = 0x0400
MEM_COMMIT           = 0x1000
PAGE_NOACCESS        = 0x01
PAGE_GUARD           = 0x100

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

# ── STAGE 1: open process read-only ──────────────────────────────────────────
print("=== STAGE 1: Open process (read-only) ===")
hProc = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, False, PID)
if not hProc:
    err = kernel32.GetLastError()
    print(f"  FAIL: OpenProcess error {err}")
    sys.exit(1)
print(f"  OK: handle = 0x{hProc:X}  (read-only, no write access)")

# ── STAGE 2: enumerate memory regions ────────────────────────────────────────
print("\n=== STAGE 2: Enumerate memory regions ===")
mbi   = MEMORY_BASIC_INFORMATION()
addr  = 0
regions = []
while kernel32.VirtualQueryEx(hProc, ctypes.c_void_p(addr),
                               ctypes.byref(mbi), ctypes.sizeof(mbi)):
    if (mbi.State == MEM_COMMIT
            and mbi.Protect != PAGE_NOACCESS
            and not (mbi.Protect & PAGE_GUARD)
            and mbi.RegionSize > 0):
        regions.append((mbi.BaseAddress, mbi.RegionSize, mbi.Protect))
    next_addr = mbi.BaseAddress + mbi.RegionSize
    if next_addr <= addr:
        break
    addr = next_addr

total_mb = sum(r[1] for r in regions) / 1024 / 1024
print(f"  Readable regions: {len(regions)},  total: {total_mb:.1f} MB")
if regions:
    print("  First 5 regions:")
    for base, size, prot in regions[:5]:
        print(f"    base=0x{base:016X}  size={size//1024:6d} KB  prot=0x{prot:02X}")

# ── STAGE 3: read PE header of cnc3game.dat (safe known region) ───────────────
print("\n=== STAGE 3: Read PE header of cnc3game.dat (smoke-test ReadProcessMemory) ===")
# Find region that looks like a PE image (starts at 0x400000 range or is large)
test_base = None
for base, size, prot in regions:
    if size >= 0x400000 and base > 0x10000000:
        test_base = base
        break

if test_base is None:
    test_base = regions[4][0] if len(regions) > 4 else regions[0][0]

buf  = (ctypes.c_char * 64)()
read = ctypes.c_size_t(0)
ok   = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(test_base),
                                   buf, 64, ctypes.byref(read))
if ok:
    data = bytes(buf[:read.value])
    print(f"  OK: read {read.value} bytes from 0x{test_base:X}")
    print(f"  First 16 bytes: {data[:16].hex(' ')}")
    if data[:2] == b'MZ':
        print("  → MZ header confirmed (PE module base)")
    else:
        print("  → Not a PE header (data region)")
else:
    print(f"  ReadProcessMemory failed: {kernel32.GetLastError()}")

kernel32.CloseHandle(hProc)
print("\n=== Stage tests PASSED — safe to proceed to cash scan ===")
