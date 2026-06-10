import ctypes, ctypes.wintypes as wt, struct
kernel32 = ctypes.windll.kernel32
hProc = kernel32.OpenProcess(0x0410, False, 11744)
rd = ctypes.c_size_t(0)

def r4(a):
    b = (ctypes.c_char*4)()
    ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(int(a)), b, 4, ctypes.byref(rd))
    return struct.unpack('<i', bytes(b))[0] if (ok and rd.value==4) else None

def rblock(a, sz):
    b = (ctypes.c_char*sz)()
    ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(int(a)), b, sz, ctypes.byref(rd))
    return bytes(b[:rd.value]) if ok else b''

# Top candidates from last scan (income + spend pattern)
TOP = [
    0x14A230B4, 0x14A230D8,   # score=20, income ~726 cr/min, spend
    0x3110238,                 # score=16, slow steady grow +24,+26
    0xD53050C,                 # score=16, tiny grow +5,+7
    0x1502FCE4,                # score=16, grow
]

print("Live reads:")
for a in TOP:
    v = r4(a)
    print(f"  0x{a:X}  = {v}")

best = 0x14A230B4
BEFORE, AFTER = 256, 512
raw = rblock(best - BEFORE, BEFORE + AFTER)

print(f"\nSTRUCT DUMP  0x{best-BEFORE:X}  cash@+{BEFORE}  ({len(raw)} bytes)")
print(f"  {'off':>7}  {'addr':>12}  {'i32':>10}  {'hex':>10}  note")
print("  " + "-"*55)

for i in range(0, len(raw)-3, 4):
    vi = struct.unpack_from('<i', raw, i)[0]
    vu = struct.unpack_from('<I', raw, i)[0]
    vf = struct.unpack_from('<f', raw, i)[0]
    ca  = best - BEFORE + i
    rel = i - BEFORE
    near = abs(rel) <= 128
    note = ""
    if rel == 0:                       note = "<<< CASH"
    elif vi == -1:                     note = "sentinel"
    elif 0x400000 <= vu <= 0x0FFFFFFF: note = "ptr"
    elif vi == 0:                      note = ""
    elif 1 <= vi <= 8:                 note = "small(%d)" % vi
    elif 100 <= vi <= 100000:          note = "resource"
    elif 0 < vf < 100000 and abs(vf-round(vf)) < 1: note = "f~%.0f" % vf
    sig = bool(note) or (1 <= vi <= 500000)
    if not (near or sig): continue
    rel_s = "CASH" if rel==0 else ("%+d" % rel)
    print("  %7s  0x%10X  %10d  0x%08X  %s" % (rel_s, ca, vi, vu, note))

print()
print("HEX DUMP +-64 bytes around cash")
ctx_raw = rblock(best-64, 132)
for row in range(0, len(ctx_raw), 16):
    ab = best-64+row; rb = ctx_raw[row:row+16]
    hp = ' '.join("%02X" % b for b in rb)
    ap = ''.join(chr(b) if 32<=b<127 else '.' for b in rb)
    mk = "<" if best-64+row <= best < best-64+row+16 else " "
    print("  %s 0x%08X  %-47s  %s" % (mk, ab, hp, ap))

print()
print("ALSO checking 0x14A230D8 (duplicate candidate):")
raw2 = rblock(0x14A230D8 - 64, 132)
print("  Offset from 0x14A230B4 to 0x14A230D8 =", 0x14A230D8 - 0x14A230B4, "bytes")
print("  Values at -4,0,+4:")
for off in (-4, 0, 4, 8, 12, 16):
    v = r4(0x14A230D8 + off)
    print("    %+4d  0x%X  = %s" % (off, 0x14A230D8+off, v))

kernel32.CloseHandle(hProc)
