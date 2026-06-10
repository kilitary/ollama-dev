"""Final cash comparison — all players, identify human by rate & amount."""
import ctypes, ctypes.wintypes as wt, struct, time
kernel32 = ctypes.windll.kernel32
hProc = kernel32.OpenProcess(0x0410, False, 22756)

def r4(a):
    b = (ctypes.c_char*4)(); rd = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(a), b, 4, ctypes.byref(rd))
    return struct.unpack('<i', bytes(b))[0] if ok and rd.value == 4 else None

def ctx(a, n=12):
    base = a - n*4; sz = (n*2+1)*4
    b = (ctypes.c_char*sz)(); rd = ctypes.c_size_t(0)
    kernel32.ReadProcessMemory(hProc, ctypes.c_void_p(base), b, sz, ctypes.byref(rd))
    raw = bytes(b[:rd.value])
    return [(base+i, struct.unpack_from('<i', raw, i)[0]) for i in range(0, len(raw)-3, 4)]

TARGETS = [
    ('CA7FACC outlier',   0xCA7FACC),
    ('41B1AE8 AI-2A',     0x41B1AE8),
    ('41BE388 AI-2B',     0x41BE388),
    ('522A88C AI-3',      0x522A88C),
    ('56F0590 AI-5',      0x56F0590),
    ('6113D60 AI-6',      0x6113D60),
    ('6353580 AI-9',      0x6353580),
    ('6490088 AI-14',     0x6490088),
    ('C9F8998 AI-15A',    0xC9F8998),
    ('CBCFA18 AI-18',     0xCBCFA18),
    ('CBE96C0 AI-19',     0xCBE96C0),
]

s = [{}, {}, {}]
for lbl, a in TARGETS:
    s[0][lbl] = r4(a)
time.sleep(2)
for lbl, a in TARGETS:
    s[1][lbl] = r4(a)
time.sleep(2)
for lbl, a in TARGETS:
    s[2][lbl] = r4(a)

print("All-player 3-sample cash (2s apart):")
print(f"  {'Label':<22} {'addr':>10}  {'t=0':>8}  {'t=2s':>8}  {'t=4s':>8}  {'delta':>7}  {'cr/min':>8}  note")
print("  " + "-"*95)
for lbl, a in TARGETS:
    v0, v1, v2 = s[0][lbl], s[1][lbl], s[2][lbl]
    if None in (v0, v1, v2):
        continue
    d  = v2 - v0
    rm = d / 4 * 60
    flag = " HUMAN?" if (v1 < v0 or v2 < v1) else ""
    print(f"  {lbl:<22} 0x{a:08X}  {v0:>8}  {v1:>8}  {v2:>8}  {d:>+7}  {rm:>8.0f}{flag}")

print("\nContext around 0xCA7FACC (outlier player):")
for ca, cv in ctx(0xCA7FACC):
    m = " <<< CASH" if ca == 0xCA7FACC else ""
    if -1 < cv < 300_000:
        print(f"  0x{ca:08X}  {cv:>10}{m}")

print("\nContext around 0x41B1AE8 (AI sample):")
for ca, cv in ctx(0x41B1AE8):
    m = " <<< here" if ca == 0x41B1AE8 else ""
    if -1 < cv < 300_000:
        print(f"  0x{ca:08X}  {cv:>10}{m}")

kernel32.CloseHandle(hProc)
